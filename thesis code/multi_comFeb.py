"""
ROS 2 communication layer for the DRL exploration environment.

This module implements `Communication_Interface`, the bridge between the Gym
environment (`simple_gymFeb.XelonaEnv`) and the ROS 2 robotic stack. It owns
every ROS 2 endpoint the environment needs -- publishers, subscribers, service
clients and action clients -- and exposes three high-level operations used by
the Gym interface:

  * reset_com()                          -- reset the simulation for a new
                                            episode (choose a world/spawn, reload
                                            the SLAM pose graph, respawn the robot)
  * execute_first_small_step()           -- take the first small motion and
                                            return the initial observation
  * execute_action_and_compute_reward()  -- send a goal to Nav2, execute it as a
                                            sequence of sub-paths, and compute the
                                            reward and next observation

Supporting ROS 2 nodes wrap the individual subsystems: SLAM Toolbox, Nav2,
Gazebo spawn/delete/physics services, odometry, the occupancy-grid map, and a
navigation-feedback monitor. A file-lock based watchdog (`try_global_reset`)
coordinates a single global restart when an environment becomes unhealthy.
"""

# region Project Imports
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, Pose2D, Pose, PoseWithCovarianceStamped
from std_srvs.srv import Empty
import tf_transformations
from slam_toolbox.srv import Pause, DeserializePoseGraph, SaveMap
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import String
import subprocess
import random
import cv2
import gymnasium as gym
from gymnasium import spaces
import time
import pdb
from matplotlib.patches import Rectangle
from PIL import Image
import io
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from rclpy.context import Context
import os
import threading
import logging
import math
from nav2_msgs.action._navigate_to_pose import NavigateToPose_FeedbackMessage
import errno
from tf2_ros import TransformException
import itertools
import copy
# endregion

# Cross-process lock: only one environment performs the global restart.
LOCKFILE = "/tmp/env_reset_lock"


logging.basicConfig(
    filename='my_app10.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def try_global_reset():
    """Atomically acquire the lockfile and, if this process wins, trigger the
    global simulation restart (stop_envs.sh). Returns True if it performed the
    restart, False if another process already holds the lock."""
    try:
        # Atomic create: succeeds only if the file does NOT already exist.
        fd = os.open(LOCKFILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)  # lock persists via file existence

        print("Acquired persistent lock. Performing global reset...")
        os.system("./stop_envs.sh")
        return True

    except OSError as e:
        if e.errno == errno.EEXIST:
            # Another process already triggered the reset.
            print("Persistent lock already exists — skipping global reset.")
            return False
        raise


def extract_world_patch(map_data, map_info, posX, posY, radius_m=0.15, margin_px=2):
    """Extract a square patch of the occupancy grid centered on a world
    coordinate, with a small margin. Cells outside the map are marked -2."""
    resolution = map_info.resolution
    side_px = int((2 * radius_m) / resolution) + 1

    # Allocate a bigger patch (with margin) then crop back.
    big_side = side_px + 2 * margin_px
    big_patch = np.full((big_side, big_side), -2, dtype=np.int8)

    # World coordinates of the top-left of the big patch.
    x0 = posX - radius_m - margin_px * resolution
    y0 = posY - radius_m - margin_px * resolution

    for i in range(big_side):
        for j in range(big_side):
            wx = x0 + j * resolution
            wy = y0 + i * resolution

            map_x = int((wx - map_info.origin.position.x) / resolution)
            map_y = int((wy - map_info.origin.position.y) / resolution)

            if 0 <= map_x < map_data.shape[1] and 0 <= map_y < map_data.shape[0]:
                big_patch[i, j] = map_data[map_y, map_x]

    patch = big_patch[margin_px:-margin_px, margin_px:-margin_px]
    return patch


# ======================================================================
#  ROS 2 nodes wrapping individual subsystems
# ======================================================================
class DistanceMonitor(Node):
    """Subscribes to Nav2 action feedback to track the remaining distance to the
    current goal (used to detect path-planning problems)."""

    def __init__(self):
        super().__init__('distance_monitor')
        self.distance_remaining = -1
        self.sub = self.create_subscription(
            NavigateToPose_FeedbackMessage,
            '/navigate_to_pose/_action/feedback',
            self.feedback_callback,
            10
        )

    def feedback_callback(self, msg):
        self.distance_remaining = msg.feedback.distance_remaining


class CubeSpawner(Node):
    """Spawns cube obstacles ("holes") in Gazebo via the /spawn_entity service."""

    def __init__(self):
        super().__init__('cube_spawner')
        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')

    def spawn_cube(self, pos_list, num_of_holes):
        for name_idx in range(num_of_holes):
            req = SpawnEntity.Request()
            req.name = f'my_cube_{name_idx+1}'
            req.xml = """
    <sdf version="1.6">
    <model name="my_cube">
        <static>false</static>
        <link name="link">
        <collision name="collision">
            <geometry>
            <box><size>1 1 1</size></box>
            </geometry>
        </collision>
        <visual name="visual">
            <geometry>
            <box><size>1 1 1</size></box>
            </geometry>
        </visual>
        </link>
    </model>
    </sdf>
    """
            pose = Pose()
            pose.position.x = pos_list[name_idx][0]
            pose.position.y = pos_list[name_idx][1]
            pose.position.z = 0.5
            pose.orientation.w = 1.0

            req.initial_pose = pose
            req.reference_frame = 'world'

            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                self.get_logger().info(f"Cube spawned successfully with name my_cube_{name_idx}")
            else:
                self.get_logger().error('Failed to spawn cube')


class CubeDeleter(Node):
    """Deletes previously-spawned cube obstacles via /delete_entity."""

    def __init__(self):
        super().__init__('cube_deleter')
        self.client = self.create_client(DeleteEntity, '/delete_entity')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /delete_entity service...')

    def delete_cube(self, num_of_holes):
        for name_idx in range(num_of_holes):
            req = DeleteEntity.Request()
            req.name = f'my_cube_{name_idx + 1}'

            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                self.get_logger().info(f'Cube deleted successfully with name my_cube_{name_idx}')
            else:
                self.get_logger().error('Failed to delete cube')


class PhysicsClientNode(Node):
    """Pauses/unpauses the Gazebo physics engine via /pause_physics and
    /unpause_physics (used to make resets deterministic)."""

    def __init__(self):
        super().__init__('physics_client')

        self.pause_client = self.create_client(Empty, '/pause_physics')
        while not self.pause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /pause_physics service...')

        self.unpause_client = self.create_client(Empty, '/unpause_physics')
        while not self.unpause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /unpause_physics service...')

    def pause_physics(self):
        req = Empty.Request()
        future = self.pause_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Physics paused')
        else:
            self.get_logger().error('Failed to pause physics')

    def unpause_physics(self):
        req = Empty.Request()
        future = self.unpause_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Physics resumed')
        else:
            self.get_logger().error('Failed to unpause physics')


class MapImageSaver(Node):
    """Subscribes to /map, keeps the latest occupancy grid, and turns it into
    the grayscale observation image. Also monitors the map->odom distance as a
    localization-health signal."""

    def __init__(self):
        super().__init__('map_saver_step')
        self.accept_updates = True
        self.sub_ = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.prev_map_data = None
        self.prev_map_info = None
        self.map_dist = 0.0
        self.localize_error = False

        # TF listener to get the robot pose.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def map_callback(self, msg):
        if not self.accept_updates:
            return

        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_dist = self.get_map_odom_distance()
        if self.map_dist > 0.55:
            self.localize_error = True

        self.map_info = msg.info

    def get_map_odom_distance(self) -> float:
        """Translational distance between the map and odom frames -- the
        correction SLAM applies over raw odometry. A large value indicates
        degraded localization."""
        try:
            t = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            return math.sqrt(x * x + y * y)
        except TransformException:
            return float('nan')

    # region AlignMaps, world_to_map_coords_Data, get_robot_pos and saves
    def world_to_map_coords(self, x, y):
        origin = self.map_info.origin.position
        resolution = self.map_info.resolution
        map_x = int((x - origin.x) / resolution)
        map_y = int((y - origin.y) / resolution)
        return map_x, map_y

    def world_to_map_coords_Data(self, x, y):
        origin = self.prev_map_info.origin.position
        resolution = self.prev_map_info.resolution
        map_x = int((x - origin.x) / resolution)
        map_y = int((y - origin.y) / resolution)
        return map_x, map_y

    def get_robot_position(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', now, timeout=rclpy.duration.Duration(seconds=1.0)
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            return x, y
        except Exception as e:
            self.get_logger().warning(f"Could not get transform: {e}")
            return None

    def save_txt_form(self, filename):
        np.savetxt(filename, self.map_data, fmt='%4d', delimiter=' ')
    # endregion

    def check_exploration_reward_Data(self, posX, posY, radius_m=0.15, need_image=False):
        """Extract the patch around a coordinate from the PREVIOUS map (before
        the action). Optionally also render it as an image."""
        if self.prev_map_data is None or self.prev_map_info is None:
            return 0, None

        patch = extract_world_patch(self.prev_map_data, self.prev_map_info, posX, posY, radius_m)

        if need_image:
            image = np.zeros_like(patch, dtype=np.uint8)
            image[patch == 0] = 255
            image[patch == 100] = 0
            image[patch == -1] = 127
            image[patch == -2] = 64
        else:
            image = None

        return image, patch

    def check_robot_exploration_status(self, posX, posY, radius_m=0.15, need_image=False):
        """Extract the patch around a coordinate from the CURRENT map (after the
        action). Optionally also render it as an image."""
        if self.map_data is None or self.map_info is None:
            return 0, None

        patch = extract_world_patch(self.map_data, self.map_info, posX, posY, radius_m)

        if need_image:
            image = np.zeros_like(patch, dtype=np.uint8)
            image[patch == 0] = 255
            image[patch == 100] = 0
            image[patch == -1] = 127
            image[patch == -2] = 64
        else:
            image = None

        return image, patch

    def GetMapCoverage(self):
        """Fraction of known (non -1) cells in the current map."""
        covered_pixels = np.count_nonzero(self.map_data != -1)
        numOfCells = self.map_data.size
        cover_metric = covered_pixels / numOfCells
        return cover_metric

    def isMapCovered(self, num_of_holes):
        """True if map coverage exceeds a threshold (lowered when obstacles are
        present, since holes reduce reachable free space)."""
        map_covered_threshhold = 0.96 - 0.035 * num_of_holes
        covered_pixels = np.count_nonzero(self.map_data != -1)
        numOfCells = self.map_data.size
        cover_metric = covered_pixels / numOfCells

        print(f" ---->Cover Metric is : {cover_metric}<----")
        if cover_metric > map_covered_threshhold:
            return True
        else:
            return False

    def plotnp_image11_obs(self, posX, posY, radius_m=0.35, ros_domainid=100):
        """Build the (1, 64, 64) grayscale observation: occupancy grid mapped to
        grayscale with the robot's position drawn as a circular mark, rendered
        and resized to 64x64."""
        if self.map_data is None or self.map_info is None:
            print("Map data or info not available.")
            return np.zeros((1, 64, 64), dtype=np.uint8)  # fallback

        pos = (posX, posY)

        # Occupancy values -> grayscale image.
        image = np.zeros_like(self.map_data, dtype=np.uint8)
        image[self.map_data == 0] = 255   # free -> white
        image[self.map_data == 100] = 0   # occupied -> black
        image[self.map_data == -1] = 127  # unknown -> gray

        # World -> map coordinates for the robot position.
        map_x, map_y = self.world_to_map_coords(*pos)
        map_x, map_y = int(map_x), int(map_y)

        # Radius in pixels.
        resolution = self.map_info.resolution
        radius_px = int(radius_m / resolution)

        # Paint a circular mark for the robot's position.
        for y in range(max(0, map_y - radius_px), min(self.map_data.shape[0], map_y + radius_px + 1)):
            for x in range(max(0, map_x - radius_px), min(self.map_data.shape[1], map_x + radius_px + 1)):
                if (x - map_x)**2 + (y - map_y)**2 <= radius_px**2:
                    image[y, x] = 198

        # Render to an off-screen buffer, then load back as grayscale and resize.
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(image, cmap='gray', origin='lower')
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert('L')  # grayscale
        img = img.resize((64, 64), Image.ANTIALIAS)
        obs = np.array(img, dtype=np.uint8)[np.newaxis, :, :]

        return obs

    def plotnp_image(self):
        """Debug helper: display the current map as an image."""
        image = np.zeros_like(self.map_data, dtype=np.uint8)
        image[self.map_data == 0] = 255   # free -> white
        image[self.map_data == 100] = 0   # occupied -> black
        image[self.map_data == -1] = 127  # unknown -> gray

        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray', origin='lower')
        plt.title("Map Snapshot")
        plt.axis('off')
        plt.show()
        print(image.shape)


class SaveMapClient(Node):
    """Client for the SLAM Toolbox /save_map service."""

    def __init__(self):
        super().__init__('map_saver_my_client')
        self.map_save_client = self.create_client(SaveMap, '/slam_toolbox/save_map')
        while not self.map_save_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /..slam../save_map srv response ..')

    def save_curr_map(self, curr_map_name):
        save_current_req = SaveMap.Request()
        save_current_req.name = curr_map_name
        future = self.map_save_client.call_async(save_current_req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Save map Request sent. ')
        if future.result() is not None:
            self.get_logger().info('Save map Done')
        else:
            self.get_logger().info('Error On Save map')


class WaffleHandleNode(Node):
    """Deletes the robot ("waffle") from Gazebo via /delete_entity, used during
    resets before respawning at a new position."""

    def __init__(self):
        super().__init__('my_gazebo_summoner')
        self.eraser_waffle_client = self.create_client(DeleteEntity, '/delete_entity')
        while not self.eraser_waffle_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /delete_entity srv response ..')

    def dlt_waffle(self):
        waffle_dlt_req = DeleteEntity.Request()
        waffle_dlt_req.name = 'waffle'
        future = self.eraser_waffle_client.call_async(waffle_dlt_req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Waffle dlt Request sent. ')
        if future.result() is not None:
            self.get_logger().info('Waffle dlt Done')
        else:
            self.get_logger().info('Error On Waffle Dlt')


class GazClientNode(Node):
    """Client for the Gazebo /reset_world service."""

    def __init__(self):
        super().__init__('my_gazebo_handler')
        self.reseter_gazebo_client = self.create_client(Empty, '/reset_world')
        while not self.reseter_gazebo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /reset_simulation srv response ..')

    def reset_gaz_sim(self):
        self.get_logger().info('Calling /reset_simulation service...')
        gaz_req = Empty.Request()
        future = self.reseter_gazebo_client.call_async(gaz_req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Gaz Request sent. ')
        if future.result() is not None:
            self.get_logger().info('Gaz Reset Done')
        else:
            self.get_logger().info('Error On Gazebo Reset')


class SlamClientNode(Node):
    """Clients for SLAM Toolbox: pause new measurements and deserialize a saved
    pose graph. Holds the per-world pose-graph map names, keyed by world index."""

    def __init__(self):
        super().__init__('my_slam_handler')

        self.pause_slam_client = self.create_client(Pause, '/slam_toolbox/pause_new_measurements')
        while not self.pause_slam_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for Pause_measurements service ..")

        self.deserialize_map_client = self.create_client(DeserializePoseGraph, '/slam_toolbox/deserialize_map')
        while not self.deserialize_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for deserialize map service ..")

        # Saved pose-graph names per world (comments give the world center).
        self.deser_map_name00 = "emptybitmap"   # (0,0) ~ center
        self.deser_map_name = "bitmap_56"       # (-5,6) ~ center
        self.deser_map_name1 = "bitmap56"       # (5,6) ~ center
        self.deser_map_name2 = "bitmap5_6"      # (5,-6) ~ center
        self.deser_map_name3 = "bitmap_5_6"     # (-5,-6) ~ center
        self.deser_map_name4 = "bitmap_15_6"    # (-15,-6) ~ center
        self.deser_map_name5 = "bitmap_156"     # (-15,6) ~ center
        self.deser_map_name6 = "bitmap15_6"     # (15,-6) ~ center

    def pause_slam_metrics(self):
        pause_slam_req = Pause.Request()
        future = self.pause_slam_client.call_async(pause_slam_req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Pause req sent.')
        if future.result() is not None:
            self.get_logger().info('Pause Done')
        else:
            self.get_logger().info('Error On Pause')

    def deser_map_slamp(self, pose2d_robot, isReset=False, curr_world=-1):
        """Deserialize the saved pose graph for the selected world, initializing
        SLAM at the robot's spawn pose."""
        serial_req = DeserializePoseGraph.Request()
        serial_req.filename = " "
        if curr_world == 0:
            serial_req.filename = self.deser_map_name
            print("serialize 0")
        if curr_world == 1:
            serial_req.filename = self.deser_map_name1
            print("serialize 1")
        if curr_world == 2:
            serial_req.filename = self.deser_map_name2
            print("serialize 2")
        if curr_world == 3:
            serial_req.filename = self.deser_map_name3
            print("serialize 3")
        if curr_world == 4:
            serial_req.filename = self.deser_map_name4
            print("serialize 4")
        if curr_world == 5:
            serial_req.filename = self.deser_map_name5
            print("serialize 5")
        if curr_world == 6:
            serial_req.filename = self.deser_map_name6
            print("serialize 6")

        serial_req.match_type = int(1)
        serial_req.initial_pose = pose2d_robot
        future = self.deserialize_map_client.call_async(serial_req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('Deserial req sent.')
        if future.result() is not None:
            self.get_logger().info('Deserial Done')
        else:
            self.get_logger().info('Error On Deserial')


class OdomSubNode(Node):
    """Subscribes to /odom and stores the latest robot pose (as Pose and
    Pose2D, with yaw from the quaternion)."""

    def __init__(self):
        super().__init__('my_odom_sub')
        self.my_odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.pose_stored = Pose()
        self.pose2D_stored = Pose2D()

    def odom_callback(self, msg):
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(orientation_list)

        self.pose2D_stored.x = position.x
        self.pose2D_stored.y = position.y
        self.pose2D_stored.theta = yaw

        self.pose_stored.position = position
        self.pose_stored.orientation = orientation_q


def set_init_pose(nav: BasicNavigator, pos_x, pos_y):
    """Build a map-frame PoseStamped at (pos_x, pos_y) with zero orientation,
    used to set the robot's initial pose after spawn."""
    pose_pos_x = pos_x
    pose_pos_y = pos_y

    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.0)
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = nav.get_clock().now().to_msg()
    initial_pose.pose.position.x = pose_pos_x
    initial_pose.pose.position.y = pose_pos_y
    initial_pose.pose.position.z = 0.01
    initial_pose.pose.orientation.x = qx
    initial_pose.pose.orientation.y = qy
    initial_pose.pose.orientation.z = qz
    initial_pose.pose.orientation.w = qw

    return initial_pose


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch import LaunchService


def spawn_turtlebot(domain_ros, x, y):
    """Spawn the TurtleBot3 in Gazebo at (x, y) via the turtlebot3_gazebo launch
    file, under the given ROS_DOMAIN_ID."""
    os.environ["ROS_DOMAIN_ID"] = str(domain_ros)

    turtlebot3_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    launch_file_path = os.path.join(
        turtlebot3_gazebo_dir, 'launch', 'spawn_turtlebot3.launch.py'
    )

    ld = LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file_path),
            launch_arguments={
                'x_pose': str(x),
                'y_pose': str(y),
            }.items()
        )
    ])

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()


def run_launch_waffle(domain_ros):
    """Alternative spawn via a subprocess `ros2 launch` call."""
    try:
        command = ["ros2", "launch", "turtlebot3_gazebo", "spawn_turtlebot3.launch.py"]
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(domain_ros)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)

        for line in process.stdout:
            print(line, end="")

        process.wait()
        print("***SPAWNING PROCESS FINSHED!***")
    except Exception as e:
        print(f"Error running launch file: {e}")


def plot_with_np(map_data):
    """Debug helper: display an occupancy grid as an image."""
    image = np.zeros_like(map_data, dtype=np.uint8)
    image[map_data == 0] = 255   # free -> white
    image[map_data == 100] = 0   # occupied -> black
    image[map_data == -1] = 127  # unknown -> gray

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray', origin='lower')
    plt.title("Map Snapshot")
    plt.axis('off')
    plt.show()


def action_gym_to_nav(nav: BasicNavigator, gym_action):
    """Convert a 2D goal coordinate into a map-frame Nav2 PoseStamped goal."""
    action_goal_pose = PoseStamped()
    action_goal_pose.header.frame_id = 'map'
    action_goal_pose.header.stamp = nav.get_clock().now().to_msg()
    gx, gy, gz, gw = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.0)
    print(f"Action's orientation is :  {gx}, {gy}, {gz} ,{gw} ")
    action_goal_pose.pose.orientation.x = gx
    action_goal_pose.pose.orientation.y = gy
    action_goal_pose.pose.orientation.z = gz
    action_goal_pose.pose.orientation.w = gw
    action_goal_pose.pose.position.x = float(gym_action[0])
    action_goal_pose.pose.position.y = float(gym_action[1])
    action_goal_pose.pose.position.z = 0.0

    return action_goal_pose


# ======================================================================
#  Communication interface: the bridge between Gym and the ROS 2 stack
# ======================================================================
class Communication_Interface:
    """Owns every ROS 2 endpoint the environment needs and implements reset,
    first-step, and action-execution/reward for one environment instance
    (identified by its ROS domain id)."""

    def __init__(self, my_domain_id):
        print(f"Hello {my_domain_id}")
        logging.info("------------------------------")
        logging.info("Env Initialized")
        logging.info("------------------------------")
        rclpy.init(domain_id=my_domain_id)
        self.envResetHealth = True

        self.domain_ros = my_domain_id
        self.temp_filename = f"temp_read{my_domain_id}"
        self.resize_x = 64
        self.resize_y = 64
        self.first_step_flag = True

        # Create instances of all communication nodes.
        self.curr_holes_num = 0
        self.holeKill = CubeDeleter()
        self.holeSpawn = CubeSpawner()
        self.map_handle = MapImageSaver()
        self.gaz_reseter = GazClientNode()
        self.slam_handler = SlamClientNode()
        self.odom_info_getter = OdomSubNode()
        self.waffle_dlter = WaffleHandleNode()
        self.navigation_info = DistanceMonitor()
        self.nav = BasicNavigator()
        self.prev_action = np.array([0.0, 0.0])
        self.spawn_x_axis = None
        self.physics_handler = PhysicsClientNode()
        self.reset_done = False
        # Local origins of the training worlds packed into a single .world file.
        self.local_origins = np.array([(-5.0, 6.0), (5.0, 6.0), (5.0, -6.0), (-5.0, -6.0)])
        self.curr_local_origin = (0.0, 0.0)
        self.circular_idx = 0
        self.crucial_point = False  # used to change direction in execute_first_small_step()
        self.holePositions = []
        self.localization_samples = []

        # Candidate obstacle ("hole") coordinates per world.
        self.holes_per_world = {
            0: {"coords": np.array([(+3.0, -3.0), (0.0, +1.5)])},
            1: {"coords": np.array([(-3.0, +3.0), (+3.0, +3.0), (+3.0, -3.0), (-3.0, -3.0), (0.0, 0.0)])},
            2: {"coords": np.array([(-1.5, 0.0), (3.0, 3.0)])},
            3: {"coords": np.array([(1.5, 0.0), (3.0, 0.0)])},
            4: {"coords": np.array([(-3.0, +3.0), (+3.0, +3.0), (+3.0, -3.0), (-3.0, -3.0)])},
            5: {"coords": np.array([(-3.0, 0.0), (0.0, 1.5)])},
            6: {"coords": np.array([(-3.0, +3.0), (+3.0, +3.0), (+3.0, -3.0), (-3.0, -3.0)])}
        }
        self.dont_sample = False

        # 5x5 grid of goal coordinates; corner goals are pulled in slightly so
        # they stay inside the walls.
        step = 1.5
        values = np.arange(-3, 4, step)
        self.action_coords = np.array(list(itertools.product(values, values)), dtype=np.float32)
        mask = self.action_coords[:, 1] == -3
        self.action_coords[mask, 1] = -2.7
        mask = self.action_coords[:, 0] == -3
        self.action_coords[mask, 0] = -2.7
        mask = self.action_coords[:, 1] == 3
        self.action_coords[mask, 1] = 2.7
        mask = self.action_coords[:, 0] == 3
        self.action_coords[mask, 0] = 2.7

    def close_communication(self):
        """Destroy clients/subscriptions and shut down rclpy."""
        self.gaz_reseter.destroy_client(self.gaz_reseter.reseter_gazebo_client)
        self.slam_handler.destroy_client(self.slam_handler.pause_slam_client)
        self.slam_handler.destroy_client(self.slam_handler.deserialize_map_client)
        self.odom_info_getter.destroy_subscription(self.odom_info_getter.my_odom_sub)
        self.navigation_info.destroy_node()
        self.map_handle.destroy_subscription(self.map_handle.sub_)
        rclpy.shutdown()

    def reset_com(self):
        """Reset the simulation for a new episode: pick a world and a valid spawn
        position, delete/respawn the robot, reload the corresponding SLAM pose
        graph, and clear the costmaps. A watchdog triggers a global restart if
        the reset takes too long."""
        self.localization_samples = []
        
        # Delete previously spawned obstacles if any.
        if self.curr_holes_num != 0:
            self.KillPrevRandHoles(self.curr_holes_num)

        self.crucial_point = False
        self.map_handle.localize_error = False
        self.reset_done = False

        # Choose a world.
        self.circular_idx = self.circular_idx + 1
        rand_world_idx = np.random.randint(len(self.local_origins))
       

        if self.circular_idx == len(self.local_origins):
            self.circular_idx = 0
        print(f"RAND WORLD IS : {rand_world_idx}")

        # Choose the local origin => current world.
        self.curr_local_origin = self.local_origins[rand_world_idx]

        # Choose a spawn position within the selected world.
        spawn_pairs = np.array([(0.0, 0.0), (3.0, 3.0), (-3.0, 3.0), (3.0, -3.0), (-3.0, -3.0)])
        spawn_yaw = 0.0

        rand_spawn_idx = np.random.randint(len(spawn_pairs))
        spawn_coords = spawn_pairs[rand_spawn_idx]

        # Some spawn positions require a special small first-move direction.
        if rand_world_idx == 2:
            if (rand_spawn_idx == 0 or rand_spawn_idx == 2):
                self.crucial_point = True
        elif rand_world_idx == 1:
            if rand_spawn_idx == 2:
                self.crucial_point = True
        elif rand_world_idx == 3:
            if rand_spawn_idx == 2:
                self.crucial_point = True

        spawn_coords_global = spawn_coords
        spawn_coords = spawn_coords + self.curr_local_origin

        self.prev_action = np.array(spawn_coords)
        spawn_x = spawn_coords[0]
        spawn_y = spawn_coords[1]
        self.spawn_x_axis = spawn_x

        timeout_sec = 40 + int(self.domain_ros) * 2
        health_flag = False

        def watchdog():
            """Timer thread that triggers a global restart if reset takes too long."""
            time.sleep(timeout_sec)
            if not health_flag:
                self.envResetHealth = False
                print(f"\033[31m[WARN] Reset taking longer than {timeout_sec}s — triggering restart!\033[0m")
                logging.info("Restart Simulation Detected - reset")
                logging.warning(f"reset with holes {self.holePositions}, and world : {rand_world_idx}, with spawn : {spawn_coords}")
                try_global_reset()
                return self.envResetHealth, False

        timer_thread = threading.Thread(target=watchdog, daemon=False)
        timer_thread.start()

        self.slam_handler.pause_slam_metrics()

        self.nav.cancelTask()
        self.nav.waitUntilNav2Active()
        self.nav.clearAllCostmaps()

        self.physics_handler.pause_physics()
        self.waffle_dlter.dlt_waffle()

        time.sleep(2)
        self.physics_handler.unpause_physics()

        # Optionally spawn random obstacles (disabled here: spawn_holes_rand = 0).
        spawn_holes_rand = 0
        self.curr_holes_num = spawn_holes_rand
        if spawn_holes_rand != 0:
            self.SpawnHolesToRandPos(spawn_coords_global, self.holes_per_world[rand_world_idx]["coords"], num=spawn_holes_rand)

        # Spawn the robot.
        spawn_turtlebot(self.domain_ros, spawn_x, spawn_y)
        time.sleep(2)

        init_pose2d = Pose2D()
        init_pose2d.x = spawn_x
        init_pose2d.y = spawn_y
        init_pose2d.theta = spawn_yaw
        self.nav.setInitialPose(set_init_pose(self.nav, spawn_x, spawn_y))

        self.physics_handler.pause_physics()
        self.slam_handler.deser_map_slamp(init_pose2d, False, curr_world=rand_world_idx)
        self.physics_handler.unpause_physics()

        time.sleep(2)
        self.slam_handler.pause_slam_metrics()
        time.sleep(1)
        self.map_handle.prev_map_data = None
        self.map_handle.prev_map_info = None

        self.first_step_flag = True

        health_flag = True
        self.nav.clearGlobalCostmap()
        self.nav.clearLocalCostmap()

        self.nav.waitUntilNav2Active()
        self.reset_done = True
        return self.envResetHealth

    def execute_first_small_step(self):
        """Take a small first motion after reset and return the first observed
        map image. A watchdog triggers a global restart if it stalls."""
        if not self.reset_done:
            print("Reset was not completed before the first step.")
            return

        timeout_sec = 25.0 + int(self.domain_ros) * 2
        health_flag = False

        def watchdog():
            """Timer thread that triggers a global restart if the step stalls."""
            time.sleep(timeout_sec)
            if not health_flag:
                self.envResetHealth = False
                print(f"\033[31m[WARN] Reset step taking longer than {timeout_sec}s — triggering restart!\033[0m")
                logging.info("Restart Simulation Detected - small step")
                try_global_reset()
                return None, self.envResetHealth

        timer_thread = threading.Thread(target=watchdog, daemon=False)
        timer_thread.start()

        rclpy.spin_once(self.odom_info_getter)
        self.MoveSlighly(self.odom_info_getter)

        init_pose2d = self.odom_info_getter.pose2D_stored
        next_state = self.map_handle.plotnp_image11_obs(init_pose2d.x, init_pose2d.y, ros_domainid=self.domain_ros)
        health_flag = True
        return next_state, self.envResetHealth

    def MoveSlighly(self, odom: OdomSubNode):
        """Nudge the robot ~0.6 m in a direction chosen from the spawn geometry,
        so the first observation is not taken from a standstill."""
        print("Try to move slightly.")
        start = time.time()
        timeout = 1.0

        while time.time() - start < timeout:
            rclpy.spin_once(odom, timeout_sec=0.1)

        init_pose = odom.pose_stored
        first_goal_pose = PoseStamped()
        first_goal_pose.header.frame_id = 'map'
        first_goal_pose.header.stamp = self.nav.get_clock().now().to_msg()
        first_goal_pose.pose = init_pose

        if self.crucial_point:
            first_goal_pose.pose.position.y = first_goal_pose.pose.position.y - float(0.6)
        else:
            if self.spawn_x_axis == self.curr_local_origin[0] or self.spawn_x_axis < self.curr_local_origin[0]:
                first_goal_pose.pose.position.x = first_goal_pose.pose.position.x + float(0.6)
            else:
                first_goal_pose.pose.position.x = first_goal_pose.pose.position.x - float(0.6)

        self.nav.goToPose(first_goal_pose)
        while not self.nav.isTaskComplete():
            rclpy.spin_once(self.map_handle)

        return

    def execute_action_and_compute_reward(self, gym_action):
        """Execute a selected goal through Nav2 (as a sequence of sub-paths) and
        compute the reward. Reward combines local exploration gain with penalties
        for localization failure, hitting an obstacle, or a non-progressive
        (repeated) action. Returns reward, next observation, health flag,
        localization-lost flag, success flag, and current coverage."""
        holes_exists = False
        running = True
        # Convert the goal into the current world's local frame.
        local_gym_action = np.add(gym_action, np.array(self.curr_local_origin))

        timeout_sec = 180 + int(self.domain_ros) * 2
        done_event = threading.Event()

        def localization_monitor():
            """Sample the map->odom distance once per second while the action runs."""
            while running:
                if self.map_handle.map_dist < 1.6 and (not self.dont_sample):
                    self.localization_samples.append(self.map_handle.map_dist)
                time.sleep(1.0)

        def watchdog():
            """Trigger a global restart if the action takes too long."""
            if not done_event.wait(timeout=timeout_sec):
                self.envResetHealth = False
                print(f"\033[31m[WARN] Action taking longer than {timeout_sec}s — triggering restart!\033[0m")
                logging.info("Restart Simulation Detected - exec action")
                running = False
                try_global_reset()
                return 0, None, self.envResetHealth

        threading.Thread(target=watchdog, daemon=False).start()
        thread_localization = threading.Thread(target=localization_monitor, daemon=False)
        thread_localization.start()

        # Let the map settle briefly before capturing the "previous" state.
        time_wait_after_move = 0.5
        end_time = time.time() + time_wait_after_move
        while time.time() < end_time:
            rclpy.spin_once(self.map_handle, timeout_sec=0.05)

        # Store the previous map before sending the goal.
        self.map_handle.prev_map_data = self.map_handle.map_data
        self.map_handle.prev_map_info = self.map_handle.map_info

        # Execute the action (as a sequence of sub-paths).
        notReached = self.ExecActionAsPath(local_gym_action)
        # Let the map settle again if the goal was reached.
        if not notReached:
            time_wait_after_move = 0.5
            end_time = time.time() + time_wait_after_move
            while time.time() < end_time:
                rclpy.spin_once(self.map_handle, timeout_sec=0.05)

        # Local patches (before/after) around the goal, for the exploration reward.
        prev_image, prev_patch = self.map_handle.check_exploration_reward_Data(local_gym_action[0], local_gym_action[1], radius_m=1.0, need_image=False)
        curr_image, curr_patch = self.map_handle.check_robot_exploration_status(local_gym_action[0], local_gym_action[1], radius_m=1.0, need_image=False)

        print(f"----> prev_action : {self.prev_action}")
        ret_localize = False
        ret_success = False
        if len(self.holePositions) > 0:
            holes_exists = True

        # Reward logic (see thesis reward system).
        if np.array_equal(self.prev_action, local_gym_action):
            # Non-progressive (repeated) action -> heavy penalty.
            rew = -5.0
        else:
            if self.map_handle.localize_error:
                if self.map_handle.map_dist > 0.75:
                    # Localization still broken -> penalty and terminate.
                    rew = -0.5
                    logging.info(">>We have localize error")
                    print("Localization issue persists.")
                    ret_localize = True
                    ret_success = False
                else:
                    self.map_handle.localize_error = False
                    rew_explore, blackPx = self.compute_exploration_reward(prev_patch, curr_patch)
                    if rew_explore < 0.6 and rew_explore > 0.15:
                        rew = +0.5
                    elif rew_explore > 0.6:
                        rew = +1.0
                    else:
                        rew = 0.0
                    ret_localize = False
                    if self.map_handle.isMapCovered(self.curr_holes_num):
                        rew = +3.0
                        ret_success = True
            else:
                rew_explore, blackPx = self.compute_exploration_reward(prev_patch, curr_patch)
                if rew_explore < 0.6 and rew_explore > 0.15:
                    rew = +0.5
                elif rew_explore > 0.6:
                    rew = +1.0
                else:
                    rew = 0.0
                ret_localize = False
                if self.map_handle.isMapCovered(self.curr_holes_num):
                    rew = +3.0
                    ret_success = True
        # Penalty for selecting a goal on top of an obstacle.
        if holes_exists:
            if np.any(np.all(self.holePositions == local_gym_action, axis=1)):
                rew = -0.5

        rob_pos_afterGoal = self.getCurrPose()
        next_state = self.map_handle.plotnp_image11_obs(rob_pos_afterGoal.pose.position.x, rob_pos_afterGoal.pose.position.y, ros_domainid=self.domain_ros)
        print(f"Position Reward : {rew}")

        done_event.set()

        if notReached:
            # Mark the previous action as invalid so the next one is not treated
            # as a repeat (matters for obstacle/hole environments).
            temp_act = (-100, -100)
            self.prev_action = np.array(temp_act)
        else:
            self.prev_action = local_gym_action

        running = False
        return rew, next_state, self.envResetHealth, ret_localize, ret_success, self.map_handle.GetMapCoverage()

    def ExecActionAsPath(self, gym_action):
        """Execute the goal not in one piece but as a sequence of shorter
        sub-paths (their number proportional to the distance to the goal). This
        made actions complete more reliably and prevented the robot from getting
        stuck in corners. Returns True if the goal was NOT reached."""
        goal_counts = 0
        odom = self.odom_info_getter
        while True:
            error_detected = False
            goal_pose = action_gym_to_nav(self.nav, gym_action)
            start = time.time()
            timeout = 0.5

            while time.time() - start < timeout:
                rclpy.spin_once(odom, timeout_sec=0.1)
            rob_pos = (odom.pose2D_stored.x, odom.pose2D_stored.y)
            robot_pos = np.array(rob_pos)
            rob_pose = action_gym_to_nav(self.nav, robot_pos)

            waypoints = self.ComputePathPoint(rob_pose, goal_pose)
            if waypoints is None:
                break

            print(len(waypoints))
            if len(waypoints) == 1:
                print(f"Goal added === {goal_counts + 1}")
                goal_counts = goal_counts + 1
                if goal_counts >= 3:
                    return False

            # Send waypoints to Nav2 one at a time.
            for w in waypoints:
                self.nav.goToPose(w)
                while not self.nav.isTaskComplete():
                    rclpy.spin_once(self.navigation_info)
                    rclpy.spin_once(self.odom_info_getter)
                    rclpy.spin_once(self.map_handle)

                    # If the remaining distance grows too large, the plan is bad;
                    # cancel and recompute.
                    if self.navigation_info.distance_remaining > 2.5:
                        self.nav.cancelTask()
                        error_detected = True
                        break
                if error_detected:
                    break
            if not error_detected:
                break

        task_result = self.nav.getResult()
        print(f"Our Task result is  {task_result}")
        if task_result != TaskResult.SUCCEEDED:
            notReached = True
        else:
            notReached = False
        return notReached

    def ComputePathPoint(self, start_pose, goal_pose):
        """Ask Nav2 for a path and subsample it into ~7 waypoints, assigning each
        an orientation toward the next. If no path exists, try nearby goals."""
        myPath = self.nav.getPath(start_pose, goal_pose)
        odom = self.odom_info_getter
        if myPath is None:
            print(f"--- WARN STATUS CODE --- is {self.nav.status} , with type of {type(self.nav.status)}")
            if self.nav.status == int(6):
                myPath = self.tryAroundGoals(goal_pose)
                if myPath is None:
                    return None

        self.nav.cancelTask()
        path = myPath.poses
        n = len(path)
        step = n // 7
        waypoints = []

        if step != 0:
            for i in range(0, n - 1, step):
                p = path[i]

                # Orientation based on the next sampled pose.
                next_p = path[min(i + step, n - 1)]
                dx = next_p.pose.position.x - p.pose.position.x
                dy = next_p.pose.position.y - p.pose.position.y
                yaw = math.atan2(dy, dx)
                q = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)

                p.pose.orientation.x = q[0]
                p.pose.orientation.y = q[1]
                p.pose.orientation.z = q[2]
                p.pose.orientation.w = q[3]

                p.header.frame_id = "map"
                p.header.stamp = self.nav.get_clock().now().to_msg()

                waypoints.append(p)

        # Append the last pose (with goal orientation computed too).
        last = path[-1]
        if n > 1:
            dx = last.pose.position.x - path[-2].pose.position.x
            dy = last.pose.position.y - path[-2].pose.position.y
            yaw = math.atan2(dy, dx)
        else:
            yaw = 0.0

        q = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
        last.pose.orientation.x = q[0]
        last.pose.orientation.y = q[1]
        last.pose.orientation.z = q[2]
        last.pose.orientation.w = q[3]

        last.header.frame_id = "map"
        last.header.stamp = self.nav.get_clock().now().to_msg()

        waypoints.append(last)

        return waypoints

    def compute_exploration_reward(self, prev_patch, curr_patch):
        """Exploration reward: fraction of valid patch cells that turned from
        unknown to known between the previous and current maps."""
        mask_valid = (curr_patch != -2)
        Total_pxNum = np.sum(mask_valid)
        if Total_pxNum == 0:
            return 0.0, 0.0

        case_A = (prev_patch == -1) & (curr_patch != -1) & mask_valid
        case_B = (prev_patch == -2) & mask_valid

        numOfPixels = np.sum(case_A | case_B)
        reward = numOfPixels / Total_pxNum
        return reward, 0.0

    def tryAroundGoals01(self, goal_pose):
        """Fallback: probe four nearby offsets to find a pose from which a path
        to the goal exists."""
        odom = self.odom_info_getter
        init_pose = odom.pose_stored
        self.nav.clearGlobalCostmap()
        self.nav.clearLocalCostmap()
        self.nav.cancelTask()
        slight_step = 1.0

        count = 0
        for i in range(4):
            count = count + 1
            if i == 0:
                x_add = float(slight_step)
                y_add = float(0.0)
                first_goal_pose0 = PoseStamped()
                first_goal_pose0.header.frame_id = 'map'
                first_goal_pose0.header.stamp = self.nav.get_clock().now().to_msg()
                first_goal_pose0.pose = copy.deepcopy(init_pose)
                first_goal_pose0.pose.position.x = first_goal_pose0.pose.position.x + float(x_add)
                first_goal_pose0.pose.position.y = first_goal_pose0.pose.position.y + float(y_add)
            elif i == 1:
                x_add = float(0.0)
                y_add = float(-slight_step)
                first_goal_pose1 = PoseStamped()
                first_goal_pose1.header.frame_id = 'map'
                first_goal_pose1.header.stamp = self.nav.get_clock().now().to_msg()
                first_goal_pose1.pose = copy.deepcopy(init_pose)
                first_goal_pose1.pose.position.x = first_goal_pose1.pose.position.x + float(x_add)
                first_goal_pose1.pose.position.y = first_goal_pose1.pose.position.y + float(y_add)
            elif i == 2:
                x_add = float(-slight_step)
                y_add = float(0.0)
                first_goal_pose2 = PoseStamped()
                first_goal_pose2.header.frame_id = 'map'
                first_goal_pose2.header.stamp = self.nav.get_clock().now().to_msg()
                first_goal_pose2.pose = copy.deepcopy(init_pose)
                first_goal_pose2.pose.position.x = first_goal_pose2.pose.position.x + float(x_add)
                first_goal_pose2.pose.position.y = first_goal_pose2.pose.position.y + float(y_add)
            elif i == 3:
                x_add = float(0.0)
                y_add = float(slight_step)
                first_goal_pose3 = PoseStamped()
                first_goal_pose3.header.frame_id = 'map'
                first_goal_pose3.header.stamp = self.nav.get_clock().now().to_msg()
                first_goal_pose3.pose = copy.deepcopy(init_pose)
                first_goal_pose3.pose.position.x = first_goal_pose3.pose.position.x + float(x_add)
                first_goal_pose3.pose.position.y = first_goal_pose3.pose.position.y + float(y_add)

        for i in range(4):
            if i == 0:
                self.nav.goToPose(first_goal_pose0)
            elif i == 1:
                self.nav.goToPose(first_goal_pose1)
            elif i == 2:
                self.nav.goToPose(first_goal_pose2)
            elif i == 3:
                self.nav.goToPose(first_goal_pose3)

            while not self.nav.isTaskComplete():
                rclpy.spin_once(self.map_handle)
                rclpy.spin_once(self.odom_info_getter)
                q = self.getCurrPose()
                myPath = self.nav.getPath(q, goal_pose)
                if self.nav.status != 6 and myPath is not None:
                    logging.info('Saved!!!')
                    return myPath

        self.tryAroundGoals1(goal_pose)
        return None

    def tryAroundGoals(self, goal_pose):
        """Fallback: circle the robot through eight nearby offsets, checking at
        each whether a path to the goal becomes available."""
        self.dont_sample = True
        self.nav.clearGlobalCostmap()
        self.nav.clearLocalCostmap()
        p = self.getCurrPose()

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.0)
        aroundGoal = PoseStamped()
        aroundGoal.header.frame_id = '/map'
        aroundGoal.header.stamp = self.nav.get_clock().now().to_msg()
        aroundGoal.pose.position.x = p.pose.position.x
        aroundGoal.pose.position.y = p.pose.position.y
        aroundGoal.pose.position.z = 0.01
        aroundGoal.pose.orientation.x = qx
        aroundGoal.pose.orientation.y = qy
        aroundGoal.pose.orientation.z = qz
        aroundGoal.pose.orientation.w = qw

        start_posx = p.pose.position.x
        start_posy = p.pose.position.y

        for i in range(8):
            if (i + 1) % 8 == 0:
                x_add = float(+1.0)
                y_add = float(0.0)
            elif (i + 1) % 8 == 1:
                x_add = float(0.0)
                y_add = float(-1.0)
            elif (i + 1) % 8 == 2:
                x_add = float(0.0)
                y_add = float(+1.0)
            elif (i + 1) % 8 == 3:
                x_add = float(-1.0)
                y_add = float(0.0)

            temp_q_goal = aroundGoal
            temp_q_goal.pose.position.x = start_posx + x_add
            temp_q_goal.pose.position.y = start_posy + y_add

            self.nav.goToPose(temp_q_goal)

            while not self.nav.isTaskComplete():
                rclpy.spin_once(self.map_handle)
                q = self.getCurrPose()
                myPath = self.nav.getPath(q, goal_pose)
                if self.nav.status != 6 and myPath is not None:
                    logging.info('Saved!!!')
                    self.dont_sample = False
                    return myPath
                rclpy.spin_once(self.odom_info_getter)

            q = self.getCurrPose()
            myPath = self.nav.getPath(q, goal_pose)
            if self.nav.status != 6 and myPath is not None:
                logging.info('Saved!!!')
                self.dont_sample = False
                return myPath

        logging.info(f'Can Find next step in worlds {self.curr_local_origin}, and stuck at position : ({q.pose.position.x}, {q.pose.position.y})')
        self.dont_sample = False
        return None

    def tryAroundGoals0(self, x, y):
        """Fallback: move the robot by a fixed (x, y) offset from its current pose."""
        odom = self.odom_info_getter
        init_pose = odom.pose_stored
        first_goal_pose = PoseStamped()
        first_goal_pose.header.frame_id = 'map'
        first_goal_pose.header.stamp = self.nav.get_clock().now().to_msg()
        first_goal_pose.pose = init_pose

        first_goal_pose.pose.position.x = first_goal_pose.pose.position.x + float(x)
        first_goal_pose.pose.position.y = first_goal_pose.pose.position.y + float(y)

        self.nav.goToPose(first_goal_pose)
        while not self.nav.isTaskComplete():
            rclpy.spin_once(self.map_handle)

        return

    def SpawnHolesToRandPos(self, robot_spawn, valid_holes_pos, num=1):
        """Spawn `num` obstacles at random valid positions (excluding the robot's
        spawn cell), recording their world-frame coordinates."""
        cur_num = num
        prev_idxs = []
        self.holePositions = []
        mask = ~((valid_holes_pos[:, 0] == robot_spawn[0]) & (valid_holes_pos[:, 1] == robot_spawn[1]))
        remaining = valid_holes_pos[mask]
        spawns_called = 0
        while cur_num > 0:
            idx = np.random.randint(len(remaining))
            while idx in prev_idxs:
                idx = np.random.randint(len(remaining))

            random_pair = remaining[idx]
            spawns_called = spawns_called + 1
            self.holePositions.append((float(random_pair[0]) + self.curr_local_origin[0], float(random_pair[1]) + self.curr_local_origin[1]))
            prev_idxs.append(idx)
            cur_num = cur_num - 1

        self.holeSpawn.spawn_cube(self.holePositions, num)
        self.curr_holes_num = spawns_called
        return

    def KillPrevRandHoles(self, num):
        """Delete previously spawned obstacles."""
        self.holeKill.delete_cube(num)

    def updateCurrMap(self):
        """Spin the map subscriber briefly to refresh the current map."""
        map_sub = self.map_handle
        start = time.time()
        timeout = 0.5

        while time.time() - start < timeout:
            rclpy.spin_once(map, timeout_sec=0.1)

    def getCurrPose(self):
        """Return the robot's current pose as a Nav2 PoseStamped (from odometry)."""
        odom = self.odom_info_getter
        start = time.time()
        timeout = 0.5

        while time.time() - start < timeout:
            rclpy.spin_once(odom, timeout_sec=0.1)
        rob_pos = (odom.pose2D_stored.x, odom.pose2D_stored.y)
        robot_pos = np.array(rob_pos)
        rob_pose = action_gym_to_nav(self.nav, robot_pos)
        return rob_pose
