#!/bin/bash
# Launch NUM_ENVS parallel simulation stacks, each in its own gnome-terminal.
# Each stack gets a distinct ROS_DOMAIN_ID and GAZEBO_MASTER_URI so the
# instances are isolated, then launches Gazebo + Navigation2 + SLAM Toolbox.
# Only the first environment opens RViz. After the stacks are up, the training
# script is started.

# Number of parallel environments.
NUM_ENVS=1

sleep 5

for ((i=1; i<=NUM_ENVS; i++))
do
  echo "Starting environment $i..."

  # Only the first environment opens the RViz GUI.
  if [ "$i" -eq 1 ]; then
    RVIZ_FLAG=True
  else
    RVIZ_FLAG=False
  fi

  gnome-terminal -- bash -c "
    echo 'Setting ROS_DOMAIN_ID and GAZEBO_MASTER_URI for env $i...';
    export ROS_DOMAIN_ID=$i;
    sleep 1;
    export GAZEBO_MASTER_URI=http://localhost:1134${i};
    sleep 1;

    echo 'Launching Gazebo for env $i...';
    ros2 launch turtlebot3_gazebo turtlebot3_d1worldwalls.launch.py   >/dev/null 2>&1 &

    sleep 5;  # wait for Gazebo to initialize

    echo 'Launching Navigation2 (RViz: $RVIZ_FLAG) for env $i...';
    ros2 launch turtlebot3_navigation2 navigation2.launch.py \
      use_sim_time:=True launch_rviz:=${RVIZ_FLAG}  >/dev/null 2>&1 &

    sleep 2;

    ros2 launch slam_toolbox online_sync_launch.py use_sim_time:=True >/dev/null 2>&1;

    exec bash
  " &
done

# Wait for the stacks to come up before starting training.
sleep 10

python3 simple_gymFeb.py  --inference True   #--new_train False
