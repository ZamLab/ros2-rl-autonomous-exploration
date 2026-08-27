"""
Gym environment and training/inference entry points for the DRL exploration agent.

This module defines `XelonaEnv`, an OpenAI Gym environment that wraps the ROS 2
communication layer (`multi_comFeb.Communication_Interface`) so that a
Stable-Baselines3 DQN agent can drive a TurtleBot3 through SLAM-based
exploration. The observation is a 64x64 grayscale image of the SLAM occupancy
grid with the robot's position marked; the action is one of 25 discrete goal
coordinates on a 5x5 grid, executed through Nav2.

Entry points (selected from __main__ via CLI flags):
  * main0                -- main training run (parallel envs)
  * main0_nextEnvType    -- resume/continue training with a fresh model whose
                            weights are seeded from a previously trained one
                            (used to keep training with different RL
                            hyperparameters, loading from the prev-train folder)
  * main / main11        -- inference: run the trained model's predictions
  * main_infer_plots*    -- run the trained agent and generate the evaluation
                            metric plots (coverage/time, success rate,
                            localization stability)
"""

import logging
import os
import argparse
import itertools
import time

import gym
import numpy as np
from gym import spaces
from gym.spaces import Dict as DictSpace
import matplotlib.pyplot as plt
import subprocess

import torch
import torch.nn as nn

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.utils.tensorboard import SummaryWriter

import multi_comFeb as ros_com


writer = SummaryWriter("my_runs/exp_01")

info = {}
terminated = False
truncated = False
done = False

# Paths for checkpointing / resuming. `save_path` holds the current run's
# checkpoints; `prev_train_file` holds a previously trained model to seed from.
tb_folder_file = "./dqn_tensorboard_multiWorld02_holes"
save_path = "./my_training_statesAgain_multiWorld01_XL/"
folder_name_str = "my_training_statesAgain_multiWorld01_XL"
last_checkpoint_file = os.path.join(save_path, "last_checkpoint.txt")
prev_train_file = "./prev_train/"
prev_train_checkpoint_file = os.path.join(prev_train_file, "last_checkpoint.txt")

logging.basicConfig(
    filename='my_app.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cross-process lock used to coordinate a single global simulation restart
# when an environment reports an unrecoverable state (see multi_comFeb).
LOCKFILE = "/tmp/env_reset_lock"


# ======================================================================
#  Feature extractor
# ======================================================================
class SmallCnn(BaseFeaturesExtractor):
    """Small 3-conv CNN feature extractor for the 64x64 grayscale map image.

    Kept deliberately small: in this system a training step involves SLAM and a
    full navigation action, so the network is never the bottleneck.
    """

    def __init__(self, observation_space, features_dim=64):
        super(SmallCnn, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 1 for greyscale
        print(f"CNN INPUT CHANNEL IS : {n_input_channels}")
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the flattened CNN output size from a sample observation.
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.linear(x)


policy_kwargs = dict(
    features_extractor_class=SmallCnn,
    features_extractor_kwargs=dict(features_dim=64),
)


# ======================================================================
#  Checkpoint callback
# ======================================================================
class CustomCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that also records the path of the latest saved model,
    so training can resume from `last_checkpoint_file` after a restart."""

    def _on_step(self) -> bool:
        result = super()._on_step()  # saves the model normally

        if self.n_calls % self.save_freq == 0:
            latest_model_path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            with open(last_checkpoint_file, "w") as f:
                f.write(latest_model_path)
            print(f" Last saved model updated: {latest_model_path}")

        return result


checkpoint_callback = CustomCheckpointCallback(
    save_freq=max(50 // 4, 1),
    save_path=save_path,
    name_prefix="rl_model",
    save_replay_buffer=True,
    verbose=2,
    save_vecnormalize=True
)


# ======================================================================
#  Action space: 25 goals on a 5x5 grid over the workspace
# ======================================================================
step = 1.5
values = np.arange(-3, 4, step)
action_coords = np.array(list(itertools.product(values, values)), dtype=np.float32)


# ======================================================================
#  Gym environment
# ======================================================================
class XelonaEnv(gym.Env):
    """Custom Gym environment wrapping the ROS 2 exploration system.

    Observation: (1, 64, 64) uint8 grayscale image of the SLAM map.
    Action:      Discrete(25) -- index into `action_coords` (goal coordinates).
    Reward:      exploration of new space, with penalties for localization
                 failure and non-progressive actions (see multi_comFeb).
    """

    def __init__(self, env_id):
        super(XelonaEnv, self).__init__()
        self.count_int_steps = 0
        self.curr_id = env_id
        self.xelona_com = ros_com.Communication_Interface(env_id)
        self.step_count = 0

        self.action_space = spaces.Discrete(len(action_coords))
        self.action_coords = action_coords

        # --- Metric bookkeeping (used by the evaluation-plot entry points) ---
        # Metric 1: simulation time to reach coverage levels.
        self.curr_episode_time = 0
        self.timePerCover = []
        self.episode_count = -1
        self.got2 = False
        self.got4 = False
        self.got5 = False
        self.got6 = False
        self.got7 = False
        self.got8 = False
        self.got9 = False
        # Metric 2: successful episodes vs. steps.
        self.steps_num_m2 = 0
        self.m2_succeed_eps = 0
        self.m2 = []
        # Metric 3: coverage at fixed step counts.
        self.m3 = []
        
        # Localization metric.
        self.localization_metric_list = []
        self.localization_lost_count = 0

        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8)

    def step(self, action):
        self.count_int_steps = self.count_int_steps + 1
        map_action = self.action_coords[action]
        self.step_count = self.step_count + 1
        truncated = False
        terminated = False
        s_time = time.time()
        reward, observation, healthyReset, lost_localization, success_map, cover = \
            self.xelona_com.execute_action_and_compute_reward(map_action)
        e_time = time.time()

        # --- Metric 1: record the time at which each coverage level is reached ---
        self.curr_episode_time = self.curr_episode_time + (e_time - s_time)
        if success_map:
            self.timePerCover.append((1.0, self.curr_episode_time))
        elif cover >= 0.8 and (not self.got8):
            self.got8 = True
            self.got6 = True
            self.got4 = True
            self.got2 = True
            self.timePerCover.append((0.8, self.curr_episode_time))
        elif cover >= 0.6 and (not self.got6):
            self.got6 = True
            self.got4 = True
            self.got2 = True
            self.timePerCover.append((0.6, self.curr_episode_time))
        elif cover >= 0.4 and (not self.got4):
            self.got4 = True
            self.got2 = True
            self.timePerCover.append((0.4, self.curr_episode_time))
        elif cover >= 0.2 and (not self.got2):
            self.got2 = True
            self.timePerCover.append((0.2, self.curr_episode_time))

        # --- Metric 2: successful episodes vs. cumulative steps ---
        self.steps_num_m2 = self.steps_num_m2 + 1
        if success_map:
            self.m2_succeed_eps = self.m2_succeed_eps + 1
            self.m2.append((self.steps_num_m2, self.m2_succeed_eps))
        if lost_localization:
            self.localization_lost_count = self.localization_lost_count + 1

        # --- Metric 3: coverage sampled at fixed step counts ---
        if not lost_localization:
            if self.steps_num_m2 == 2:
                self.m3.append((cover, 2))
            if self.steps_num_m2 == 4:
                self.m3.append((cover, 4))
            if self.steps_num_m2 == 6:
                self.m3.append((cover, 6))
            if self.steps_num_m2 == 8:
                self.m3.append((cover, 8))
            if self.steps_num_m2 == 10:
                self.m3.append((cover, 10))

        # Terminate on localization loss or on successful full coverage.
        if lost_localization or success_map:
            self.step_count = 0
            truncated = False
            terminated = True

        # Truncate at the episode horizon (14 internal steps here).
        if self.step_count == 14:
            self.step_count = 0
            truncated = True
            terminated = False

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.steps_num_m2 = 0
        self.got2 = False
        self.got4 = False
        self.got6 = False
        self.got8 = False
        self.curr_episode_time = 0
        self.episode_count = self.episode_count + 1

        # Reset the ROS 2 side (respawn robot, reload map/pose graph, etc.).
        healthyReset = self.xelona_com.reset_com()

        # First observation after reset.
        observation, healthyReset = self.xelona_com.execute_first_small_step()

        # If the reset was unhealthy, trigger a coordinated global restart.
        if not healthyReset:
            if not os.path.exists(LOCKFILE):
                subprocess.call(["./stop_envs.sh"])

        return observation, {}

    def close(self):
        self.xelona_com.close_communication()


def make_env(env_id):
    """Factory returning a thunk that builds a XelonaEnv (for SubprocVecEnv)."""
    def _init():
        env = XelonaEnv(env_id)
        return env
    return _init


# ======================================================================
#  Entry points
# ======================================================================
def main():
    """Quick inference sanity check: run the trained model for a few steps and
    compare its cumulative reward against a random policy."""
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)

    num_envs = 1
    envs = SubprocVecEnv([make_env(i + 1) for i in range(num_envs)])
    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file, "r") as f:
            last_saved_model = f.read().strip()

        logging.info(f"Resuming training from checkpoint: {last_saved_model}")
        print(f" Resuming training from: {last_saved_model}")
        model = DQN.load(last_saved_model, envs, device="cpu")
        saved_steps_str = str(model.num_timesteps)
        print(saved_steps_str)
        replay_buffer_path = f"./my_training_states10test/rl_model_replay_buffer_{saved_steps_str}_steps.pkl"
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
        else:
            print("Replay buffer file not found, continuing with empty buffer.")

        obs = envs.reset()
        first_obs = obs
        reward_ofTrained = 0
        reward_ofRandom = 0
        for _ in range(6):
            actions, _states = model.predict(obs, deterministic=True)
            print(f"Action  Done is : {actions}")
            obs, rew, dones, infos = envs.step(actions)
            print(first_obs)
            print("-----------------")
            reward_ofTrained = reward_ofTrained + rew

        for _ in range(6):
            actions = envs.action_space.sample  # just sample
            obs, rew, dones, infos = envs.step(actions)
            reward_ofRandom = reward_ofRandom + rew

        print(f"Inference Data: Robot Score : {reward_ofTrained} VS Random Score : {reward_ofRandom}")


def main11():
    """Interactive inference loop: repeatedly predict and step the environment
    (used to visually test that a trained model behaves as expected)."""
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)

    num_envs = 1
    envs = SubprocVecEnv([make_env(i + 1) for i in range(num_envs)])

    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file, "r") as f:
            last_saved_model = f.read().strip()

        logging.info(f"Resuming training from checkpoint: {last_saved_model}")
        print(f" Resuming training from: {last_saved_model}")
        model = DQN.load(last_saved_model, envs, device="cpu")
        saved_steps_str = str(model.num_timesteps)
        print(saved_steps_str)
        replay_buffer_path = f"./{folder_name_str}/rl_model_replay_buffer_{saved_steps_str}_steps.pkl"
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
        else:
            print("Replay buffer file not found, continuing with empty buffer.")
    user_input = 0
    obs = envs.reset()

    while user_input != 100:
        user_input = 88
        if user_input == 101:
            actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
            obs, rew, dones, infos = envs.step(actions)
        elif user_input == 80:
            obs = envs.reset()
        elif user_input == 99:
            count = 0
            for el in action_coords:  # print the action coordinate table
                print(f"coord {el} is count : {count}")
                count = count + 1
            print("-----------------------------------")
        elif user_input == 88:  # predict
            actions, _states = model.predict(obs, deterministic=True)
            print(f"Current action is {actions} with type {type(actions)}")
            obs, rew, dones, infos = envs.step(actions)
        else:
            action = np.array([user_input], dtype=int)
            obs, rew, dones, infos = envs.step(action)


def main0_nextEnvType():
    """Continue training with a fresh model (new RL hyperparameters) whose
    weights and optimizer state are seeded from a previously trained model
    loaded from the prev-train folder."""
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)

    num_envs = 4
    envs = SubprocVecEnv([make_env(i + 1) for i in range(num_envs)])
    envs = VecMonitor(envs)

    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file, "r") as f:
            last_saved_model = f.read().strip()

        logging.info(f"Resuming training from checkpoint: {last_saved_model}")
        print(f" Resuming training from: {last_saved_model}")
        new_model = DQN.load(last_saved_model, envs, device="cpu")
        saved_steps_str = str(new_model.num_timesteps)
        print(saved_steps_str)
        replay_buffer_path = f"./{folder_name_str}/rl_model_replay_buffer_{saved_steps_str}_steps.pkl"
        if os.path.exists(replay_buffer_path):
            new_model.load_replay_buffer(replay_buffer_path)
        else:
            print("Replay buffer file not found, continuing with empty buffer.")
    else:
        print(" No checkpoint found, starting fresh training.")
        # Fresh model with the new hyperparameters.
        new_model = DQN("CnnPolicy", envs, device="cpu", buffer_size=2000, exploration_initial_eps=0.9,
                        exploration_fraction=0.3, gamma=0.85, learning_rate=2e-4, target_update_interval=2000,
                        train_freq=1, exploration_final_eps=0.10, learning_starts=1200, gradient_steps=4,
                        tensorboard_log=tb_folder_file, policy_kwargs=policy_kwargs)

        # Seed weights/optimizer from a previously trained model.
        if os.path.exists(prev_train_checkpoint_file):
            with open(prev_train_checkpoint_file, "r") as f:
                last_saved_model = f.read().strip()
                print("Old model Loaded <--------------------------------------------")
            model = DQN.load(last_saved_model, envs, device="cpu")

        new_model.policy.load_state_dict(model.policy.state_dict())
        new_model.policy.optimizer.load_state_dict(
            model.policy.optimizer.state_dict()
        )

    print("Hello, NEW <-- model stats as following :")
    print("-----------------------------------------")
    print(f"My models buffer size is : {new_model.replay_buffer.pos}")
    print(f"Models exploration rate is : {new_model.exploration_rate}")
    print(f"Models learning rate is : {new_model.learning_rate}")
    print(f"Models current step is : {new_model.num_timesteps}")
    print("-----------------------------------------")

    new_model.learn(total_timesteps=40_000, callback=checkpoint_callback,
                    reset_num_timesteps=False, tb_log_name="DQN_run")


def main000():
    """Small manual test loop calling the communication layer's test hook."""
    envs = XelonaEnv(1)
    user_input = 0
    while user_input != 100:
        user_input = int(input("Please enter an integer: "))
        action = np.array([user_input], dtype=int)
        envs.xelona_com.main_test00()


def main0():
    """Main training run: 4 parallel environments, DQN, resume from checkpoint
    if one exists, otherwise start a fresh model."""
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)

    num_envs = 4
    envs = SubprocVecEnv([make_env(i + 1) for i in range(num_envs)])
    envs = VecMonitor(envs)

    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file, "r") as f:
            last_saved_model = f.read().strip()

        logging.info(f"Resuming training from checkpoint: {last_saved_model}")
        print(f" Resuming training from: {last_saved_model}")
        model = DQN.load(last_saved_model, envs, device="cpu")
        saved_steps_str = str(model.num_timesteps)
        print(saved_steps_str)
        replay_buffer_path = f"./{folder_name_str}/rl_model_replay_buffer_{saved_steps_str}_steps.pkl"
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
        else:
            print("Replay buffer file not found, continuing with empty buffer.")
    else:
        print(" No checkpoint found, starting fresh training.")
        logging.info("No checkpoint found. Starting fresh training.")
        model = DQN("CnnPolicy", envs, device="cpu", buffer_size=5000, exploration_fraction=0.4, gamma=0.85,
                    learning_rate=3e-4, target_update_interval=2000, train_freq=1, exploration_final_eps=0.1,
                    learning_starts=1250, gradient_steps=4, tensorboard_log=tb_folder_file,
                    policy_kwargs=policy_kwargs)

    print("Hello, model stats as following :")
    print("-----------------------------------------")
    print(f"My models buffer size is : {model.replay_buffer.pos}")
    print(f"Models exploration rate is : {model.exploration_rate}")
    print(f"Models learning rate is : {model.learning_rate}")
    print(f"Models current step is : {model.num_timesteps}")
    print("-----------------------------------------")

    model.learn(total_timesteps=40_000, callback=checkpoint_callback,
                reset_num_timesteps=False, tb_log_name="DQN_run")


def main_infer_plots():
    """Full evaluation: run the trained agent and a random baseline, and
    generate the coverage/time (metric 1), success-rate (metric 2) and
    localization-stability plots comparing the two."""
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)

    envs = XelonaEnv(1)
    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file, "r") as f:
            last_saved_model = f.read().strip()

        logging.info(f"Resuming training from checkpoint: {last_saved_model}")
        print(f" Resuming training from: {last_saved_model}")
        model = DQN.load(last_saved_model, envs, device="cpu")
        saved_steps_str = str(model.num_timesteps)
        print(saved_steps_str)
        replay_buffer_path = f"./{folder_name_str}/rl_model_replay_buffer_{saved_steps_str}_steps.pkl"
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
        else:
            print("Replay buffer file not found, continuing with empty buffer.")
    user_input = 0
    obs, info = envs.reset()

    # Reset metrics.
    envs.episode_count = 0
    envs.timePerCover = []

    # --- Metric 1: RL agent, mean time per coverage level ---
    while envs.episode_count < 10:
        print(f">>>>EPISODE COUNT IS : {envs.episode_count}<<<<")
        actions, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = envs.step(actions)
        if terminated or truncated:
            obs, info = envs.reset()

    res1 = np.array(envs.timePerCover)
    values20 = res1[res1[:, 0] == 0.2, 1]
    values40 = res1[res1[:, 0] == 0.4, 1]
    values60 = res1[res1[:, 0] == 0.6, 1]
    values80 = res1[res1[:, 0] == 0.8, 1]
    values100 = res1[res1[:, 0] == 1.0, 1]

    mean_value20 = values20.mean() if len(values20) > 0 else None
    mean_value40 = values40.mean() if len(values40) > 0 else None
    mean_value60 = values60.mean() if len(values60) > 0 else None
    mean_value80 = values80.mean() if len(values80) > 0 else None
    mean_value100 = values100.mean() if len(values100) > 0 else None

    x = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    y_rl = np.array([mean_value20, mean_value40, mean_value60, mean_value80, mean_value100])

    # --- Metric 1: random agent, mean time per coverage level ---
    envs.episode_count = 0
    envs.timePerCover = []
    envs.num_envs = 1
    while envs.episode_count < 10:
        x = np.random.randint(0, 25)  # 0 to 24
        action = np.array(x, dtype=int)
        obs, reward, terminated, truncated, info = envs.step(action)
        if terminated or truncated:
            obs, info = envs.reset()

    res1 = np.array(envs.timePerCover)
    values20 = res1[res1[:, 0] == 0.2, 1]
    values40 = res1[res1[:, 0] == 0.4, 1]
    values60 = res1[res1[:, 0] == 0.6, 1]
    values80 = res1[res1[:, 0] == 0.8, 1]
    values100 = res1[res1[:, 0] == 1.0, 1]

    mean_value20 = values20.mean() if len(values20) > 0 else None
    mean_value40 = values40.mean() if len(values40) > 0 else None
    mean_value60 = values60.mean() if len(values60) > 0 else None
    mean_value80 = values80.mean() if len(values80) > 0 else None
    mean_value100 = values100.mean() if len(values100) > 0 else None

    x = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    y_random = np.array([mean_value20, mean_value40, mean_value60, mean_value80, mean_value100])

    plt.figure(figsize=(8, 6))
    plt.plot(x, y_rl, marker='o', color='blue', label='RL Agent')
    plt.plot(x, y_random, marker='s', color='red', label='Random Agent')
    plt.xlabel("Coverage")
    plt.ylabel("Mean simulation time ( x 10.0 speed ) ")
    plt.title("Mean time per coverage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mean_time_per_coverage.png")
    plt.close()

    # --- Localization metric: RL agent ---
    while envs.episode_count < 10:
        actions, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = envs.step(actions)
        if terminated or truncated:
            envs.localization_metric_list.extend(envs.xelona_com.localization_samples)
            obs, info = envs.reset()

    alpha = 0.95
    envs.localization_metric_list.pop(0)
    smoothed1 = [envs.localization_metric_list[0]]
    for x in envs.localization_metric_list[1:]:
        smoothed1.append(alpha * smoothed1[-1] + (1 - alpha) * x)

    plt.figure(figsize=(8, 5))
    plt.plot(envs.localization_metric_list, color="lightblue", linewidth=1.5, label="Raw")
    plt.plot(smoothed1, color="blue", linewidth=0.8, label="Smoothed")
    plt.xlabel(" Samples ")
    plt.ylabel("Map - Odom Distance")
    plt.title("RL Agent - Samples")
    plt.grid(True)
    plt.legend()
    plt.savefig("Localizationw4RL0.png")
    plt.close()

    # --- Localization metric: random agent ---
    smoothed = []
    envs.localization_metric_list = []
    envs.episode_count = 0
    while envs.episode_count < 10:
        stp_x = np.random.randint(0, 25)  # 0 to 24
        action = np.array(stp_x, dtype=int)
        obs, reward, terminated, truncated, info = envs.step(action)
        if terminated or truncated:
            envs.localization_metric_list.extend(envs.xelona_com.localization_samples)
            obs, info = envs.reset()

    alpha = 0.95
    envs.localization_metric_list.pop(0)
    smoothed = [envs.localization_metric_list[0]]
    for x in envs.localization_metric_list[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * x)

    plt.figure(figsize=(8, 5))
    plt.plot(envs.localization_metric_list, color="lightgreen", linewidth=1.5, label="Raw")
    plt.plot(smoothed, color="green", linewidth=0.8, label="Smoothed")
    plt.xlabel("Samples ")
    plt.ylabel("Map - Odom Distance")
    plt.title("Random Agent -  Samples")
    plt.grid(True)
    plt.legend()
    plt.savefig("RandomLocalizationw4_0.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(smoothed1, color="blue", linewidth=1, label="RL Agent")
    plt.plot(smoothed, color="green", linewidth=1, label="Random Agent")
    plt.xlabel("Samples")
    plt.ylabel("Map - Odom Distance")
    plt.title("Localization Metric Comparison")
    plt.grid(True)
    plt.legend()
    plt.savefig("Localizationw4Comparison0.png")
    plt.close()

    np.save("smoothedRLw3.npy", np.array(smoothed1))
    np.save("smoothedRandw3.npy", np.array(smoothed))

    # --- Metric 3: RL agent, coverage at fixed step counts ---
    while envs.episode_count < 15:
        print(f">>>>EPISODE RL COUNT IS : {envs.episode_count}<<<<")
        actions, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = envs.step(actions)
        if terminated or truncated:
            obs, info = envs.reset()
    res = np.array(envs.m3)

    x_values = [2, 4, 6, 8, 10]
    rl_avg_n = []
    for x in x_values:
        values = res[res[:, 1] == x, 0]
        rl_mean_val = values.mean() if len(values) > 0 else None
        rl_avg_n.append(rl_mean_val)

    # --- Metric 3: random agent ---
    envs.m3 = []
    rand_avg_n = []
    envs.episode_count = 0
    while envs.episode_count < 15:
        print(f">>>>EPISODE Random COUNT IS : {envs.episode_count}<<<<")
        stp_x = np.random.randint(0, 25)  # 0 to 24
        action = np.array(stp_x, dtype=int)
        obs, reward, terminated, truncated, info = envs.step(action)
        if terminated or truncated:
            obs, info = envs.reset()

    res = np.array(envs.m3)
    for x in x_values:
        values = res[res[:, 1] == x, 0]
        rand_mean_val = values.mean() if len(values) > 0 else None
        rand_avg_n.append(rand_mean_val)

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, rl_avg_n, marker='o', color='blue', label='RL Agent', markersize=4)
    plt.plot(x_values, rand_avg_n, marker='o', color='red', label='Random Agent', markersize=4)
    plt.xlabel("Steps")
    plt.ylabel("Map Coverage")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Train_m3_w0.png")
    plt.close()


def main_infer_plots0():
    """Evaluation variant: run the trained agent only and save the localization,
    coverage-time and coverage-steps plots (plus the raw arrays as .npy)."""
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)

    envs = XelonaEnv(1)
    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file, "r") as f:
            last_saved_model = f.read().strip()

        logging.info(f"Resuming training from checkpoint: {last_saved_model}")
        print(f" Resuming training from: {last_saved_model}")
        model = DQN.load(last_saved_model, envs, device="cpu")
        saved_steps_str = str(model.num_timesteps)
        print(saved_steps_str)
        replay_buffer_path = f"./{folder_name_str}/rl_model_replay_buffer_{saved_steps_str}_steps.pkl"
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
        else:
            print("Replay buffer file not found, continuing with empty buffer.")
    user_input = 0
    obs, info = envs.reset()

    # Reset metrics.
    envs.episode_count = 0
    envs.timePerCover = []
    envs.localization_metric_list = []

    while envs.episode_count < 2:
        print(f">>>>EPISODE COUNT IS : {envs.episode_count}<<<<")
        actions, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = envs.step(actions)
        if terminated or truncated:
            envs.localization_metric_list.extend(envs.xelona_com.localization_samples)
            obs, info = envs.reset()

    alpha = 0.95
    envs.localization_metric_list.pop(0)
    smoothed1 = [envs.localization_metric_list[0]]
    for x_loc in envs.localization_metric_list[1:]:
        smoothed1.append(alpha * smoothed1[-1] + (1 - alpha) * x_loc)

    rl_mean = np.mean(smoothed1)
    plt.figure(figsize=(8, 5))
    plt.plot(envs.localization_metric_list, color="lightblue", linewidth=1.5, label="Raw")
    plt.plot(smoothed1, color="blue", linewidth=0.8, label=f"Smoothed (mean={rl_mean:.3f})")
    plt.xlabel(" Samples ")
    plt.ylabel("Map - Odom Distance")
    plt.title("RL Agent - Samples")
    plt.grid(True)
    plt.legend()
    plt.savefig("train_localizew2.png")
    plt.close()

    np.save("train_smoothed_loc_w2.npy", np.array(smoothed1))
    np.save("train_loc_w2.npy", envs.localization_metric_list)

    res1 = np.array(envs.timePerCover)
    values20 = res1[res1[:, 0] == 0.2, 1]
    values40 = res1[res1[:, 0] == 0.4, 1]
    values60 = res1[res1[:, 0] == 0.6, 1]
    values80 = res1[res1[:, 0] == 0.8, 1]
    values100 = res1[res1[:, 0] == 1.0, 1]

    mean_value20 = values20.mean() if len(values20) > 0 else None
    mean_value40 = values40.mean() if len(values40) > 0 else None
    mean_value60 = values60.mean() if len(values60) > 0 else None
    mean_value80 = values80.mean() if len(values80) > 0 else None
    mean_value100 = values100.mean() if len(values100) > 0 else None

    x_m1 = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    y_rl = np.array([mean_value20, mean_value40, mean_value60, mean_value80, mean_value100])

    res = np.array(envs.m3)
    x_values = [2, 4, 6, 8, 10]
    rl_avg_n = []
    for x in x_values:
        values = res[res[:, 1] == x, 0]
        rl_mean_val = values.mean() if len(values) > 0 else None
        rl_avg_n.append(rl_mean_val)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x_m1, y_rl, marker='o')
    ax1.set_xlabel("Coverage")
    ax1.set_ylabel("Mean time")
    ax1.set_title("Mean time per coverage")
    ax1.grid(True)
    ax2.plot(x_values, rl_avg_n, marker='o', color='blue', label='RL Agent', markersize=4)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Map Coverage")
    ax2.set_title("Map Coverage per steps")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.savefig("train_timesteps2.png")
    plt.close(fig)

    np.save("xtrain_time_w2.npy", x_m1)
    np.save("ytrain_time_w2.npy", y_rl)
    np.save("xtrain_steps_w2", np.array(x_values))
    np.save("ytrain_steps_w2.npy", np.array(rl_avg_n))


def main_infer_plots1():
    """Evaluation variant: RL vs. random success-rate scatter (metric 2) over
    40 episodes each, plus localization-loss counts."""
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)

    envs = XelonaEnv(1)
    if os.path.exists(last_checkpoint_file):
        with open(last_checkpoint_file, "r") as f:
            last_saved_model = f.read().strip()

        logging.info(f"Resuming training from checkpoint: {last_saved_model}")
        print(f" Resuming training from: {last_saved_model}")
        model = DQN.load(last_saved_model, envs, device="cpu")
        saved_steps_str = str(model.num_timesteps)
        print(saved_steps_str)
        replay_buffer_path = f"./{folder_name_str}/rl_model_replay_buffer_{saved_steps_str}_steps.pkl"
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
        else:
            print("Replay buffer file not found, continuing with empty buffer.")
    user_input = 0
    obs, info = envs.reset()

    # Reset metrics.
    envs.episode_count = 0
    envs.timePerCover = []
    envs.localization_metric_list = []
    rl_loc_lost = envs.localization_lost_count
    envs.localization_lost_count = 0

    # --- Metric 2: RL agent ---
    while envs.episode_count < 40:
        actions, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = envs.step(actions)
        if terminated or truncated:
            obs, info = envs.reset()

    res2 = np.array(envs.m2)
    x = res2[:, 0]
    y_rl = res2[:, 1]
    success_rate = envs.m2_succeed_eps / envs.episode_count

    # Reset for the random agent.
    envs.episode_count = 0
    envs.m2_succeed_eps = 0
    envs.m2 = []

    while envs.episode_count < 40:
        stp_x = np.random.randint(0, 25)  # 0 to 24
        action = np.array(stp_x, dtype=int)
        obs, reward, terminated, truncated, info = envs.step(action)
        if terminated or truncated:
            obs, info = envs.reset()

    res2 = np.array(envs.m2)
    x_rand = res2[:, 0]
    y_rand = res2[:, 1]
    success_rate_rand = envs.m2_succeed_eps / envs.episode_count
    np.save("m2_rlx.npy", x)
    np.save("m2_rly.npy", y_rl)
    np.save("m2_rlx.npy", x_rand)
    np.save("m2_rly.npy", y_rand)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y_rl, color='blue', label='RL Agent', s=20, alpha=0.5)
    plt.scatter(x_rand, y_rand, color='red', label='Random Agent', s=20, alpha=0.5)
    plt.xlabel("Steps")
    plt.ylabel("Successful Episodes")
    plt.title(f"RL Success Rate: {success_rate:.3f}, Random Success Rate: {success_rate_rand:.3f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Metric2.png", dpi=300)
    plt.close()
    print(f"In 40 Episodes we lost localization for RL : {rl_loc_lost} \\ for Random : {envs.localization_lost_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent")
    # --inference : run the trained model's predictions (test that it works)
    parser.add_argument("--inference", type=bool, default=False)
    # --new_train : continue training an already-trained model with different RL
    #               hyperparameters (seeded from the prev-train folder)
    parser.add_argument("--new_train", type=bool, default=False)

    args = parser.parse_args()

    print(args.inference)
    print(args.new_train)

    if args.inference:
        main_infer_plots()
        main11()
    else:
        if args.new_train:
            main0_nextEnvType()
        else:
            main0()

    print("bye")
