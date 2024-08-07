#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@habitat.registry.register_env(name="myEnv")` for reusability
"""

import importlib
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np

import habitat
from habitat import Dataset
from habitat.gym.gym_wrapper import HabGymWrapper
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    ORBSLAMTopDownMapMeasurementConfig,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


RLTaskEnvObsType = Union[np.ndarray, Dict[str, np.ndarray]]


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return habitat.registry.get_env(env_name)


class RLTaskEnv(habitat.RLEnv):
    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        super().__init__(config, dataset)
        self._reward_measure_name = self.config.task.reward_measure
        self._success_measure_name = self.config.task.success_measure
        assert (
            self._reward_measure_name is not None
        ), "The key task.reward_measure cannot be None"
        assert (
            self._success_measure_name is not None
        ), "The key task.success_measure cannot be None"

    def reset(
        self, *args, return_info: bool = False, **kwargs
    ) -> Union[RLTaskEnvObsType, Tuple[RLTaskEnvObsType, Dict]]:
        return super().reset(*args, return_info=return_info, **kwargs)

    def step(
        self, *args, **kwargs
    ) -> Tuple[RLTaskEnvObsType, float, bool, dict]:
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward = self.config.task.slack_reward

        reward += current_measure

        if self._episode_success():
            reward += self.config.task.success_reward

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        if self.config.task.end_on_success and self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self._env.get_metrics()


@habitat.registry.register_env(name="GymRegistryEnv")
class GymRegistryEnv(gym.Wrapper):
    """
    A registered environment that wraps a gym environment to be
    used with habitat-baselines
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        for dependency in config["env_task_gym_dependencies"]:
            importlib.import_module(dependency)
        env_name = config["env_task_gym_id"]
        gym_env = gym.make(env_name)
        super().__init__(gym_env)


@habitat.registry.register_env(name="GymHabitatEnv")
class GymHabitatEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        base_env = RLTaskEnv(config=config, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_frames: int = 1):
        super().__init__(env)
        self.n_frames = n_frames

    def step(self, action):
        total_reward = 0.0
        n_skipped = 0
        while n_skipped < self.n_frames:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            n_skipped += 1
            if done:
                break
        return obs, total_reward, done, info


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, forward_metres=0.25, turn_degrees=30):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(4)
        self.disc_to_cont = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, forward_metres], dtype=np.float32),
            np.array([turn_degrees, 0.0], dtype=np.float32),
            np.array([-turn_degrees, 0.0], dtype=np.float32),
        ]

    def action(self, act):
        return self.disc_to_cont[act]


class LocNavWrapper(HabGymWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def _direct_hab_step(self, action):
        obs, reward, done, info = self.env.step(action=action)
        if "orbslam_pointgoal_with_gps_compass_transformed" in obs:
            obs["pointgoal_with_gps_compass"] = obs[
                "orbslam_pointgoal_with_gps_compass_transformed"
            ]
        self._last_obs = obs
        obs = self._transform_obs(obs)
        return obs, reward, done, info


@habitat.registry.register_env(name="LocNavEnv")
class LocNavEnv(gym.Wrapper):
    def __init__(self, config, dataset=None):
        env = RLTaskEnv(config=config, dataset=dataset)
        env = LocNavWrapper(env)
        env = FrameSkipWrapper(env, n_frames=10)
        env = DiscreteActionWrapper(env, turn_degrees=10)
        super().__init__(env)


@habitat.registry.register_env(name="LocNavEnvContinuous")
class LocNavEnvContinuous(gym.Wrapper):
    def __init__(self, config, dataset):
        env = RLTaskEnv(config=config, dataset=dataset)
        env = LocNavWrapper(env)
        env = FrameSkipWrapper(env, n_frames=10)
        super().__init__(env)
