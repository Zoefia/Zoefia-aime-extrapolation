
import logging
import os
import time

import gym
import numpy as np

from aime.data import ArrayDict

log = logging.getLogger("env")


def cheetah_obs_to_state_fn(obs):
    x_pos = np.zeros(1)
    pos = obs["position"]
    vel = obs.get("velocity", np.zeros(9))
    return np.concatenate([x_pos, pos, vel])


obs_to_state_fns = {"cheetah": cheetah_obs_to_state_fn}


class DMC(gym.Env):

    """gym environment for dm_control, adapted from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/envs.py"""

    def __init__(
        self, name, action_repeat=1, size=(64, 64), camera=None, render=True, seed=None
    ):
        domain, task = name.split("-", 1)
        self._reward_fn = None
        self._obs_to_state_fn = obs_to_state_fns.get(domain, None)
        if domain == "manip":
            from dm_control import manipulation

            self._env = manipulation.load(task + "_vision", seed=seed)
        elif domain == "locom":
            from dm_control.locomotion.examples import basic_rodent_2020

            self._env = getattr(basic_rodent_2020, task)(np.random.RandomState(seed))
        else:
            from dm_control import suite

            if domain == "cheetah":
                self._env = suite.load("cheetah", "run", task_kwargs={"random": seed})
                if task == "run":
                    self._reward_fn = None
                elif task == "runbackward":
                    self._reward_fn = lambda obs: max(
                        0, min(-obs["velocity"][0] / 10, 1)
                    )
                elif task == "flip":
                    self._reward_fn = lambda obs: max(0, min(obs["velocity"][2] / 5, 1))
                elif task == "flipbackward":
                    self._reward_fn = lambda obs: max(
                        0, min(-obs["velocity"][2] / 5, 1)
                    )
            elif domain == "walker":
                if task in ["stand", "walk", "run"]:
                    self._env = suite.load(domain, task, task_kwargs={"random": seed})
                    self._reward_fn = None
                elif task == "walkbackward":
                    self._env = suite.load(
                        "walker", "stand", task_kwargs={"random": seed}
                    )
                    self._reward_fn = (
                        lambda obs: obs["reward"]
                        * (5 * max(0, min(-obs["velocity"][0] / 1, 1)) + 1)
                        / 6
                    )
                elif task == "runbackward":
                    self._env = suite.load(
                        "walker", "stand", task_kwargs={"random": seed}
                    )
                    self._reward_fn = (
                        lambda obs: obs["reward"]
                        * (5 * max(0, min(-obs["velocity"][0] / 8, 1)) + 1)
                        / 6
                    )
                elif task == "flip":
                    self._env = suite.load(
                        "walker", "stand", task_kwargs={"random": seed}
                    )
                    self._reward_fn = lambda obs: max(0, min(obs["velocity"][2] / 5, 1))
                elif task == "flipbackward":
                    self._env = suite.load(
                        "walker", "stand", task_kwargs={"random": seed}
                    )
                    self._reward_fn = lambda obs: max(
                        0, min(-obs["velocity"][2] / 5, 1)
                    )
                elif task == "jump":
                    self._env = suite.load(
                        "walker", "stand", task_kwargs={"random": seed}
                    )
                    self._reward_fn = lambda obs: max(0, min(obs["height"][0] / 5, 1))
            else:
                self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self._action_repeat = action_repeat
        self._size = size
        self._render = render
        if camera in (-1, None):
            camera = {
                "quadruped-walk": 2,
                "quadruped-run": 2,
                "quadruped-escape": 2,
                "quadruped-fetch": 2,
                "locom_rodent-maze_forage": 1,
                "locom_rodent-two_touch": 1,
            }.get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

        # setup observation and action space
        spec = self._env.action_spec()
        self.act_space = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

        spaces = {
            "reward": gym.spaces.Box(0, self._action_repeat, (1,), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=bool),
        }
        if self._render:
            spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(
                    -np.inf, np.inf, (int(np.prod(value.shape)),), np.float32
                )
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(
                    0, 255, (int(np.prod(value.shape)),), np.uint8
                )
            else:
                raise NotImplementedError(value.dtype)
        spaces["pre_action"] = self.act_space
        self.obs_space = spaces

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    def step(self, action):
        assert np.isfinite(action).all(), action
        action = np.clip(action, self.act_space.low, self.act_space.high)
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break