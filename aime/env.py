
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