from copy import deepcopy

import numpy as np
import torch
from einops import rearrange

from aime.data import ArrayDict


class RandomActor:
    """Actor that random samples from the action space"""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def __call__(self, obs):
        return self.action_space.sample()

    def reset(self):
        pass


class PolicyActor:
    """Model-based policy for taking actions"""

    def __init__(self, ssm, policy, eval=True) -> None:
        """
        ssm          : a state space model
        policy       : a policy take a hidden state and output th