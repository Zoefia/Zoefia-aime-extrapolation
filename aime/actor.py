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
        policy       : a policy take a hidden state and output the distribution of actions
        """  # noqa: E501
        self.ssm = ssm
        self.policy = policy
        self.eval = eval

    def reset(self):
        self.state = self.ssm.reset(1)
        self.model_parameter = list(self.ssm.parameters())[0]

    def __call__(self, obs):
        obs = ArrayDict(deepcopy(obs))
        obs.to_torch()
        obs.expand_dim_equal_()
        obs.to(self.model_parameter)
        obs.vmap_(lambda v: v.unsqueeze(dim=0))

        self.state, _ = self.ssm.posterior_step(obs, obs["pre_action"], self.state)
        state_feature = self.ssm.get_state_feature(self.state)
        action_dist = self.policy(state_feature)
     