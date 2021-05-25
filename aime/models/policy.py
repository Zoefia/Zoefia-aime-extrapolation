import torch

from aime.dist import Normal, TanhNormal

from .base import MIN_STD, MLP


class TanhGaussianPolicy(torch.nn.Module):
    def __init__(
        self, state_d