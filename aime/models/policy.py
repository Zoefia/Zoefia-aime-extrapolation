import torch

from aime.dist import Normal, TanhNormal

from .base import MIN_STD, MLP


class TanhGaussianPolicy(torch.nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_size=32, hidden_layers=2, min_std=None
    ) -> None:
        super().__init__()
       