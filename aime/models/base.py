
import torch
import torchvision
from torch import nn
from torch.functional import F

from aime.dist import Normal

MIN_STD = 1e-6


class MLP(nn.Module):
    r"""
    Multi-layer Perceptron
    Inputs:
        in_features : int, features numbers of the input
        out_features : int, features numbers of the output
        hidden_size : int, features numbers of the hidden layers
        hidden_layers : int, numbers of the hidden layers
        norm : str, normalization method between hidden layers, default : None
        hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu'
        output_activation : str, activation function used in output layer, default : 'identity'
    """  # noqa: E501

    ACTIVATION_CREATORS = {
        "relu": lambda: nn.ReLU(inplace=True),
        "elu": lambda: nn.ELU(),
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
        "identity": lambda: nn.Identity(),
        "gelu": lambda: nn.GELU(),
        "swish": lambda: nn.SiLU(),
        "softplus": lambda: nn.Softplus(),
    }

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        hidden_size: int = 32,
        hidden_layers: int = 2,
        norm: str = None,