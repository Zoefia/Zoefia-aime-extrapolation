
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
        have_head: bool = True,
        hidden_activation: str = "elu",
        output_activation: str = "identity",
        zero_init: bool = False,
    ):
        super(MLP, self).__init__()

        if out_features is None:
            out_features = hidden_size
        self.output_dim = out_features

        hidden_activation_creator = self.ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = self.ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            assert have_head, "you have to have a head when there is no hidden layers!"
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features), output_activation_creator()
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(
                    nn.Linear(in_features if i == 0 else hidden_size, hidden_size)
                )
                if norm:
                    if norm == "ln":
                        net.append(nn.LayerNorm(hidden_size))
                    elif norm == "bn":
                        net.append(nn.BatchNorm1d(hidden_size))
                    else:
                        raise NotImplementedError(f"{norm} does not supported!")
                net.append(hidden_activation_creator())
            if have_head:
                net.append(nn.Linear(hidden_size, out_features))
                if zero_init:
                    with torch.no_grad():
                        net[-1].weight.fill_(0)
                        net[-1].bias.fill_(0)
                net.append(output_activation_creator())
            self.net = nn.Sequential(*net)

    def forward(self, x):
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        head_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = self.net(x)
        out = out.view(*head_shape, out.shape[-1])
        return out


class CNNEncoderHa(nn.Module):
    """
    The structure is introduced in Ha and Schmidhuber, World Model.
    NOTE: The structure only works for 64 x 64 image.
    """

    def __init__(self, image_size, width=32, *args, **kwargs) -> None:
        super().__init__()

        self.resize = torchvision.transforms.Resize(64)
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 4, 2),
            nn.ReLU(True),  # This relu is problematic
            nn.Conv2d(width, width * 2, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(width * 2, width * 4, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(width * 4, width * 8, 4, 2),
            nn.Flatten(start_dim=-3, end_dim=-1),
        )

        self.output_dim = 4 * width * 8

    def forward(self, image):
        """forward process an image, the return feature is 1024 dims"""
        head_dims = image.shape[:-3]
        image = image.view(-1, *image.shape[-3:])
        image = self.resize(image)
        output = self.net(image)
        return output.view(*head_dims, self.output_dim)


class IndentityEncoder(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim

    def forward(self, x):
        return x


encoder_classes = {
    "mlp": MLP,