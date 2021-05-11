
import torch
from torch.distributions import TransformedDistribution
from torch.distributions.kl import kl_divergence, register_kl
from torch.distributions.transforms import TanhTransform

TANH_CLIP = 0.999
NUM_KL_APPROXIMATE_SAMPLES = 1024


class Normal(torch.distributions.Normal):
    @property  # make pytorch < 1.12 compatible with the mode api
    def mode(self):
        return self.mean

    def detach(self):