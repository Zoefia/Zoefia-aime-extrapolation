from copy import deepcopy

import numpy as np
import torch
from einops import rearrange

from aime.data import ArrayDict


class RandomActor:
    """Actor that random samples from the action space"""

    def 