
import logging
import os
import random

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

from aime.data import ArrayDict

log = logging.getLogger("utils")


def setup_seed(seed=42):
    """Fix the common random source in deep learning programs"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    log.info(f"global seed is set to {seed}")


class AverageMeter:
    """Class to collect and average a sequence of metrics"""

    def __init__(self) -> None:
        self.storage = None

    def add(self, metrics):
        if self.storage is None:
            self.storage = {k: [v] for k, v in metrics.items()}
        else:
            for k in metrics.keys():
                self.storage[k].append(metrics[k])

    def get(
        self,
    ):
        if self.storage is None:
            return {}
        return {k: np.mean(v) for k, v in self.storage.items()}


def get_sensor_shapes(example_data):
    shapes = {}
    for k, v in example_data.items():