
import json
import logging
import os
from pprint import pformat

from torch.utils.tensorboard import SummaryWriter


def get_default_logger(root: str):
    logger = ListLogger(root)
    logger.add(TerminalLogger)
    logger.add(TensorboardLogger)
    logger.add(JsonlLogger)
    return logger

