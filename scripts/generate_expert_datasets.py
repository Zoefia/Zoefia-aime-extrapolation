import logging as log
import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from aime.actor import PolicyActor
from aime.env import DMC, SaveTrajectories, TerminalSummaryWrapper
from aime.models.policy import TanhGaussianPolicy
from ai