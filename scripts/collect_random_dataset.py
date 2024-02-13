import logging as log
import os
from argparse import ArgumentParser

from omegaconf import OmegaConf

from aime.actor import RandomActor
from aime.env import DMC, SaveTrajectories, TerminalSummaryWrapper
from aime.utils import CONFIG_PATH, DATA_PATH, interact_with_environment, setup_seed

log.basicConfig(level=log.INF