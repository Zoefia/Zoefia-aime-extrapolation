import logging as log
import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf

from aime.actor import PolicyActor
from aime.env import DMC, SaveTrajectories, TerminalSummaryWrapper
from aime.models.policy import TanhGaussianPolicy
from aime.models.ssm import ssm_classes
from aime.utils import (
    get_image_sensors,
    interact_with_environment,
    parse_world_model_config,
    setup_seed,
)

log.basicConfig(level=log.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--seed", type=int, defau