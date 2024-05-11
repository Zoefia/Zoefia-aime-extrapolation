
import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from aime.data import NPZFolder, get_sample_loader
from aime.env import env_classes
from aime.logger import get_default_logger
from aime.models.ssm import ssm_classes
from aime.utils import (
    CONFIG_PATH,
    DATA_PATH,
    MODEL_PATH,
    OUTPUT_PATH,
    AverageMeter,
    eval_prediction,
    generate_prediction_videos,
    get_image_sensors,
    parse_world_model_config,
    setup_seed,
)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="model-only")
def main(config: DictConfig):
    setup_seed(config["seed"])

    log.info("using the following config:")
    log.info(config)