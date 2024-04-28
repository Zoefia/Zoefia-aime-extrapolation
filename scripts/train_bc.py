
import logging
import os

import hydra
import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from aime.actor import StackPolicyActor
from aime.data import NPZFolder, get_epoch_loader
from aime.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aime.logger import get_default_logger
from aime.models.base import MLP, MultimodalEncoder
from aime.utils import (
    CONFIG_PATH,
    DATA_PATH,
    OUTPUT_PATH,
    AverageMeter,
    get_inputs_outputs,
    get_sensor_shapes,
    interact_with_environment,
    setup_seed,
)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="bc")
def main(config: DictConfig):
    setup_seed(config["seed"])

    log.info("using the following config:")
    log.info(config)

    stack = 1 if config["environment_setup"] == "mdp" else config["stack"]

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    demonstration_dataset_folder = os.path.join(
        DATA_PATH, config["demonstration_dataset_name"]
    )
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    log.info("Creating environment ...")
    env_config = config["env"]
    env_class_name = env_config["class"]
    test_env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"] * 2,
        render=True,
    )
    test_env = SaveTrajectories(test_env, eval_folder)
    test_env = TerminalSummaryWrapper(test_env)

    log.info("Loading datasets ...")
    demonstration_dataset = NPZFolder(
        demonstration_dataset_folder, stack + 1, overlap=True
    )
    demonstration_dataset.keep(config["num_expert_trajectories"])
    eval_dataset = NPZFolder(eval_folder, stack + 1, overlap=False)
    data = demonstration_dataset[0]

    log.info("Creating models ...")
    sensor_layout = env_config["sensors"]
    encoder_configs = config["encoders"]
    sensor_shapes = get_sensor_shapes(data)
    input_sensors, _, _ = get_inputs_outputs(sensor_layout, config["environment_setup"])
    multimodal_encoder_config = [
        (k, sensor_shapes[k], dict(encoder_configs[sensor_layout[k]["modility"]]))
        for k in input_sensors
    ]
    image_sensors = [k for k, v in sensor_layout.items() if v["modility"] == "visual"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")

    policy_encoder = MultimodalEncoder(multimodal_encoder_config)
    policy_encoder = policy_encoder.to(device)

    policy_config = config["policy"]
    policy = MLP(
        policy_encoder.output_dim * stack,
        sensor_shapes["pre_action"],
        output_activation="tanh",
        **policy_config,
    )
    policy = policy.to(device)

    loss_fn = torch.nn.MSELoss()

    logger = get_default_logger(output_folder)

    policy_optim = torch.optim.Adam(
        [*policy.parameters(), *policy_encoder.parameters()], lr=config["policy_lr"]
    )

    log.info("Training Policy ...")
    train_size = int(
        len(demonstration_dataset) * config["train_validation_split_ratio"]
    )
    val_size = len(demonstration_dataset) - train_size
    (
        demonstration_dataset_train,
        demonstration_dataset_val,