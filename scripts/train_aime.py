
import logging
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from aime.actor import PolicyActor
from aime.data import NPZFolder, get_sample_loader
from aime.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aime.logger import get_default_logger
from aime.models.policy import TanhGaussianPolicy
from aime.models.ssm import ssm_classes
from aime.utils import (
    CONFIG_PATH,
    DATA_PATH,
    MODEL_PATH,
    OUTPUT_PATH,
    AverageMeter,
    generate_prediction_videos,
    get_image_sensors,
    interact_with_environment,
    need_render,
    parse_world_model_config,
    setup_seed,
)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="aime")
def main(config: DictConfig):
    setup_seed(config["seed"])

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    model_folder = os.path.join(MODEL_PATH, config["model_name"])
    dataset_folder = os.path.join(DATA_PATH, config["demonstration_dataset_name"])
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    env_config = config["env"]
    env_class_name = env_config["class"]
    render = env_config["render"] or need_render(config["environment_setup"])
    test_env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"] + 100,
        render=render,
    )
    test_env = SaveTrajectories(test_env, eval_folder)
    test_env = TerminalSummaryWrapper(test_env)

    dataset = NPZFolder(dataset_folder, config["horizon"], overlap=True)
    dataset.keep(config["num_expert_trajectories"])
    log.info(f"Training on {len(dataset.trajectories)} expert trajectories!")
    eval_dataset = NPZFolder(eval_folder, config["horizon"], overlap=False)
    data = dataset[0]

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(config, sensor_layout, data, False)
    world_model_name = world_model_config.pop("name")
    image_sensors, used_image_sensors = get_image_sensors(
        world_model_config, sensor_layout
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ssm_classes[world_model_name](**world_model_config)
    model.load_state_dict(
        torch.load(os.path.join(model_folder, "model.pt"), map_location="cpu"),
        strict=False,
    )
    model = model.to(device)
    if "reward" in model.decoders.keys():
        model.decoders.pop("reward")
    if config["freeze_model"]:
        log.info("freeze the model weights")
        model.requires_grad_(False)

    # load the pretrained policy
    policy_config = config["policy"]
    policy_file = os.path.join(model_folder, "policy.pt")
    if os.path.exists(policy_file):
        policy = TanhGaussianPolicy(
            model.state_feature_dim, world_model_config["action_dim"], **policy_config
        )
        policy.load_state_dict(
            torch.load(os.path.join(model_folder, "policy.pt"), map_location="cpu")
        )
        policy = policy.to(device)

        # directly test this model and policy on the new task
        log.info("Evaluating the pretrained model and policy ...")
        with torch.no_grad():
            actor = PolicyActor(model, policy)
            interact_with_environment(test_env, actor, image_sensors)
        eval_dataset.update()

    # reinitialize the policy to random policy
    if config["random_policy"]:
        policy = TanhGaussianPolicy(
            model.state_feature_dim, world_model_config["action_dim"], **policy_config
        )
        policy = policy.to(device)
        log.info("Evaluating the pretrained model and random policy ...")
        with torch.no_grad():
            actor = PolicyActor(model, policy)
            interact_with_environment(test_env, actor, image_sensors)
        eval_dataset.update()

    if config["use_idm"]:
        idm = model.idm
        # remove the idm from the model, so that it won't be count twice in optimizor.
        model.idm = None
        # idm.requires_grad_(True)
    else:
        idm = None

    logger = get_default_logger(output_folder)

    parameters = [*policy.parameters(), *model.parameters()]