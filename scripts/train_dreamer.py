
import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from aime.actor import GuassianNoiseActorWrapper, PolicyActor, RandomActor
from aime.data import NPZFolder, get_sample_loader
from aime.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aime.logger import get_default_logger
from aime.models.base import MLP
from aime.models.policy import TanhGaussianPolicy
from aime.models.ssm import ssm_classes
from aime.utils import (
    CONFIG_PATH,
    MODEL_PATH,
    OUTPUT_PATH,
    AverageMeter,
    generate_prediction_videos,
    get_image_sensors,
    interact_with_environment,
    lambda_return,
    need_render,
    parse_world_model_config,
    setup_seed,
)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="dreamer")
def main(config: DictConfig):
    setup_seed(config["seed"])

    log.info("using the following config:")
    log.info(config)

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    dataset_folder = os.path.join(output_folder, "train_trajectories")
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    env_config = config["env"]
    env_class_name = env_config["class"]
    env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"],
        render=env_config["render"] or need_render(config["environment_setup"]),
    )
    env = SaveTrajectories(env, dataset_folder)
    env = TerminalSummaryWrapper(env)
    env.action_space.seed(config["seed"])
    test_env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"] * 2,
        render=True,
    )
    test_env = SaveTrajectories(test_env, eval_folder)
    test_env = TerminalSummaryWrapper(test_env)

    # collect initial dataset
    for _ in range(config["prefill"]):
        actor = RandomActor(env.action_space)
        interact_with_environment(env, actor, [])

    dataset = NPZFolder(dataset_folder, config["horizon"], overlap=True)
    eval_dataset = NPZFolder(eval_folder, config["horizon"], overlap=False)
    data = dataset[0]

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(config, sensor_layout, data)
    world_model_name = world_model_config.pop("name")
    image_sensors, used_image_sensors = get_image_sensors(
        world_model_config, sensor_layout
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")
    model = ssm_classes[world_model_name](**world_model_config)
    if config["pretrained_model_name"] is not None:
        pretrained_paramters = torch.load(
            os.path.join(MODEL_PATH, config["pretrained_model_name"], "model.pt"),
            map_location="cpu",
        )
        model.load_state_dict(pretrained_paramters, strict=False)
        if config["freeze_pretrained_parameters"]:
            for name, parameter in model.named_parameters():
                if name in pretrained_paramters.keys():
                    parameter.requires_grad_(False)
    model = model.to(device)

    policy_config = config["policy"]
    policy = TanhGaussianPolicy(
        model.state_feature_dim, world_model_config["action_dim"], **policy_config
    )
    policy = policy.to(device)

    vnet_config = config["vnet"]
    vnet = MLP(model.state_feature_dim, 1, **vnet_config)
    vnet = vnet.to(device)

    logger = get_default_logger(output_folder)

    model_optim = torch.optim.Adam(model.parameters(), lr=config["model_lr"])
    model_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])
    policy_optim = torch.optim.Adam(policy.parameters(), lr=config["policy_lr"])
    policy_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])