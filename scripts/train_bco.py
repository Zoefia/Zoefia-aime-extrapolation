
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
    need_render,
    setup_seed,
)

log = logging.getLogger("main")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="bco")
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
    embodiment_dataset_folder = os.path.join(
        DATA_PATH, config["embodiment_dataset_name"]
    )
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
        render=need_render(config["environment_setup"]) or config["render"],
    )
    test_env = SaveTrajectories(test_env, eval_folder)
    test_env = TerminalSummaryWrapper(test_env)

    log.info("Loading datasets ...")
    embodiment_dataset = NPZFolder(embodiment_dataset_folder, stack + 1, overlap=True)
    demonstration_dataset = NPZFolder(
        demonstration_dataset_folder, stack + 1, overlap=True
    )
    demonstration_dataset.keep(config["num_expert_trajectories"])
    eval_dataset = NPZFolder(eval_folder, stack + 1, overlap=False)
    data = embodiment_dataset[0]

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
    idm_encoder = MultimodalEncoder(multimodal_encoder_config)
    idm_encoder = idm_encoder.to(device)

    # not sure whether we should share the encoder weights
    policy_encoder = MultimodalEncoder(multimodal_encoder_config)
    policy_encoder = policy_encoder.to(device)

    idm_config = config["idm"]
    idm = MLP(
        idm_encoder.output_dim * (stack + 1),
        sensor_shapes["pre_action"],
        output_activation="tanh",
        **idm_config,
    )
    idm = idm.to(device)

    policy_config = config["policy"]
    policy = MLP(
        idm_encoder.output_dim * stack,
        sensor_shapes["pre_action"],
        output_activation="tanh",
        **policy_config,
    )
    policy = policy.to(device)

    loss_fn = torch.nn.MSELoss()

    logger = get_default_logger(output_folder)

    idm_optim = torch.optim.Adam(
        [*idm.parameters(), *idm_encoder.parameters()], lr=config["idm_lr"]
    )
    policy_optim = torch.optim.Adam(
        [*policy.parameters(), *policy_encoder.parameters()], lr=config["policy_lr"]
    )

    log.info("Training IDM ...")
    train_size = int(len(embodiment_dataset) * config["train_validation_split_ratio"])
    val_size = len(embodiment_dataset) - train_size
    embodiment_dataset_train, embodiment_dataset_val = torch.utils.data.random_split(
        embodiment_dataset, [train_size, val_size]
    )
    train_loader = get_epoch_loader(
        embodiment_dataset_train, config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = get_epoch_loader(
        embodiment_dataset_val, config["batch_size"], shuffle=False, num_workers=4
    )
    e = 0
    s = 0
    best_val_loss = float("inf")
    convergence_count = 0
    while True:
        log.info(f"Starting epcoh {e}")
        metrics = {}
        train_metric_tracker = AverageMeter()
        for data in tqdm(iter(train_loader)):
            data = data.to(device)
            emb = idm_encoder(data)
            emb = rearrange(emb, "t b f -> b (t f)")
            predict_action = idm(emb)
            loss = loss_fn(predict_action, data[-1]["pre_action"])

            idm_optim.zero_grad()
            loss.backward()
            idm_optim.step()
            s += 1

            train_metric_tracker.add({"train/idm_loss": loss.item()})

        metrics.update(train_metric_tracker.get())

        val_metric_tracker = AverageMeter()
        with torch.no_grad():
            for data in tqdm(iter(val_loader)):
                data = data.to(device)
                emb = idm_encoder(data)
                emb = rearrange(emb, "t b f -> b (t f)")
                predict_action = idm(emb)
                loss = loss_fn(predict_action, data[-1]["pre_action"])

                val_metric_tracker.add({"val/idm_loss": loss.item()})

        metrics.update(val_metric_tracker.get())

        logger(metrics, e)

        e += 1

        if metrics["val/idm_loss"] < best_val_loss:
            best_val_loss = metrics["val/idm_loss"]
            torch.save(
                idm_encoder.state_dict(), os.path.join(output_folder, "idm_encoder.pt")
            )
            torch.save(idm.state_dict(), os.path.join(output_folder, "idm.pt"))
            convergence_count = 0
        else:
            convergence_count += 1
            if (
                convergence_count >= config["patience"]
                and e >= config["min_idm_epoch"]
                and s >= config["min_idm_steps"]
            ):
                break

    log.info(f"IDM training finished in {e} epoches!")

    # restore the best idm
    idm_encoder.load_state_dict(
        torch.load(os.path.join(output_folder, "idm_encoder.pt"))
    )
    idm.load_state_dict(torch.load(os.path.join(output_folder, "idm.pt")))
    idm_encoder.requires_grad_(False)
    idm.requires_grad_(False)

    # I think reusing the weight is a good thing to go
    policy_encoder.load_state_dict(idm_encoder.state_dict())

    log.info("Training Policy ...")
    train_size = int(
        len(demonstration_dataset) * config["train_validation_split_ratio"]
    )
    val_size = len(demonstration_dataset) - train_size
    (
        demonstration_dataset_train,
        demonstration_dataset_val,
    ) = torch.utils.data.random_split(demonstration_dataset, [train_size, val_size])
    train_loader = get_epoch_loader(
        demonstration_dataset_train, config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = get_epoch_loader(
        demonstration_dataset_val, config["batch_size"], shuffle=False, num_workers=4
    )
    e = 0
    s = 0
    best_val_loss = float("inf")
    convergence_count = 0
    while True:
        log.info(f"Starting epcoh {e}")

        metrics = {}
        train_metric_tracker = AverageMeter()
        for data in tqdm(iter(train_loader)):
            data = data.to(device)
            emb_idm = idm_encoder(data)
            emb_idm = rearrange(emb_idm, "t b f -> b (t f)")
            idm_action = idm(emb_idm)