
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
        shape = v.shape
        if len(shape) == 1 or len(shape) == 2:
            shapes[k] = shape[-1]
        elif len(shape) == 3 or len(shape) == 4:
            shapes[k] = shape[-2:]
    return shapes


def get_inputs_outputs(sensor_layout, environment_setup):
    assert environment_setup in ["lpomdp", "pomdp", "mdp", "exp", "visual", "full", "real"]
    if environment_setup == "mdp":
        input_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "tabular"
        ]
        output_sensors = input_sensors
        probe_sensors = []
    elif environment_setup == "lpomdp" or environment_setup == "pomdp":
        input_sensors = [
            k
            for k, v in sensor_layout.items()
            if v["modility"] == "tabular" and v["order"] == "first"
        ]
        output_sensors = input_sensors
        probe_sensors = [
            k
            for k, v in sensor_layout.items()
            if v["modility"] == "tabular" and v["order"] == "second"
        ]
    elif environment_setup == "visual":
        input_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "visual"
        ]
        output_sensors = input_sensors
        probe_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "tabular"
        ]

    return input_sensors, output_sensors, probe_sensors


def parse_world_model_config(config, sensor_layout, example_data, predict_reward=True):
    world_model_config = dict(config["world_model"])
    input_sensors, output_sensors, probe_sensors = get_inputs_outputs(
        sensor_layout, config["environment_setup"]
    )
    sensor_shapes = get_sensor_shapes(example_data)
    sensor_layout = dict(sensor_layout)
    encoder_configs = world_model_config.pop("encoders")
    decoder_configs = world_model_config.pop("decoders")
    probe_configs = world_model_config.pop("probes")
    world_model_config["input_config"] = [
        (k, sensor_shapes[k], dict(encoder_configs[sensor_layout[k]["modility"]]))
        for k in input_sensors
    ]
    world_model_config["output_config"] = [
        (k, sensor_shapes[k], dict(decoder_configs[sensor_layout[k]["modility"]]))
        for k in output_sensors
    ]
    if predict_reward:
        world_model_config["output_config"] = world_model_config["output_config"] + [
            ("reward", 1, dict(decoder_configs["tabular"]))
        ]
    world_model_config["probe_config"] = [
        (k, sensor_shapes[k], dict(probe_configs[sensor_layout[k]["modility"]]))
        for k in probe_sensors
    ]
    world_model_config["action_dim"] = sensor_shapes["pre_action"]
    return world_model_config


def get_image_sensors(world_model_config, sensor_layout):
    image_sensors = [k for k, v in sensor_layout.items() if v["modility"] == "visual"]
    used_sensors = [config[0] for config in world_model_config["output_config"]]
    used_image_sensors = [
        image_sensor for image_sensor in image_sensors if image_sensor in used_sensors
    ]
    return image_sensors, used_image_sensors

def load_pretrained_model(model_root):
    from aime.env import env_classes
    from aime.models.ssm import ssm_classes

    config = OmegaConf.load(os.path.join(model_root, 'config.yaml'))
    env_config = config["env"]
    env_class_name = env_config["class"]
    env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"],
        render=True
    )

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(config, sensor_layout, env.observation_space, False)
    world_model_name = world_model_config.pop("name")
    model = ssm_classes[world_model_name](**world_model_config)