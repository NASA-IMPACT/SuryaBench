import argparse
import os
from logging import Logger
from time import strftime

import torch
import yaml

from utils import distributed
from utils.config import ExperimentConfig


def make_folder_name(config: ExperimentConfig) -> str:
    param_folder = f"wpt-c{config.data.input_size_time}-s{config.data.input_size_time}"
    return param_folder


def create_config_yaml(args: argparse.Namespace, job_id: str, save_path: str) -> dict:
    params_dict = dict(vars(args))
    # Add more info
    params_dict["job_id"] = job_id
    params_dict["checkpoint_dir"] = save_path
    # params_dict['visualization_dir'] = vis_dir
    params_dict["tensorboard_dir"] = os.path.join(
        args.base_log_dir, make_folder_name(args), job_id, "tensorboard"
    )
    params_dict["csv_dir"] = os.path.join(
        args.base_log_dir, make_folder_name(args), job_id, "csv"
    )
    params_dict["world_size"] = int(os.environ["WORLD_SIZE"])
    params_dict["job_start_time"] = f"{strftime('%Y-%m-%d %H:%M:%S')}"
    params_dict["job_finish_time"] = "NA"
    return params_dict


def dump_config_yaml(config: ExperimentConfig, logger: Logger):
    fname = os.path.join(
        config.path_base_log, make_folder_name(config), config.job_id, "config.yaml"
    )
    config.path_experiment = fname
    with open(fname, "w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=True)
    if distributed.is_main_process():
        logger.info(f"Configuration saved at: {fname}")


def create_folders(config: ExperimentConfig):
    os.makedirs(config.path_weights, mode=0o775, exist_ok=True)


def print_model_diagram(
    model: torch.nn.Module,
    save_path: str,
    format: str = "pdf",
    **kwargs,
):
    """Save a diagram with model's structure."""
    from torchview import draw_graph

    with torch.inference_mode():
        model_graph_1 = draw_graph(model, **kwargs)
    fig = model_graph_1.visual_graph.pipe(format=format)
    with open(save_path, "wb") as f:
        f.write(fig)


def load_checkpoint(path, model_without_ddp, optimizer, scheduler, logger):
    start_epoch = 1
    if distributed.is_main_process():
        print(f"Resuming Training from {path}...")
    checkpoint = torch.load(path, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])

    if "optimizer" in checkpoint and "epoch" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    if distributed.is_main_process():
        print(f"\nStarting training at epoch {start_epoch}...")

    return start_epoch


def save_checkpoint(save_dict: dict, file_path: str):
    # Guaranteeing checkpoint directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(save_dict, file_path)
