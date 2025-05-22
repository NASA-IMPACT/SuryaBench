from typing import Dict

import numpy as np
import torch

from datasets.transformations import Transformation, StandardScaler
from utils.config import DataConfig
from utils.misc import class_from_name, view_as_windows


def calc_num_windows(raw_size: int, win_size: int, stride: int) -> int:
    return (raw_size - win_size) // stride + 1


def get_scalers_info(dataset) -> dict:
    return {
        k: (type(v).__module__, type(v).__name__, v.to_dict())
        for k, v in dataset.scalers.items()
    }


def build_scalers_pressure(info: dict) -> Dict[str, Transformation]:
    ret_dict = {k: dict() for k in info.keys()}
    for var_key, var_d in info.items():
        for p_key, p_val in var_d.items():
            ret_dict[var_key][p_key] = class_from_name(
                p_val["base"], p_val["class"]
            ).from_dict(p_val)
    return ret_dict


def build_scalers(info: dict) -> Dict[str, Transformation]:
    ret_dict = {k: None for k in info.keys()}
    for p_key, p_val in info.items():
        ret_dict[p_key]: StandardScaler = class_from_name(
            p_val["base"], p_val["class"]
        ).from_dict(p_val)
    return ret_dict


def break_batch_5d(
    data: list, lat_size: int, lon_size: int, time_steps: int
) -> np.ndarray:
    """
    data: list of samples, each sample is [C, T, L, H, W]
    """
    num_levels = data[0].shape[2]
    num_vars = data[0].shape[0]
    big_batch = np.stack(data, axis=0)
    vw = view_as_windows(
        big_batch,
        [1, num_vars, time_steps, num_levels, lat_size, lon_size],
        step=[1, num_vars, time_steps, num_levels, lat_size, lon_size],
    ).squeeze()
    # To check if it is correctly reshaping
    # idx = 30
    # (big_batch[0, :, idx:idx+2, :, 40:80, 40:80]-vw[idx//2, 1, 1]).sum()
    vw = vw.reshape(-1, num_vars, time_steps, num_levels, lat_size, lon_size)
    # How to test:
    # (big_batch[0, :, :2, :, :40, :40] - vw[0]).sum()
    # (big_batch[0, :, :2, :, :40, 40:80] - vw[1]).sum()
    # (big_batch[0, :, :2, :, 40:80, :40] - vw[2]).sum()

    # Need to move axis because Weather model is expecting [C, L, T, H, W] instead of [C, T, L, H, W]
    vw = np.moveaxis(vw, 3, 2)
    vw = torch.tensor(vw, dtype=torch.float32)
    return vw


def break_batch_5d_aug(data: list, cfg: DataConfig, max_batch: int = 256) -> np.ndarray:
    num_levels = data[0].shape[2]
    num_vars = data[0].shape[0]
    big_batch = np.stack(data, axis=0)

    y_step, x_step, t_step = (
        cfg.patch_size_lat // 2,
        cfg.patch_size_lon // 2,
        cfg.patch_size_time // 2,
    )
    y_max = calc_num_windows(big_batch.shape[4], cfg.input_size_lat, y_step)
    x_max = calc_num_windows(big_batch.shape[5], cfg.input_size_lon, x_step)
    t_max = calc_num_windows(big_batch.shape[2], cfg.input_size_time, t_step)
    max_batch = min(max_batch, y_max * x_max * t_max)

    batch = np.empty(
        (
            max_batch,
            num_vars,
            cfg.input_size_time,
            num_levels,
            cfg.input_size_lat,
            cfg.input_size_lon,
        ),
        dtype=np.float32,
    )
    for j, i in enumerate(np.random.permutation(np.arange(max_batch))):
        t, y, x = np.unravel_index(
            i,
            (
                t_max,
                y_max,
                x_max,
            ),
        )
        batch[j] = big_batch[
            :,  # batch_id
            :,  # vars
            t * t_step : t * t_step + cfg.input_size_time,
            :,  # levels
            y * y_step : y * y_step + cfg.input_size_lat,
            x * x_step : x * x_step + cfg.input_size_lon,
        ]

    batch = np.moveaxis(batch, 3, 2)
    batch = torch.tensor(batch, dtype=torch.float32)
    return batch
