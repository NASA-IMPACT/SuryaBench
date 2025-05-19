import warnings

from ds_datasets.ar import AREmergenceDataset

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm

# Resolve HelioFM path based on script's location
sys.path.insert(0, "../../HelioFM")
from utils.config import get_config
from utils.data import build_scalers
from torch.utils.data import DataLoader, Subset
from torchvision import models

# Now try imports
import torch

# set all seeds
import random
import numpy as np
import wandb
import os

from ar_models.baselines import TestModel

# from ds_models.fl_unet import UNet


def custom_collate_fn(batch):
    """
    Custom collate function for handling batches of data and metadata in a PyTorch DataLoader.

    This function separately processes the data and metadata from the input batch.

    - The `data_batch` is collated using PyTorch's `default_collate`. If collation fails due to incompatible data types,
    the batch is returned as-is.

    - The `metadata_batch` is assumed to be a dictionary, where each key corresponds to a list of values across the batch.
    Each key is collated using `default_collate`. If collation fails for a particular key, the original list of values
    is retained.

    Example usage for accessing collated metadata:
        - `collated_metadata['timestamps_input'][batch_idx][input_time]`
        - `collated_metadata['timestamps_input'][batch_idx][rollout_step]`

    Args:
        batch (list of tuples): Each tuple contains (data, metadata), where:
            - `data` is a tensor or other data structure used for training.
            - `metadata` is a dictionary containing additional information.

    Returns:
        tuple: (collated_data, collated_metadata)
            - `collated_data`: The processed batch of data.
            - `collated_metadata`: The processed batch of metadata.
    """

    # Unpack batch into separate lists of data and metadata
    data_batch, metadata_batch = zip(*batch)

    # Attempt to collate the data batch using PyTorch's default collate function
    try:
        collated_data = torch.utils.data.default_collate(data_batch)
    except TypeError:
        # If default_collate fails (e.g., due to incompatible types), return the data batch as-is
        collated_data = data_batch

    # Handle metadata collation
    if isinstance(metadata_batch[0], dict):
        collated_metadata = {}
        for key in metadata_batch[0].keys():
            values = [d[key] for d in metadata_batch]
            try:
                # Attempt to collate values under the current key
                collated_metadata[key] = torch.utils.data.default_collate(values)
            except TypeError:
                # If collation fails, keep the values as a list
                collated_metadata[key] = values
    else:
        # If metadata is not a dictionary, try to collate it as a whole
        try:
            collated_metadata = torch.utils.data.default_collate(metadata_batch)
        except TypeError:
            # If collation fails, return metadata as-is
            collated_metadata = metadata_batch

    return collated_data, collated_metadata


def main(config, use_gpu: bool, use_wandb: bool, profile: bool):

    train_subset = AREmergenceDataset(
        config.data.train_data_path,
    )
    valid_subset = AREmergenceDataset(
        config.data.valid_data_path,
    )

    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(valid_subset)}")

    dl_kwargs = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_data_workers,
        prefetch_factor=config.data.prefetch_factor,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    train_loader = DataLoader(
        dataset=train_subset,
        # sampler=StatefulDistributedSampler(train_subset, drop_last=True),
        **dl_kwargs,
    )

    valid_loader = DataLoader(
        dataset=valid_subset,
        # sampler=StatefulDistributedSampler(valid_subset, drop_last=True),
        **dl_kwargs,
    )

    if "test_model" in config.model.model_type:
        model = TestModel()
    else:
        raise NotImplementedError("Please choose from [persistence, average]")

    device = torch.device("cuda" if use_gpu else "cpu")
    criterion = torch.nn.MSELoss()
    running_loss = torch.tensor(0.0, device=device)
    running_batch = torch.tensor(0, device=device)

    for i, (batch, metadata) in enumerate(train_loader):

        data, target = batch["input"], batch["output"]

        # Move data to device
        data, target = data.to(device), target.to(device).float()

        outputs = model(data)
        loss = criterion(outputs, target)

        running_loss += loss
        running_batch += data.shape[0]
        if i % config.wandb_log_train_after == 0:
            print(f"Batch {i+1}/{len(train_loader)}: Loss = {loss.item()}")

    print(f"Total loss: {running_loss.item()/running_batch.item()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("SpectFormer Training")
    parser.add_argument(
        "--config_path", type=str, help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--gpu", default=False, action="store_true", help="Run on GPU CUDA."
    )
    parser.add_argument(
        "--wandb", default=False, action="store_true", help="Log into WanDB."
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    config = get_config(args.config_path)

    if config.dtype == "float16":
        config.dtype = torch.float16
    elif config.dtype == "bfloat16":
        config.dtype = torch.bfloat16
    elif config.dtype == "float32":
        config.dtype = torch.float32
    else:
        raise NotImplementedError("Please choose from [float16,bfloat16,float32]")

    if not args.gpu:
        raise ValueError(
            "Training scripts are not configured for CPU use. Please set the `--gpu` flag."
        )

    main(config=config, use_gpu=args.gpu, use_wandb=args.wandb, profile=args.profile)
    torch.distributed.destroy_process_group()
