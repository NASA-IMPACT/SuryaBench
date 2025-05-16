import os
import sys
import torch
import wandb
import random
import argparse
import warnings
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

# Project-specific imports
from ds_datasets.ar import AREmergenceDataset
from ar_models.baselines import TestModel
from ar_models.spatio_temporal_attention import SpatioTemporalAttention

# HelioFM utilities
sys.path.insert(0, "../../HelioFM")
from utils.config import get_config
from utils.io import create_folders

warnings.filterwarnings("ignore", category=UserWarning)


def custom_collate_fn(batch):
    """
    Custom collate function to handle nested metadata.
    """
    data_batch, metadata_batch = zip(*batch)
    try:
        collated_data = torch.utils.data.default_collate(data_batch)
    except TypeError:
        collated_data = data_batch

    if isinstance(metadata_batch[0], dict):
        collated_metadata = {}
        for key in metadata_batch[0].keys():
            values = [d[key] for d in metadata_batch]
            try:
                collated_metadata[key] = torch.utils.data.default_collate(values)
            except TypeError:
                collated_metadata[key] = values
    else:
        try:
            collated_metadata = torch.utils.data.default_collate(metadata_batch)
        except TypeError:
            collated_metadata = metadata_batch

    return collated_data, collated_metadata


def validate_model_mse(model, valid_loader, device, criterion):
    """
    Evaluate the model and return average MSE.

    Args:
        model: Trained model.
        valid_loader: Validation dataloader.
        device: torch.device.
        criterion: Loss function.

    Returns:
        float: Validation loss (MSE).
    """
    model.eval()
    running_loss = 0.0
    running_batch = 0

    with torch.no_grad():
        for batch, _ in valid_loader:
            x, y = batch["input"], batch["output"]
            x, y = x.to(device), y.to(device).float()

            preds = model(x)
            loss = criterion(preds, y)

            running_loss += loss.item()
            running_batch += x.size(0)

    return running_loss / running_batch


def main(config, use_gpu: bool, use_wandb: bool, profile: bool):
    """
    Main training routine.
    """
    create_folders(config)
    device = torch.device("cuda" if use_gpu else "cpu")
    mode = "online" if use_wandb else "disabled"

    slurm_job_id = os.getenv("SLURM_JOB_ID", "local")

    run = wandb.init(
        project=config.wandb_project,
        entity="nasa-impact",
        name=f"[JOB: {slurm_job_id}] {config.job_id}",
        config=config.to_dict(),
        mode=mode,
    )

    train_dataset = AREmergenceDataset(config.data.train_data_path)
    valid_dataset = AREmergenceDataset(config.data.valid_data_path)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")

    dl_kwargs = dict(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_data_workers,
        prefetch_factor=config.data.prefetch_factor or 2,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    train_loader = DataLoader(train_dataset, **dl_kwargs)
    valid_loader = DataLoader(valid_dataset, **dl_kwargs)

    if config.model.model_type == "test_model":
        model = TestModel()
    elif config.model.model_type == "spatio_temporal_attention":
        model = SpatioTemporalAttention()
    else:
        raise NotImplementedError(
            "Unsupported model_type. Choose from ['test_model', 'spatio_temporal_attention']"
        )

    model.to(device)
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)

    model.train()
    for epoch in range(config.optimizer.max_epochs):
        running_loss = torch.tensor(0.0, device=device)
        running_batch = torch.tensor(0, device=device)

        for batch, _ in train_loader:
            x, y = batch["input"], batch["output"]
            x, y = x.to(device), y.to(device).float()

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss
            running_batch += x.size(0)

        epoch_loss = running_loss.item() / running_batch.item()
        print(f"Epoch {epoch+1} completed. Average Loss = {epoch_loss:.6f}")
        wandb.log({"train_loss": epoch_loss}, step=epoch + 1)

        if (epoch + 1) % config.validate_after_epoch == 0:
            valid_loss = validate_model_mse(model, valid_loader, device, criterion)
            print(f"Validation Loss = {valid_loss:.6f}")
            wandb.log({"valid_loss": valid_loss}, step=epoch + 1)

        if (epoch + 1) % config.save_wt_after_iter == 0:
            save_path = os.path.join(config.path_weights, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SpectFormer Training")
    parser.add_argument(
        "--config_path", type=str, required=True, help="YAML config file path."
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling (not implemented)."
    )
    args = parser.parse_args()

    config = get_config(args.config_path)
    config.config_path = args.config_path  # For wandb.save()

    # Handle dtype
    if config.dtype == "float16":
        config.dtype = torch.float16
    elif config.dtype == "bfloat16":
        config.dtype = torch.bfloat16
    elif config.dtype == "float32":
        config.dtype = torch.float32
    else:
        raise NotImplementedError("dtype must be one of: float16, bfloat16, float32")

    wandb.save(args.config_path)
    main(config=config, use_gpu=args.gpu, use_wandb=args.wandb, profile=args.profile)
    wandb.finish()
    print("Training completed.")
