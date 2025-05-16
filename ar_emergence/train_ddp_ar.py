"""
Training script for spectral forecasting using ResNet-based binary classifier and
custom PyTorch data loaders with distributed training support.
"""

import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch
from torch import nn
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import wandb

from ds_datasets.ar import AREmergenceDataset
from utils.config import get_config
from utils.data import build_scalers
from utils.distributed import (
    init_ddp,
    print0,
    save_model_singular,
    set_global_seed,
    StatefulDistributedSampler,
)
from utils.io import create_folders
from utils.log import log
from ar_models.baselines import Average, Persistence


def custom_collate_fn(batch):
    """
    Collate function that handles (data, metadata) pairs for DataLoader.

    - Attempts to default-collate the data portion.
    - For metadata dicts, collates each key separately, falling back to list if collation fails.

    Args:
        batch (list of tuple): Each element is (data, metadata).

    Returns:
        tuple: (collated_data, collated_metadata)
    """
    data_batch, metadata_batch = zip(*batch)

    try:
        collated_data = torch.utils.data.default_collate(data_batch)
    except TypeError:
        collated_data = data_batch

    if isinstance(metadata_batch[0], dict):
        collated_metadata = {}
        for key in metadata_batch[0]:
            values = [m[key] for m in metadata_batch]
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


class ResNetBinaryClassifier(nn.Module):
    """
    Binary classifier using a pretrained ResNet backbone adapted for 13-channel input.

    Attributes:
        resnet (nn.Module): Modified ResNet model.
    """

    SUPPORTED_RESNETS = (
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    )

    def __init__(self, resnet_type="resnet152"):
        """
        Initialize the classifier.

        Args:
            resnet_type (str): Name of ResNet variant to use.
        """
        super().__init__()
        print0(f"Using ResNet type: {resnet_type}")

        if resnet_type not in self.SUPPORTED_RESNETS:
            raise ValueError(
                f"Unsupported ResNet type '{resnet_type}'. "
                f"Choose from: {', '.join(self.SUPPORTED_RESNETS)}"
            )

        constructor = getattr(models, resnet_type)
        self.resnet = constructor(pretrained=True)
        # Adapt first conv layer for 13-channel input
        self.resnet.conv1 = nn.Conv2d(
            in_channels=13,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # Single-output head for binary classification
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits.
        """
        return self.resnet(x)


def evaluate_model(dataloader, model, device, run, criterion=nn.BCELoss()):
    """
    Evaluate model on validation set and log metrics.

    Args:
        dataloader (DataLoader): Validation data loader.
        model (nn.Module): Trained model.
        device (torch.device): Compute device.
        run (wandb.Run): WandB run object (or None).
        criterion (nn.Module): Loss function.

    Returns:
        tuple: (accuracy, avg_loss)
    """
    model.eval()
    total, correct = 0, 0
    running_loss, num_batches = 0.0, 0
    all_preds, all_labels = [], []
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        for batch_idx, (batch, metadata) in enumerate(dataloader):
            if config.iters_per_epoch_valid == batch_idx:
                break

            data = batch["ts"].squeeze(2).to(device)
            target = batch["target"].unsqueeze(1).to(device).float()

            with autocast(device_type="cuda", dtype=config.dtype):
                outputs = model(data)
                loss = criterion(outputs, target)

            running_loss += loss.item()
            num_batches += 1

            preds = (sigmoid(outputs) > 0.5).float()
            correct += (preds == target).sum().item()
            total += target.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    tp = int(((preds_np == 1) & (labels_np == 1)).sum())
    tn = int(((preds_np == 0) & (labels_np == 0)).sum())
    fp = int(((preds_np == 1) & (labels_np == 0)).sum())
    fn = int(((preds_np == 0) & (labels_np == 1)).sum())

    stats = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "loss_sum": running_loss,
        "num_batches": num_batches,
        "total": float(total),
    }

    for key, value in stats.items():
        tensor = torch.tensor(value, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        stats[key] = tensor.item()

    accuracy = (stats["tp"] + stats["tn"]) / sum(
        stats[k] for k in ("tp", "tn", "fp", "fn")
    )
    precision = (
        stats["tp"] / (stats["tp"] + stats["fp"]) if stats["tp"] + stats["fp"] else 0.0
    )
    recall = (
        stats["tp"] / (stats["tp"] + stats["fn"]) if stats["tp"] + stats["fn"] else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_loss = stats["loss_sum"] / stats["num_batches"]

    print0(
        f"Validation Accuracy: {accuracy:.4f} "
        f"Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, "
        f"F1: {f1:.4f}, "
        f"Avg Loss: {avg_loss:.4f}"
    )

    if dist.get_rank() == 0:
        log(
            run,
            {
                "valid/accuracy": accuracy,
                "valid/precision": precision,
                "valid/recall": recall,
                "valid/f1": f1,
                "valid/loss": avg_loss,
                "confusion/tp": stats["tp"],
                "confusion/tn": stats["tn"],
                "confusion/fp": stats["fp"],
                "confusion/fn": stats["fn"],
            },
        )

    return accuracy, avg_loss


def wrap_all_checkpoints(model):
    """
    Apply activation checkpointing to eligible submodules.

    Args:
        model (nn.Module): The model to wrap.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.Sequential, nn.Linear, nn.Conv2d)):
            wrapped = checkpoint_wrapper(
                module, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )
            setattr(model, name, wrapped)


def main(config, use_gpu: bool, use_wandb: bool, profile: bool):
    """
    Main training loop with distributed setup, logging, and checkpointing.

    Args:
        config (omegaconf.DictConfig): Configuration object.
        use_gpu (bool): Flag to enable CUDA.
        use_wandb (bool): Flag to enable WandB logging.
        profile (bool): Flag to enable profiling.
    """
    run = None
    local_rank, rank = init_ddp(use_gpu)
    print(f"RANK: {rank}; LOCAL_RANK: {local_rank}")

    scalers = build_scalers(info=config.data.scalers)

    if dist.get_rank() == 0:
        create_folders(config)

    if use_wandb and dist.get_rank() == 0:
        slurm_job_id = os.getenv("SLURM_JOB_ID", "")
        run = wandb.init(
            project=config.wandb_project,
            entity="nasa-impact",
            name=f"[JOB:{slurm_job_id}] {config.job_id}",
            config=config.to_dict(),
            mode="online",
        )
        wandb.save(args.config_path)

    train_ds = AREmergenceDataset(config.data.train_data_path)
    valid_ds = AREmergenceDataset(config.data.valid_data_path)

    print0(f"Training set size: {len(train_ds)}")
    print0(f"Validation set size: {len(valid_ds)}")

    dl_kwargs = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_data_workers,
        "prefetch_factor": config.data.prefetch_factor,
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": custom_collate_fn,
    }

    train_loader = DataLoader(
        dataset=train_ds,
        sampler=StatefulDistributedSampler(train_ds, drop_last=True),
        **dl_kwargs,
    )
    valid_loader = DataLoader(
        dataset=valid_ds,
        sampler=StatefulDistributedSampler(valid_ds, drop_last=True),
        **dl_kwargs,
    )

    model_type = config.model.model_type.lower()
    if "resnet" in model_type:
        model = ResNetBinaryClassifier(resnet_type=config.model.model_type).to(
            local_rank
        )
    elif "persistence" in model_type:
        model = Persistence().to(local_rank)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")

    if config.model.checkpoint_layers:
        print0("Applying checkpointing...")
        wrap_all_checkpoints(model)

    total_params = sum(p.numel() for p in model.parameters())
    print0(f"Total parameters: {total_params:,}")

    model = DistributedDataParallel(
        model, device_ids=[torch.cuda.current_device()], find_unused_parameters=False
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)
    scaler = GradScaler()

    for epoch in range(config.optimizer.max_epochs):
        epoch_loss = torch.tensor(0.0, device=local_rank)
        epoch_batches = torch.tensor(0, device=local_rank)

        for i, (batch, metadata) in enumerate(train_loader):
            if config.iters_per_epoch_train == i:
                break

            model.train()
            data = batch["input"].to(local_rank)
            target = batch["output"].to(local_rank).float()

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=config.dtype):
                outputs = model(data)
                loss = criterion(outputs, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            reduced_loss = loss.detach()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= dist.get_world_size()

            epoch_loss += reduced_loss
            epoch_batches += 1

            if i % config.wandb_log_train_after == 0 and dist.get_rank() == 0:
                print(f"Epoch {epoch}, batch {i}, loss {reduced_loss.item():.4f}")
                log(run, {"train/loss": reduced_loss.item()})

            if (i + 1) % config.save_wt_after_iter == 0:
                print0(f"Saving checkpoint at iteration {i + 1}")
                path = os.path.join(config.path_weights, "checkpoint.pth")
                save_model_singular(model, path, parallelism=config.parallelism)

        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_batches, op=dist.ReduceOp.SUM)

        log(run, {"epoch/loss": (epoch_loss / epoch_batches).item()})

        epoch_path = os.path.join(config.path_weights, f"epoch_{epoch}.pth")
        save_model_singular(model, epoch_path, parallelism=config.parallelism)
        print0(f"Epoch {epoch} saved: {epoch_path}")

        evaluate_model(valid_loader, model, local_rank, run, criterion)


if __name__ == "__main__":
    set_global_seed(0)

    parser = argparse.ArgumentParser(description="SpectFormer Training")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to config YAML."
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (CUDA) usage.")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable performance profiling."
    )
    args = parser.parse_args()

    config = get_config(args.config_path)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if args.gpu:
        config.dtype = dtype_map.get(config.dtype, None)
        if config.dtype is None:
            raise NotImplementedError(f"Unsupported dtype {config.dtype}")
    else:
        raise ValueError("CPU mode not supported. Use --gpu flag.")

    main(config, use_gpu=args.gpu, use_wandb=args.wandb, profile=args.profile)
    torch.distributed.destroy_process_group()
