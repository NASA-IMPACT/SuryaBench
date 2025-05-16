import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from torch.utils.data import DataLoader
from ds_datasets.ar import AREmergenceDataset
from train_baselines import custom_collate_fn
from ar_models.spatio_temporal_attention import SpatioTemporalAttention


def validate_model_mse(model, valid_loader, device, criterion):
    """
    Compute MSE on the validation set.

    Args:
        model: Trained PyTorch model.
        valid_loader: Validation DataLoader.
        device: torch.device.
        criterion: Loss function (e.g. MSELoss).

    Returns:
        float: Average MSE across all samples.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch, metadata in valid_loader:
            x, y = batch["input"], batch["output"]
            x = x[:, :, :, 27:37]
            y = y[:, 27:37]
            x, y = x.to(device), y.to(device).float()

            preds = model(x)
            loss = criterion(preds, y)

            total_loss += loss.item()
            total_samples += x.size(0)

    return total_loss / total_samples


def validate_model_graphs(model, valid_loader, device, criterion):
    """
    Runs inference on the validation set and plots predictions vs. targets
    for specific ARs (based on month filtering).
    """
    model.eval()
    all_outputs, all_targets, all_times = [], [], []

    with torch.no_grad():
        for batch, metadata in valid_loader:
            x, y = batch["input"], batch["output"]
            x = x[:, :, :, 37:47]
            y = y[:, 37:47]
            x, y = x.to(device), y.to(device).float()

            preds = model(x)

            all_outputs.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_times.append(np.array(metadata["output_time"]))

    outputs = np.concatenate(all_outputs)
    targets = np.concatenate(all_targets)
    times = np.concatenate(all_times)

    # Sort by timestamps
    sorted_idx = np.argsort(times)
    outputs, targets, times = (
        outputs[sorted_idx],
        targets[sorted_idx],
        times[sorted_idx],
    )

    # Filter by month (March → AR11698, April → AR11726)
    mask_11698 = np.array(
        [int(str(t).split("-")[1]) == 3 if "-" in str(t) else False for t in times]
    )
    mask_11726 = np.array(
        [int(str(t).split("-")[1]) == 4 if "-" in str(t) else False for t in times]
    )

    plot_ar11698(outputs[mask_11698], targets[mask_11698], times[mask_11698])
    plot_ar11726(outputs[mask_11726], targets[mask_11726], times[mask_11726])


def plot_ar(outputs, targets, times, save_path, title_prefix, starting_tile=0):
    """
    Plot predictions vs targets for specific tiles.

    Args:
        outputs: Model predictions [N, D].
        targets: Ground truth targets [N, D].
        times: Timestamps [N].
        save_path: Path to save the plot.
        title_prefix: Title label prefix.
        starting_tile: Tile index offset.
    """
    num_tiles = outputs.shape[1]
    times = [
        (
            datetime.strptime(t.decode("utf-8"), "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(t, bytes)
            else pd.Timestamp(t).to_pydatetime()
        )
        for t in times
    ]

    fig, axes = plt.subplots(num_tiles, 1, figsize=(12, 3 * num_tiles), sharex=True)

    for i in range(num_tiles):
        ax = axes[i] if num_tiles > 1 else axes
        ax.plot(times, outputs[:, i], label="Model Output", color="blue")
        ax.plot(times, targets[:, i], label="Target", color="red")

        ax.set_ylabel("Value")
        ax.set_ylim(0, 1)
        ax.set_title(f"{title_prefix} (Tile {i + starting_tile})")
        ax.legend()

        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ar11698(outputs, targets, times):
    plot_ar(
        outputs=outputs[:, 1:7],
        targets=targets[:, 1:7],
        times=times,
        save_path="ar11698_val_plot.png",
        title_prefix="AR11698",
        starting_tile=9,
    )


def plot_ar11726(outputs, targets, times):
    plot_ar(
        outputs=outputs[:, 2:8],
        targets=targets[:, 2:8],
        times=times,
        save_path="ar11726_val_plot.png",
        title_prefix="AR11726",
        starting_tile=10,
    )


def main(args, device):
    """
    Main entry point for model inference and evaluation.
    """
    # Load model
    if args.model_type == "spatio_temporal_attention":
        model = SpatioTemporalAttention(spatial_cells=10)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Load weights
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    print(f"Model loaded from {args.checkpoint_path}")
    model.to(device)

    # Dataset and DataLoader
    dataset = AREmergenceDataset(args.valid_data_path)
    dl_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_data_workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": custom_collate_fn,
    }
    val_loader = DataLoader(dataset, **dl_kwargs)

    # Run validation
    criterion = torch.nn.MSELoss(reduction="sum")
    mse = validate_model_mse(model, val_loader, device, criterion)
    validate_model_graphs(model, val_loader, device, criterion)
    print(f"Validation Loss - MSE: {mse:.4f} | RMSE: {np.sqrt(mse):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for AR validation")
    parser.add_argument(
        "--valid_data_path", type=str, required=True, help="Path to validation dataset"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_data_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--model_type",
        type=str,
        default="spatio_temporal_attention",
        help="Model type (default: spatio_temporal_attention)",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args, device)
