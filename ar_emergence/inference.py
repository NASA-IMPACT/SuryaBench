import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ds_datasets.ar import AREmergenceDataset
from torch.utils.data import DataLoader, Subset
from train_baselines import custom_collate_fn
import pandas as pd
import matplotlib.dates as mdates  # Import mdates for date handling

from ar_models.spatio_temporal_attention import SpatioTemporalAttention
from datetime import datetime, timedelta


def validate_model_mse(model, valid_loader, device, criterion):
    """
    Validate the model on the validation set and compute the RSME.

    Args:
        model: The trained model to be validated.
        valid_loader: DataLoader for the validation dataset.
        device: Device to perform computations on (CPU or GPU).

    Returns:
        float: The computed RSME value.
    """
    model.eval()
    running_loss = 0.0
    running_batch = 0

    with torch.no_grad():
        for i, (batch, metadata) in enumerate(valid_loader):
            data, target = batch["input"], batch["output"]

            # Move data to device
            data, target = data.to(device), target.to(device).float()

            # Forward pass
            outputs = model(data)
            # Compute loss
            loss = criterion(outputs, target)
            running_loss += loss.item()
            running_batch += data.shape[0]
    epoch_loss = running_loss / running_batch

    return epoch_loss


def validate_model_graphs(model, valid_loader, device, criterion):

    model.eval()
    running_loss = 0.0
    running_batch = 0
    all_outputs = []
    all_targets = []
    all_times = []

    with torch.no_grad():
        for i, (batch, metadata) in enumerate(valid_loader):
            data, target = batch["input"], batch["output"]

            # Move data to device
            data, target = data.to(device), target.to(device).float()

            # Forward pass
            outputs = model(data)
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_times.append(np.array(metadata["output_time"]))

    # Concatenate all collected data
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_times = np.concatenate(all_times, axis=0)

    # Reorder based on all_times
    sorted_indices = np.argsort(all_times)
    all_outputs = all_outputs[sorted_indices]
    all_targets = all_targets[sorted_indices]
    all_times = all_times[sorted_indices]

    # Split based on the month
    ar11698_mask = np.array(
        [
            int(str(time).split("-")[1]) == 3 if "-" in str(time) else False
            for time in all_times
        ]
    )
    ar11726_mask = np.array(
        [
            int(str(time).split("-")[1]) == 4 if "-" in str(time) else False
            for time in all_times
        ]
    )

    # Split outputs, targets, and times
    outputs_ar11698 = all_outputs[ar11698_mask]
    targets_ar11698 = all_targets[ar11698_mask]
    times_ar11698 = all_times[ar11698_mask]
    plot_ar11698(outputs_ar11698, targets_ar11698, times_ar11698)

    outputs_ar11726 = all_outputs[ar11726_mask]
    targets_ar11726 = all_targets[ar11726_mask]
    times_ar11726 = all_times[ar11726_mask]
    plot_ar11726(outputs_ar11726, targets_ar11726, times_ar11726)
    plot_ar(outputs_ar11726, targets_ar11726, times_ar11726)


def plot_ar(outputs, targets, times):
    plt.figure(figsize=(12, 6))
    plt.plot(times, outputs, label="Model Outputs", color="blue", alpha=0.7)
    plt.plot(times, targets, label="Targets", color="red", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Model Outputs vs Targets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "/nobackupnfs1/sroy14/processed_data/Helio/aremerge_skasapis/code/downstream-heliofm/downstream_apps/ar_kasapis/combined_plot.png"
    )


def plot_ar11698(outputs, targets, times):
    num_tiles = 6
    starting_tile = 46 - 9
    outputs = outputs[:, starting_tile : starting_tile + num_tiles]
    targets = targets[:, starting_tile : starting_tile + num_tiles]
    num_dimensions = outputs.shape[1]
    fig, axes = plt.subplots(
        num_dimensions, 1, figsize=(10, 3 * num_dimensions), sharex=True
    )
    times = [
        (
            datetime.strptime(t.decode("utf-8"), "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(t, bytes)
            else pd.Timestamp(t).to_pydatetime()
        )
        for t in times
    ]

    for dim in range(num_dimensions):
        ax = axes[dim] if num_dimensions > 1 else axes
        ax.plot(
            times,
            outputs[:, dim],
            label=f"Model Output (Tile {dim+starting_tile+8})",
            color="blue",
        )
        ax.plot(
            times,
            targets[:, dim],
            label=f"Target (Tile {dim+starting_tile+8})",
            color="red",
        )
        # Set x-axis to interpret values as dates
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.DayLocator())  # Major ticks once per day
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d")
        )  # Format as 'YYYY-MM-DD'
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"Model Output vs Target for AR11698 (Tile {dim+starting_tile+8})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        "/nobackupnfs1/sroy14/processed_data/Helio/aremerge_skasapis/code/downstream-heliofm/downstream_apps/ar_kasapis/ar11698_val_plot.png"
    )


def plot_ar11726(outputs, targets, times):
    num_tiles = 6
    starting_tile = 37 - 9
    outputs = outputs[:, starting_tile : starting_tile + num_tiles]
    targets = targets[:, starting_tile : starting_tile + num_tiles]
    num_dimensions = outputs.shape[1]
    fig, axes = plt.subplots(
        num_dimensions, 1, figsize=(10, 3 * num_dimensions), sharex=True
    )
    times = [
        (
            datetime.strptime(t.decode("utf-8"), "%Y-%m-%d %H:%M:%S.%f")
            if isinstance(t, bytes)
            else pd.Timestamp(t).to_pydatetime()
        )
        for t in times
    ]

    for dim in range(num_dimensions):
        ax = axes[dim] if num_dimensions > 1 else axes
        ax.plot(
            times,
            outputs[:, dim],
            label=f"Model Output (Tile {dim+starting_tile+8})",
            color="blue",
        )
        ax.plot(
            times,
            targets[:, dim],
            label=f"Target (Tile {dim+starting_tile+8})",
            color="red",
        )
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.DayLocator())  # Major ticks once per day
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d")
        )  # Format as 'YYYY-MM-DD'
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"Model Output vs Target for AR11698 (Tile {dim+starting_tile+8})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        "/nobackupnfs1/sroy14/processed_data/Helio/aremerge_skasapis/code/downstream-heliofm/downstream_apps/ar_kasapis/ar11726_val_plot.png"
    )


def main(args, device):
    """
    Main function to run the inference script.
    """

    # Load the model
    match args.model_type:
        case "spatio_temporal_attention":
            model = SpatioTemporalAttention()
        case _:
            raise ValueError(f"Unknown model type: {args.model_type}")

    model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
    print("Model loaded from checkpoint", args.checkpoint_path)

    model.to(device)

    valid_subset = AREmergenceDataset(args.valid_data_path)
    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_data_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    valid_loader = DataLoader(
        dataset=valid_subset,
        # sampler=StatefulDistributedSampler(valid_subset, drop_last=True),
        **dl_kwargs,
    )

    criterion = torch.nn.MSELoss(reduction="sum")
    epoch_loss = validate_model_mse(model, valid_loader, device, criterion)
    validate_model_graphs(model, valid_loader, device, criterion)

    print(f"Validation Loss: \t MSE: {epoch_loss:.4f} \t RMSE: {epoch_loss**0.5:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference script for the model.")
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=True,
        help="Path to the validation data.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for validation."
    )
    parser.add_argument(
        "--num_data_workers", type=int, default=4, help="Number of data workers."
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=2, help="Prefetch factor for DataLoader."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="spatio_temporal_attention",
        help="Type of the model to be used.",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args, device)
