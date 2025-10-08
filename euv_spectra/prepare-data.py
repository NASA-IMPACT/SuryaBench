"""
Data Preparation Script for EVE Spectra Dataset

This script:
1. Downloads the AIA_EVE_dataset_combined.nc file from HuggingFace if not present
2. Splits the dataset into train/val/test/leaky_val sets
3. Prepares training data (X_train, Y_train) for neural network training
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import xarray as xr
import torch
import os
import requests
from tqdm import tqdm

# Append base path. May need to be modified if the folder structure changes
sys.path.append("../../HelioFM")

from train_spectformer import get_config
from utils.data import build_scalers
from datasets.helio import HelioNetCDFDataset
import eve_dataloader


# ===============================
# Configuration
# ===============================
CONFIG_PATH = "./ds_configs/config_resnet_18.yaml"
DATA_DIR = "../../hfmds/data/"
DATASET_FILENAME = "AIA_EVE_dataset_combined.nc"
DATASET_PATH = os.path.join(DATA_DIR, DATASET_FILENAME)
HUGGINGFACE_URL = "https://huggingface.co/datasets/nasa-ibm-ai4science/euv-spectra/resolve/main/archive/AIA_EVE_dataset_combined.nc"

X_TRAIN_FILE = "X_train.pt"
Y_TRAIN_FILE = "Y_train.csv"
X_VAL_FILE = "X_val.pt"
Y_VAL_FILE = "Y_val.csv"
X_TEST_FILE = "X_test.pt"
Y_TEST_FILE = "Y_test.csv"


def download_file(url, dest_path, overwrite=False):
    """
    Download a file from a URL with progress bar.
    
    Parameters:
    -----------
    url : str
        URL to download from
    dest_path : str
        Destination file path
    """
    print(f"Downloading {dest_path}...")

    dest_path = Path(dest_path)
    if dest_path.exists() and not overwrite:
        print(f"⚠️ File already exists: {dest_path}")
        print("Use overwrite=True to replace it.")
        return dest_path
    
    # Create directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    # Create progress bar (dynamic ncols for clean look)
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, ncols=80)

    # Stream write the file in chunks
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:  # Filter out keep-alive chunks
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()

    # Verify final file size if known
    if total_size and dest_path.stat().st_size != total_size:
        print(f"⚠️ Warning: Download incomplete. Expected {total_size} bytes, got {dest_path.stat().st_size} bytes.")
    else:
        print(f"✅ Download complete: {dest_path}")


def download_dataset_if_needed():
    download_file(HUGGINGFACE_URL, DATASET_PATH, overwrite=False)


def assign_split(t: pd.Timestamp) -> str:
    """
    Assign a split label (train/val/test/leaky_val) based on timestamp.
    
    Rules:
    - Jan 1-14 and Feb 1-14 → leaky_validation
    - Jan 15-31 in 2011, 2014 → test
    - Jan 15-31 in 2012, 2013 → validation
    - Jan 15-31 other years → training
    - Feb 15+ and Mar+ → training
    
    Parameters:
    -----------
    t : pd.Timestamp
        Timestamp to assign split label
        
    Returns:
    --------
    str : Split label ('training', 'validation', 'test', 'leaky_validation')
    """
    y, m, d = t.year, t.month, t.day
    
    # Jan 1-14 and Feb 1-14 → leaky_validation
    if (m == 1 and 1 <= d <= 14) or (m == 2 and 1 <= d <= 14):
        return "leaky_validation"
    
    # Jan 15-31:
    # 2011, 2014 → test
    # 2012, 2013 → validation
    if m == 1 and 15 <= d <= 31:
        if y in (2011, 2014):
            return "test"
        elif y in (2012, 2013):
            return "validation"
        else:
            return "training"
    
    # After Feb 15 → training
    if (m == 2 and d >= 15) or (m >= 3):
        return "training"
    
    return "training"


# ===== Cell 4: Define split_dataset =====
def split_dataset():
    """
    Split the dataset into train/val/test/leaky_val sets based on timestamps.
    Saves:
      - dataset_splits.csv
      - training_times.csv, validation_times.csv, test_times.csv, leaky_validation_times.csv
    """
    print("\nSplitting dataset into train/val/test/leaky_val...")
    print(f"Loading dataset from {DATASET_PATH}...")

    ds = xr.open_dataset(DATASET_PATH)

    # Extract 'time' coordinate/variable robustly
    if 'time' in ds.coords:
        times = pd.to_datetime(ds['time'].values)
    elif 'time' in ds.variables:
        times = pd.to_datetime(ds['time'].values)
    else:
        time_vars = [var for var in ds.variables if 'time' in var.lower()]
        if time_vars:
            times = pd.to_datetime(ds[time_vars[0]].values)
        else:
            ds.close()
            raise ValueError("Could not find time variable in dataset")

    print(f"Found {len(times)} timestamps in dataset")
    print(f"Time range: {times[0]} to {times[-1]}")

    # Apply split assignment
    splits = [assign_split(t) for t in times]

    # Build DataFrame
    df_splits = pd.DataFrame({'time': times, 'split': splits})
    
    # Print distribution
    print("\nSplit distribution:")
    total = len(df_splits)
    for split_name in ['training', 'validation', 'test', 'leaky_validation']:
        count = (df_splits['split'] == split_name).sum()
        pct = (count / total) * 100 if total else 0.0
        print(f"  {split_name:17s}: {count:6d} ({pct:5.2f}%)")

    # Save outputs
    splits_file = os.path.join(DATA_DIR, "dataset_splits.csv")
    df_splits.to_csv(splits_file, index=False)
    print(f"\nSaved split information to {splits_file}")

    for split_name in ['training', 'validation', 'test', 'leaky_validation']:
        split_df = df_splits[df_splits['split'] == split_name][['time']]
        split_file = os.path.join(DATA_DIR, f"{split_name}_times.csv")
        split_df.to_csv(split_file, index=False)
        print(f"Saved {split_name} times to {split_file}")

    ds.close()
    print("\nDataset splitting complete!")


def prepare_data_for_phase(phase="train", config=None, scalers=None):
    """
    Prepare X and Y data for a given phase (train, val, or test).
    
    Parameters:
    -----------
    phase : str
        One of 'train', 'val', or 'test'
    config : object
        Configuration object
    scalers : dict
        Scalers for normalization
        
    Returns:
    --------
    X : torch.Tensor
        Input tensor of shape (N, 13, 4096, 4096)
    Y : torch.Tensor
        Target tensor of shape (N, 1343)
    """
    # Map phase to appropriate time column and data path
    phase_config = {
        "train": {
            "time_column": "train_time",
            "data_path": config.data.train_data_path
        },
        "val": {
            "time_column": "val_time",
            "data_path": config.data.val_data_path
        },
        "test": {
            "time_column": "test_time",
            "data_path": config.data.test_data_path
        }
    }
    
    if phase not in phase_config:
        raise ValueError(f"Phase must be one of {list(phase_config.keys())}")
    
    time_column = phase_config[phase]["time_column"]
    data_path = phase_config[phase]["data_path"]
    
    print(f"\n{'='*50}")
    print(f"Preparing {phase.upper()} data")
    print(f"{'='*50}")
    
    # Initialize dataset
    dataset = eve_dataloader.EVEDSDataset(
        # Required by parent HelioNetCDFDataset class
        index_path=data_path,
        time_delta_input_minutes=config.data.time_delta_input_minutes,
        time_delta_target_minutes=config.data.time_delta_target_minutes,
        n_input_timestamps=config.data.n_input_timestamps,
        rollout_steps=config.rollout_steps,
        channels=config.data.channels,
        drop_hmi_probablity=config.drop_hmi_probablity,
        num_mask_aia_channels=config.num_mask_aia_channels,
        use_latitude_in_learned_flow=config.use_latitude_in_learned_flow,
        scalers=scalers,
        phase=phase,
        # Downstream specific parameters
        ds_eve_index_path=DATASET_PATH,
        ds_time_column=time_column,
        ds_time_tolerance="6m",
        ds_match_direction="forward"
    )
    
    print(f"Sample Size: {len(dataset)}")
    
    # Initialize storage lists
    X_list = []  # For input tensors (ts[:, 0, :, :])
    Y_list = []  # For target tensors (spectra)
    
    # Loop over the dataset
    for i in range(len(dataset)):
        # Load the i-th sample from the dataset
        item, _ = dataset[i]
        
        # Input tensor: item['ts'] is shape (13, 2, 4096, 4096)
        # We use only the first time slice on axis 1 → ts[:, 0, :, :]
        # Resulting shape: (13, 4096, 4096)
        ts = item['ts']
        ts_single = ts[:, 0, :, :]  # Extract time slice 0 for all channels
        
        # Ensure it's a PyTorch tensor
        ts_tensor = torch.tensor(ts_single) if not isinstance(ts_single, torch.Tensor) else ts_single
        X_list.append(ts_tensor)
        
        # Output tensor: item['target'] is shape (1343,)
        spectra = item['target']
        spectra_tensor = torch.tensor(spectra) if not isinstance(spectra, torch.Tensor) else spectra
        Y_list.append(spectra_tensor)
        
        if (i + 1) % 100 == 0 or i == len(dataset) - 1:
            print(f"Processed {i + 1}/{len(dataset)} samples")
    
    # Stack into single tensors
    X = torch.stack(X_list)  # Final shape: (N, 13, 4096, 4096)
    Y = torch.stack(Y_list)  # Final shape: (N, 1343)
    
    print(f"X_{phase} shape: {X.shape}")
    print(f"Y_{phase} shape: {Y.shape}")
    
    return X, Y



def main():
    """
    Main execution function.
    """
    print("="*60)
    print("EVE Spectra Dataset Preparation")
    print("="*60)
    
    # Step 1: Download dataset if needed
    download_dataset_if_needed()
    
    # Step 2: Split dataset
    split_dataset()
    
    # Step 3: Load configuration and build scalers
    print("\nLoading configuration...")
    config = get_config(CONFIG_PATH)
    scalers = build_scalers(info=config.data.scalers)
    
    # Step 4: Prepare training data
    X_train, Y_train = prepare_data_for_phase("train", config, scalers)
    
    # Save training data
    print(f"\nSaving training data...")
    torch.save(X_train, X_TRAIN_FILE)
    df_y_train = pd.DataFrame(Y_train.cpu().numpy())
    df_y_train.to_csv(Y_TRAIN_FILE, index=False)
    print(f"Saved {X_TRAIN_FILE} and {Y_TRAIN_FILE}")
    
    # Step 5: Prepare validation data
    X_val, Y_val = prepare_data_for_phase("val", config, scalers)
    
    # Save validation data
    print(f"\nSaving validation data...")
    torch.save(X_val, X_VAL_FILE)
    df_y_val = pd.DataFrame(Y_val.cpu().numpy())
    df_y_val.to_csv(Y_VAL_FILE, index=False)
    print(f"Saved {X_VAL_FILE} and {Y_VAL_FILE}")
    
    # Step 6: Prepare test data
    X_test, Y_test = prepare_data_for_phase("test", config, scalers)
    
    # Save test data
    print(f"\nSaving test data...")
    torch.save(X_test, X_TEST_FILE)
    df_y_test = pd.DataFrame(Y_test.cpu().numpy())
    df_y_test.to_csv(Y_TEST_FILE, index=False)
    print(f"Saved {X_TEST_FILE} and {Y_TEST_FILE}")
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)


if __name__ == "__main__":
    main()


