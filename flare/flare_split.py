"""
Data Preparation Script for Flare Prediction Dataset

This script:
1. Downloads the data.csv file from HuggingFace if not present
2. Splits the dataset into train/val/test/leaky_val sets based on timestamps
3. Saves split files to splits directory
"""

import pandas as pd
import os
import requests
from pathlib import Path
from tqdm import tqdm


# ===============================
# Configuration
# ===============================
DATA_DIR = Path("./data")
DATASET_FILENAME = "data.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
HUGGINGFACE_URL = "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting/resolve/main/data.csv"

OUTPUT_DIR = DATA_DIR / "splits"


def download_file(url, dest_path):
    """
    Download a file from a URL with progress bar.

    Parameters:
    -----------
    url : str
        URL to download from
    dest_path : Path or str
        Destination file path
    """
    print(f"Downloading {dest_path}...")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Stream download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"✓ Download complete: {dest_path}")


def download_dataset_if_needed():
    """
    Download the dataset from HuggingFace if it doesn't exist locally.
    """
    if DATASET_PATH.exists():
        print(f"✓ Dataset already exists at {DATASET_PATH}")
        return

    print(f"Dataset not found at {DATASET_PATH}")
    print(f"Downloading from {HUGGINGFACE_URL}...")
    download_file(HUGGINGFACE_URL, DATASET_PATH)


def assign_split(t):
    """
    Assign a split label (train/val/test/leaky_val) based on timestamp.

    Rules:
    - >= 2020-01-01 → test
    - <= 2010 → training
    - 2011-2019:
        - Jan 1-14 or Feb 1-14 → leaky_validation
        - Jan 15-31 → validation
        - Other dates → training

    Parameters:
    -----------
    t : pd.Timestamp
        Timestamp to assign split label

    Returns:
    --------
    str : Split label ('training', 'validation', 'test', 'leaky_validation')
    """
    if t >= pd.Timestamp("2020-01-01"):
        return "test"
    if t.year <= 2010:
        return "training"
    if 2011 <= t.year <= 2019:
        m, d = t.month, t.day
        if (m == 1 and 1 <= d <= 14) or (m == 2 and 1 <= d <= 14):
            return "leaky_validation"
        if (m == 1 and 15 <= d <= 31):
            return "validation"
        return "training"
    return "training"


def split_dataset():
    """
    Split the dataset into train/val/test/leaky_val sets based on timestamps.

    This function:
    1. Loads the data.csv file
    2. Assigns split labels based on date rules
    3. Saves split files to OUTPUT_DIR
    """
    print("\n" + "="*60)
    print("Splitting dataset into train/val/test/leaky_val...")
    print("="*60)

    # Load the dataset
    print(f"\nLoading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    # Ensure timestamp column is datetime type
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    print(f"✓ Loaded {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Apply split assignment
    print("\nApplying split logic...")
    df["split"] = df["timestamp"].apply(assign_split)

    # Count splits
    split_counts = df["split"].value_counts()
    print("\nSplit distribution:")
    for split_name in ['training', 'validation', 'test', 'leaky_validation']:
        count = split_counts.get(split_name, 0)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {split_name:20s}: {count:6d} samples ({percentage:5.2f}%)")

    # Columns to save (exclude 'split' column)
    to_save = [c for c in df.columns if c != "split"]

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save split files
    print(f"\nExporting split files to {OUTPUT_DIR}...")

    train_df = df[df["split"] == "training"][to_save]
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    print(f"  ✓ train.csv: {len(train_df)} rows")

    val_df = df[df["split"] == "validation"][to_save]
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    print(f"  ✓ validation.csv: {len(val_df)} rows")

    leaky_val_df = df[df["split"] == "leaky_validation"][to_save]
    leaky_val_df.to_csv(OUTPUT_DIR / "leaky_validation.csv", index=False)
    print(f"  ✓ leaky_validation.csv: {len(leaky_val_df)} rows")

    test_df = df[df["split"] == "test"][to_save]
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    print(f"  ✓ test.csv: {len(test_df)} rows")

    print(f"\n✓ All split files saved to: {OUTPUT_DIR}")


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("Flare Prediction Dataset Preparation")
    print("="*60)

    # Step 1: Download dataset if needed
    download_dataset_if_needed()

    # Step 2: Split dataset
    split_dataset()

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
