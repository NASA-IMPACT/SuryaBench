"""Create index files for Active Region (AR) segmentation data.

This script creates index files for AR segmentation data by:
1. Finding all AR mask files in a given directory
2. Extracting timestamps from filenames
3. Matching with input data timestamps
4. Creating yearly index files
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd


def parse_filename(file_path: list):
    file_name = os.path.basename(file_path)

    # Basic validation to ensure filename has at least 6 characters for year+month
    if len(file_name) < 6 or not file_name[:6].isdigit():
        raise ValueError(f"Filename '{file_name}' must start with YYYYMM")

    year, month = file_name[:4], file_name[4:6]
    return os.path.join("assets", year, month, file_name)


def index_create(
    mask_path: str,
    input_dir: str,
    file_ext: str,
    start: str,
    stop: str,
    cadence: str,
    savepath: str,
):
    """Create index files for AR segmentation data.

    Parameters
    ----------
    mask_path : str
        Path to directory containing AR mask files.
    input_dir : str
        Path to directory containing input data files.
    file_ext : str
        File extension of AR mask files.
    start : str
        Start date in format 'YYYY-MM-DD HH:MM:SS'.
    stop : str
        Stop date in format 'YYYY-MM-DD HH:MM:SS'.
    cadence : str
        Time cadence for sampling (e.g. '1h' for hourly).
    savepath : str
        Path to save the index files.
    """
    # Create dataframe from file paths
    files = glob.glob(mask_path + "*/*/*." + file_ext)
    df = pd.DataFrame()
    df["file_path"] = [parse_filename(file) for file in files]
    df["timestep"] = df["file_path"].str.extract(r"(\d{8}_\d{4})")
    df["timestep"] = pd.to_datetime(df["timestep"], format="%Y%m%d_%H%M%S")
    df["present"] = np.ones((len(df),), dtype=int)

    # Create input file paths
    path_fmt = "%Y/%m/hmi.m_720s.%Y%m%d_%H0000_TAI.1.magnetogram.h5"
    path_template = os.path.join(input_dir, path_fmt)
    df["input_path"] = df["timestep"].dt.strftime(path_template)

    # Create dataframe for all times between start and stop
    df_time = pd.DataFrame()
    time_index = pd.date_range(start=start, end=stop, freq=cadence)
    df_time["timestep"] = pd.Series(time_index)

    # Merge dataframes to get final index
    merge_cols = ["timestep", "timestep"]
    df_all = df_time.merge(
        df, how="left", left_on=merge_cols[0], right_on=merge_cols[1]
    )
    df_all.loc[df_all["present"].isnull(), "present"] = 0
    # df_all = df_all.loc[df_all["present"] == 1, :]
    df_all = df_all[["timestep", "file_path", "present"]]

    # Save index files
    df_all.to_csv(savepath + "ar_mask_index.csv", index=False)
    split_dataset(df_all, savepath=savepath)

# create splits based on 14-17-14 day splits


def split_dataset(df: pd.DataFrame, savepath: str = "/"):
    """
    Split the dataset into 4 different subsets and save each as CSV.

    Leaky validation: Jan 1-14 and Feb 1-14 of each year from 2010 to 2019
    Validation: Jan 15-31 of each year from 2010 to 2019
    Training: Feb 15 onward of each year from 2010 to 2019
    Testing: All data from Jan 1, 2020 to Dec 31, 2024

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with 'timestep' and label columns.
    savepath : str
        Path to save the partitioned CSV files.

    Returns:
    --------
    dict: Dictionary containing the split DataFrames
    """
    # Ensure datetime conversion
    df = df.copy()
    df["timestep"] = pd.to_datetime(df["timestep"], errors="coerce")
    df['month'] = df['timestep'].dt.month
    df['day'] = df['timestep'].dt.day

    # Define year masks
    train_val_years = df["timestep"].dt.year.between(2010, 2019)
    test_years = df["timestep"].dt.year >= 2020

    # Define splits
    splits = {}

    # Leaky validation: Jan 1-14 and Feb 1-14 for years 2010-2019
    leaky_jan = train_val_years & (df['month'] == 1) & df['day'].between(1, 14)
    leaky_feb = train_val_years & (df['month'] == 2) & df['day'].between(1, 14)
    splits["leaky_validation"] = df[leaky_jan | leaky_feb]

    # Validation: Jan 15-31 for years 2010-2019
    splits["validation"] = df[train_val_years & (
        df['month'] == 1) & df['day'].between(15, 31)]

    # Training: Feb 15 onward for years 2010-2019
    splits["training"] = df[train_val_years & (
        ((df['month'] == 2) & (df['day'] >= 15)) | (df['month'] >= 3))]

    # Testing: All data from 2020 onwards
    splits["testing"] = df[test_years]

    # Save to CSV
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        filenames = {
            "leaky_validation": "leaky_validation.csv",
            "validation": "validation.csv",
            "training": "train.csv",
            "testing": "test.csv"
        }

        for name, subset in splits.items():
            fname = filenames[name]
            path = os.path.join(savepath, fname)
            subset[["timestep", "file_path", "present"]].to_csv(
                path, index=False)
            print(f"Saved {name} ({len(subset)} rows) to {path}")

    return splits


if __name__ == "__main__":
    desc = "Create AR segmentation index files"
    parser = argparse.ArgumentParser(description=desc)

    default_mask_path = "./aripod/out/pil/"
    default_save_path = "./SuryaBench/ar_segmentation/"

    parser.add_argument(
        "--file_path",
        type=str,
        default=default_mask_path,
        help="Path to AR mask files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help="Path to save index files",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-05-12 00:00:00",
        help="Start date (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default="2024-12-31 23:59:59",
        help="End date (YYYY-MM-DD HH:MM:SS)",
    )
    args = parser.parse_args()

    index_create(
        mask_path=args.file_path,
        input_dir=default_mask_path,
        file_ext="h5",
        start=args.start,
        stop=args.stop,
        cadence="1h",
        savepath=args.save_path,
    )
