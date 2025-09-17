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


def index_create(
    mask_path: str,
    input_dir: str,
    sdo_index: str,
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
    df["file_path"] = files
    df["timestep"] = df["file_path"].str.extract(r"(\d{8}_\d{4})")
    df["timestep"] = pd.to_datetime(df["timestep"], format="%Y%m%d_%H%M%S")
    df["present"] = np.ones((len(df),), dtype=int)

    # Create input file paths
    path_fmt = "%Y/%m/hmi.m_720s.%Y%m%d_%H0000_TAI.1.magnetogram.h5"
    path_template = os.path.join(input_dir, path_fmt)
    df["input_path"] = df["timestep"].dt.strftime(path_template)

    # Read valid input index file
    df_input = pd.read_csv(sdo_index)
    df_input = df_input.loc[df_input["present"] == 1, :]
    fmt = "%Y-%m-%d %H:%M:%S"
    df_input["timestep"] = pd.to_datetime(df_input["timestep"], format=fmt)

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
    df_all = df_all.loc[df_all["present"] == 1, :]

    # Merge with input index file
    df_all = df_all.merge(
        df_input, how="inner", left_on="timestep", right_on="timestep"
    )
    df_all = df_all[["timestep", "file_path", "present_x"]]
    df_all.rename(columns={"present_x": "present"}, inplace=True)

    # Save index files
    df_all.to_csv(savepath + "ar_mask_index.csv", index=False)
    split_dataset(df_all, savepath=savepath)


# Creating time-segmented 4 tri-monthly partitions
def split_dataset(df: pd.DataFrame, savepath: str = "/"):
    """
    Split the dataset into 4 different subsets
    and save each as CSV.

    First buffer: Weeks 1-2 of each year from 2010 to 2019
    validation: Weeks 3-4 of each year from 2010 to 2019
    Second buffer: Weeks 5-6 of each year from 2010 to 2019
    Training: Weeks 7~52 of each year from 2010 to 2019
    Testing: Weeks 1~52 of each year from Jan 1, 2020 to Dec 31st, 2024

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with 'timestep' and label columns.
    savepath : str
        Path to save the partitioned CSV files.

    Returns:
    --------
    None
    """
    # Ensure datetime conversion
    df = df.copy()
    df["timestep"] = pd.to_datetime(df["timestep"], errors="coerce")

    # Extract ISO calendar fields
    iso = df["timestep"].dt.isocalendar()

    # Define year masks
    train_val_years = df["timestep"].dt.year.between(2010, 2019)
    test_years = df["timestep"].dt.year.between(2020, 2024)

    # Define splits
    splits = {
        "first_buffer": df[train_val_years & iso.week.between(1, 2)],
        "validation": df[train_val_years & iso.week.between(3, 4)],
        "second_buffer": df[train_val_years & iso.week.between(5, 6)],
        "training": df[train_val_years & (iso.week >= 7)],
        "testing": df[test_years & iso.week.between(1, 52)],
    }

    # Create leaky validation (combination of both buffers)
    splits["leaky_validation"] = pd.concat(
        [splits["first_buffer"], splits["second_buffer"]],
        axis=0
    ).sort_values("timestep")

    # Save to CSV if requested
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        for name, subset in splits.items():
            fname = f"ar_binary_mask_index_{name}_1h.csv"
            path = os.path.join(savepath, fname)
            subset.to_csv(path, index=False)
            print(f"Saved {name} ({len(subset)} rows) to {path}")

    return splits

# def split_dataset_by_year(df, savepath="/"):
#     """Split dataset into yearly files.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame containing AR mask data.
#     savepath : str, optional
#         Path to save the yearly files, by default "/".
#     """
#     fmt = "%Y-%m-%d %H:%M:%S"
#     df["timestep"] = pd.to_datetime(df["timestep"], format=fmt)

#     for year, group in df.groupby(df["timestep"].dt.year):
#         file_path = os.path.join(savepath, f"ar_{year}.csv")
#         group.to_csv(file_path, index=False)
#         print(f"Saved: {file_path}")


if __name__ == "__main__":
    desc = "Create AR segmentation index files"
    parser = argparse.ArgumentParser(description=desc)

    default_mask_path = "/nobackupp17/jhong15/project/aripod/out/pil/"
    default_sdo_idex_path = "/nobackupp17/jhong15/project/SuryaBench/ar_segmentation/index_file/index_all.csv"
    default_save_path = "/nobackupp17/jhong15/project/SuryaBench/ar_segmentation/"

    parser.add_argument(
        "--file_path",
        type=str,
        default=default_mask_path,
        help="Path to AR mask files",
    )
    parser.add_argument(
        "--sdo_index_path",
        type=str,
        default=default_sdo_idex_path,
        help="Path to sdo data index file",
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

    # data_dir = "/nobackupp17/jhong15/project/aripod/out/pil"
    index_create(
        mask_path=args.file_path,
        input_dir=default_mask_path,
        sdo_index=args.sdo_index_path,
        file_ext="h5",
        start=args.start,
        stop=args.stop,
        cadence="1h",
        savepath=args.save_path,
    )
