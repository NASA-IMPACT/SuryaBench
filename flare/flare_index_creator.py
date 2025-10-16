import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def sub_class_num(df: pd.DataFrame):
    """
    Convert GOES flare class strings (e.g., C3.2, M5.1)
    into numeric values based on magnitude.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the 'fl_goescls' column with flare class strings.

    Returns:
    --------
    np.ndarray
        Array of numeric flare magnitudes: C=x, M=10x, X=100x.
    """

    if not df["fl_goescls"].str.match(r"^[ABCMX]\d+(\.\d+)?$").all():
        raise ValueError("fl_goescls contains unexpected values.")

    # Extract the numeric part after the class (C/M/X) and convert to float
    numeric_part = df["fl_goescls"].str[1:].astype(float)

    # Use np.select for multiple conditions to avoid repeated operations
    conditions = [
        df["fl_goescls"].str.startswith("C"),
        df["fl_goescls"].str.startswith("M"),
        df["fl_goescls"].str.startswith("X"),
    ]

    # Corresponding choices based on class
    choices = [
        numeric_part,  # C-class: same value
        10 * numeric_part,  # M-class: multiply by 10
        100 * numeric_part,  # X-class: multiply by 100
    ]

    return np.select(conditions, choices, default=None)


def rolling_window(
    df_fl: pd.DataFrame,
    valid_input_df: bool,
    save_path: str,
    start: str,
    stop: str,
    cadence: dict,
    windowsize: dict,
    thres_max: str = "M1.0",
    thres_cum: float = 10,
) -> pd.DataFrame:
    """
    Generate a rolling window dataset from flare catalog data
    with binary labels based on max class and cumulative flare intensity.

    Parameters:
    -----------
    df_fl : pd.DataFrame
        DataFrame containing GOES flare data.
    valid_input_df : bool
        Whether to validate the generated data against an input file.
    save_path : str
        Path to save the output datasets.
    start : str
        Start date in "YYYY-MM-DD" format.
    stop : str
        Stop date in "YYYY-MM-DD HH:MM:SS" format.
    cadence : dict
        Dictionary to define the rolling window stride (e.g., {'hours': 1}).
    windowsize : dict
        Dictionary to define window length (e.g., {'hours': 24}).
    thres_max : str
        Threshold GOES class string for max label (e.g., "M1.0").
    thres_cum : float
        Threshold for cumulative flare intensity.

    Returns:
    --------
    pd.DataFrame
        Time-indexed dataset with binary labels.
    """

    # Datetime
    df_fl["event_starttime"] = pd.to_datetime(
        df_fl["event_starttime"], format="%Y-%m-%d %H:%M:%S"
    )
    window_start = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(stop, "%Y-%m-%d %H:%M:%S")

    # Create sub-class column
    df_fl["sub_cls"] = sub_class_num(df_fl)
    result = []
    while window_start < stop:
        print(f"Processing, {window_start}")
        window = df_fl[
            (df_fl.event_starttime > window_start)
            & (df_fl.event_starttime <= window_start + timedelta(**windowsize))
        ]

        window_sorted = window.sort_values("fl_goescls", ascending=False)
        top_row = window_sorted.head(1)

        if not top_row.empty:
            Maximum_index = top_row.squeeze(axis=0)
        else:
            Maximum_index = None  # Or handle appropriately

        cumulative_index = window["sub_cls"].sum()

        # 1) define binary index from max flare class
        if window.empty:
            ins = "FQ"
            target = 0
        else:
            ins = Maximum_index.fl_goescls

            if ins >= thres_max:  # FQ and A class flares
                target = 1
            else:
                target = 0

        # 2) define binary index from cumulative flare class
        if cumulative_index >= thres_cum:
            target_cumulative = 1
        else:
            target_cumulative = 0

        result.append([window_start, ins, cumulative_index, target, target_cumulative])
        window_start += timedelta(**cadence)

    cols = ["timestamp", "max_goes_class", "cumulative_index", "label_max", "label_cum"]

    df = pd.DataFrame(result, columns=cols)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # if valid_input_df is True
    # Merge two data from input index file and flare index file.
    # Read valid input index file
    if valid_input_df:
        df_input = pd.read_csv("./index_all.csv")
        df_input = df_input.loc[df_input["present"] == 1, :]

        df = df.merge(df_input, how="inner", left_on="timestamp", right_on="timestamp")
        df = df[cols]

    print(f"Total {len(df)} instances!")

    # save the dataset based on tri-montly partitioning
    # partition 1 : January to March
    # partition 2 : April to June
    # partition 3 : July to September
    # partition 4 : Octomber to December
    split_dataset(df, savepath=save_path)

    return df


def assign_split(t):
    """
    Assign a split label (train/val/test/leaky_val) based on timestamp.

    Data Split Strategy:
        Training Set: 
            - All days in 2010
            - All days from 2011 to 2019, excluding the dates used in the validation and leaky validation sets
        Validation Set: January 15–31 for each year from 2011 to 2019
        Leaky Validation Set: January 1–14 or February 1–14 for each year from 2011 to 2019
        Test Set: All days from January 1, 2020 onward

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


def split_dataset(df: pd.DataFrame, savepath: str = "/"):
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

    # Save split files
    print(f"\nExporting split files to {savepath}...")

    train_df = df[df["split"] == "training"][to_save]
    train_df.to_csv(savepath + "train.csv", index=False)
    print(f"  ✓ train.csv: {len(train_df)} rows")

    val_df = df[df["split"] == "validation"][to_save]
    val_df.to_csv(savepath + "validation.csv", index=False)
    print(f"  ✓ validation.csv: {len(val_df)} rows")

    leaky_val_df = df[df["split"] == "leaky_validation"][to_save]
    leaky_val_df.to_csv(savepath + "leaky_validation.csv", index=False)
    print(f"  ✓ leaky_validation.csv: {len(leaky_val_df)} rows")

    test_df = df[df["split"] == "test"][to_save]
    test_df.to_csv(savepath + "test.csv", index=False)
    print(f"  ✓ test.csv: {len(test_df)} rows")

    print(f"\n✓ All split files saved to: {savepath}")


if __name__ == "__main__":

    # Load Original source for Goes Flare X-ray Flux
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="./flare/",
        help="File path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./flare/",
        help="Save path",
    )
    parser.add_argument(
        "--start", type=str, default="2010-05-12", help="start time of the dataset"
    )
    parser.add_argument(
        "--stop",
        type=str,
        default="2024-12-31 23:59:59",
        help="end time of the dataset",
    )
    args = parser.parse_args()

    df = pd.read_csv(
        args.file_path + "flare_catalog_2010-2024.csv",
        usecols=["event_starttime", "fl_goescls"],
    )

    # Calling functions in order
    df_res = rolling_window(
        df_fl=df,
        valid_input_df=False,
        save_path=args.save_path,
        start=args.start,
        stop=args.stop,
        cadence={"hours": 1, "minutes": 0, "seconds": 0},
        windowsize={"hours": 23, "minutes": 59, "seconds": 59},
        thres_max="M1.0",
        thres_cum=10,
    )
