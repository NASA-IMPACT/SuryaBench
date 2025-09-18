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

        # result.append([window_start, ins, cumulative_index, target, target_cumulative])
        result.append([window_start, target, target_cumulative])

        window_start += timedelta(**cadence)

    # cols = ["timestep", "max_goes_class", "cumulative_index", "label_max", "label_cum"]
    cols = ["timestep", "label_max", "label_cum"]

    df = pd.DataFrame(result, columns=cols)
    df["timestep"] = df["timestep"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # if valid_input_df is True
    # Merge two data from input index file and flare index file.
    # Read valid input index file
    if valid_input_df:
        df_input = pd.read_csv("./index_all.csv")
        df_input = df_input.loc[df_input["present"] == 1, :]

        df = df.merge(df_input, how="inner", left_on="timestep", right_on="timestep")
        df = df[cols]

    print(f"Total {len(df)} instances!")

    # save the dataset based on tri-montly partitioning
    # partition 1 : January to March
    # partition 2 : April to June
    # partition 3 : July to September
    # partition 4 : Octomber to December
    split_dataset(df, savepath=save_path)

    return df


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
    df['day_of_year'] = df['timestep'].dt.dayofyear - 1

    # Define year masks
    train_val_years = df["timestep"].dt.year.between(2010, 2019)
    test_years = df["timestep"].dt.year.between(2020, 2024)

    splits = {
        "first_buffer": df[train_val_years & df['day_of_year'].between(0, 13)],    # days 0–13 → weeks 1–2
        "validation": df[train_val_years & df['day_of_year'].between(14, 27)],    # days 14–27 → weeks 3–4
        "second_buffer": df[train_val_years & df['day_of_year'].between(28, 41)], # days 28–41 → weeks 5–6
        "training": df[train_val_years & (df['day_of_year'] >= 42)],             # day 42 → week 7 onwards
        "testing": df[test_years]                                                # all days in 2020–2024
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
            fname = f"flare_cls_{name}_1h.csv"
            path = os.path.join(savepath, fname)
            subset[["timestep", "label_cum", "label_max"]].to_csv(path, index=False)
            print(f"Saved {name} ({len(subset)} rows) to {path}")

    return splits


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
