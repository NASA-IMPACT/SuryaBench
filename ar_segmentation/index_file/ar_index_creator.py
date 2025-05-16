import os
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def index_create(
    mask_path: str,
    input_dir: str,
    file_ext: str,
    start: str,
    stop: str,
    cadence: str,
    savepath: str,
):

    # create dataframe from file path
    # list all the files that exist
    df = pd.DataFrame()
    files = glob.glob(mask_path + "*/*/*." + file_ext)
    df["file_path"] = files
    df["timestep"] = df["file_path"].str.extract(r"(\d{8}_\d{6})")
    df["timestep"] = pd.to_datetime(df["timestep"], format="%Y%m%d_%H%M%S")
    df["present"] = np.ones((len(df),), dtype=int)
    df["input_path"] = df["timestep"].dt.strftime(
        input_dir + "%Y/%m/hmi.m_720s.%Y%m%d_%H0000_TAI.1.magnetogram.h5"
    )

    # Read valid input index file
    df_input = pd.read_csv("./index_all.csv")
    df_input = df_input.loc[df_input["present"] == 1, :]
    df_input["timestep"] = pd.to_datetime(
        df_input["timestep"], format="%Y-%m-%d %H:%M:%S"
    )

    # create dataframe which represents all the time space between start and stop
    df_time = pd.DataFrame()
    time_index = pd.date_range(start=start, end=stop, freq=cadence)
    df_time["timestep"] = pd.Series(time_index)

    # merge two dataframes
    df_all = df_time.merge(df, how="left", left_on="timestep", right_on="timestep")
    df_all.loc[df_all["present"].isnull(), "present"] = 0
    df_all = df_all.loc[df_all["present"] == 1, :]

    # Merge two data from input index file and ar index file.
    df_all = df_all.merge(
        df_input, how="inner", left_on="timestep", right_on="timestep"
    )
    df_all = df_all[["timestep", "file_path", "present_x"]]
    df_all.rename(columns={"present_x": "present"}, inplace=True)

    df_all.to_csv(savepath + "ar_mask_index.csv", index=False)
    split_dataset_by_year(df_all, savepath=savepath)


def split_dataset_by_year(df, savepath="/"):
    df["timestep"] = pd.to_datetime(df["timestep"], format="%Y-%m-%d %H:%M:%S")

    for year, group in df.groupby(df["timestep"].dt.year):
        file_path = os.path.join(savepath, f"ar_{year}.csv")
        group.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")


if __name__ == "__main__":

    # Load Original source for Goes Flare X-ray Flux
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="/home/jh/2python_pr/aripod/out/ar_binary_masks/pil/",
        help="File path",
    )  # default="/workspace/data/"
    parser.add_argument(
        "--save_path",
        type=str,
        default="./downstream_apps/segment_yang/ds_data/",
        help="Save path",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-05-12 00:00:00",
        help="start time of the dataset",
    )
    parser.add_argument(
        "--stop",
        type=str,
        default="2024-12-31 23:59:59",
        help="end time of the dataset",
    )
    args = parser.parse_args()

    # Calling functions in order
    df_res = index_create(
        mask_path=args.file_path,
        input_dir="/nobackupnfs1/sroy14/processed_data/Helio/ar_detection/",
        file_ext="h5",
        start=args.start,
        stop=args.stop,
        cadence="1h",
        savepath=args.save_path,
    )
