"""
Split the solar wind dataset into training, validation, and test sets.

This script processes a solar wind dataset CSV file and splits it into:
- Training set: First 8 months of each year
- Validation set: Portion of remaining months
- Test set: Specific test cases and portion of remaining months

The split is done to ensure balanced validation and test sets while preserving
specific test cases of interest.
"""

import os

import numpy as np
import pandas as pd
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries


def get_indices(start, end, time_series):
    """Get indices of time series data within a date range.

    Parameters
    ----------
    start : str
        Start date in the format 'YYYY-MM-DD'.
    end : str
        End date in the format 'YYYY-MM-DD'.
    time_series : pandas.DataFrame
        The time series data with an 'Epoch' column containing timestamps.

    Returns
    -------
    list
        List of indices that fall within the start and end dates.
    """
    # Convert start and end dates to datetime objects
    start = np.datetime64(pd.to_datetime(start, unit="ns"))
    end = np.datetime64(pd.to_datetime(end, unit="ns"))

    # Get indices within the date range
    times = pd.to_datetime(time_series["Epoch"].values, unit="ns").to_numpy()
    start = np.argmin(np.abs(times - start))
    end = np.argmin(np.abs(times - end))
    indices = np.arange(len(times))[start:end]
    return indices


def main(solar_wind_path):
    """Split solar wind data into train, validation and test sets.

    Parameters
    ----------
    solar_wind_path : str
        Path to the solar wind data CSV file.
    """
    savepath = "/".join(solar_wind_path.split("/")[:-1]) + "/"
    # Load the solar wind data
    solar_wind = pd.read_csv(solar_wind_path)

    # Define specific test cases
    test_dates = [
        ("2011-08-04 00:00", "2011-08-08 00:00"),
        ("2015-03-16 00:00", "2015-03-20 00:00"),
        ("2017-09-25 00:00", "2017-09-29 00:00"),
    ]

    # Extract indices for specific test cases
    test_indices = []
    for ii, (start, end) in enumerate(test_dates):
        indices = get_indices(start, end, solar_wind)
        small_df = solar_wind.iloc[indices]
        small_df.to_csv(f"{savepath}test_data_cases_{ii}.csv", index=False)
        test_indices.extend(indices)

    # Split remaining data into train/val/test
    # Training: First 8 months of each year
    # Validation/Test: Last 4 months split to balance test set size
    train_indices = []
    val_indices = []

    for year in range(2010, 2024):
        # Training period: Jan-Aug
        start = f"{year}-01-01 00:00"
        end = f"{year}-08-31 23:59"
        indices = get_indices(start, end, solar_wind)
        indices = [i for i in indices if i not in test_indices]
        train_indices.extend(indices)

        # Validation/Test period: Sep-Dec
        start = f"{year}-09-01 00:00"
        end = f"{year}-12-31 23:59"
        indices = get_indices(start, end, solar_wind)
        indices = [i for i in indices if i not in test_indices]
        val_indices.extend(indices)

    # Save training data
    train_df = solar_wind.iloc[train_indices]
    train_df.to_csv(f"{savepath}train_data.csv", index=False)

    # Balance validation and test sets
    N_test = len(test_indices)
    N_val = len(val_indices)
    N_total = N_test + N_val

    # Split validation set to achieve balanced test set
    N_test_split = N_total // 2 - N_test
    N_val_split = N_total - N_test_split
    test_indices = test_indices + val_indices[N_val_split:]
    val_indices = val_indices[:N_val_split]

    # Save test and validation data
    test_df = solar_wind.iloc[test_indices]
    test_df.to_csv(f"{savepath}test_data.csv", index=False)
    val_df = solar_wind.iloc[val_indices]
    val_df.to_csv(f"{savepath}val_data.csv", index=False)

    # Print dataset sizes
    print(f"Number of samples in training set: {len(train_indices)}")
    print(f"Number of samples in validation set: {len(val_indices)}")
    print(f"Number of samples in test set: {len(test_indices)}")


if __name__ == "__main__":
    # Path to the solar wind data file.
    solar_wind_path = "../data/solar_wind_data/solar_wind_data.csv"
    main(solar_wind_path)
