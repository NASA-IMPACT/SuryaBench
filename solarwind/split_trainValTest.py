"""
Split the dataset into training, validation, and test sets.
"""

from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
import os
import pandas as pd
import numpy as np


def get_indices(start, end, time_series):
    """
    Get the indices of the time series that fall within the start and end dates.

    Parameters
    ----------
    start : str
        Start date in the format 'YYYY-MM-DD'.
    end : str
        End date in the format 'YYYY-MM-DD'.
    time_series : pandas.DataFrame
        The time series data.

    Returns
    -------
    list
        A list of indices that fall within the start and end dates.
    """
    # Convert start and end dates to datetime objects.
    start = np.datetime64(pd.to_datetime(start, unit="ns"))
    end = np.datetime64(pd.to_datetime(end, unit="ns"))
    # Get the indices of the time series that fall within the start and end dates.
    times = pd.to_datetime(time_series["Epoch"].values, unit="ns").to_numpy()
    start = np.argmin(np.abs(times - start))
    end = np.argmin(np.abs(times - end))
    indices = np.arange(len(times))[start:end]
    return indices


def main(solar_wind_path):
    savepath = "/".join(solar_wind_path.split("/")[:-1]) + "/"
    # Load the solar wind data.
    solar_wind = pd.read_csv(solar_wind_path)
    test_dates = [
        ("2011-08-04 00:00", "2011-08-08 00:00"),
        ("2015-03-16 00:00", "2015-03-20 00:00"),
        ("2017-09-25 00:00", "2017-09-29 00:00"),
    ]
    # Get the indices of the time series that fall within the test dates.
    test_indices = []
    for ii, (start, end) in enumerate(test_dates):
        indices = get_indices(start, end, solar_wind)
        small_df = solar_wind.iloc[indices]
        # Save the test data to a CSV file.
        small_df.to_csv(f"{savepath}test_data_cases_{ii}.csv", index=False)
        # Append the indices to the test_indices list.
        test_indices.extend(indices)

    # Get the indices of the time series that fall within the training and validation dates.
    # Our training set is the first 8 months of each year, while the last 3 are split into validation and test sets.
    # The test set should be of the same length as validation set. But, we already have a subset of the test set, which we must account for.
    train_indices = []
    val_indices = []
    for year in range(2010, 2024):
        start = f"{year}-01-01 00:00"
        end = f"{year}-08-31 23:59"
        indices = get_indices(start, end, solar_wind)
        indices = [i for i in indices if i not in test_indices]
        train_indices.extend(indices)
        start = f"{year}-09-01 00:00"
        end = f"{year}-12-31 23:59"
        indices = get_indices(start, end, solar_wind)
        indices = [i for i in indices if i not in test_indices]
        val_indices.extend(indices)
    # Save the training and validation data to CSV files.
    train_df = solar_wind.iloc[train_indices]
    train_df.to_csv(f"{savepath}train_data.csv", index=False)
    N_test = len(test_indices)
    N_val = len(val_indices)
    N_total = N_test + N_val
    # Split the validation set into unequal parts, to have similar lengths of val and test sets.
    N_test_split = N_total // 2 - N_test
    N_val_split = N_total - N_test_split
    test_indices = test_indices + val_indices[N_val_split:]
    val_indices = val_indices[:N_val_split]
    # Save the test and validation data to CSV files.
    test_df = solar_wind.iloc[test_indices]
    test_df.to_csv(f"{savepath}test_data.csv", index=False)
    val_df = solar_wind.iloc[val_indices]
    val_df.to_csv(f"{savepath}val_data.csv", index=False)
    # Print the number of samples in each set.
    print(f"Number of samples in training set: {len(train_indices)}")
    print(f"Number of samples in validation set: {len(val_indices)}")
    print(f"Number of samples in test set: {len(test_indices)}")


if __name__ == "__main__":
    # Path to the solar wind data file.
    solar_wind_path = "../data/solar_wind_data/solar_wind_data.csv"
    main(solar_wind_path)
