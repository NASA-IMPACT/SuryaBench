"""
Download solar wind data using sunpy.
The arguments are all given default values to generate the solar wind dataset in SuryaBench.
"""

import os

import matplotlib.pyplot as plt
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries


def pull_solarwind(
    T_start,
    T_end,
    ID_dataset="OMNI2_H0_MRG1HR",
    vars=["V", "BX_GSE", "BY_GSM", "BZ_GSM", "N"],
    save_dir=None,
):
    """Download the solar wind data from OMNI database. This script will download the data, drop nans, and return a pandas dataframe.

    Args:
        T_start (str): Starting datetime for solar wind data.
        T_end (str): Ending datetime for solar wind data
        ID_dataset (str, optional): The OMNI time series to be pulled. Defaults to 'OMNI2_H0_MRG1HR'.
        vars (list, optional): Set of variables to be returned. This must be consistent with the variables returned by OMNI data.
                               Defaults to ['V','BX_GSE','BY_GSM','BZ_GSM','N'].
        save_dir (str, optional): Path of directory to save the files. Defaults to None, which would mean the path where sunpy is installed.
    Returns:
        solar_wind (pandas.df): The time series object containing the solar wind data.
    """
    # We are selecting a date range from 2017/09/25 to 2017/09/29. This is a HSE
    trange = a.Time(T_start, T_end)
    print(f"Downloading solar wind data from {T_start} to {T_end}...")
    dataset = a.cdaweb.Dataset(ID_dataset)
    result = Fido.search(trange, dataset)
    downloaded_files = Fido.fetch(result, path=save_dir)
    solar_wind = TimeSeries(downloaded_files, concatenate=True)
    solar_wind = solar_wind.to_dataframe()[vars]
    solar_wind = solar_wind.dropna()
    return solar_wind


def main():
    T_start = "2010/01/01"
    T_end = "2023/12/31"
    ID_dataset = "OMNI2_H0_MRG1HR"
    save_dir = "../data/solar_wind_data/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    solar_wind = pull_solarwind(T_start, T_end, ID_dataset, save_dir=save_dir)
    solar_wind.plot(subplots=True, figsize=(12, 8))
    plt.savefig(os.path.join(save_dir, "solar_wind_data.png"))
    solar_wind.to_csv(os.path.join(save_dir, "solar_wind_data.csv"))


if __name__ == "__main__":
    main()
