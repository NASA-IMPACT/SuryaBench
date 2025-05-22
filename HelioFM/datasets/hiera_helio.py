"""
LightningDataModule to load Heliophysics data from NPZ files using numpy.
"""

import glob
import os

import lightning as L
import numpy as np
import pandas as pd
import torch
import xarray as xr


# %%
class HelioDataset(torch.utils.data.Dataset):
    """
    Pytorch dataset to load a curated data set from the NASA Solar Dynamics
    Observatory (SDO) mission stored as NPZ files.

    Loading data from three instruments:
    - Atmospheric Imaging Assembly (AIA)
    - Helioseismic and Magnetic Imager (HMI)
    - EUV Variability Experiment (EVE)

    References:
    - Galvez, R., Fouhey, D. F., Jin, M., Szenicer, A., Muñoz-Jaramillo, A.,
      Cheung, M. C. M., Wright, P. J., Bobra, M. G., Liu, Y., Mason, J., & Thomas, R.
      (2019). A Machine-learning Data Set Prepared from the NASA Solar Dynamics
      Observatory Mission. The Astrophysical Journal Supplement Series, 242(1), 7.
      https://doi.org/10.3847/1538-4365/ab1005
    """

    def __init__(self, dirpath: str = "data"):
        self.dirpath = dirpath

        # Atmospheric Imaging Assembly (AIA) data
        self.wavelength_bands: list[int] = [
            94,
            131,
            171,
            193,
            211,
            304,
            335,
            1600,
            1700,
            4500,
        ]

        # Initialize pandas.DataFrame with 4500 wavelength data at hourly resolution
        # pd.date_range(start="2010-01-01", end="2018-12-31", freq="1h")
        wavelength_data: dict[str, str] = {
            "AIA_4500": sorted(
                glob.glob(pathname=f"{self.dirpath}/AIA_4500_??????/**/*.npz")
            )
        }
        self.df_wavelength = pd.DataFrame.from_dict(
            data=wavelength_data, orient="columns"
        )

        # Get the datetime based on filename (hourly cadence)
        self.df_wavelength["YYYYMMDD_HHMM"] = (
            self.df_wavelength["AIA_4500"]
            .str.split("/")
            .str.get(-1)
            .str.extract(pat=r"AIA(\d{8}_\d{4})_")
        )
        self.df_wavelength["datetime"] = pd.to_datetime(
            self.df_wavelength["YYYYMMDD_HHMM"], format="%Y%m%d_%H%M"
        )

        # Get npz filepaths for all the other AIA wavelength bands
        for aia_band in self.wavelength_bands[:-1]:  # every band except 4500
            # E.g. 'data/AIA_0094_201812/31/AIA20181231_1900_0094.npz'
            self.df_wavelength[f"AIA_{aia_band:04}"] = (
                self.dirpath
                + "/"
                + f"AIA_{aia_band:04}_"
                + self.df_wavelength.datetime.dt.year.astype(pd.StringDtype())
                + self.df_wavelength.datetime.dt.month.astype(pd.StringDtype())
                + "/"
                + self.df_wavelength.datetime.dt.day.astype(pd.StringDtype()).str.zfill(
                    width=2
                )
                + "/"
                + "AIA"
                + self.df_wavelength["YYYYMMDD_HHMM"]
                + f"_{aia_band:04}.npz"
            )

        # Helioseismic and Magnetic Imager (HMI) data
        for hmi_band in ["bx", "by", "bz"]:
            # E.g. 'data/HMI_Bz_201812/01/HMI20181201_0000_bz.npz'
            self.df_wavelength[f"HMI_{hmi_band}"] = (
                self.dirpath
                + "/"
                + f"HMI_{hmi_band.title()}_"
                + self.df_wavelength.datetime.dt.year.astype(pd.StringDtype())
                + self.df_wavelength.datetime.dt.month.astype(pd.StringDtype())
                + "/"
                + self.df_wavelength.datetime.dt.day.astype(pd.StringDtype()).str.zfill(
                    width=2
                )
                + "/"
                + "HMI"
                + self.df_wavelength["YYYYMMDD_HHMM"]
                + f"_{hmi_band}.npz"
            )

        # Remove rows where npz filepaths don't exist
        data_cols: list[str] = [
            col
            for col in list(self.df_wavelength.columns)
            if col.startswith("AIA_") or col.startswith("HMI_")
        ]
        filepath_exists_mask: np.ndarray = np.logical_and.reduce(
            array=[self.df_wavelength[col].map(os.path.exists) for col in data_cols]
        )
        self.df_wavelength: pd.DataFrame = self.df_wavelength.loc[
            filepath_exists_mask
        ].reset_index(drop=True)

        # Match t=0 rows with t+1hour rows
        df_has_tplusone = (
            self.df_wavelength["datetime"]
            .astype(np.int64)
            .rolling(window=2, closed="both")
            .apply(lambda t: t.iloc[1] - t.iloc[0])
        ).shift(periods=-1) == 3.6e12
        self.df_t0 = self.df_wavelength[df_has_tplusone]
        self.df_t1 = self.df_wavelength.loc[self.df_t0.index + 1]

    def __len__(self) -> int:
        return len(self.df_t0) - 1

    def __getitem__(self, index: int = 0) -> dict[str, torch.Tensor]:
        """
        Get a tensor of shape (Channels, Depth, Height, Width).
        """
        wavelengths: list = [
            *[f"AIA_{wv:04}" for wv in self.wavelength_bands],
            "HMI_bx",
            "HMI_by",
            "HMI_bz",
        ]

        wavelength_t0_series: pd.Series = self.df_t0.iloc[index]
        wavelength_t1_series: pd.Series = self.df_t0.iloc[index]

        tensor_t0: torch.Tensor = torch.from_numpy(
            np.stack(
                arrays=[
                    np.load(file=wavelength_t0_series[wavelength])["x"]
                    for wavelength in wavelengths
                ],
                axis=0,
            )
        )
        tensor_t0: torch.Tensor = tensor_t0.unsqueeze(
            dim=1
        )  # add extra dim (T) for frame
        assert tensor_t0.shape == (len(self.df_wavelength.columns) - 2, 1, 512, 512)
        tensor_t1: torch.Tensor = torch.from_numpy(
            np.stack(
                arrays=[
                    np.load(file=wavelength_t1_series[wavelength])["x"]
                    for wavelength in wavelengths
                ],
                axis=0,
            )
        )
        tensor_t1: torch.Tensor = tensor_t1.unsqueeze(
            dim=1
        )  # add extra dim (T) for frame
        assert tensor_t1.shape == (len(self.df_wavelength.columns) - 2, 1, 512, 512)

        return {"t0": tensor_t0, "t1": tensor_t1}


class HelioNetCDFDataset(torch.utils.data.Dataset):
    """
    Pytorch dataset to load a curated data set from the NASA Solar Dynamics
    Observatory (SDO) mission stored as NetCDF files.
    """

    def __init__(self, dirpath: str = "data"):
        self.dirpath = dirpath

        nc_paths: list = sorted(glob.glob(pathname=f"{self.dirpath}/*.nc"))
        self.ds: xr.Dataset = xr.open_mfdataset(
            paths=nc_paths, engine="h5netcdf", combine="nested", concat_dim="time"
        )

    def __len__(self) -> int:
        return len(self.ds.time) - 1

    def __getitem__(self, index: int = 0) -> torch.Tensor:
        """
        Get a tensor of shape (Channels, Depth, Height, Width).
        """
        ds_t0: xr.Dataset = self.ds.isel(time=index)
        tensor_t0: torch.Tensor = torch.as_tensor(
            data=ds_t0.value.data.astype(dtype="float32").compute()
        )
        # assert tensor_t0.shape == (8, 4096, 4096)
        tensor_t0: torch.Tensor = tensor_t0.unsqueeze(
            dim=1
        )  # add extra dim (T) for frame
        # assert tensor.shape == (8, 1, 4096, 4096)

        ds_t1: xr.Dataset = self.ds.isel(time=index + 1)
        tensor_t1: torch.Tensor = torch.as_tensor(
            data=ds_t1.value.data.astype(dtype="float32").compute()
        )
        # assert tensor_t1.shape == (8, 4096, 4096)
        tensor_t1: torch.Tensor = tensor_t1.unsqueeze(
            dim=1
        )  # add extra dim (T) for frame
        # assert tensor_t1.shape == (8, 1, 4096, 4096)

        return {"t0": tensor_t0, "t1": tensor_t1}


class HelioDataModule(L.LightningDataModule):
    """
    LightningDataModule to load NASA Solar Dynamics Observatory (SDO) mission data from
    NPZ files.
    """

    def __init__(
        self,
        train_data_path: str = "data",
        valid_data_path: str = "data",
        batch_size: int = 8,
        num_data_workers: int = 8,
    ):
        """
        Go from multiple NPZ files to time-series tensors!

        Also does mini-batching and train/validation splits.

        Parameters
        ----------
        batch_size : int
            Size of each mini-batch. Default is 8.
        num_data_workers : int
            How many subprocesses to use for data loading. 0 means that the data will be
            loaded in the main process. Default is 8.
        """
        super().__init__()
        self.train_dirpath = train_data_path
        self.valid_dirpath = valid_data_path

        self.batch_size: int = batch_size
        self.num_workers: int = num_data_workers

    def setup(
        self, stage: str | None = None
    ) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.
        """
        if stage == "fit":  # training/validation loop
            self.dataset_train = HelioNetCDFDataset(dirpath=self.train_dirpath)
            self.dataset_val = HelioNetCDFDataset(dirpath=self.valid_dirpath)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the training loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Loads the data used in the validation loop.
        """
        return torch.utils.data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
