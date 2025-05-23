"""Active Region (AR) Dataset for Heliophysics.

This module provides a dataset class for handling Active Region segmentation
data in conjunction with the HelioNetCDFDataset base class.
"""

import h5py
import pandas as pd

from datasets.helio import HelioNetCDFDataset


class ArDSDataset(HelioNetCDFDataset):
    """Dataset class for Active Region segmentation tasks.

    This class extends HelioNetCDFDataset to provide functionality specific to
    Active Region segmentation, including loading and processing AR masks.

    Parameters
    ----------
    index_path : str
        Path to the index file containing timestamps.
    time_delta_input_minutes : list[int]
        List of time deltas in minutes for input data.
    time_delta_target_minutes : int
        Time delta in minutes for target data.
    n_input_timestamps : int
        Number of input timestamps to use.
    rollout_steps : int
        Number of steps to roll out in prediction.
    scalers : object, optional
        Scaler objects for data normalization.
    num_mask_aia_channels : int, optional
        Number of AIA channels to mask, by default 0.
    drop_hmi_probablity : float, optional
        Probability of dropping HMI data, by default 0.
    use_latitude_in_learned_flow : bool, optional
        Whether to use latitude in flow learning, by default False.
    channels : list[str], optional
        List of channels to use, by default None.
    phase : str, optional
        Dataset phase (train/val/test), by default "train".
    ds_ar_index_paths : list, optional
        List of paths to AR index files.
    """

    def __init__(
        self,
        index_path: str,
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        n_input_timestamps: int,
        rollout_steps: int,
        scalers=None,
        num_mask_aia_channels=0,
        drop_hmi_probablity=0,
        use_latitude_in_learned_flow=False,
        channels: list[str] | None = None,
        phase="train",
        ds_ar_index_paths: list = None,
    ):
        super().__init__(
            index_path=index_path,
            time_delta_input_minutes=time_delta_input_minutes,
            time_delta_target_minutes=time_delta_target_minutes,
            n_input_timestamps=n_input_timestamps,
            rollout_steps=rollout_steps,
            scalers=scalers,
            num_mask_aia_channels=num_mask_aia_channels,
            drop_hmi_probablity=drop_hmi_probablity,
            use_latitude_in_learned_flow=use_latitude_in_learned_flow,
            channels=channels,
            phase=phase,
        )

        # Load AR index and find intersection with HelioFM index
        self.ar_index = pd.DataFrame()
        print(ds_ar_index_paths)
        all_data = [pd.read_csv(file) for file in ds_ar_index_paths]
        self.ar_index = pd.concat(all_data, ignore_index=True)
        self.ar_index = self.ar_index.loc[self.ar_index["present"] == 1, :]

        # Convert timesteps to datetime and sort
        timesteps = self.ar_index["timestep"]
        dt_values = pd.to_datetime(timesteps).values
        self.ar_index["timestep"] = dt_values.astype("datetime64[ns]")
        self.ar_index.sort_values("timestep", inplace=True)

        # Create HelioFM valid indices and find closest match to AR index
        self.ar_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")

        self.ar_valid_indices = pd.merge(
            self.ar_index,
            self.ar_valid_indices,
            how="inner",
            left_on="timestep",
            right_on="valid_indices",
        )

        # Override valid indices to reflect matches between HelioFM and AR
        self.valid_indices = [
            pd.Timestamp(date) for date in self.ar_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.ar_valid_indices.set_index("valid_indices", inplace=True)

    def __len__(self):
        """Return the length of the dataset."""
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            Dictionary containing:
            - All keys from HelioNetCDFDataset
            - 'target': AR mask from h5 file
        """
        # Get base dictionary from parent class
        base_dictionary, metadata = super().__getitem__(idx=idx)

        # Load AR mask
        file_path = self.ar_valid_indices.iloc[idx]["file_path"]
        mask = h5py.File(file_path, "r")
        base_dictionary["target"] = mask["union_with_intersect"]

        return base_dictionary, metadata
