"""Solar Flare Dataset for Heliophysics.

This module provides a dataset class for handling solar flare data in conjunction
with the HelioNetCDFDataset base class.
"""

import pandas as pd
from datasets.helio import HelioNetCDFDataset


class FlareDSDataset(HelioNetCDFDataset):
    """Dataset class for solar flare prediction tasks.

    This class extends HelioNetCDFDataset to provide functionality specific to
    flare prediction, including loading and processing flare labels.

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
    ds_flare_index_paths : list, optional
        List of paths to flare index files.
    df_flare_label_type : str, optional
        Type of flare label to use ('maximum' or 'cumulative').
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
        ds_flare_index_paths: list = None,
        df_flare_label_type: str = None,
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

        # Load flare index and find intersection with HelioFM index
        self.fl_index = pd.DataFrame()
        all_data = [pd.read_csv(file) for file in ds_flare_index_paths]
        self.fl_index = pd.concat(all_data, ignore_index=True)

        # Convert timesteps to datetime and sort
        timesteps = self.fl_index["timestep"]
        dt_values = pd.to_datetime(timesteps).values
        self.fl_index["timestep"] = dt_values.astype("datetime64[ns]")
        self.fl_index.set_index("timestep", inplace=True)
        self.fl_index.sort_index(inplace=True)

        # Choose label type (maximum or cumulative)
        label_map = {"maximum": "label_max", "cumulative": "label_cum"}

        try:
            self.target_type = label_map[df_flare_label_type]
        except KeyError:
            msg = "Please check type, either 'maximum' or 'cumulative'"
            raise ValueError(msg)

        # Create HelioFM valid indices and find closest match to flare index
        valid_indices_df = pd.DataFrame({"valid_indices": self.valid_indices})
        self.fl_valid_indices = valid_indices_df.sort_values("valid_indices")

        # Merge flare index with valid indices
        merge_cols = ["valid_indices", "timestep"]
        self.fl_valid_indices = self.fl_valid_indices.merge(
            self.fl_index,
            how="inner",
            left_on=merge_cols[0],
            right_on=merge_cols[1],
        )

        # Override valid indices to reflect matches between HelioFM and flares
        valid_dates = self.fl_valid_indices["valid_indices"]
        self.valid_indices = [pd.Timestamp(date) for date in valid_dates]
        self.adjusted_length = len(self.valid_indices)
        self.fl_valid_indices.set_index("valid_indices", inplace=True)

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
            - 'target': Flare intensity label
        """
        # Get base dictionary from parent class
        base_dictionary, metadata = super().__getitem__(idx=idx)

        # Add flare intensity label
        target_idx = self.fl_valid_indices.iloc[idx]
        base_dictionary["target"] = target_idx[self.target_type]

        return base_dictionary, metadata
