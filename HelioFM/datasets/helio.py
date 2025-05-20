import random
import torch
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset


class RandomChannelMaskerTransform:
    def __init__(self, num_channels, num_mask_aia_channels, phase, drop_hmi_probablity):
        """
        Initialize the RandomChannelMaskerTransform class as a transform.

        Args:
        - num_channels: Total number of channels in the input (3rd dimension of the tensor).
        - num_mask_aia_channels: Number of channels to randomly mask.
        """
        self.num_channels = num_channels
        self.num_mask_aia_channels = num_mask_aia_channels
        self.drop_hmi_probablity = drop_hmi_probablity

        # print(f'[{phase}] Using Random Channel Masker with num_mask_aia_channels = {self.num_mask_aia_channels}')

    def __call__(self, input_tensor):

        C, T, H, W = input_tensor.shape  # Unpacking the correct 5 dimensions

        # Randomly select channels to mask
        channels_to_mask = random.sample(range(C), self.num_mask_aia_channels)

        # Create an in-place mask of shape [1, 1, num_channels, 1, 1]
        mask = torch.ones((C, 1, 1, 1))
        mask[channels_to_mask, ...] = 0  # Set selected channels to zero

        # Apply the mask in-place for memory efficiency
        masked_tensor = input_tensor * mask  # Modify input_tensor directly

        if self.drop_hmi_probablity > random.random():
            masked_tensor[-1, ...] = 0

        return masked_tensor


class HelioNetCDFDataset(Dataset):
    """
    PyTorch dataset to load a curated dataset from the NASA Solar Dynamics
    Observatory (SDO) mission stored as NetCDF files, with handling for variable timesteps.

    Internally maintains two databases. The first is`self.index`. This takes the form
                                                                        path  present
        timestep
        2011-01-01 00:00:00  /lustre/fs0/scratch/shared/data/2011/01/Arka_2...        1
        2011-01-01 00:12:00  /lustre/fs0/scratch/shared/data/2011/01/Arka_2...        1
        ...                                                                ...      ...
        2012-11-30 23:48:00  /lustre/fs0/scratch/shared/data/2012/11/Arka_2...        1

    The second is `self.valid_indices`. This is simply a list of timesteps -- entries in the
    index of `self.index` -- which define valid samples. A sample is valid when all timestamps
    that can be reached by entris in time_delta_input_minutes and time_delta_target_minutes
    can be reached from it are present.
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
    ):
        self.scalers = scalers
        self.phase = phase
        self.channels = channels
        self.num_mask_aia_channels = num_mask_aia_channels
        self.drop_hmi_probablity = drop_hmi_probablity
        self.n_input_timestamps = n_input_timestamps
        self.rollout_steps = rollout_steps
        self.use_latitude_in_learned_flow = use_latitude_in_learned_flow

        if self.channels is None:
            # AIA + HMI channels
            self.channels = [
                "0094",
                "0131",
                "0171",
                "0193",
                "0211",
                "0304",
                "0335",
                "hmi",
            ]
        self.in_channels = len(self.channels)

        self.masker = RandomChannelMaskerTransform(
            num_channels=self.in_channels,
            num_mask_aia_channels=self.num_mask_aia_channels,
            phase=self.phase,
            drop_hmi_probablity=self.drop_hmi_probablity,
        )

        # Convert time delta to numpy timedelta64
        self.time_delta_input_minutes = sorted(
            np.timedelta64(t, "m") for t in time_delta_input_minutes
        )
        self.time_delta_target_minutes = [
            np.timedelta64(iroll * time_delta_target_minutes, "m")
            for iroll in range(1, rollout_steps + 2)
        ]

        # Create the index
        self.index = pd.read_csv(index_path)
        self.index = self.index[self.index["present"] == 1]
        self.index["timestep"] = pd.to_datetime(self.index["timestep"]).values.astype(
            "datetime64[ns]"
        )
        self.index.set_index("timestep", inplace=True)
        self.index.sort_index(inplace=True)

        # Filter out rows where the sequence is not fully present
        self.valid_indices = self.filter_valid_indices()
        self.adjusted_length = len(self.valid_indices)

    def filter_valid_indices(self):
        """
        Extracts timestamps from the index of self.index that define valid
        samples.

        Args:
        Returns:
            List of timestamps.
        """

        valid_indices = []
        time_deltas = np.unique(
            self.time_delta_input_minutes + self.time_delta_target_minutes
        )

        for reference_timestep in self.index.index:
            required_timesteps = reference_timestep + time_deltas

            if all(t in self.index.index for t in required_timesteps):
                valid_indices.append(reference_timestep)

        return valid_indices

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                forecast (torch.Tensor):          C, L, H, W
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        # start_time = time.time()

        time_deltas = np.array(
            sorted(
                random.sample(
                    self.time_delta_input_minutes[:-1], self.n_input_timestamps - 1
                )
            )
            + [self.time_delta_input_minutes[-1]]
            + self.time_delta_target_minutes
        )
        reference_timestep = self.valid_indices[idx]
        required_timesteps = reference_timestep + time_deltas
        sequence_data = [
            self.transform_data(
                self.load_nc_data(
                    self.index.loc[timestep, "path"], timestep, self.channels
                )
            )
            for timestep in required_timesteps
        ]

        # Split sequence_data into inputs and target
        inputs = sequence_data[: -self.rollout_steps - 1]
        targets = sequence_data[-self.rollout_steps - 1 :]

        stacked_inputs = np.stack(inputs, axis=1)
        stacked_targets = np.stack(targets, axis=1)

        timestamps_input = required_timesteps[: -self.rollout_steps - 1]
        timestamps_targets = required_timesteps[-self.rollout_steps - 1 :]

        if self.num_mask_aia_channels > 0 or self.drop_hmi_probablity:
            # assert 0 < self.num_mask_aia_channels < self.in_channels, \
            #     f'num_mask_aia_channels = {self.num_mask_aia_channels} should lie between 0 and {self.in_channels}'

            stacked_inputs = self.masker(stacked_inputs)

        time_delta_input_float = (
            time_deltas[-self.rollout_steps - 2]
            - time_deltas[: -self.rollout_steps - 1]
        ) / np.timedelta64(1, "h")
        time_delta_input_float = time_delta_input_float.astype(np.float32)

        lead_time_delta_float = (
            time_deltas[-self.rollout_steps - 2]
            - time_deltas[-self.rollout_steps - 1 :]
        ) / np.timedelta64(1, "h")
        lead_time_delta_float = lead_time_delta_float.astype(np.float32)

        # print('LocalRank', int(os.environ["LOCAL_RANK"]),
        #       'GlobalRank', int(os.environ["RANK"]),
        #       'worker', torch.utils.data.get_worker_info().id,
        #       f': Processed Input: {idx} ',time.time()- start_time)

        metadata = {
            "timestamps_input": timestamps_input,
            "timestamps_targets": timestamps_targets,
        }

        if self.use_latitude_in_learned_flow:
            from sunpy.coordinates.ephemeris import get_earth

            sequence_latitude = [
                get_earth(timestep).lat.value for timestep in required_timesteps
            ]
            input_latitudes = sequence_latitude[: -self.rollout_steps - 1]
            target_latitude = sequence_latitude[-self.rollout_steps - 1 :]

            return {
                "ts": stacked_inputs,
                "time_delta_input": time_delta_input_float,
                "input_latitudes": input_latitudes,
                "forecast": stacked_targets,
                "lead_time_delta": lead_time_delta_float,
                "forecast_latitude": target_latitude,
            }, metadata

        return {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float,
            "forecast": stacked_targets,
            "lead_time_delta": lead_time_delta_float,
        }, metadata

    def load_nc_data(
        self, filepath: str, timestep: pd.Timestamp, channels: list[str]
    ) -> np.ndarray:
        """
        Args:
            filepath: String or Pathlike. Points to NetCDF file to open.
            timestep: Identifies timestamp to retrieve.
        Returns:
            Numpy array of shape (C, H, W).
        """
        with xr.open_dataset(
            filepath, chunks=None, cache=False
        ) as ds:
            data = ds[channels].to_array().values
        return data

    def transform_data(self, data: np.ndarray) -> np.ndarray:
        """
        Applies scalers.

        Args:
            data: Numpy array of shape (C, H, W)
        Returns:
            Tensor of shape (C, H, W). Data type float32.
        """
        for idx, channel in enumerate(self.channels):
            data[idx] = self.scalers[channel].signum_log_transform(data[idx])

        return data.astype(np.float32)
