import sys
import os
import numpy as np
import pandas as pd

# Append base path.  May need to be modified if the folder structure changes
from datasets.helio import HelioNetCDFDataset


class FlareDSDataset(HelioNetCDFDataset):
    def __init__(
        self,
        #### All these lines are required by the parent HelioNetCDFDataset class
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
        #### Put your donwnstream (DS) specific parameters below this line
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

        # Load ds index and find intersection with HelioFM index
        self.fl_index = pd.DataFrame()
        all_data = [pd.read_csv(file) for file in ds_flare_index_paths]
        self.fl_index = pd.concat(all_data, ignore_index=True)

        self.fl_index["timestep"] = pd.to_datetime(
            self.fl_index["timestep"]
        ).values.astype("datetime64[ns]")
        self.fl_index.set_index("timestep", inplace=True)
        self.fl_index.sort_index(inplace=True)

        # Choose a label between maximum and cumulative label.
        label_map = {"maximum": "label_max", "cumulative": "label_cum"}

        try:
            self.target_type = label_map[df_flare_label_type]
        except KeyError:
            raise ValueError("Please check type, either 'maximum' or 'cumulative'")

        # Create HelioFM valid indices and find closest match to DS index
        self.fl_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")
        self.fl_valid_indices = self.fl_valid_indices.merge(
            self.fl_index, how="inner", left_on="valid_indices", right_on="timestep"
        )

        # Override valid indices variables to reflect matches between HelioFM and DS
        self.valid_indices = [
            pd.Timestamp(date) for date in self.fl_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.fl_valid_indices.set_index("valid_indices", inplace=True)

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary with following keys. The values are tensors with shape as follows:
                # HelioFM keys--------------------------------
                ts (torch.Tensor):                C, T, H, W
                time_delta_input (torch.Tensor):  T
                input_latitude (torch.Tensor):    T
                forecast (torch.Tensor):          C, L, H, W
                lead_time_delta (torch.Tensor):   L
                forecast_latitude (torch.Tensor): L
                # HelioFM keys--------------------------------
                flare_intensity_target
            C - Channels, T - Input times, H - Image height, W - Image width, L - Lead time.
        """

        # This lines assembles the dictionary that HelioFM's dataset returns (defined above)
        base_dictionary, metadata = super().__getitem__(idx=idx)

        # We now add the flare intensity label
        base_dictionary["target"] = self.fl_valid_indices.iloc[idx][self.target_type]

        return base_dictionary, metadata
