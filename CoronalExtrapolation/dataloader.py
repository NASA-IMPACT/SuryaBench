import os
import pickle
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from tqdm import tqdm

## Append base path.  May need to be modified if the folder structure changes
sys.path.insert(0, "../HelioFM")
from datasets.helio import HelioNetCDFDataset
from utils.config import get_config
from utils.data import build_scalers


def main():
    # os.chdir("../..")
    # config_path = "downstream_configs/config_spectformer_dgx.yaml"
    config_path = "./ds_configs/config_resnet_18.yaml"

    config = get_config(config_path)
    scalers = build_scalers(info=config.data.scalers)

    train_dataset = ThreeDMagDSDataset(
        #### All these lines are required by the parent HelioNetCDFDataset class
        index_path="/nobackupnfs1/sroy14/processed_data/Helio/csv_files/index_201005_to_201812.csv",
        time_delta_input_minutes=config.data.time_delta_input_minutes,
        time_delta_target_minutes=config.data.time_delta_target_minutes,
        n_input_timestamps=config.data.n_input_timestamps,
        rollout_steps=config.rollout_steps,
        channels=config.data.channels,
        drop_hmi_probablity=config.drop_hmi_probablity,
        num_mask_aia_channels=config.num_mask_aia_channels,
        use_latitude_in_learned_flow=config.use_latitude_in_learned_flow,
        scalers=scalers,
        phase="train",
        #### Put your donwnstream (DS) specific parameters below this line
        ds_3dmag_index_path="/nobackupnfs1/sroy14/processed_data/Helio/daniel/heliofm_downstream_wsa_train_index_new.csv",
        ds_3dmag_wsa_root="/nobackupnfs1/sroy14/processed_data/Helio/daniel",
        ds_index_cache="./ds_index_cache.pickle",
        ds_input_timesteps=[0, -7, -14],
    )

    print("Dataset length", len(train_dataset))

    item, metadata = train_dataset[0]

    print("item keys", item.keys())
    print("item target shape", item["target"].shape)
    print("item target", item["target"])


class ThreeDMagDSDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset to show an example of how to create a
    dataset for donwstream applications. It includes both the necessary parameters
    to initialize the parent class, as well as those of the child

    HelioFM Parameters
    ------------------
    index_path : str
        Path to HelioFM index
    time_delta_input_minutes : list[int]
        Input delta times to define the input stack in minutes from the present
    time_delta_target_minutes : int
        Target delta time to define the output stack on rollout in minutes from the present
    n_input_timestamps : int
        Number of input timestamps
    rollout_steps : int
        Number of rollout steps
    scalers : optional
        scalers used to perform input data normalization, by default None
    num_mask_aia_channels : int, optional
        Number of aia channels to mask during training, by default 0
    drop_hmi_probablity : int, optional
        Probability of removing hmi during training, by default 0
    use_latitude_in_learned_flow : bool, optional
        Switch to provide heliographic latitude for each datapoint, by default False
    channels : list[str] | None, optional
        Input channels to use, by default None
    phase : str, optional
        Descriptor of the phase used for this database, by default "train"

    Downstream (DS) Parameters
    --------------------------
    ds_flare_index_path : str, optional
        DS index.  In this example a flare dataset, by default None
    ds_time_column : str, optional
        Name of the column to use as datestamp to compare with HelioFM's index, by default None
    ds_time_tolerance : str, optional
        How much time difference is tolerated when finding matches between HelioFM and the DS, by default None
    ds_match_direction : str, optional
        Direction used to find matches using pd.merge_asof possible values are "forward", "backward",
        or "nearest".  For causal relationships is better to use "forward", by default "forward"

    Raises
    ------
    ValueError
        Error is raised if there is not overlap between the HelioFM and DS indices
        given a tolerance
    """

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
        ds_3dmag_index_path: str = None,
        ds_3dmag_wsa_root: str = None,
        ds_index_cache="./ds_index_cache.pickle",
        ds_input_timesteps: list = [0, -7, -14],
    ):

        ## Initialize parent class
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

        self.ds_input_timesteps = ds_input_timesteps
        self.ds_3dmag_wsa_root = ds_3dmag_wsa_root
        if os.path.exists(ds_index_cache):
            print(f"Loading from {ds_index_cache}")
            with open(ds_index_cache, "rb") as fh:
                self.df_index = pickle.load(fh)
        else:
            self.df_index = self.get_df_index(ds_3dmag_index_path, ds_input_timesteps)

            print(f"Wrote to {ds_index_cache}")

            with open(ds_index_cache, "wb") as fh:
                pickle.dump(self.df_index, fh)

    def get_df_index(self, ds_3dmag_index_path, ds_input_timesteps):
        """
        Load 3DMag Dataframe and pair in time with the SDO indices
        based on ds_input_timesteps
        """
        raw_index = pd.read_csv(ds_3dmag_index_path, parse_dates=["timestamp"])
        pair_threshold = timedelta(hours=2)
        valid_indices = self.valid_indices.copy()
        valid_indices.sort()
        valid_indices = pd.Series(valid_indices)
        rows = []

        for _, row in tqdm(list(raw_index.iterrows())):
            if not row["present"]:
                continue

            row = row.copy()
            unable_to_pair = False
            for input_timestep in ds_input_timesteps:
                target_timestamp = row["timestamp"] + timedelta(days=input_timestep)
                sdo_idx = np.searchsorted(valid_indices, target_timestamp)
                # sdo_idx = np.argmin(np.abs(valid_indices - target_timestamp))
                pair_dist = abs(valid_indices[sdo_idx] - target_timestamp)
                if pair_dist > pair_threshold:
                    unable_to_pair = True
                    # print(input_timestep, valid_indices[sdo_idx], target_timestamp)
                    break

                key = "sdo" + str(abs(input_timestep))
                row[key] = sdo_idx

            if unable_to_pair:
                continue

            rows.append(row)

        df_index = pd.DataFrame(rows)

        print(
            "Dropped ", len(raw_index.index) - len(rows), "out of", len(raw_index.index)
        )

        return df_index

    def __len__(self):
        return len(self.df_index.index)

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Index of sample to load. (Pytorch standard.)
        Returns:
            Dictionary
        """
        row = self.df_index.iloc[idx]

        # Load SDO data
        base_dictionary = {}
        metadata = {}

        for i, input_timestep in enumerate(self.ds_input_timesteps):
            # print(f'Loading SDO input {i+1} out of {len(self.ds_input_timesteps)}')

            key = "sdo" + str(abs(input_timestep))
            cur_dictionary, cur_metadata = super().__getitem__(idx=row[key])

            base_dictionary[f"input{i}"] = cur_dictionary
            metadata[f"input{i}"] = cur_metadata

        # Load WSA Data
        # print('Loading WSA coefficients')
        wsa_path = os.path.join(self.ds_3dmag_wsa_root, row.wsa_file)
        fits_file = fits.open(wsa_path)
        sph_data = fits_file[3].data.copy()
        fits_file.close()

        base_dictionary["target"] = torch.from_numpy(
            np.array(
                [
                    sph_data[0, :, :][np.triu_indices(sph_data.shape[1])],
                    sph_data[1, :, :][np.triu_indices(sph_data.shape[1])],
                ]
            )
        )

        return base_dictionary, metadata


if __name__ == "__main__":
    main()
