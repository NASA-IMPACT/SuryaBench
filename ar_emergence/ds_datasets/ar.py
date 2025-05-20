import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


# Dataset for AR Emergence
class AREmergenceDataset(Dataset):
    def __init__(self, data_dir="/rgroup/aifm/aremerge_skasapis"):
        self.file = h5py.File(data_dir, "r")
        # List all index groups (e.g., 'index_0', 'index_1', ...)
        self.indices = list(self.file.keys())

    def process(self, data, data_type="input"):
        """
        Process the data by normalizing it using the provided min and max values.

        Args:
            data (torch.Tensor): The input data to be normalized.

        Returns:
            torch.Tensor: The normalized data.
        """
        # Normalize the data using min-max scaling
        if data_type == "input":
            # Normalize input data
            self.data_min = torch.tensor(
                [-7.4745e07, -3.6508e08, -1.6605e08, -3.5536e07, -7.0342e01]
            )
            self.data_max = torch.tensor(
                [2.3280e07, 1.4658e08, 5.8470e07, 2.7218e07, 4.9013e02]
            )

            self.data_min = self.data_min.view(1, 5, 1)
            self.data_max = self.data_max.view(1, 5, 1)
            # Ensure data is in the same shape as min and max
            # if data.shape[1] != self.data_min.shape[1]:
            #     raise ValueError(f"Data shape {data.shape} does not match min/max shape {self.data_min.shape}")
            # Normalize the data

        elif data_type == "output":
            self.data_min = torch.tensor(-12419.5938)
            self.data_max = torch.tensor(2505.3042)

        normalized_data = (data - self.data_min) / (self.data_max - self.data_min)
        return normalized_data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        metadata = {}
        data = {}
        group = self.file[self.indices[idx]]
        input_matrix = group["input_matrix"][:]
        output_matrix = group["output_matrix"][:]
        input_times = group["input_times"][:]
        output_time = group["outout_time"][()]
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_matrix.astype(np.float32))
        output_tensor = torch.from_numpy(output_matrix.astype(np.float32))

        metadata["input_times"] = input_times
        metadata["output_time"] = output_time

        data["input"] = input_tensor
        data["output"] = output_tensor

        # input_times_tensor = torch.from_numpy(input_times.astype(np.float32))
        # output_time_tensor = torch.tensor(output_time, dtype=torch.float32)
        return data, metadata
