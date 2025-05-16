import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class AREmergenceDataset(Dataset):
    """
    PyTorch Dataset for Active Region (AR) Emergence from HDF5 files.

    Each sample contains a multichannel time series input and a scalar output.

    Args:
        data_dir (str): Path to the HDF5 file.
    """

    def __init__(self, data_dir: str = "/rgroup/aifm/aremerge_skasapis"):
        super().__init__()
        self.file = h5py.File(data_dir, "r")
        self.indices = list(self.file.keys())

    def process(self, data: torch.Tensor, data_type: str = "input") -> torch.Tensor:
        """
        Normalize input/output data using predefined min-max statistics.

        Args:
            data (torch.Tensor): Input data tensor.
            data_type (str): One of {'input', 'output'} to apply the correct normalization.

        Returns:
            torch.Tensor: Normalized data.
        """
        if data_type == "input":
            data_min = torch.tensor(
                [-7.4745e07, -3.6508e08, -1.6605e08, -3.5536e07, -7.0342e01]
            ).view(1, 5, 1)

            data_max = torch.tensor(
                [2.3280e07, 1.4658e08, 5.8470e07, 2.7218e07, 4.9013e02]
            ).view(1, 5, 1)

        elif data_type == "output":
            data_min = torch.tensor(-12419.5938)
            data_max = torch.tensor(2505.3042)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        return (data - data_min) / (data_max - data_min)

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of samples.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple:
                - data (dict): Contains 'input' and 'output' tensors.
                - metadata (dict): Contains 'input_times' and 'output_time'.
        """
        group = self.file[self.indices[idx]]
        input_matrix = group["input_matrix"][:]
        output_matrix = group["output_matrix"][:]
        input_times = group["input_times"][:]
        output_time = group["output_time"][()]

        input_tensor = torch.from_numpy(input_matrix.astype(np.float32))
        output_tensor = torch.from_numpy(output_matrix.astype(np.float32))

        data = {"input": input_tensor, "output": output_tensor}

        metadata = {"input_times": input_times, "output_time": output_time}

        return data, metadata
