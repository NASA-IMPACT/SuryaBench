import torch
from tqdm import tqdm
from ds_datasets.ar import AREmergenceDataset

def compute_dataset_min_max(h5_file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the global min and max values across all output samples
    in the AREmergenceDataset.

    Args:
        h5_file_path (str): Path to the HDF5 dataset.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Global minimum and maximum values.
    """
    dataset = AREmergenceDataset(h5_file_path)

    data_min = None
    data_max = None

    for d, _ in tqdm(dataset, desc="Computing min/max"):
        output = d["output"]  # Shape: (some_dims), assumed scalar or 1D

        batch_min = output.min()
        batch_max = output.max()

        if data_min is None:
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max


if __name__ == "__main__":
    train_data_path = (
        "/rgroup/aifm/aremerge_skasapis/"
        "train_indexed_data_ar_emergence_kasapis_rohit.h5"
    )

    min_val, max_val = compute_dataset_min_max(train_data_path)

    print("Channel-wise Min:", min_val)
    print("Channel-wise Max:", max_val)
