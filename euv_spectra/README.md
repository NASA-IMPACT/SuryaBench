# EVE-AIA Dataset Preparation and Loader for HelioFM

This repository provides utilities for preparing and loading EUV irradiance spectra from NASA's EVE instrument, aligned with input data from the HelioFM framework. The pipeline enables ML-ready formatting and timestamp-synchronized dataset loading using PyTorch.

---

## Repository Structure

- `prepare_data.ipynb`  
  Jupyter notebook to load, preprocess, and save training input (`X_train`) and target (`Y_train`) tensors. Handles tensor conversion, shape formatting, and disk saving.

- `eve_dataloader.py`  
  A PyTorch `Dataset` class (`EVEDSDataset`) that extends `HelioNetCDFDataset`. It aligns EVE spectra from a NetCDF file with HelioFM's temporal index, performs log-scaling and global normalization, and supports train/val/test splits.

- `euv_wavelengths.csv`  
  Contains the EUV wavelength grid (1343 wavelengths) corresponding to the irradiance spectra, saved for reference or postprocessing.

---

## Setup

Ensure the following dependencies are installed:

```bash
pip install numpy pandas xarray torch netCDF4

Ensure that the HelioFM repository is available locally and its modules are importable in your environment (e.g., via sys.path.append()).

How to Use:

Run the prepare_data.ipynb notebook to generate:
X_train.pt: input tensor of shape (N, 13, 4096, 4096)
Y_train.pt: corresponding target spectra of shape (N, 1343)
Both files will be saved in the current directory.
Optionally, Y_train.csv will also be saved to allow inspection of the spectra in tabular format.

Inside the prepare_data.ipynb notebook, we use eve_dataloader
To construct the PyTorch-ready dataset aligned with EVE spectra, use the EVEDSDataset class like so:

from eve_dataloader import EVEDSDataset
train_dataset = EVEDSDataset(
    index_path="path/to/heliofm/index.nc",
    time_delta_input_minutes=[-12, -6, 0],
    time_delta_target_minutes=0,
    n_input_timestamps=3,
    rollout_steps=1,
    channels=["aia_171", "aia_193", "aia_211"],
    phase="train",
    ds_eve_index_path="AIA_EVE_dataset_combined_NAS.nc",
    ds_time_column="train_time",
    ds_time_tolerance="6m",  # Match timestamps within a 6-minute window
)

To load validation or test data, just change:

phase → "val" or "test"
ds_time_column → "val_time" or "test_time"
You can also modify ds_time_tolerance to change the matching window (e.g., "1m", "10s", "15m").

Preprocessing Details
Zero handling: Zero values in EVE spectra are replaced by the wavelength-wise minimum (avoids -inf when taking log10)
Log-scaling: Intensities are compressed using log10 for dynamic range reduction
Normalization: Spectra are scaled globally using predefined min/max values (-9.00 to -1.96 in log10 space)

Output Format
Each training sample returns:
ts: 3D temporal image data from HelioFM inputs (shape: (13, 4096, 4096))
target: Normalized EVE spectrum vector (length 1343)