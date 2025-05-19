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
```

Ensure that the HelioFM repository is available locally and its modules are importable in your environment (e.g., via sys.path.append()). Additionally, please make sure the configuration file `config_spectformer_dgx_test.yaml` is present and configured appropriately. This file provides the necessary paths and settings for the dataloader to locate the corresponding AIA and HMI files within the HelioFM dataset.

## How to Use

Run the prepare_data.ipynb notebook to generate:

    X_train.pt: input tensor of shape (N, 13, 4096, 4096)

    Y_train.csv: corresponding target spectra of shape (N, 1343)

Both files will be saved in the current directory.

Inside the prepare_data.ipynb notebook, we use eve_dataloader to construct the PyTorch-ready dataset aligned with EVE spectra, use the EVEDSDataset class like so:

```bash
from eve_dataloader import EVEDSDataset
train_dataset = eve_dataloader.EVEDSDataset(
    #### All these lines are required by the parent HelioNetCDFDataset class
    index_path=config.data.train_data_path,
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
    ds_eve_index_path= "../../hfmds/data/AIA_EVE_dataset_combined.nc",
    ds_time_column="train_time",
    ds_time_tolerance = "6m",
    ds_match_direction = "forward"    
)
```
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

## Active Region Emergence Prediction

This  contains code and model implementations for forecasting continuum intensity from spatiotemporal solar data. The dataset includes physical measurements from active regions on the Sun, preprocessed into a format that captures both spatial and temporal dynamics.

---

### 📊 Dataset Description

**Dataset can be found at [NASA-IMPACT HuggingFace Repository](https://huggingface.co/datasets/nasa-impact/ar_emergence)**

Each sample in the dataset corresponds to a tracked active region and is structured as follows:
- Input shape: (120, 5, 63)
- 120 timesteps per sample (≈24 hours at 12-minute cadence)
- 5 physical quantities:
- 1: Mean unsigned magnetic flux
- 2–5: Doppler velocity acoustic power in frequency bands: 2–3, 3–4, 4–5, and 5–6 mHz
- 63 spatial tiles, extracted from a 9×9 grid with top and bottom rows removed (7×9 = 63).
- Input timestamps: (120,)
- Output shape: (63,)
- Scalar continuum intensity prediction per tile
- Output timestamp:  (single value per prediction)


### 🚀 Example Usage

For training run the below code

1. **SpatioTemporalAttention Transformer**
```
python train_baselines.py --config_path ./ds_configs/config_spectformer_ar_sta.yaml --gpu 
```

2. **SpatioTemporalResNet**
```
python train_baselines.py --config_path ./ds_configs/config_ar_stresnet.yaml --gpu 
```

### 🧠 Models

1. **SpatioTemporalAttention Transformer**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 63)`

    A two-stage transformer architecture:
    - Temporal Transformer: models per-tile temporal evolution.
    - Spatial Transformer: models spatial interactions at each timestep.

    Core features:
    - Sinusoidal positional encodings for time and space.
    - Per-tile temporal encoding.
    - Per-timestep spatial encoding.
    - Mean-pooling over time followed by per-cell regression.


2. **SpatioTemporalResNet**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 63)`

    A 3D ResNet-18 variant adapted for spatiotemporal input:
    - Uses PyTorch’s r3d_18 as the backbone.
    - First 3D convolution modified to accept 5 channels.
    - Output layer adapted to predict 63 values (one per tile).