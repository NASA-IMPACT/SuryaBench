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

## EVE EUV Spectra Prediction 

This  contains code and model implementations for predicting the EVE spectra from spatiotemporal solar data. The dataset contains hundreds of thousands of temporally aligned AIA image cubes and EUV spectra, covering Solar Cycle 24 and parts of Solar Cycle 25, and including both quiet-Sun and active-region conditions. 

---

### 📊 Dataset Description

**Dataset can be found at [NASA-IMPACT HuggingFace Repository](https://huggingface.co/datasets/nasa-impact/surya-bench-euv-spectra)**

The dataset consists of three splits (70-15-15): train, val, and test, each containing:
- A timestamp
- A 1343-dimensional spectrum corresponding to EUV wavelengths ranging from approximately 6.5 to 33.3 nm, observed at a 1-minute cadence
- EUV measurements from both flare and quiet sun conditions
- Input shape: (1, 13, 4096, 4096)
- Output shape: (1, 1343)


### 🚀 Example Usage

For training run the below code

1. **Resnet Models**
Multiple resnet models as described below have been used to train the baseline. The only change needed to run the baselines in the given command is to change the model name.
- Resnet18
- Resnet34
- Resnet50
- Resnet101
- Resnet152
```
python train_baseline.py --config_path ./ds_configs/config_resnet_18.yaml --gpu 
```

```
torchrun --nnodes=1 --nproc_per_node=1 train_baseline.py --config_path ./ds_configs/config_resnet_18.yaml --gpu
```

2. **AttentionUNet**
```
python train_baseline.py --config_path ./ds_configs/config_attention_unet.yaml --gpu 
```

```
torchrun --nnodes=1 --nproc_per_node=1 train_baseline.py --config_path ./ds_configs/config_attention_unet.yaml --gpu
```

3. **UNet**
```
python train_baseline.py --config_path ./ds_configs/config_unet.yaml --gpu 
```

```
torchrun --nnodes=1 --nproc_per_node=1 train_baseline.py --config_path ./ds_configs/config_unet.yaml --gpu
```