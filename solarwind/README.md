# Download solar wind data

This folder contains scripts for downloading the solar wind data and splits them into train-val-test sets. For convenience, we have `download_solar_wind_data.sh` that creates a virtual environment, installs required packages, downloads data, splits into sets, and destroys the environment. The data would be saved in `../data/`

1. First step is to download the solar wind data. For this, use data_process/download_sw_data.py. This needs the exact keyword variable by selecting OMNI from: [https://cdaweb.gsfc.nasa.gov/index.html](https://cdaweb.gsfc.nasa.gov/index.html). Be sure to select the right variables to get the correct data.
2. If you want to just run the scripts, you must run `download_sw_data.py` for downloading the solar wind data.
3. `split_trainValTest.py` splits the dataset into train-val-test sets. These sets are defined in the paper.

## Solar Wind Prediction

This contains code and model implementations for predicting the solar wind velocity. includes Speed, ("V"), Bx (GSE), By (GSM), Bz (GSM) and number density (N). For this task, we only consider the wind speed from the dataset.

---

### 📊 Dataset Description

**Dataset can be found at [NASA-IMPACT HuggingFace Repository](https://huggingface.co/datasets/nasa-impact/Surya-bench-solarwind)**

The dataset it stored as `.csv` files. Each sample in the dataset corresponds to a tracked active region and is structured as follows:
- Input shape: (1, 13, 4096, 4096)
- Temporal coverage of the dataset is `2010-05-01` to `2023-12-31`
- 5 physical quantities: V, Bx(GSE), By(GSM), Bz(GSM), Number Density (N)
- Input timestamps: (120748,)
- cadence: Hourly
- Output shape: (1)
- Output prediction:  (single value per prediction)


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