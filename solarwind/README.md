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
- Input shape: (1, 5)
- Temporal coverage of the dataset is `2010-05-01` to `2023-12-31`
- 5 physical quantities: V, Bx(GSE), By(GSM), Bz(GSM), Number Density (N)
- Input timestamps: (120748,)
- cadence: Hourly
- Output shape: (1, 1)
- Output prediction:  (single value per prediction)


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

1. **Attention UNet**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 1)`

    An UNet model with multiple downsampling and upsampling layers along with attention mechanism.
    - 4 Downsampling blocks with 2 Convolutional Layers
    - 4 Upsampling blocks
    - Attention Gate 
    - Final out convolutional layer with adaptive average pooling to reduce the image dimensions to an integer value.


2. **UNet**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 63)`

     An UNet model with multiple downsampling and upsampling layers.
    - 4 Downsampling blocks with 2 Convolutional Layers
    - 4 Upsampling blocks
    - Final out convolutional layer with adaptive average pooling to reduce the image dimensions to an integer value.

3. **ResNet152**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 63)`

    A ResNet-18 variant:
    - Uses PyTorch’s resnet_152 pretrained_weights.
    - First 3D convolution modified to accept 13 channels.
    - Output layer adapted to predict 1 value.

4. **ResNet101**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 63)`

    A 3D ResNet-18 variant adapted for spatiotemporal input:
    - Uses PyTorch’s r3d_18 as the backbone.
    - First 3D convolution modified to accept 5 channels.
    - Output layer adapted to predict 63 values (one per tile).

5. **ResNet50**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 63)`

    A 3D ResNet-18 variant adapted for spatiotemporal input:
    - Uses PyTorch’s r3d_18 as the backbone.
    - First 3D convolution modified to accept 5 channels.
    - Output layer adapted to predict 63 values (one per tile).

6. **ResNet34**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 63)`

    A 3D ResNet-18 variant adapted for spatiotemporal input:
    - Uses PyTorch’s r3d_18 as the backbone.
    - First 3D convolution modified to accept 5 channels.
    - Output layer adapted to predict 63 values (one per tile).

7. **ResNet18**

    Input shape: `(B, 120, 5, 63)`
    Output shape: `(B, 63)`

    A 3D ResNet-18 variant adapted for spatiotemporal input:
    - Uses PyTorch’s r3d_18 as the backbone.
    - First 3D convolution modified to accept 5 channels.
    - Output layer adapted to predict 63 values (one per tile).