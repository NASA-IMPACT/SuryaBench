# Coronal Field Extrapolation

This downstream application uses the foundation model to emulate the physics-based ADAPT-WSA PFSS model. The parameters to predict are spherical harmonic coefficients which represent the magnetic potential for a domain between the photosphere and te source surface (set to 2.51 Rs). 

`make_train_test_index.py`: Make training and validation indeces from a directory holding the coronal field extrapolation benchmark data
`dataloader.py`: Dataset which can be used with PyTorch

## Authors
Daniel da Silva, [daniel.e.dasilva@nasa.gov](daniel.e.dasilva@nasa.gov)

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