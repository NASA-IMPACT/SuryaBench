
For training below code can be used. 

1. **SpatioTemporalAttention Transformer**
```
python train_baselines.py --config_path ./ds_configs/config_spectformer_ar_sta.yaml --gpu 
```

2. **SpatioTemporalResNet**
```
python train_baselines.py --config_path ./ds_configs/config_ar_stresnet.yaml --gpu 
```

### Models

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