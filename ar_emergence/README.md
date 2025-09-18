# Active Region Emergence Dataset 

This repository contains the preprocessing code and pipeline used to generate the **Solar Active Region Emergence Dataset (SOLARED)**. The dataset is designed for machine learning applications in heliophysics, enabling the prediction of continuum intensity variations from spatiotemporal solar observations.  

The dataset is publicly available on [NASA-IMPACT HuggingFace Repository](https://huggingface.co/datasets/nasa-impact/ar_emergence).  

---

## i. General Dataset Description

The SOLARED dataset consists of time series derived from **57 emerging solar active regions (ARs)** observed by the **Helioseismic and Magnetic Imager (HMI)** onboard the **Solar Dynamics Observatory (SDO)**.  

- Each AR was selected based on:
  - Emergence within ±30° longitude from central meridian.  
  - Persistence on the visible disk for at least **4 days**.  
  - Reaching a minimum area of **200 millionths of a solar hemisphere**.  
- The dataset covers NOAA ARs from **AR11130 (2010)** through **AR13213 (2024)**.  

This dataset provides **6-channel spatiotemporal sequences**:
1. Four acoustic power channels (2–3, 3–4, 4–5, 5–6 mHz).  
2. Unsigned magnetic flux.  
3. Continuum intensity.  

Each AR is represented by **240 time steps** (12-minute cadence, ≈2 days). The dataset enables research on the physics of AR emergence and provides a benchmark for predictive modeling in space weather.

---

## ii. Repository Structure

ar_emergence/
│
├── preprocessing/ # Dataset preprocessing pipeline
│ ├── create_datasets.py # Create train/valid/test splits
│ ├── functions.py # Utility functions for preprocessing
│ ├── mean_tiles.py # Generate mean tile time series
│ ├── preprocess_functions.py # Core preprocessing functions
│ └── README.md
│
├── ar_models/ # Model architectures (e.g., SpatioTemporalAttention, ResNet)
├── ds_configs/ # YAML configuration files for experiments
├── ds_datasets/ # Dataset classes and dataloaders
├── shell_scripts/ # Helper bash scripts for running jobs
│
├── compute_stats.py # Script to compute dataset statistics
├── get_baselines.py # Utility for retrieving baseline results
├── inference.py # Run inference using trained models
├── train_baselines.py # Train baseline models on AR Emergence Dataset
│
├── init.py
└── README.md # Main documentation (this file)


The raw input files (SDO/HMI Dopplergrams, magnetograms, continuum intensity) are obtained from [JSOC](http://jsoc.stanford.edu/). The outputs are compressed `.npz` files containing per-tile timelines for each AR.  

---

## iii. Mathematical Description of Dataset

Each AR is tracked in a **512×512 pixel patch** centered on the AR location. The preprocessing follows a **five-step pipeline**:  

1. **Tracking:** Extract 512×512 pixel cutouts of HMI Doppler velocity, magnetic flux, and continuum intensity.  
2. **Acoustic Power Maps:** Compute Dopplergram differences to remove solar rotation:  

   \[
   \Delta V_{\text{dop}}[i,x,y] = V_{\text{dop}}[i+1,x,y] - V_{\text{dop}}[i,x,y],
   \]

   followed by Fourier transform to compute power spectra:  

   \[
   V_{\text{dop}}^{\text{FFT}}[k,x,y] = \left(\frac{dt^2}{T}\right) \left| \mathcal{F} \{ \Delta V_{\text{dop}}[:,x,y] \}[k] \right|^2
   \]

   where \(dt = 45\) sec, \(T = 28800\) sec, \(k=1,\dots,320\).  
   
   Frequency bands: 2–3, 3–4, 4–5, 5–6 mHz.  

3. **Downsampling:** Divide each 512×512 patch into a **9×9 grid** of tiles. Remove top and bottom rows → **63 tiles** remain.  
4. **Geometric Correction:** Remove foreshortening and projection effects from solar sphere curvature.  
5. **Timeline Creation:** Compute mean power, unsigned magnetic flux, and continuum intensity per tile across all timesteps.  

**Final dataset structure per AR:**
- Input tensor: `(120, 5, 63)` (120 timesteps, 5 input channels, 63 tiles).  
- Output tensor: `(63,)` (predicted continuum intensity per tile).  

Dynamic ranges:  
- Acoustic power: \([-7.5 \times 10^7, \, 5.8 \times 10^7]\)  
- Magnetic flux: \([-1.4 \times 10^2, \, 5.3 \times 10^2]\)  
- Continuum intensity: \([-1.7 \times 10^4, \, 4.0 \times 10^3]\)  

---

## iv. Exploratory Data Analysis (EDA)

- **Coverage:** 56 ARs from 2010–2024.  
- **Latitude/Longitude:** ARs span a broad latitude range; longitudes constrained to ±30° from disk center.  
- **Persistence:** All ARs lived ≥4 days, enabling well-sampled emergence profiles.  
- **Size distribution:** Minimum \(200 \, \mu\)Hem area; some exceeded \(1000 \, \mu\)Hem.  
- **Class balance:** ARs span Hale classes (α, β, βγ, βγδ).  

Visualizations included in the dataset paper show:  
- AR latitude vs emergence time.  
- Distribution of AR lifetimes.  
- Boxplots of location, size, and visibility times.  

---

## v. Requirements & Reproducibility

To reproduce the dataset:  

**Dependencies**
- Python ≥ 3.9  
- NumPy, SciPy  
- Astropy  
- SunPy  
- h5py  
- scikit-learn  
- tqdm  

**Example usage**
```bash
# Step 1: Download raw data from JSOC
# Step 2: Run preprocessing
python mean_tiles.py --ar_id 11130
# Step 3: Create dataset splits
python create_datasets.py

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


## vi. Contact

Spiridon Kasapis, PhD (skasapis@princeton.edu)









