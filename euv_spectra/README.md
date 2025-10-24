# EUV Spectra Modeling Dataset Preparation

## Description

This repository provides utilities for preparing and loading **Extreme Ultraviolet (EUV) irradiance spectra** from NASA’s **EVE instrument** (onboard SDO). 
It includes scripts for ML-ready dataset preparation for SuryaBench, timestamp synchronization, and PyTorch dataloaders.

Temporal coverage of the dataset is from 2010 to 2014 (during EVE MEGS-A was operational) with a 12-minute cadence. Spectral Range is 6.5–33.3 nm (1343 wavelength bins, with 0.02nm spectral resolution). This dataset is available on HuggingFace: NASA-IMPACT SuryaBench EUV Spectra [https://huggingface.co/datasets/nasa-ibm-ai4science/euv-spectra]

## Repository Structure
```bash
EVE-AIA-Dataset/
├── prepare_data.py # Downloading, preprocessing and dataset split script
├── eve_dataloader.py # PyTorch Dataset (EVEDSDataset) extending HelioNetCDFDataset
├── euv_wavelengths.csv # EUV wavelength grid (1343 bins)
├── config_spectformer_dgx_test.yaml # Config file with paths and parameters
└── README.md # Project documentation
```

## Features

- Performs temporal-alignment of EUV spectra & AIA image cubes (compatible with SuryaBench).  
- Uses predefined splits (for trainin, validation, testing, and buffer (leaky_validation))
- Performs normalization & preprocessing**:  
  - Zero-value replacement with wavelength-wise minimum.  
  - Log-scaling (`log10`) for dynamic range compression.  
  - Global normalization between -9.00 and -1.96 (log10 space).  
- Creates a dataloader: adjustable matching window (`ds_time_tolerance`), split selection (`phase`), and downstream configs.  


## Usage

### 1. Install dependencies
```bash
pip install numpy pandas xarray torch netCDF4
```
<!-- 2. Prepare dataset

-->

## Requirements

    Python 3.8+

    Required packages:
        numpy
        pandas
        xarray
        torch
        netCDF4


## Contact

Shah Bahauddin [Shah.Bahauddin@lasp.colorado.edu]

## References
Woods, T. N., Eparvier, F. G., Hock, R., Jones, A. R., Woodraska, D., Judge, D., ... & Viereck, R. (2012). Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of science objectives, instrument design, data products, and model developments. The solar dynamics observatory, 115–143. doi:10.1007/s11207-009-9487-6
