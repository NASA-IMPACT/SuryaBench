# Core SDO Machine Learning Ready Dataset Generation

## Description

This repository provides scripts to **download AIA and HMI data** from the **SDO spacecraft** and preprocess it to create a **homogenized, machine learning–ready Core-SDO dataset** used in the training of **Surya**.

The data is sourced from **JSOC**, and several preprocessing steps are applied to bring AIA and HMI data into Level 1.5, ready for machine learning applications.

- **Source Data:** JSOC Full Disk FITS files 
- **Cadence:** 12 minutes 
- **Channels:** 
  - **8 AIA wavelengths:** 94, 131, 171, 193, 211, 304, 335, 1600 Å 
  - **5 HMI variables:** LOS Magnetogram, LOS Dopplergram, Bx, By, Bz 
- **Purpose:** Training machine learning models on SDO observations.

---

##  Repository Structure

```bash
core_sdo/
│
├── core_sdo_download.py              # Script to download AIA/HMI FITS data via JSOC
├── core_sdo_mlready_processing_multi.py  # Preprocessing pipeline to create ML-ready NetCDF4 files
├── helio.py                          # Utility functions for solar disk scaling, HMI vector conversion, etc.
├── plot_nc.py                        # Quicklook plotter for processed NetCDF4 files
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation (this file)
```


## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
---

## Usage

### 1. Download a sample dataset from JSOC

Make sure to provide your email address in the script for JSOC access.
If it's your first time, JSOC will send a confirmation email — reply with "yes" to activate access.

```bash
cd core_sdo
python core_sdo_download.py
```

### 2. Process downloaded fits files
```bash
cd core_sdo
python core_sdo_mlready_processing_multi.py
```
### 3 Plot a processed maps in the NetCDF4 file
```bash
cd core_sdo
python plot_nc.py 'filename.nc'
```

## Contact
Dinesha Hegde, [dinesha.hegde@uah.edu]

