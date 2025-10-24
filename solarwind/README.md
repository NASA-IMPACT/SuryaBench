# Solar Wind Forecasting Dataset Generation


## Description
This repository provides scripts to download data and create sampling splits for generating the **Solar Wind Forecasting Dataset**.  
The dataset is derived from **NASA OMNI solar wind and IMF data** and curated by removing Interplanetary Coronal Mass Ejections (ICMEs) to retain only background solar wind.  

- **Source Data:** NASA OMNI dataset ([CDAWeb](https://cdaweb.gsfc.nasa.gov/index.html))  
- **Coverage:** 2010-05-13 to 2024-12-31  
- **Cadence:** Hourly measurements at L1  
- **Variables:** Solar wind speed (V), IMF components (Bx, By, Bz), proton density (N)  
- **Purpose:** Training and benchmarking solar wind forecasting models  

---

## Repository Structure
```bash
solarwind/
│
├── data_process/
│ ├── download_sw_data.py # Download OMNI solar wind data
│ ├── remove_icme_omni.py # Remove ICME intervals from data
│ ├── split_omni_icme.py # Split data into train/val/test sets
│ ├── plot_omni_csvs.py # Plot dataset for inspection
│ └── download_solar_wind_data.sh # Automated setup + data processing script
│
├── csv_files/ # Processed dataset (CSV format, train/val/test splits)
│
├── plots/ # Visualizations of data (time-series plots)
│
├── models/ # Example model implementations (baseline forecasting)
│
└── README.md # Project documentation
```


---

## Features

- Hourly-resolution solar wind and IMF dataset with ICME events removed  
- Predefined splits: **train, validation, leaky validation, test**  
- Variables included:  
  - Solar wind speed (V, km/s)  
  - IMF components: Bx (GSE), By (GSM), Bz (GSM) (nT)  
  - Proton number density (N, cm⁻³)  
- Dataset size: **119,478 samples** across all splits  
- CSV format, compatible with pandas, NumPy, and ML frameworks  

## Usage

### 1. Download dataset from HuggingFace
```bash
huggingface-cli download nasa-impact/Surya-bench-solarwind --repo-type dataset --local-dir csv_files
```

### 1.a Process data locally (alternative)
```bash
cd data_process
bash download_solar_wind_data.sh
```
### 2 Load dataset in Python
```python
import pandas as pd

# Load training set
df_train = pd.read_csv("csv_files/train.csv")
print(df_train.head())

# Example: plot solar wind speed
import matplotlib.pyplot as plt
plt.plot(pd.to_datetime(df_train["Epoch"]), df_train["V"])
plt.title("Solar Wind Speed (Train Set)")
plt.xlabel("Time")
plt.ylabel("Speed (km/s)")
plt.show()
```

## Requirements
- pandas
- numpy
- matplotlib
- requests

Install dependencies:
```bash
pip install pandas numpy matplotlib requests
```

## Contact
Vishal Upendran, [vupendran@seti.org]

