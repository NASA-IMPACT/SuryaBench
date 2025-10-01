#  Full-disk Solar Flare Forecasting Dataset Generator

## Description 

This Python script generates flaring activity labels using a 24h rolling prediction window. The output includes binary labels based on both the **maximum flare class** and **cumulative flare intensity** within a given time window.

## Project Structure
```bash
flare/
├── `flare_catalog_2010-2024.csv` – Input CSV file containing flare data  
├── `flare_index_creator.py` – Main script for generating flare labels  
└── `flare.py` – (Dataloader) Loads 4096×4096 input raster images and corresponding flare labels
```
*If you're interested in how the `flare_catalog_2010-2024.csv` file was created, please visit the repository [here](https://bitbucket.org/gsudmlab/flare_list_creator/src/main/).*


## Features

- Converts GOES flare classes (C/M/X) into numeric sub-class values
- Applies a rolling window to segment flare data by timestamp
- Generates binary labels based on:
  - Maximum flare class in each window
  - Cumulative flare sub-class values in each window
- Creates temporally non-overlapping, non-chronological splits for training, validation, and testing
- Optionally filters with an external index file based on the availability of the input files in SuryaBench

## Converting Flare Classes to Sub-Class Values

| Flare Class | Multiplier | Example |
|-------------|------------|---------|
| C           | ×1         | C3.2 → 3.2 |
| M           | ×10        | M1.0 → 10.0 |
| X           | ×100       | X2.5 → 250.0 |

## Requirements

- Python 3.7+
- pandas
- numpy

Install dependencies:
```bash
pip install pandas numpy
```

### Download and Train with Surya AR Flare Dataset
- Download the processed data from huggingface: 
```
hf download nasa-ibm-ai4science/surya-bench-flare-forecasting --repo-type dataset --local-dir surya_flare_forecasting
```
- A link to train model using Surya/Unet https://github.com/NASA-IMPACT/Surya/tree/main/downstream_examples/solar_flare_forcasting

## Contact
Jinsu Hong [jhong36@gsu.edu]
Kang Yang [kyang30@student.gsu.edu]
Berkay Aydin [baydin2@gsu.edu]