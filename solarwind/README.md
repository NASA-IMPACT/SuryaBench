# Download solar wind data

This folder contains scripts for downloading the solar wind data and splits them into train-val-test sets. For convenience, we have `download_solar_wind_data.sh` that creates a virtual environment, installs required packages, downloads data, splits into sets, and destroys the environment. The data would be saved in `../data/`

1. First step is to download the solar wind data. For this, use data_process/download_sw_data.py. This needs the exact keyword variable by selecting OMNI from: [https://cdaweb.gsfc.nasa.gov/index.html](https://cdaweb.gsfc.nasa.gov/index.html). Be sure to select the right variables to get the correct data.
2. If you want to just run the scripts, you must run `download_sw_data.py` for downloading the solar wind data.
3. `split_trainValTest.py` splits the dataset into train-val-test sets. These sets are defined in the paper. 