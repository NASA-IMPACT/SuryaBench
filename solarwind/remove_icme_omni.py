"""
    Author: Vishal Upendran [uvishal1995@gmail.com]
    Script to remove ICME data from the OMNI dataset. 
    The ICME data is from Chris Moestl, given by Dinesh Hegde.
"""

import numpy as np
import pandas as pd 
from tqdm import tqdm
import sys

ICME_PATH = "csv_files/"
path_ICME = f"{ICME_PATH}/HELIO4CAST_ICMECAT_v23.csv"
icme = pd.read_csv(path_ICME)
#SELECT ICME observed by Wind.
icme_wind = icme[icme['sc_insitu'] == 'Wind']
# Subsample icme_wind between 2010-01-01 and 2024-12-31
start_date = pd.Timestamp('2010-01-01').tz_localize('UTC') 
end_date = pd.Timestamp('2024-12-31').tz_localize('UTC') 
icme_wind = icme_wind[
    (pd.to_datetime(icme_wind['icme_start_time']) >= start_date) &
    (pd.to_datetime(icme_wind['icme_start_time']) <= end_date)
]

#Load OMNI
save_dir = f"csv_files/"
omni_file = "omni_solar_wind.csv"
omni_solar_wind = pd.read_csv(f"{save_dir}/{omni_file}")

# Ensure the time columns are in datetime format
omni_solar_wind['Epoch'] = pd.to_datetime(omni_solar_wind['Epoch'])
icme_wind['icme_start_time'] = pd.to_datetime(icme_wind['icme_start_time']).dt.tz_localize(None)
icme_wind['mo_end_time'] = pd.to_datetime(icme_wind['mo_end_time']).dt.tz_localize(None)
print(f"Length of OMNI data: {omni_solar_wind.shape}")
# Iterate through each time window in icme_wind and remove rows from omni_solar_wind
for _, row in tqdm(icme_wind.iterrows(), total=len(icme_wind)):
    start_time = row['icme_start_time']
    end_time = row['mo_end_time']
    omni_solar_wind = omni_solar_wind[
        ~((omni_solar_wind['Epoch'] >= start_time) & (omni_solar_wind['Epoch'] <= end_time))
    ]
print(f"After ICME removal: {omni_solar_wind.shape}")

# Save the filtered dataset to a new CSV file
omni_solar_wind.to_csv(f'{save_dir}/omni_icme_removed_solar_wind.csv', index=False)