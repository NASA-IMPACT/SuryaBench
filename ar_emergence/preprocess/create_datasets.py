# Data Preprocessing for Solar Power Maps Creation
# Spyros Kasapis - NASA Ames Research Center (Advanced Supercomputing Division)
from preprocess_functions import get_data, calculate_fft, calculate_power_map, save_sasha_fits
#from matplotlib.animation import FuncAnimation
#from matplotlib.animation import FFMpegWriter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import glob
import sys
import os
import time

# Record the start time
start_time = time.time()

ar_num = sys.argv[1] # Select the desired active region
print(); print('AR #: ',ar_num)
timelines = True # Do you want the data correspongind timelines produced?
power_maps = True # Do you want the powermaps produced?
sasha = False # Do you want to save the .fits files for Sasha?
freq_ranges = [[2,3],[3,4],[4,5],[5,6]] # Select Power Map Frequency Ranges

# Dopplergrams
dopplergram_files = glob.glob(os.path.join("/nobackup/skasapis/AR{}/dopplergrams/".format(ar_num), "*.fits")) # get a list of all files with a .fits extension in the directory
dopplergram_files = sorted(dopplergram_files, key=os.path.getmtime, reverse=True) # sort the files by their modified time (latest file first)
dopplergram_files.reverse()

# Continuous Intensity
intensity_files = glob.glob(os.path.join("/nobackup/skasapis/AR{}/cont_intensity/".format(ar_num), "*.fits")) # get a list of all files with a .fits extension in the directory
intensity_files = sorted(intensity_files, key=os.path.getmtime, reverse=True) # sort the files by their modified time (latest file first)
intensity_files.reverse()

# Magnetograms
magnetogram_files = glob.glob(os.path.join("/nobackup/skasapis/AR{}/magnetograms/".format(ar_num), "*.fits")) # get a list of all files with a .fits extension in the directory
magnetogram_files = sorted(magnetogram_files, key=os.path.getmtime, reverse=True) # sort the files by their modified time (latest file first)
magnetogram_files.reverse()

frames_num = int(len(dopplergram_files)); 
if (len(dopplergram_files) != len(magnetogram_files)) and (len(dopplergram_files) != len(intensity_files)): print('Data Length Missmatch')
#print(len(dopplergram_files),len(magnetogram_files),len(intensity_files))

# Create Pure Timelines and Power Maps
all_dopplergrams = np.zeros((frames_num,512,512)) # Initialize
all_intensities = np.zeros((frames_num,512,512))
all_magnetograms = np.zeros((frames_num,512,512))
all_pm_dop = np.zeros((len(freq_ranges),frames_num,512,512)) # Initialize
all_pm_int = np.zeros((len(freq_ranges),frames_num,512,512))
all_dates_dop = [] # Initialize
all_dates_mag = [] # Initialize
all_dates_int = [] # Initialize

for file_num in range(0,frames_num): # Go through every .fits file

    # Get the data from every file
    dop_grams, dop_headers = get_data(dopplergram_files[file_num])
    mag_grams, mag_headers = get_data(magnetogram_files[file_num])
    int_grams, int_headers = get_data(intensity_files[file_num])
    print(np.shape(dop_grams))#,mag_headers,int_headers)
    sys.exit()
    
    # Check if all times are correct and then add to dates
    if dop_headers['MIDTIME'] == int_headers['MIDTIME'] and int_headers['MIDTIME'] == mag_headers['MIDTIME']: 
        print('File Num {}: All Times Align'.format(file_num))
    else: 
        print('File Num {}: There is a time missmatch problem'.format(file_num))
        print(dop_headers['MIDTIME'], mag_headers['MIDTIME'], int_headers['MIDTIME'])

    # Create time series: Parse the string to a datetime object
    all_dates_dop.append(datetime.strptime(dop_headers['MIDTIME'], '%Y.%m.%d_%H:%M:%S.%f_TAI'))
    all_dates_mag.append(datetime.strptime(mag_headers['MIDTIME'], '%Y.%m.%d_%H:%M:%S.%f_TAI'))
    all_dates_int.append(datetime.strptime(int_headers['MIDTIME'], '%Y.%m.%d_%H:%M:%S.%f_TAI'))

    if timelines: # Create Datasets
        all_dopplergrams[file_num,:,:] = dop_grams[int(frames_num/2),:,:]
        all_intensities[file_num,:,:] = int_grams[int(frames_num/2),:,:]
        all_magnetograms[file_num,:,:] = np.absolute(mag_grams[int(frames_num/2),:,:])
    if power_maps: # Create Power Maps
        dop_fft = calculate_fft(dop_grams) # Calculate the FFT of the dopplergrams series
        int_fft = calculate_fft(int_grams) # Calculate the FFT of the intensity series
        #if sasha: save_sasha_fits(dop_fft,dop_headers,int_fft,int_headers,all_dates)
        for rng_num in range(0,len(freq_ranges)):
            all_pm_dop[rng_num,file_num,:,:] = calculate_power_map(dop_fft,freq_ranges[rng_num]) # Calculate the power maps for the dopplergrams
            all_pm_int[rng_num,file_num,:,:] = calculate_power_map(int_fft,freq_ranges[rng_num]) # Calculate the power maps for the continuous intensities

# Save the data in .npz files
output_dir = '/nobackup/skasapis/AR{}'.format(ar_num)
if not os.path.exists(output_dir): os.makedirs(output_dir) # if the AR directory doesnt exist, create it

# Put in order (Dopplergrams)
all_dates_dop = np.array(all_dates_dop) # Convert your pm_time list to a numpy array
sorted_indices = np.argsort(all_dates_dop) # Get the indices that would sort the pm_time_np array
all_dates_dop_sorted = all_dates_dop[sorted_indices]; all_dates_dop = all_dates_dop_sorted # Use these indices to sort both pm_time_np and pm56_means
all_dopplergrams_sorted = all_dopplergrams[sorted_indices, :, :]; all_dopplergrams = all_dopplergrams_sorted
all_pm_dop_sorted = all_pm_dop[:,sorted_indices,:,:]; all_pm_dop = all_pm_dop_sorted

# Put in order (Cont Int)
all_dates_int = np.array(all_dates_int) # Convert your pm_time list to a numpy array
sorted_indices = np.argsort(all_dates_int) # Get the indices that would sort the pm_time_np array
all_dates_int_sorted = all_dates_int[sorted_indices]; all_dates_int = all_dates_int_sorted # Use these indices to sort both pm_time_np and pm56_means
all_intensities_sorted = all_intensities[sorted_indices, :, :]; all_intensities = all_intensities_sorted
all_pm_int_sorted = all_pm_int[:,sorted_indices,:,:]; all_pm_int = all_pm_int_sorted

# Put in order (Magnetograms)
all_dates_mag = np.array(all_dates_mag) # Convert your pm_time list to a numpy array
sorted_indices = np.argsort(all_dates_mag) # Get the indices that would sort the pm_time_np array
all_dates_mag_sorted = all_dates_mag[sorted_indices]; all_dates_mag = all_dates_mag_sorted # Use these indices to sort both pm_time_np and pm56_means
all_magnetograms_sorted = all_magnetograms[sorted_indices, :, :]; all_magnetograms = all_magnetograms_sorted

if timelines: # Save Pure Timelines 
    np.savez(output_dir+'/dopplergrams{}.npz'.format(ar_num,ar_num),all_dopplergrams,all_dates_dop)
    np.savez(output_dir+'/intensities{}.npz'.format(ar_num,ar_num),all_intensities,all_dates_int)
    np.savez(output_dir+'/magnetograms{}.npz'.format(ar_num,ar_num),all_magnetograms,all_dates_mag)
    print('Pure Timelines Saved')
if power_maps: # Save Power Maps
    power_maps_dop = []; power_maps_int = [] # Initialize
    for rng_num in range(0,len(freq_ranges)): # Create list of saved items
        power_maps_dop.append(all_pm_dop[rng_num,:,:,:]) 
        power_maps_int.append(all_pm_int[rng_num,:,:,:])
    np.savez(output_dir+'/power_maps_dop{}.npz'.format(ar_num), *power_maps_dop, all_dates_dop)
    np.savez(output_dir+'/power_maps_int{}.npz'.format(ar_num), *power_maps_int, all_dates_int)
    print('Power Maps Saved')

# Record the end time
end_time = time.time()
# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

        


