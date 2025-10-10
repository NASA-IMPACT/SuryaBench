# Functions for create_datasets.py

import matplotlib.pyplot as plt
from datetime import datetime
from astropy.io import fits
from PIL import Image
import numpy as np
import sys

# Function designed to read FITS image data and headers from a specified file, and return the data and associated headers as output
def get_data(file):
    with fits.open(file) as file_read:
        mag_grams = file_read[1].data # get the images
        headers = file_read[1].header # get the headers associated with the images
    return mag_grams, headers

# Function that calculates the FFT of a set of dopplergrams or continuous intensity mages and returns the resulting frequency-domain representation of the data in fft_maps.
def calculate_fft(series):
    D_series = np.zeros((np.shape(series)[0]-1,np.shape(series)[1],np.shape(series)[2])) #initialize
    for i in range(0,np.shape(series)[0]-1): D_series[i,:,:] = series[i+1,:,:]-series[i,:,:] #subtract previous dopplergram with next one
    dt = float(45) #sec (this is the sampling interval = 1/sampling rate)
    T = float(8*60*60) #sec
    # Take the FFT for all points in dopplergram
    fft_maps = np.zeros((320,512,512)) #initialize (half of frequency axis)
    for x in range(0,512): # Go through the x axis of the picture
        for y in range(0,512): fft_maps[:,x,y] = (dt**2/T)*np.square(np.absolute(np.fft.rfft(D_series[:,x,y]))) # Go through the y axis of the picture and calculate the FFT
    return fft_maps

# Function that takes the frequency-domain representation of a set of dopplergrams/cont_intensities and a frequency range of interest, and returns the power map for that frequency range, which represents the distribution of power in that frequency range across the solar surface.
def calculate_power_map(dop_fft,rng):
    T = float(8*60*60) #sec
    n = np.arange(np.shape(dop_fft)[0]) # number of points (640)
    freq = (n/T)*1000 # in mHz 
    # CONSTRUCT POWER MAP for 2-3 mHz
    freq_range = np.where(np.logical_and(freq>=rng[0], freq<=rng[1])) #find indexes in range
    power_map_range = np.squeeze(dop_fft[freq_range,:,:]) # seclude the range
    # Integration for desired interval
    power_map = np.zeros((512,512)) #initialize
    for x in range(0,512): # Go through the x axis of the picture
        for y in range(0,512): power_map[x,y] = np.trapz(power_map_range[:,x,y]) # Go through the y axis of the picture and integrate using trapezoidal rule   
    return power_map

# Function that saves in .fits files the frequency-domain representation of the doplergrams or continuous intensity images
def save_sasha_fits(dop_fft,dop_headers,int_fft,int_headers,time):

    # Create a FITS header with some metadata
    header = fits.Header() #take header from sasha READ FROM ORIGINAL FILES AND EDIT THE ONES THAT ARE DIFFERENT, KEEP THE REST SAME
    header['OBJECT'] = 'Example FITS file'
    header['AUTHOR'] = 'Your name'
    header['DATE'] = '2022-05-09'
    header['COMMENT'] = 'This is an example FITS file created using astropy'

    # Create a FITS HDU (Header/Data Unit) with the data and header
    dop_fft_file = fits.PrimaryHDU(dop_fft, header)
    int_fft_file = fits.PrimaryHDU(int_fft, header)

    # Write the FITS file to disk
    dop_fft_file.writeto('/nobackup/skasapis/sasha_fft/dopplergram_fft.fits', overwrite=True)
    int_fft_file.writeto('/nobackup/skasapis/sasha_fft/cont_int_fft.fits', overwrite=True)

    t = 1
