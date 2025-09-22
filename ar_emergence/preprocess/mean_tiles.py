# Full pipeline (power maps, magnetograms and intensity)
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys
from PIL import Image
from functions import split_image, get_piece_means, dtws
import time
import ast
#from matplotlib.dates import DayLocator

# Power Maps
ar_num = sys.argv[1]
print('AR{}'.format(ar_num))

for f in ['True','False']: 
    for d in ['True','False']: 
        flatten = ast.literal_eval(f) #False 
        derivative = ast.literal_eval(d) #False
        source = 'dop' #'int'   # # #    #CALCULATE CONT INTENSITY UNITS FOR POWER MAPS W/m^2/sr/Hz
        units = {'dop':r'PM Tiles Mean ($\frac{m^{2}}{s^{2} \cdot Hz}$)','int':r'PM Tiles Mean ($\frac{W^2 \cdot sr^2}{m^2 \cdot Hz^2}$)'}
        power_maps = np.load('/nobackup/skasapis/AR{}/power_maps/power_maps_{}{}.npz'.format(ar_num,source,ar_num),allow_pickle=True) #to see objects in load: lst2 = magnetograms.files lst = power_maps.files
        print(np.shape(power_maps))
        print(np.shape(power_maps['arr_0']))
        print(np.shape(power_maps['arr_1']))
        print(np.shape(power_maps['arr_2']))
        print(np.shape(power_maps['arr_3']))
        print(np.shape(power_maps['arr_4']))
        sys.exit()
        # Magnetograms
        magnetograms = np.load('/nobackup/skasapis/AR{}/power_maps/magnetograms{}.npz'.format(ar_num,ar_num),allow_pickle=True) #to see objects in load: lst2 = magnetograms.files
        # Intensity
        intensities = np.load('/nobackup/skasapis/AR{}/power_maps/intensities{}.npz'.format(ar_num,ar_num),allow_pickle=True)

        # Split the image into pieces and calculate their mean
        size = 9
        pm23_means = get_piece_means(power_maps['arr_0'],size)
        pm34_means = get_piece_means(power_maps['arr_1'],size)
        pm45_means = get_piece_means(power_maps['arr_2'],size)
        pm56_means = get_piece_means(power_maps['arr_3'],size)
        mag_means = get_piece_means(magnetograms['arr_0'],size)
        int_means = get_piece_means(intensities['arr_0'],size)

        if flatten:
            pm23_means = pm23_means - dtws(size,pm23_means)
            pm34_means = pm34_means - dtws(size,pm34_means)
            pm45_means = pm45_means - dtws(size,pm45_means)
            pm56_means = pm56_means - dtws(size,pm56_means)
            mag_means = mag_means - dtws(size,mag_means)
            int_means = int_means - dtws(size,int_means)

        if derivative:
            pm23_means = np.gradient(pm23_means,axis=1)
            pm34_means = np.gradient(pm34_means,axis=1)
            pm45_means = np.gradient(pm45_means,axis=1)
            pm56_means = np.gradient(pm56_means,axis=1)
            mag_means = np.gradient(mag_means,axis=1)
            int_means = np.gradient(int_means,axis=1)

        # Convert string to datetime object
        pm_time = []
        for time_obj in power_maps['arr_4']: pm_time.append(datetime.strptime(str(time_obj), '%Y-%m-%d %H:%M:%S.%f'))
        # Convert string to datetime object
        mag_time = []
        for time_obj in magnetograms['arr_1']: mag_time.append(datetime.strptime(str(time_obj), '%Y-%m-%d %H:%M:%S.%f'))
        # Convert string to datetime object
        int_time = []
        for time_obj in intensities['arr_1']: int_time.append(datetime.strptime(str(time_obj), '%Y-%m-%d %H:%M:%S.%f'))
        
        if flatten == False and derivative == False:
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_pm{}{}.npz'.format(ar_num,source,ar_num),pm23_means,pm34_means,pm45_means,pm56_means,pm_time)
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_int{}.npz'.format(ar_num,ar_num),int_means,int_time)
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_mag{}.npz'.format(ar_num,ar_num),mag_means,mag_time)
            print('Saved Normal')
        elif flatten == True and derivative == False:
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_pm{}{}_flat.npz'.format(ar_num,source,ar_num),pm23_means,pm34_means,pm45_means,pm56_means,pm_time)
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_int{}_flat.npz'.format(ar_num,ar_num),int_means,int_time)
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_mag{}_flat.npz'.format(ar_num,ar_num),mag_means,mag_time)
            print('Saved Flat')
        elif flatten == True and derivative == True:
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_pm{}{}_dflat.npz'.format(ar_num,source,ar_num),pm23_means,pm34_means,pm45_means,pm56_means,pm_time)
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_int{}_dflat.npz'.format(ar_num,ar_num),int_means,int_time)
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_mag{}_dflat.npz'.format(ar_num,ar_num),mag_means,mag_time)
            print('Saved D_Flat')
        else:
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_pm{}{}_d.npz'.format(ar_num,source,ar_num),pm23_means,pm34_means,pm45_means,pm56_means,pm_time)
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_int{}_d.npz'.format(ar_num,ar_num),int_means,int_time)
            np.savez('/nobackup/skasapis/AR{}/mean_tiles9/mean_mag{}_d.npz'.format(ar_num,ar_num),mag_means,mag_time)
            print('Saved D_Normal')