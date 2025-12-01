"""
Various utility functions for processing SDO/AIA and SDO/HMI data.
Includes functions for scaling solar disk, computing timestamps,
processing AIA and HMI maps, disambiguating
HMI azimuth angles, converting HMI vector magnetogram maps to components,
and saving the resulting maps.
Authors: Dinesha Hegde, dinesha.hegde@gmail.com"""

# Import necessary libraries
import re
import os.path
import numpy as np
from glob import glob
from pprint import pprint
import datetime as dt
from sunpy.map import Map
import sunpy
import aiapy.calibrate as ac
from aiapy.util.exceptions import AiapyUserWarning
from aiapy.calibrate import normalize_exposure, register, update_pointing, correct_degradation
from skimage.transform import SimilarityTransform, warp
from sunpy.util.exceptions import SunpyUserWarning,SunpyMetadataWarning
import astropy.units as u
from astropy.io.fits.verify import VerifyWarning
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from astropy.io.fits import CompImageHDU

import warnings
warnings.filterwarnings('ignore', message=".*dubious year.*")
warnings.simplefilter('ignore', category=SunpyMetadataWarning)
warnings.simplefilter('ignore', category=VerifyWarning)
warnings.simplefilter('ignore', category=AiapyUserWarning)
warnings.simplefilter('ignore', category=VerifyWarning)


#Setting constants
saturation = 16383 # 14 bit int max used in the SDO/CCD camera
target_ang_size = 976.0
width = 4096
height = 4096
image_shape = (width, height)

 
def scale_solardisk(m, map_type, target_solar_radius=976.0):
    """
    Scales the solar disk radius to the target angular size in a SunPy map.

    Parameters:
    - m: A SunPy map object.
    - map_type: The type of the map, either 'aia' or 'hmi'.
    - target_solar_radius: The target radius of the solar disk in arcseconds.

    Returns:
    - A new SunPy map object with the scaled solar disk.
    """
    
    # Data
    mdata = m.data

    # Create a valid mask; this will be used to correct for downscaling when interpolating
    if map_type == "aia":
        valid_mask = (mdata > 0).astype(float)
        mdata[mdata <= 0.0] = 0.0
    elif map_type == "hmi":
        valid_mask = (mdata < 1.E6).astype(float)
    else:
        raise ValueError('The type of the map is not recognized.')

    # Calculate the scale factor
    rad = m.meta['RSUN_OBS']
    scale_factor = target_solar_radius / rad

    # Calculate the translation needed to keep the center
    shape_center = mdata.shape[0] / 2.0
    translation = (shape_center - scale_factor * shape_center,) * 2

    # Create the similarity transformation
    transform = SimilarityTransform(scale=scale_factor, translation=translation)

    # Apply the transformation to both the data and the valid mask
    scaled_mdata = warp(mdata, transform.inverse, preserve_range=True, mode='edge', output_shape=mdata.shape, order=0)
    scaled_validmask = warp(valid_mask, transform.inverse, preserve_range=True, mode='edge', output_shape=mdata.shape, order=0)

    # Correct for interpolating over valid pixels
    scaled_mdata /= (scaled_validmask + 1e-8)

    # Update the map metadata
    new_meta = m.meta.copy()
    new_meta['RSUN_FIX'] = target_solar_radius

    # Comment for the new RSUN_FIX keyword
    new_meta['RFIX_COM'] = '[arcsec] Target solar radius achieved by scaling the solar disk.'

    # Create a new map instance with the scaled data and updated metadata
    map_scaled = m._new_instance(scaled_mdata, new_meta)

    return map_scaled


def compute_timestamp(fname):
    """
    Compute the timestamp string from the SDO FITS filename.
    Parameters:
    - fname: The FITS filename as a string.
    Returns:
    - A timestamp string in the format 'yyyymmdd_HHMM'.
    Raises:
    - ValueError: If the filename does not match known patterns.
    3 patterns are supported:
    1) AIA-style names:
       e.g., aia.lev1_euv_12s.2012-01-30T215937Z.171.image_lev1.fits
    2) HMI-style names:
         e.g., hmi.b_720s.20120130_220000_TAI.azimuth.fits
    3) Generic fallback: any filename containing 'yyyymmdd_HHMM' pattern.
    Comments:
    - For AIA-style names, the timestamp is rounded to the nearest 12-minute interval.
    - For HMI-style names, the timestamp is taken directly from the filename.
    """
    # 1) try AIA-style names:
    #    aia.lev1_euv_12s.2012-01-30T215937Z.171.image_lev1.fits
    pattern_aia = re.compile(r'(s\.)(.*)Z\.(\d+)(\.image)')
    m = pattern_aia.search(fname)
    if m:
        t = m.group(2)  # '2012-01-30T215937'
        format_string = "%Y-%m-%dT%H%M%S"
        t_obs = dt.datetime.strptime(t, format_string)
        # round up to the nearest 12 min (your original code)
        nearest_12min = round(t_obs.minute / 12) * 12
        deltaT = nearest_12min * 60 - (60 * t_obs.minute + t_obs.second)
        t = t_obs + dt.timedelta(seconds=deltaT)
        t_key = f"{t.year}{t.month:02d}{t.day:02d}_{t.hour:02d}{t.minute:02d}"
        return t_key

    # 2) else try HMI-style names:
    #    hmi.b_720s.20120130_220000_TAI.azimuth.fits
    pattern_hmi = re.compile(r'\.(\d{8})_(\d{6})_TAI')
    m = pattern_hmi.search(fname)
    if m:
        ymd = m.group(1)     # 20120130
        hms = m.group(2)     # 220000
        # HMI is already at 12-min cadence in the filename -> just drop seconds
        return f"{ymd}_{hms[:4]}"

    # 3) optional generic fallback: grab 20120130_2200 if it's in the name
    m = re.search(r'(\d{8}_\d{4})', fname)
    if m:
        return m.group(1)

    # if nothing matched, raise so you know there's a new pattern
    raise ValueError(f"compute_timestamp: cannot parse timestamp from '{fname}'")


def process_aia_map(map_lev1, pointing_tbl=None):

    """
    Process a full-disk level 1 AIA map to level 1.5. This includes updating the pointing,
    registering the map. Then to make it mlready, level 1.5 map is normalized for exposure time,
    corrected for degradation, and scaled to a fixed solar disk angular size.
    Parameters:
    - map_lev1: A SunPy map object for the level 1 AIA data.
    - pointing_tbl: An optional pointing table for updating the pointing.
    Returns:
    - A SunPy map object for the processed level 1.5 AIA data.
    """

    #1: update poining
    m_updated_pointing = update_pointing(map_lev1, pointing_table=pointing_tbl)
    
    #2: Register the AIA map
    map_lev15 = register(m_updated_pointing, missing=None) 

    #3: make sure the map not shrink after register, pad the map if necessary
    if (map_lev15.data).shape != image_shape:
        # make a copy of the map
        temp_map = map_lev15

        #xdata = np.pad(temp_map.data, pad_width=1, mode='constant', constant_values=0)
        xdata = np.pad(temp_map.data, ((1,1),(1,1)), mode='constant')

        # update crpix1 and crpix2, learn more from below
        # https://aiapy.readthedocs.io/en/stable/api/aiapy.calibrate.register.html
        temp_map.meta['naxis1'], temp_map.meta['naxis2'] = image_shape
        temp_map.meta['crpix1'] += 1
        temp_map.meta['crpix2'] += 1
        padded_map = temp_map._new_instance(xdata, temp_map.meta)

        map_lev15 = padded_map
        #print_stats(map_lev15)

    # normalize exposure time so all frame data value is per one second exposure time
    aia_map = normalize_exposure(map_lev15)

    aia_corrected = correct_degradation(aia_map)

    # scale the solar disk radius to the target angular size
    aia_map_scaled = scale_solardisk(aia_corrected, map_type='aia', target_solar_radius=target_ang_size) 

    # make sure data are within the possbile range of the instrument (camera)
    aia_map_scaled.data[aia_map_scaled.data > saturation] = saturation
    aia_map_scaled.data[aia_map_scaled.data<0] = 0

    # returns level 1.5 map data
    return aia_map_scaled


def process_hmi_map(map_lev15, aia_wcs):
    """
    Process a full-disk level 1.5 HMI map to align with the AIA map WCS and scale the solar disk for fixed angular size.
    Parameters:
    - map_lev15: A SunPy map object for the level 1.5 HMI data.
    - aia_wcs: The WCS of the corresponding AIA map for alignment.
    """

    # align hmi with the aia map
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SunpyUserWarning)

    #mp = map_lev15.reproject_to(aia_wcs, algorithm="exact") #for more accurate reprojection but slower
    mp = map_lev15.reproject_to(aia_wcs) #bilinear interpolation by default

    #Add the RSUN_OBS keyword to the aligned HMI map metadata
    mp.meta['RSUN_OBS'] = map_lev15.meta['RSUN_OBS']

    # scaling the disk
    hmi_map_scaled = scale_solardisk(mp, map_type='hmi', target_solar_radius=target_ang_size)

    return hmi_map_scaled


def hmi_disambig(azimuth_map, disambig_map, method):
    """
    This is the direct conversion of the IDL code to Python for the disambiguation of the azimuth angle.
    https://git.smce.nasa.gov/cicci/hmi/-/blob/main/lib/hmi_disambig.pro

    & https://arxiv.org/abs/1605.03500
    
    Parameters:
    - azimuth_map: A SunPy map object of the azimuth angle.
    - disambig_map: A SunPy map object of the disambiguation information.
    - method: An integer to specify the disambiguation method.
    azimuth:    azimuth image, with values between 0 and 180.
    disambig:   bit mask with the same size as azimuth
                bit 1 indicates the azimuth needs to be flipped (plus 180)
                For full disk, three bits are set, giving three disambiguation
                solutions from different methods. Note the strong field pixels
                have the same solution (000 or 111, that is, 0 or 7) from the 
                annealing method. The weak field regions differ 
                lowest: potential field acute
                middle: radial acute (default)
                highest: random

    """

    # Extract data from SunPy maps
    azimuth = azimuth_map.data.copy()  # Make a copy of the azimuth data
    disambig = disambig_map.data

    # Check dimensions only
    # No further WCS check implemented
    nx, ny = azimuth.shape
    nx1, ny1 = disambig.shape
    if nx != nx1 or ny != ny1:
        print('Dimensions of two images do not agree')
        return azimuth_map

    # Convert disambig to integer type
    disambig = disambig.astype(int)

    # Check method
    if method < 0 or method > 2:
        method = 2
        print('Invalid disambiguation method, set to default method = 2')

    # Perform disambiguation using integer division
    disambig = disambig // (2 ** method)

    # Create an odd value index mask
    odd_value_index_mask = (disambig % 2 != 0)

    # Directly add 180 degrees to the azimuth where the odd value index mask is True
    azimuth[odd_value_index_mask] += 180

    # Create a new SunPy map with the modified data
    disambiguated_azimuth_map = azimuth_map._new_instance(azimuth, azimuth_map.meta)

    return disambiguated_azimuth_map

# Function to convert the HMI vector magnetogram maps to Bx, By, Bz maps
def vector2components(field_map, inclination_map, azimuth_disambiguated_map):
    """
    Convert the HMI vector magnetogram maps to Bx, By, Bz maps.

    Parameters:
    - field_map: A SunPy map object for the HMI vector field data.
    - inclination_map: A SunPy map object for the HMI vector inclination data.
    - azimuth_disambiguated_map: A SunPy map object for the HMI vector azimuth data after disambiguation.
    
    -Assumption: Accordinf to the HMI vector magnetogram data description from https://arxiv.org/abs/1309.2392v3
    -The three base vectors here are (ê ξ , ê η , ê ζ ), with +ξ referring to +x on the CCD array, +η as +y, and +ζ as +z out of
    the image plane. Note that +ζ coincides with the line-of-sight (LOS) direction. The field vectors are expressed as
    field strength (B), inclination (γ) and azimuth (ψ). The inclination γ is defined with respect to +ζ, with 0 ◦ pointing
    out of the image, 180 ◦ into the image. The azimuth ψ is defined with respect to +η, with 0 ◦ at +η and increase counterclockwise. 
    https://link.springer.com/article/10.1007/s11207-014-0516-8 (section:6.1.2)

    Returns:
    - bx_map: A SunPy map object for the HMI Bx component data.
    - by_map: A SunPy map object for the HMI By component data.
    - bz_map: A SunPy map object for the HMI Bz component data.
    - Conversion: Following https://doi.org/10.3847/1538-4365/ab1005
    - The three components of the magnetic field vector are given by Bx = B sin γ sin ψ, By = −B sin γ cos ψ, and Bz = B cos γ.
     Where, The + xdirection points to the solar west,+y to the north, and +z out of the image plane(i.e., line-of-sight)
    """

    # Extract data from SunPy maps
    field_data = field_map.data.copy()  # Make a copy of the field data
    inclination_data = inclination_map.data.copy()
    azimuth_data = azimuth_disambiguated_map.data.copy()

    # Convert degrees to radians
    dtor = np.pi / 180.0

    # Create a boolean mask for bad indices where field_data < -1.e7
    bad_index_mask = field_data < -1.e7

    # bad indicies to NaN
    field_data[bad_index_mask] = np.nan

    # Compute Bx, By, Bz
    Bx = field_data * np.sin(inclination_data * dtor) * np.sin(azimuth_data * dtor)
    By = -field_data * np.sin(inclination_data * dtor) * np.cos(azimuth_data * dtor)
    Bz = field_data * np.cos(inclination_data * dtor)

    # Update Bx, By, Bz at the bad indices using the boolean mask as NaN
    Bx[bad_index_mask] = np.nan
    By[bad_index_mask] = np.nan
    Bz[bad_index_mask] = np.nan

    # Create new SunPy maps with the modified data
    Bx_map = field_map._new_instance(Bx, field_map.meta)
    By_map = field_map._new_instance(By, field_map.meta)
    Bz_map = field_map._new_instance(Bz, field_map.meta)

    return Bx_map, By_map, Bz_map


def make_hmi_vector(azimuth_file, disambig_file, field_file, inclination_file,hmi_disambig_method = 2):

    """
    Create HMI vector magnetogram components Bx, By, Bz from the input files.
    Parameters:
    - azimuth_file: azimuth FITS file path.
    - disambig_file: disambiguation FITS file path.
    - field_file: field FITS file path.
    - inclination_file: inclination FITS file path.
    - hmi_disambig_method: method for disambiguation (default is 2).
    Returns:
    - Bx_map: A SunPy map object for the HMI Bx component data.
    - By_map: A SunPy map object for the HMI By component data.
    - Bz_map: A SunPy map object for the HMI Bz component data
    """
    
    # read in the maps
    field_map = Map(field_file)
    inclination_map = Map(inclination_file)
    azimuth_map = Map(azimuth_file)
    disambig_map = Map(disambig_file)
    
    # Apply disambiguation on the azimuth angle
    azimuth_disambiguated_map = hmi_disambig(azimuth_map, disambig_map, hmi_disambig_method)

    # Convert the HMI vector magnetogram maps to Bx, By, Bz maps
    Bx_map, By_map, Bz_map = vector2components(field_map, inclination_map, azimuth_disambiguated_map)

    return Bx_map, By_map, Bz_map



