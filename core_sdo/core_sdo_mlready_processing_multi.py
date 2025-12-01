"""
Convert SDO (AIA + HMI) FITS files to AI/ML-ready NetCDF4 (4K) files.
Each output NetCDF4 file contains 13 variables:
  - 8 AIA wavelengths: 94, 131, 171, 193, 211, 304, 335, 1600 Angstrom
  - 5 HMI variables: LOS Magnetogram, LOS Dopplergram, Bx, By, Bz

Author: Dinesha Hegde, dinesha.hegde@gmail.com
Date: Nov 6, 2025
"""

import os
import re
import gc
import json
import datetime as dt
from glob import glob

import numpy as np
import xarray as xr
from sunpy.map import Map, contains_full_disk
import hdf5plugin  # for LZ4 compression/ filter 32004
import helio  

# CONFIG

# Get the folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The FITS files are one folder up, in ../core_sdo_fits
fits_path = os.path.join(BASE_DIR, "../core_sdo_fits")
if not os.path.exists(fits_path):
    raise FileNotFoundError(f"FITS root folder not found: {fits_path}")

# Output folder for NC4 files inside the current repo
out_dir = os.path.join(BASE_DIR, "../mlready_core_sdo_nc4")
os.makedirs(out_dir, exist_ok=True)
# -----------------------------------------------------------------------------


def sdo_fits_to_mlready_netcdf_for_timestamp(fits_path, timestamp, nc_outfile=None, nc_compression_type='LZ4'):
    """
    Process 13 SDO (AIA + HMI) FITS files in `fits_path` that belong to ONE timestamp
    and write a single NetCDF4 (4K) file, LZ4-compressed.
    Parameters:
    - fits_path: Path to the folder containing SDO FITS files.
    - timestamp: Timestamp string in 'YYYYMMDD_HHMM' format.
    - nc_outfile: Optional output NetCDF4 file path. If None, a default name is used.
    - nc_compression_type: Compression type for NetCDF4 file ('LZ4' or 'zlib').
    Returns:
    - None
    """
    # pick only files that match the timestamp
    aia_files = sorted(
        f for f in glob(os.path.join(fits_path, "aia*.fits"))
        if helio.compute_timestamp(os.path.basename(f)) == timestamp
    )
    hmi_files = sorted(
        f for f in glob(os.path.join(fits_path, "hmi*.fits"))
        if helio.compute_timestamp(os.path.basename(f)) == timestamp
    )

    print(f"\n=== Processing timestamp {timestamp} ===")
    print("AIA files:", len(aia_files))
    #print("AMI files:", aia_files)
    print("HMI files:", len(hmi_files))
    #print("HMI files:", hmi_files)
    
    if len(aia_files) == 0:
        print(f"No AIA FITS found for {timestamp}, skipping.")
        return

    if nc_outfile is None:
        nc_outfile = os.path.join(out_dir, f"{timestamp}.nc")

    data_arrays = {}
    encoding = {}

    # -------------------------------------------------------------------------
    # 1) AIA
    # -------------------------------------------------------------------------
    aia_wcs = None

    for fname in aia_files:
        m = Map(fname)
        wavelnth = m.meta["wavelnth"]
        this_timestamp = helio.compute_timestamp(os.path.basename(fname))

        if this_timestamp != timestamp:
            print(f"Timestamp mismatch: {fname} has {this_timestamp}, expected {timestamp}")
            return

        if m.meta["quality"] != 0:
            print(f"Bad AIA file {fname}, quality={m.meta['quality']} -> skipping this timestamp")
            return

        # original meta
        meta0 = m.meta
        
        # process AIA map
        aia_map = helio.process_aia_map(m)
        
        # processed meta
        meta1 = aia_map.meta

        # get data as float32
        xdata = aia_map.data.astype(np.float32)

        # replace NaN with 0.0
        np.nan_to_num(xdata, copy=False, nan=0.0)

        var_name = f"aia{wavelnth}"
        description = f"Level-1.5 AIA image for wavelength {wavelnth} Angstrom"

        print(f"add var: {var_name} (AIA) to data_arrays...")
        data_arrays[var_name] = xr.DataArray(
            xdata,
            name=var_name,
            dims=["y", "x"],
            attrs={
                "unit": "DN/s",
                "t_obs": meta0.get("t_obs", ""),
                "qflag": meta0.get("quality", 0),
                "description": description,
                "meta_0": json.dumps(meta0),
                "meta_1": json.dumps(meta1),
            },
        )
        # Apply compression
        if nc_compression_type == 'zlib':
            encoding[var_name] = {'zlib': True, 'complevel': 5, 'chunksizes': (4096, 4096)}
        else:  # default to LZ4
            encoding[var_name] = {
                "compression": 32004,        # LZ4
                "compression_opts": (),
                "chunksizes": (4096, 4096),
                "_FillValue": 0.0,
            }

        if wavelnth == 171:
            aia_wcs = aia_map.wcs # save WCS for HMI processing

        del xdata

    # If aia_wcs is still None, process the first AIA map to get WCS
    if aia_wcs is None:
        first_aia_map = helio.process_aia_map(Map(aia_files[0]))
        aia_wcs = first_aia_map.wcs

    # -------------------------------------------------------------------------
    # 2) HMI
    # -------------------------------------------------------------------------
    hmi_maps = {}
    field_file = disambig_file = inclination_file = azimuth_file = None

    for fname in hmi_files:
        if "Dopplergram" in fname:
            hmi_maps["V"] = Map(fname)
        elif "magnetogram" in fname:
            hmi_maps["M"] = Map(fname)
        elif "field" in fname:
            field_file = fname
        elif "disambig" in fname:
            disambig_file = fname
        elif "inclination" in fname:
            inclination_file = fname
        elif "azimuth" in fname:
            azimuth_file = fname

    # Create Bx, By, Bz maps if all required files are present
    if field_file and disambig_file and inclination_file and azimuth_file:
        Bx_map, By_map, Bz_map = helio.make_hmi_vector(
            azimuth_file, disambig_file, field_file, inclination_file)
        hmi_maps["Bx"] = Bx_map
        hmi_maps["By"] = By_map
        hmi_maps["Bz"] = Bz_map

        # check timestamp consistency
        pattern = re.compile(r"\d{8}_\d{4}")
        match = pattern.search(os.path.basename(azimuth_file))
        if match:
            hmi_ts2 = match.group(0)
            if hmi_ts2 != timestamp:
                print(f"HMI timestamp {hmi_ts2} != AIA timestamp {timestamp}")

    # loop through hmi_maps
    for suffix, m in hmi_maps.items():
        if not contains_full_disk(m):
            print(f"Bad HMI {suffix}, not full disk -> skipping this timestamp")
            return

        quality = m.meta.get("quality", 0)
        if quality != 0:
            print(f"HMI {suffix} quality={quality} -> skipping this timestamp")
            return

        # original meta
        meta0 = m.meta

        # process HMI map
        hmi_map = helio.process_hmi_map(m, aia_wcs)
        
        # processed meta
        meta1 = hmi_map.meta

        # get data as float32
        xdata = hmi_map.data.astype(np.float32)
        
        # replace NaN with 0.0
        np.nan_to_num(xdata, copy=False, nan=0.0)

        # determine var_name, unit, description based on suffix
        if suffix == "V":
            var_name = "hmi_v"
            unit = "m/s"
            description = "HMI LOS Dopplergrams"
        elif suffix == "M":
            var_name = "hmi_m"
            unit = "Gauss"
            description = "HMI LOS Magnetograms"
        elif suffix == "Bx":
            var_name = "hmi_bx"
            unit = "Gauss"
            description = "x-component of HMI vector magnetic field"
        elif suffix == "By":
            var_name = "hmi_by"
            unit = "Gauss"
            description = "y-component of HMI vector magnetic field"
        elif suffix == "Bz":
            var_name = "hmi_bz"
            unit = "Gauss"
            description = "z-component of HMI vector magnetic field"
        else:
            print(f"Unknown HMI suffix {suffix}, skipping")
            continue
        
        # add variables to data_arrays
        print(f"add var: {var_name} (HMI) to data_arrays...")
        data_arrays[var_name] = xr.DataArray(
            xdata,
            name=var_name,
            dims=["y", "x"],
            attrs={
                "unit": unit,
                "t_obs": meta0.get("t_obs", ""),
                "qflag": meta0.get("quality", 0),
                "description": description,
                "meta_0": json.dumps(meta0),
                "meta_1": json.dumps(meta1),
            },
        )

        # Apply compression
        if nc_compression_type == 'zlib':
            encoding[var_name] = {'zlib': True, 'complevel': 5, 'chunksizes': (4096, 4096)}
        else:  # default to LZ4
            encoding[var_name] = {
                "compression": 32004,        # LZ4
                "compression_opts": (),
                "chunksizes": (4096, 4096),
                "_FillValue": 0.0,
            }

        # clean up
        del xdata

    # sanity check
    if len(data_arrays) != 13:
        print(f"only {len(data_arrays)} out of 13 dataset available for {timestamp}. Skipping.")
        return

    # Global attributes
    attrs = {
        "title": "AI/ML ready level-1.5 SDO data in 4K Resolution",
        "author": (
            "Dinesha Hegde,dinesha.hegde@uah.edu;"
            "Sujit Roy, sujit.roy@uah.edu; "
            "Amy Lin, amy.lin@uah.edu"
        ),
        "institution": "NASA MSFC IMPACT Project; ESSC & CSPAR, University of Alabama in Huntsville",
        "data_time": timestamp,
        "production_date": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # wrap into Dataset
    ds = xr.Dataset(data_arrays, attrs=attrs)

    # clean up
    gc.collect()
    
    # write to NetCDF4 file
    ds.to_netcdf(
        path=nc_outfile,
        format="NETCDF4",
        engine="h5netcdf",
        encoding=encoding,
        mode="w",
    )

    # final clean up
    ds.close()
    del ds
    print(f"made {nc_outfile}")


def process_multi_timestamps(fits_path):
    """
    Find all timestamps present in AIA FITS in this folder and build NC for each.
    Parameters:
    - fits_path: Path to the folder containing SDO FITS files.
    Returns:
    - None
    """
    aia_files = glob(os.path.join(fits_path, "aia*.fits"))
    timestamps = sorted(
        {helio.compute_timestamp(os.path.basename(f)) for f in aia_files}
    )
    print(f"Found {len(timestamps)} timestamps: {timestamps}")

    for ts in timestamps:
        sdo_fits_to_mlready_netcdf_for_timestamp(fits_path, ts, nc_outfile=None, nc_compression_type='zlib')


if __name__ == "__main__":
    process_multi_timestamps(fits_path)
