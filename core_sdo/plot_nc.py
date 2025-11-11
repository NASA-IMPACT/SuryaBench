"""
Quicklook plotter for NC4 files.
"""
import hdf5plugin
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys

def quicklook(nc_file):
    ds = xr.open_dataset(nc_file, engine='h5netcdf')
    variables = list(ds.data_vars)
    variables.sort()
    print(f" Variables: {variables}")

    idx = 0
    while True:
        var = variables[idx]
        data = ds[var].values

        vmin = np.nanpercentile(data, 1)
        vmax = np.nanpercentile(data, 99)

        plt.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f"{var} ({idx+1}/{len(variables)})")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        cmd = input("[Enter]=next | p=prev | q=quit: ").strip().lower()
        if cmd == 'q':
            break
        elif cmd == 'p':
            idx = (idx - 1) % len(variables)
        else:
            idx = (idx + 1) % len(variables)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quicklook_nc4.py <file.nc>")
        sys.exit(1)
    quicklook(sys.argv[1])
