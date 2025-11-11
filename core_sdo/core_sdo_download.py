#!/usr/bin/env python3
"""
Download 8 AIA + 5 HMI FITS for one time stamp from JSOC (DRMS)
and save them to one folder.

- Uses method='url', protocol='fits' for full headers.
- Skips files that contain 'spikes' in the URL.
- Skips downloading files that already exist locally.

Authors: Dinesha Hegde: dinesha.hegde@uah.edu
Date: November 3, 2025
"""

import os
import requests
import drms

# ================== CONFIG ==================
"""
# replace with your email, if you are using for the first time in 
JSOC you will get a confirmation email from them. Just reply 'yes' to confirm.
"""
JSOC_EMAIL = "username@domain.com"  
# ============================================

# time range
start_time = "2012.01.30_22:00:00_TAI"
duration   = "24m"
cadence    = "12m"

# Get the directory where the current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the relative folder path
TARGET_FOLDER = os.path.join(BASE_DIR, "../core_sdo_fits")

# Create the target folder if it doesn't exist
os.makedirs(TARGET_FOLDER, exist_ok=True)

# ================== DRMS CLIENT =============
client = drms.Client()

# ================== HELPER ==================
def download_export(export, target_folder):
    """Download all files from a DRMS export, skipping 'spikes' and existing files."""

    print(f"Request URL: {export.request_url}")
    print(f"{len(export.urls)} files available for download.")
    for i in range(len(export.urls)):
        url = export.urls.url[i]
        fname = export.urls.filename[i]
        out_path = os.path.join(target_folder, fname)

        # Skip spike files
        if "spikes" in url:
            print(f"Skipping spike file: {url}")
            continue

        # Skip existing files
        if os.path.exists(out_path):
            print(f"Skipping (already exists): {fname}")
            continue

        print(f"Downloading {url}")
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            print(f"  -> saved to {out_path}")
        else:
            print(f"  -> FAILED ({resp.status_code})")


# ================== CHANNEL LIST =============
channels = [
    # --- 7 AIA EUV (12 s) ---
    {"series": "aia.lev1_euv_12s", "wavel": "94"},
    {"series": "aia.lev1_euv_12s", "wavel": "131"},
    {"series": "aia.lev1_euv_12s", "wavel": "171"},
    {"series": "aia.lev1_euv_12s", "wavel": "193"},
    {"series": "aia.lev1_euv_12s", "wavel": "211"},
    {"series": "aia.lev1_euv_12s", "wavel": "304"},
    {"series": "aia.lev1_euv_12s", "wavel": "335"},

    # --- AIA UV 1600 (24 s) ---
    {"series": "aia.lev1_uv_24s", "wavel": "1600"},

    # --- HMI LOS magnetogram ---
    {"series": "hmi.M_720s"},

    # --- HMI Doppler ---
    {"series": "hmi.V_720s"},

    # --- HMI vector (4 segments) ---
    {
        "series": "hmi.B_720s",
        "segments": ["inclination", "azimuth", "disambig", "field"],
    },
]


# ================== MAIN LOOP ===============
for ch in channels:
    series = ch["series"]

    # 1) AIA with wavelength
    if "wavel" in ch:
        wavel = ch["wavel"]
        query = f"{series}[{start_time}/{duration}@{cadence}][{wavel}]"
    # 2) HMI vector with segments
    elif "segments" in ch:
        segs = ",".join(ch["segments"])
        query = f"{series}[{start_time}/{duration}@{cadence}]{{{segs}}}"
    # 3) plain HMI series
    else:
        query = f"{series}[{start_time}/{duration}@{cadence}]"

    print(f"\nSubmitting export for {query}")
    export = client.export(
        query,
        method="url",          # full export for proper headers
        protocol="fits",       # full FITS metadata
        email=JSOC_EMAIL,
    )

    download_export(export, TARGET_FOLDER)
    print("Saving to:", os.path.abspath(TARGET_FOLDER))

print("\nAll downloads finished")
# ================== END OF FILE ==================
