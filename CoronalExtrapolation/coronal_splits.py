"""
Data Preparation Script for Coronal Field Extrapolation Dataset

This script:
1. Downloads the data.tar.gz file from HuggingFace if not present (with resume support)
2. Extracts and scans all WSA FITS files from the tar archive
3. Splits the dataset into train/val/test/leaky_val sets based on timestamps
4. Saves split files (CSV indexes) to splits directory
"""

import os
import re
import tarfile
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import time


# ===============================
# Configuration
# ===============================
DATA_DIR = Path("./data")
EXTRACTED_DIR = DATA_DIR / "extracted"  # Extracted directory
DATASET_FILENAME = "data.tar.gz"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
HUGGINGFACE_URL = "https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-coronal-extrapolation/resolve/main/data.tar.gz"

OUTPUT_DIR = DATA_DIR / "splits"

# File pattern: wsa_YYYYMMDDHH[MM]Rxxx_ahmi.fits
FILENAME_PATTERN = re.compile(r"^wsa_(\d{10}|\d{12})R(\d{3})_ahmi\.fits$")
R_COLS = [f"R{str(i).zfill(3)}" for i in range(12)]

# Time range filter
START_CUTOFF = datetime(2010, 5, 13, 0, 0, 0)
END_CUTOFF = datetime(2025, 1, 1, 0, 0, 0)  # Up to 2024-12-31 24:00:00


def download_file(url, dest_path, max_retries=5):
    """
    Download a file from a URL with progress bar and resume capability.

    Parameters:
    -----------
    url : str
        URL to download from
    dest_path : Path or str
        Destination file path
    max_retries : int
        Maximum number of retry attempts
    """
    dest_path = Path(dest_path)

    # Create directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file for partial downloads
    temp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')

    # Check if partial download exists
    resume_byte_pos = temp_path.stat().st_size if temp_path.exists() else 0

    for attempt in range(max_retries):
        try:
            # Set up headers for resume
            headers = {}
            mode = 'wb'

            if resume_byte_pos > 0:
                headers['Range'] = f'bytes={resume_byte_pos}-'
                mode = 'ab'
                print(
                    f"\nResuming download from {resume_byte_pos / (1024**3):.2f} GB...")
            else:
                print(f"\nDownloading {dest_path}...")

            # Make request with headers
            response = requests.get(
                url, headers=headers, stream=True, timeout=60)

            # Check if resume is supported
            if resume_byte_pos > 0 and response.status_code not in [206, 200]:
                print(f"Resume not supported, starting from beginning...")
                resume_byte_pos = 0
                headers = {}
                response = requests.get(url, stream=True, timeout=60)

            response.raise_for_status()

            # Get total size
            if 'Content-Range' in response.headers:
                total_size = int(
                    response.headers['Content-Range'].split('/')[-1])
            else:
                total_size = int(response.headers.get('content-length', 0))

            block_size = 8192

            # Download with progress bar
            with open(temp_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=resume_byte_pos,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading (Attempt {attempt + 1}/{max_retries})"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Download successful, rename temp file to final name
            temp_path.rename(dest_path)
            print(f"✓ Download complete: {dest_path}")
            return

        except (requests.exceptions.RequestException, IOError) as e:
            print(
                f"\n✗ Download failed (Attempt {attempt + 1}/{max_retries}): {str(e)}")

            if attempt < max_retries - 1:
                # Update resume position for next attempt
                if temp_path.exists():
                    resume_byte_pos = temp_path.stat().st_size
                    print(
                        f"Will retry from {resume_byte_pos / (1024**3):.2f} GB...")
                    import time
                    time.sleep(5)  # Wait 5 seconds before retry
            else:
                print(f"\n✗ Failed to download after {max_retries} attempts")
                print(f"Please try using huggingface-cli:")
                print(f"  pip install -U huggingface_hub")
                print(f"  huggingface-cli download nasa-ibm-ai4science/surya-bench-coronal-extrapolation data.tar.gz --repo-type dataset --local-dir data/ --resume-download")
                raise


def check_and_prepare_dataset():
    """
    Check for dataset in the following order:
    1. Check if extracted directory exists with WSA FITS files
    2. Check if data.tar.gz exists locally (and validate integrity)
    3. Download data.tar.gz from HuggingFace

    Returns:
    --------
    tuple: (data_source, data_path)
        data_source: "extracted" or "tar"
        data_path: Path to the data
    """
    print("\n" + "="*60)
    print("Checking for dataset...")
    print("="*60)

    # Check 1: Look for extracted directory with WSA FITS files
    if EXTRACTED_DIR.exists():
        fits_files = list(EXTRACTED_DIR.glob("**/wsa*.fits"))
        if fits_files:
            print(f"✓ Found extracted directory: {EXTRACTED_DIR}")
            print(f"  Contains {len(fits_files)} WSA FITS files")
            return "extracted", EXTRACTED_DIR
        else:
            print(f"⚠ Found extracted directory but no WSA FITS files inside")

    # Check 2: Look for data.tar.gz and validate
    if DATASET_PATH.exists():
        file_size_gb = DATASET_PATH.stat().st_size / (1024**3)  # Convert to GB
        print(f"✓ Found tar archive: {DATASET_PATH}")
        print(f"  Size: {file_size_gb:.2f} GB")

        # Check if file size seems reasonable (should be ~36GB)
        if file_size_gb < 30:
            print(f"⚠ Warning: File size is unusually small (expected ~36GB)")
            print(f"  The file may be corrupted or incomplete.")

            user_input = input(
                "  Delete and re-download? (y/n): ").strip().lower()
            if user_input == 'y':
                print(f"  Deleting corrupted file...")
                DATASET_PATH.unlink()
            else:
                print(f"  Attempting to use existing file (may fail)...")
                return "tar", DATASET_PATH
        else:
            return "tar", DATASET_PATH

    # Check 3: Download from HuggingFace
    print(f"✗ No local data found")
    print(f"\nDownloading from HuggingFace...")
    print(f"URL: {HUGGINGFACE_URL}")
    print(f"Note: This is a large file (~36GB), download may take a while...")
    download_file(HUGGINGFACE_URL, DATASET_PATH)
    return "tar", DATASET_PATH


def parse_filename(filename):
    """
    Parse WSA FITS filename to extract timestamp and realization ID.

    Parameters:
    -----------
    filename : str
        Filename like 'wsa_201005130800R005_ahmi.fits'

    Returns:
    --------
    tuple: (datetime, realization_id, filename) or (None, None, None) if invalid
    """
    base = os.path.basename(filename)
    match = FILENAME_PATTERN.match(base)

    if not match:
        return None, None, None

    timestamp_str, realization_num = match.groups()

    # Determine datetime format based on timestamp length
    date_format = "%Y%m%d%H" if len(timestamp_str) == 10 else "%Y%m%d%H%M"
    dt = datetime.strptime(timestamp_str, date_format)

    realization_id = f"R{realization_num}"

    return dt, realization_id, base


def keep_time(dt):
    """
    Check if datetime is within the valid time range.

    Parameters:
    -----------
    dt : datetime
        Datetime object to check

    Returns:
    --------
    bool: True if within range, False otherwise
    """
    return (dt >= START_CUTOFF) and (dt < END_CUTOFF)


def scan_tar_archive(tar_path):
    """
    Scan tar.gz archive and extract metadata for all FITS files.

    Parameters:
    -----------
    tar_path : Path or str
        Path to tar.gz file

    Returns:
    --------
    dict: Dictionary mapping timestamps to realization file paths
    """
    print(f"\n{'='*60}")
    print(f"Scanning tar archive: {tar_path}")
    print(f"{'='*60}")

    timestamp_map = {}
    file_count = 0
    skipped_count = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        # Get total number of members for progress bar
        members = tar.getmembers()

        for member in tqdm(members, desc="Scanning files", unit="files"):
            if not member.isfile():
                continue

            dt, realization_id, filename = parse_filename(member.name)

            if dt is None or not keep_time(dt):
                skipped_count += 1
                continue

            # Convert datetime to ISO format string
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")

            # Initialize timestamp entry if not exists
            if timestamp_str not in timestamp_map:
                timestamp_map[timestamp_str] = {}

            # Store file path for this realization
            if realization_id not in timestamp_map[timestamp_str]:
                timestamp_map[timestamp_str][realization_id] = f"data/{filename}"
                file_count += 1

    print(f"\n✓ Scan complete:")
    print(f"  - Valid files found: {file_count}")
    print(f"  - Files skipped (out of range): {skipped_count}")
    print(f"  - Unique timestamps: {len(timestamp_map)}")

    return timestamp_map


def scan_dataset(data_source, data_path):
    """
    Scan dataset based on data source type.

    Parameters:
    -----------
    data_source : str
        "extracted" or "tar"
    data_path : Path
        Path to the data

    Returns:
    --------
    dict: Dictionary mapping timestamps to realization file paths
    """
    if data_source == "extracted":
        return scan_extracted_directory(data_path)
    elif data_source == "tar":
        return scan_tar_archive(data_path)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    """
    Scan extracted directory for FITS files and extract metadata.

    Parameters:
    -----------
    data_path : Path
        Path to extracted directory

    Returns:
    --------
    dict: Dictionary mapping timestamps to realization file paths
    """
    print(f"\n{'='*60}")
    print(f"Scanning extracted directory: {data_path}")
    print(f"{'='*60}")

    timestamp_map = {}
    file_count = 0
    skipped_count = 0

    # Find all FITS files
    fits_files = list(data_path.glob("**/*.fits"))

    for fits_file in tqdm(fits_files, desc="Scanning files", unit="files"):
        dt, realization_id, filename = parse_filename(fits_file.name)

        if dt is None or not keep_time(dt):
            skipped_count += 1
            continue

        # Convert datetime to ISO format string
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        # Initialize timestamp entry if not exists
        if timestamp_str not in timestamp_map:
            timestamp_map[timestamp_str] = {}

        # Store relative file path for this realization
        relative_path = fits_file.relative_to(data_path)
        if realization_id not in timestamp_map[timestamp_str]:
            timestamp_map[timestamp_str][realization_id] = str(relative_path)
            file_count += 1

    print(f"\n✓ Scan complete:")
    print(f"  - Valid files found: {file_count}")
    print(f"  - Files skipped (out of range): {skipped_count}")
    print(f"  - Unique timestamps: {len(timestamp_map)}")

    return timestamp_map
    """
    Scan tar.gz archive and extract metadata for all FITS files.

    Parameters:
    -----------
    tar_path : Path or str
        Path to tar.gz file

    Returns:
    --------
    dict: Dictionary mapping timestamps to realization file paths
    """
    print(f"\n{'='*60}")
    print(f"Scanning tar archive: {tar_path}")
    print(f"{'='*60}")

    timestamp_map = {}
    file_count = 0
    skipped_count = 0

    with tarfile.open(tar_path, "r:gz") as tar:
        # Get total number of members for progress bar
        members = tar.getmembers()

        for member in tqdm(members, desc="Scanning files", unit="files"):
            if not member.isfile():
                continue

            dt, realization_id, filename = parse_filename(member.name)

            if dt is None or not keep_time(dt):
                skipped_count += 1
                continue

            # Convert datetime to ISO format string
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")

            # Initialize timestamp entry if not exists
            if timestamp_str not in timestamp_map:
                timestamp_map[timestamp_str] = {}

            # Store file path for this realization
            if realization_id not in timestamp_map[timestamp_str]:
                timestamp_map[timestamp_str][realization_id] = f"data/{filename}"
                file_count += 1

    print(f"\n✓ Scan complete:")
    print(f"  - Valid files found: {file_count}")
    print(f"  - Files skipped (out of range): {skipped_count}")
    print(f"  - Unique timestamps: {len(timestamp_map)}")

    return timestamp_map


def build_dataframe(timestamp_map):
    """
    Build DataFrame from timestamp map.

    Parameters:
    -----------
    timestamp_map : dict
        Dictionary mapping timestamps to realization file paths

    Returns:
    --------
    pd.DataFrame: DataFrame with columns [timestamp, R000, R001, ..., R011]
    """
    print(f"\nBuilding DataFrame...")

    rows = []
    for timestamp in sorted(timestamp_map.keys()):
        realization_files = timestamp_map[timestamp]
        row = [timestamp] + [realization_files.get(r, "") for r in R_COLS]
        rows.append(row)

    df = pd.DataFrame(rows, columns=["timestamp"] + R_COLS)

    print(f"✓ DataFrame created with {len(df)} rows")
    return df


def assign_split(t):
    """
    Assign a split label (train/val/test/leaky_val) based on timestamp.

    Rules:
    - >= 2020-01-01 → test
    - <= 2010 → training
    - 2011-2019:
        - Jan 1-14 or Feb 1-14 → leaky_validation
        - Jan 15-31 → validation
        - Other dates → training

    Parameters:
    -----------
    t : pd.Timestamp
        Timestamp to assign split label

    Returns:
    --------
    str : Split label ('training', 'validation', 'test', 'leaky_validation')
    """
    if t >= pd.Timestamp("2020-01-01"):
        return "test"
    if t.year <= 2010:
        return "training"
    if 2011 <= t.year <= 2019:
        m, d = t.month, t.day
        if (m == 1 and 1 <= d <= 14) or (m == 2 and 1 <= d <= 14):
            return "leaky_validation"
        if (m == 1 and 15 <= d <= 31):
            return "validation"
        return "training"
    return "training"


def split_dataset(df):
    """
    Split the dataset into train/val/test/leaky_val sets based on timestamps.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp column

    Returns:
    --------
    None: Saves split files to OUTPUT_DIR
    """
    print("\n" + "="*60)
    print("Splitting dataset into train/val/test/leaky_val...")
    print("="*60)

    # Ensure timestamp column is datetime type
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Apply split assignment
    print("Applying split logic...")
    df["split"] = df["timestamp"].apply(assign_split)

    # Count splits
    split_counts = df["split"].value_counts()
    print("\nSplit distribution:")
    for split_name in ['training', 'validation', 'test', 'leaky_validation']:
        count = split_counts.get(split_name, 0)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {split_name:20s}: {count:6d} timestamps ({percentage:5.2f}%)")

    # Columns to save (exclude 'split' column)
    to_save = [c for c in df.columns if c != "split"]

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save split files
    print(f"\nExporting split files to {OUTPUT_DIR}...")

    train_df = df[df["split"] == "training"][to_save]
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    print(f"  ✓ train.csv: {len(train_df)} rows")

    val_df = df[df["split"] == "validation"][to_save]
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    print(f"  ✓ validation.csv: {len(val_df)} rows")

    leaky_val_df = df[df["split"] == "leaky_validation"][to_save]
    leaky_val_df.to_csv(OUTPUT_DIR / "leaky_validation.csv", index=False)
    print(f"  ✓ leaky_validation.csv: {len(leaky_val_df)} rows")

    test_df = df[df["split"] == "test"][to_save]
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    print(f"  ✓ test.csv: {len(test_df)} rows")

    print(f"\n✓ All split files saved to: {OUTPUT_DIR}")


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("Coronal Field Extrapolation Dataset Preparation")
    print("="*60)

    # Step 1: Check and prepare dataset (extracted dir / tar / download)
    data_source, data_path = check_and_prepare_dataset()

    # Step 2: Scan dataset and extract metadata
    timestamp_map = scan_dataset(data_source, data_path)

    # Step 3: Build DataFrame
    df = build_dataframe(timestamp_map)

    # Step 4: Split dataset and save
    split_dataset(df)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR / 'train.csv'}")
    print(f"  - {OUTPUT_DIR / 'validation.csv'}")
    print(f"  - {OUTPUT_DIR / 'leaky_validation.csv'}")
    print(f"  - {OUTPUT_DIR / 'test.csv'}")
    print(f"\nData source used: {data_source}")
    if data_source == "extracted":
        print(f"  Location: {data_path}")
    elif data_source == "tar":
        print(f"  Archive: {data_path}")


if __name__ == "__main__":
    main()
