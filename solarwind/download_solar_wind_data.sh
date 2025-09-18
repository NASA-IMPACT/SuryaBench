#!/bin/bash

# Exit immediately if a command fails
set -e

# Define environment name and temp directory
ENV_NAME="temp_env"
ENV_DIR="./$ENV_NAME"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install sunpy[all] numpy matplotlib astropy pandas

echo "Downloading solar wind data"
python download_sw_data.py

echo "Remove ICME data"
python remove_icme_omni.py

echo "Splitting solar wind data into train-val-test"
python split_omni_icme.py

deactivate
rm -rf "$ENV_DIR"

echo "Data downloaded"
