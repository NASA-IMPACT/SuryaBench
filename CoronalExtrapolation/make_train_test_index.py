import glob
import os
import re
from datetime import datetime

import pandas as pd
from astropy.io import fits
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

N_JOBS = 16
TRAIN_OUT_FILE = "./heliofm_downstream_wsa_train_index_new.csv"
TEST_OUT_FILE = "./heliofm_downstream_wsa_test_index_new.csv"

PATH_TO_WSA_TRAIN_DATA = "/data/dedasilv/SuryaBench/CoronalFieldExtrapolation/CoronalFieldExtrapolation_train"
PATH_TO_WSA_TEST_DATA = (
    "/data/dedasilv/SuryaBench/CoronalFieldExtrapolation/CoronalFieldExtrapolation_test"
)


def main():
    make_csv(PATH_TO_WSA_TRAIN_DATA, TRAIN_OUT_FILE, "Training Data")
    make_csv(PATH_TO_WSA_TEST_DATA, TEST_OUT_FILE, "Test Data")


def make_csv(data_path, out_file, message):
    # Use Glob to get list of WSA files
    wsa_files = glob.glob(f"{data_path}/wsa*.fits")
    wsa_files.sort()

    n_files = len(wsa_files)
    print(f"Found {n_files} files")

    # Collect parallel tasks
    tasks = []

    for wsa_file in wsa_files:
        tasks.append(delayed(wsa_file_to_row_dict)(wsa_file))

    # Do parallel processing
    with joblib_progress("Preparing index...", total=n_files):
        row_dicts = Parallel(n_jobs=N_JOBS)(tasks)

    # Split to train/test dataframes
    df = pd.DataFrame(row_dicts)
    df["present"] = 1

    print(message)
    print(df.head().to_string())
    print()

    df.to_csv(out_file, index=0)

    print(f"Wrote to {out_file}")


def wsa_file_to_row_dict(wsa_file):
    time = datetime.strptime(
        os.path.basename(wsa_file.split("wsa_")[1][:12]), "%Y%m%d%H%M"
    )
    row_dict = {}
    row_dict["timestamp"] = str(time)
    row_dict["month"] = time.month

    for i in range(0, 12):
        if f"R{i:03d}" in wsa_file:
            row_dict["realization"] = i

    row_dict["wsa_file"] = wsa_file

    # fits_file = fits.open(wsa_file)
    # G = fits_file[3].data[0, :]
    # H = fits_file[3].data[1, :]
    # fits_file.close()

    # for i in range(G.shape[0]):
    #     for j in range(G.shape[1]):
    #         row_dict[f'G[{i},{j}]'] = G[i, j]
    #         row_dict[f'H[{i},{j}]'] = H[i, j]

    return row_dict


if __name__ == "__main__":
    main()
