import os
import sys
import csv
import h5py
import numpy as np

# Base directory
base_dir = "/nobackupnfs1/sroy14/processed_data/Helio/aremerge_skasapis"
output_file = os.path.join(
    base_dir, "train_indexed_data_ar_emergence_kasapis_rohit_norm.csv"
)
output_file_test = os.path.join(
    base_dir, "test_indexed_data_ar_emergence_kasapis_rohit_norm.csv"
)

# Constants
input_size = 120
forecast_window = 12
test_ARs = ["AR11698", "AR11726"]
ar_folders = sorted([f for f in os.listdir(base_dir) if f.startswith("AR")])
index = 0
test_index = 0


def save_to_csv(csv_file, index, input_matrix, output_matrix, input_times, output_time):
    """
    Save a data entry into a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        index (int): Sample index.
        input_matrix (np.ndarray): Input matrix data.
        output_matrix (np.ndarray): Output matrix data.
        input_times (np.ndarray): Input timestamps.
        output_time (str or bytes): Output timestamp.
    """
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if isinstance(output_time, bytes):
            output_time = output_time.decode("utf-8")
        writer.writerow(
            [
                index,
                input_matrix.tolist(),
                output_matrix.tolist(),
                input_times.tolist(),
                output_time,
            ]
        )
    print(f"Data saved for index {index}")


def compute_global_min_max(ar_folders):
    """
    Compute global min/max for each variable across AR folders.

    Returns:
        dict: Mapping of variable names to (min, max) tuples.
    """
    stats = {"pm_23": [], "pm_34": [], "pm_45": [], "pm_56": [], "mag": [], "int": []}

    for ar_folder in ar_folders:
        if ar_folder == "AR12494":
            break

        ar_path = os.path.join(base_dir, ar_folder)
        try:
            pm = np.load(
                os.path.join(ar_path, f"mean_pmdop{ar_folder[2:]}_flat.npz"),
                allow_pickle=True,
            )
            mag = np.load(
                os.path.join(ar_path, f"mean_mag{ar_folder[2:]}_flat.npz"),
                allow_pickle=True,
            )
            inten = np.load(
                os.path.join(ar_path, f"mean_int{ar_folder[2:]}_flat.npz"),
                allow_pickle=True,
            )
        except Exception as e:
            print(f"Skipping {ar_folder}: {e}")
            continue

        stats["pm_23"].extend([np.min(pm["arr_0"]), np.max(pm["arr_0"])])
        stats["pm_34"].extend([np.min(pm["arr_1"]), np.max(pm["arr_1"])])
        stats["pm_45"].extend([np.min(pm["arr_2"]), np.max(pm["arr_2"])])
        stats["pm_56"].extend([np.min(pm["arr_3"]), np.max(pm["arr_3"])])
        stats["mag"].extend([np.min(mag["arr_0"]), np.max(mag["arr_0"])])
        stats["int"].extend([np.min(inten["arr_0"]), np.max(inten["arr_0"])])

    return {key: (min(vals[::2]), max(vals[1::2])) for key, vals in stats.items()}


def normalize_and_write_to_csv(global_stats, ar_folders):
    """
    Normalize data and save to CSV using global min/max statistics.

    Args:
        global_stats (dict): Dictionary of (min, max) per feature.
        ar_folders (list): List of AR directories.
    """
    global index, test_index
    for ar_folder in ar_folders:
        if ar_folder == "AR12494":
            break

        ar_path = os.path.join(base_dir, ar_folder)
        try:
            pm = np.load(
                os.path.join(ar_path, f"mean_pmdop{ar_folder[2:]}_flat.npz"),
                allow_pickle=True,
            )
            mag = np.load(
                os.path.join(ar_path, f"mean_mag{ar_folder[2:]}_flat.npz"),
                allow_pickle=True,
            )
            inten = np.load(
                os.path.join(ar_path, f"mean_int{ar_folder[2:]}_flat.npz"),
                allow_pickle=True,
            )
        except Exception as e:
            print(f"Skipping {ar_folder}: {e}")
            continue

        pm_data = [
            (pm["arr_0"].T - global_stats["pm_23"][0])
            / (global_stats["pm_23"][1] - global_stats["pm_23"][0]),
            (pm["arr_1"].T - global_stats["pm_34"][0])
            / (global_stats["pm_34"][1] - global_stats["pm_34"][0]),
            (pm["arr_2"].T - global_stats["pm_45"][0])
            / (global_stats["pm_45"][1] - global_stats["pm_45"][0]),
            (pm["arr_3"].T - global_stats["pm_56"][0])
            / (global_stats["pm_56"][1] - global_stats["pm_56"][0]),
        ]
        mag_data = (mag["arr_0"].T - global_stats["mag"][0]) / (
            global_stats["mag"][1] - global_stats["mag"][0]
        )
        int_data = (inten["arr_0"].T - global_stats["int"][0]) / (
            global_stats["int"][1] - global_stats["int"][0]
        )
        time = pm["arr_4"]

        # Skip if NaNs present
        if any(np.isnan(d).any() for d in pm_data + [mag_data, int_data]):
            continue

        is_test = ar_folder in test_ARs
        out_csv = output_file_test if is_test else output_file
        counter = test_index if is_test else index

        for i in range(len(pm_data[0])):
            if i + input_size + forecast_window >= len(pm_data[0]):
                break

            input_matrix = np.stack(
                [d[i : i + input_size, 9:72] for d in pm_data]
                + [mag_data[i : i + input_size, 9:72]],
                axis=1,
            )

            output_matrix = int_data[i + input_size + forecast_window - 1, 9:72]
            input_times = np.array(
                [str(t).encode("utf-8") for t in time[i : i + input_size]]
            )
            output_time = np.array(
                str(time[i + input_size + forecast_window - 1]).encode("utf-8")
            )

            save_to_csv(
                out_csv, counter, input_matrix, output_matrix, input_times, output_time
            )

            if is_test:
                test_index += 1
            else:
                index += 1


def get_item_ar_emergence_skasapis(idx: int, train: bool = True):
    """
    Retrieve an item by index from the HDF5 dataset.

    Args:
        idx (int): Index to access.
        train (bool): Whether to access train or test file.

    Returns:
        tuple: input, output, input_times, output_time
    """
    data_file = os.path.join(
        base_dir,
        (
            "train_indexed_data_ar_emergence_kasapis_rohit.h5"
            if train
            else "test_indexed_data_ar_emergence_kasapis_rohit.h5"
        ),
    )

    with h5py.File(data_file, "r") as f:
        group = f[f"index_{idx}"]
        input_matrix = group["input_matrix"][:]
        output_matrix = group["output_matrix"][:]
        input_times = group["input_times"][:]
        output_time = group["output_time"][()]
    return input_matrix, output_matrix, input_times, output_time


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("Computing global min/max...")
    global_stats = compute_global_min_max(ar_folders)
    print("Global stats:", global_stats)

    print("Normalizing and writing to CSV...")
    normalize_and_write_to_csv(global_stats, ar_folders)

    # Example usage
    input_data, output_data, in_times, out_time = get_item_ar_emergence_skasapis(0)
    print("Input shape:", input_data.shape)
    print("Output shape:", output_data.shape)
    print("Input times:", in_times.shape)
    print("Output time:", out_time)
