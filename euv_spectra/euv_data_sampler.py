from huggingface_hub import snapshot_download
from pathlib import Path
import pandas as pd

# directory to store hugging face dataset
download_dir = Path("./euv_data").expanduser().resolve()

local_dir = snapshot_download(
    repo_id="nasa-ibm-ai4science/euv-spectra",
    repo_type="dataset",
    local_dir=download_dir,
    local_dir_use_symlinks=False
)
print("all files downloaded to:", download_dir)

# access the 'archive' subfolder of the hugging face dataset
data_dir = download_dir / "archive"

# combine csv files into a data.csv
dfs = []
for split in ["train_eve_spectra.csv", "val_eve_spectra.csv", "test_eve_spectra.csv"]:
    f = data_dir / split
    if not f.exists():
        raise FileNotFoundError(f"expected file not found: {f}")
    df_tmp = pd.read_csv(f)
    dfs.append(df_tmp)

df_all = pd.concat(dfs, ignore_index=True)

time_col = "timestamp"
df_all[time_col] = pd.to_datetime(df_all[time_col], errors="coerce")

df_all = (
    df_all.dropna(subset=[time_col])
          .drop_duplicates(subset=[time_col])
          .sort_values(time_col)
          .reset_index(drop=True)
)

# filter based on timestamp (after 2010-05-13)
start = pd.Timestamp("2010-05-13 00:00:00")
df_all = df_all[df_all[time_col] >= start]

print("total rows:", len(df_all))
print("time range:", df_all[time_col].min(), "→", df_all[time_col].max())

# data.csv
out_path = data_dir / "data.csv"
df_all.to_csv(out_path, index=False)
print(f"saved merged file to {out_path}")
print(df_all.head())

# split logic


def assign_split(t: pd.Timestamp) -> str:
    y, m, d = t.year, t.month, t.day

    # jan 1–14 and feb 1–14 → leaky_validation
    if (m == 1 and 1 <= d <= 14) or (m == 2 and 1 <= d <= 14):
        return "leaky_validation"

    # jan 15–31:
    #   2011, 2014 → test
    #   2012, 2013 → validation
    if m == 1 and 15 <= d <= 31:
        if y in (2011, 2014):
            return "test"
        elif y in (2012, 2013):
            return "validation"
        else:
            return "training"

    # after feb 15 → training
    if (m == 2 and d >= 15) or (m >= 3):
        return "training"

    return "training"


# run split logic and save as separate csv files
df_all["split"] = df_all[time_col].apply(assign_split)
print(df_all.head())

to_save = [c for c in df_all.columns if c != "split"]
save_dir = download_dir / "splits"
save_dir.mkdir(parents=True, exist_ok=True)

df_all[df_all["split"] == "training"][to_save].to_csv(
    save_dir / "train.csv", index=False)
df_all[df_all["split"] == "validation"][to_save].to_csv(
    save_dir / "validation.csv", index=False)
df_all[df_all["split"] == "leaky_validation"][to_save].to_csv(
    save_dir / "leaky_validation.csv", index=False)
df_all[df_all["split"] == "test"][to_save].to_csv(
    save_dir / "test.csv", index=False)
