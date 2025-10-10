#!/usr/bin/env python3
"""
    Author: Vishal Upendran[uvishal1995@gmail.com]

    Create train/val/buffer/test splits from omni_icme_removed_solar_wind.csv
    using the Surya-style buffered, non-chronological temporal sampling:

    For years 2010-2019 (inclusive), within each year split by day-of-year index (0-based):
    - days [0:14)   -> buffer
    - days [14:28)  -> validation
    - days [28:42)  -> buffer
    - days [42:]     -> training

    For years 2020-2024 (inclusive): all days -> testing

    Outputs are CSV files written to the output directory.
    For a dry run, perform: `python3 solarwind/split_omni_icme.py --dry-run`
    For the correct run and splitting, run: `python3 split_omni_icme.py --input csv_files/omni_icme_removed_solar_wind.csv --out-dir csv_files/`
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split omni ICME CSV into train/val/buffer/test using Surya schema")
    p.add_argument("--input", "-i", required=False, default="csv_files/omni_icme_removed_solar_wind.csv",
                   help="Path to input CSV (default: csv_files/omni_icme_removed_solar_wind.csv)")
    p.add_argument("--out-dir", "-o", required=False, default="csv_files/",
                   help="Output directory for split CSVs (default: csv_files/)")
    p.add_argument("--date-col", required=False, default=None,
                   help="Name of the date column in the CSV. If omitted, will try common names.")
    p.add_argument("--dry-run", action="store_true", help="Don't write files; only report counts")
    return p.parse_args()

def split_dataframe(df: pd.DataFrame, date_col: str):
    # Ensure datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        nbad = int(df[date_col].isna().sum())
        raise ValueError(f"{nbad} rows could not be parsed as datetimes in column '{date_col}'")

    # filter to requested overall date range: 2010-05-01 .. 2024-12-31
    min_dt = pd.to_datetime("2010-05-01")
    max_dt = pd.to_datetime("2024-12-31")
    before_count = len(df)
    df = df[(df[date_col] >= min_dt) & (df[date_col] <= max_dt)].reset_index(drop=True)
    after_count = len(df)
    dropped = before_count - after_count
    if dropped > 0:
        print(f"Filtered data to {min_dt.date()}..{max_dt.date()}: dropped {dropped} rows ({before_count} -> {after_count})")

    # add helpers
    df["year"] = df[date_col].dt.year
    # dayofyear is 1-based; convert to 0-based index
    df["day_idx"] = df[date_col].dt.dayofyear - 1

    train_parts = []
    val_parts = []
    buffer_parts = []
    test_parts = []

    for year, group in df.groupby("year"):
        if 2010 <= year <= 2019:
            # map day_idx to segments
            g = group.copy()
            # create assignment column
            def assign(day_idx: int) -> str:
                if 0 <= day_idx < 14:
                    return "buffer"
                if 14 <= day_idx < 28:
                    return "val"
                if 28 <= day_idx < 42:
                    return "buffer"
                # day_idx >= 42
                return "train"

            g["_split"] = g["day_idx"].apply(assign)
            train_parts.append(g[g["_split"] == "train"])
            val_parts.append(g[g["_split"] == "val"])
            buffer_parts.append(g[g["_split"] == "buffer"])
        elif 2020 <= year <= 2024:
            test_parts.append(group)
        else:
            # years outside required ranges will be ignored but kept as buffer for safety
            buffer_parts.append(group)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    buffer_df = pd.concat(buffer_parts, ignore_index=True) if buffer_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)

    # drop helper cols before returning
    for d in (train_df, val_df, buffer_df, test_df):
        for c in ("year", "day_idx", "_split"):
            if c in d.columns:
                d.drop(columns=[c], inplace=True)

    return train_df, val_df, buffer_df, test_df


def write_splits(out_dir: str, base_name: str, splits: dict[str, pd.DataFrame], dry_run: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for name, df in splits.items():
        fname = os.path.join(out_dir, f"{base_name}_{name}.csv")
        results[name] = (fname, len(df))
        if not dry_run:
            df.to_csv(fname, index=False)
    return results


def main():
    args = parse_args()
    inp = args.input
    out_dir = args.out_dir

    if not os.path.exists(inp):
        raise FileNotFoundError(f"Input file not found: {inp}")

    df = pd.read_csv(inp)
    date_col = "Epoch"

    train_df, val_df, buffer_df, test_df = split_dataframe(df, date_col)

    splits = {"training": train_df, "validation": val_df, "leaky_validation": buffer_df, "test": test_df}

    results = write_splits(out_dir, "omni_icme_removed", splits, dry_run=args.dry_run)

    # report
    print("Split results:")
    total = 0
    for name, (path, cnt) in results.items():
        print(f" - {name}: {cnt} rows -> {path if not args.dry_run else '(dry-run)'}")
        total += cnt
    print(f"Total rows assigned: {total}")


if __name__ == "__main__":
    main()
