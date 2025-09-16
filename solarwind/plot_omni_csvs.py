#!/usr/bin/env python3
"""
plot_omni_csvs.py

Scan CSV files in `solarwind/csv_files/` and create simple time-series plots
for numeric columns for visual inspection. Saves PNGs to `solarwind/plots/`.

Usage:
  python3 plot_omni_csvs.py --input-dir solarwind/csv_files --out-dir solarwind/plots --date-col Epoch

If --date-col is not provided, the script will attempt to auto-detect a datetime column.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot numeric columns from omni csv files for visual inspection")
    p.add_argument("--input-dir", default="solarwind/csv_files", help="Directory with csv files")
    p.add_argument("--out-dir", default="solarwind/plots", help="Directory to save plots")
    p.add_argument("--date-col", default=None, help="Name of datetime column (optional)")
    p.add_argument("--sample", type=int, default=1, help="Plot every Nth point to reduce size (default 1 = all)")
    return p.parse_args()


COMMON_DATE_COLS = ["Epoch", "date", "datetime", "time", "timestamp", "Date"]


def find_date_column(df: pd.DataFrame, hint: str | None = None) -> str | None:
    if hint and hint in df.columns:
        return hint
    for c in COMMON_DATE_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # try parse
    for c in df.columns:
        try:
            pd.to_datetime(df[c].iloc[0])
            return c
        except Exception:
            continue
    return None


def plot_file(path: Path, out_dir: Path, date_col_hint: str | None = None, sample: int = 1):
    df = pd.read_csv(path)
    date_col = find_date_column(df, date_col_hint)
    if date_col is None:
        print(f"[skip] could not find date column for {path.name}")
        return
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        print(f"[skip] no numeric columns in {path.name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot each numeric column in its own subplot (vertical stack)
    n = len(numeric)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric):
        ax.plot(df[date_col].values[::sample], df[col].values[::sample], linewidth=0.6)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(date_col)
    fig.suptitle(path.name)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = out_dir / f"{path.stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {out_path} ({n} vars)")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {input_dir}")
        return
    for f in files:
        try:
            plot_file(f, out_dir, date_col_hint=args.date_col, sample=args.sample)
        except Exception as e:
            print(f"Error plotting {f}: {e}")


if __name__ == "__main__":
    main()
