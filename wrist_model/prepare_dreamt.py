"""
prepare_dreamt.py

Converts DREAMT v2.2.0 subject CSVs into .npz feature files for WristSleepNet.

Each SXXX_whole_df.csv contains all signals at 64 Hz in one file:
    TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, IBI,
    Sleep_Stage, Obstructive_Apnea, Central_Apnea, Hypopnea, Multiple_Events

This script segments into 30-second epochs and extracts 12 features per epoch.
Bonus: also saves a sleep_disorders array (apnea/hypopnea flags per epoch).

Usage:
    python wrist_model/prepare_dreamt.py \
        --raw_dir data/dreamt/raw \
        --out_dir data/dreamt/processed
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

FS = 64                        # Hz
EPOCH_SEC = 30
SAMPLES_PER_EPOCH = FS * EPOCH_SEC   # 1920 samples

STAGE_MAP = {
    "W":   0,   # Wake
    "N1":  1,   # Light
    "N2":  1,   # Light
    "N3":  2,   # Deep
    "R":   3,   # REM
    "REM": 3,
    "P":  -1,   # Preparation — ignore
}


# ── Feature extraction ────────────────────────────────────────────────────────

def hrv_from_ibi(ibi_series: np.ndarray) -> tuple:
    """RMSSD, SDNN, pNN50 from IBI values in milliseconds."""
    valid = ibi_series[~np.isnan(ibi_series)]
    if len(valid) < 2:
        return 0.0, 0.0, 0.0
    diffs = np.diff(valid)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    sdnn  = float(np.std(valid))
    pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100.0)
    return rmssd, sdnn, pnn50


def extract_epoch_features(epoch_df: pd.DataFrame) -> list:
    """Extract 12 features from a 1920-row (30s @ 64Hz) epoch DataFrame."""

    # HR: use pre-computed HR column (forward-filled)
    hr_vals = epoch_df["HR"].dropna().values
    hr_mean = float(np.mean(hr_vals)) if len(hr_vals) else 65.0
    hr_std  = float(np.std(hr_vals))  if len(hr_vals) else 2.0
    hr_min  = float(np.min(hr_vals))  if len(hr_vals) else hr_mean * 0.94
    hr_max  = float(np.max(hr_vals))  if len(hr_vals) else hr_mean * 1.06

    # HRV: IBI is in seconds in the CSV → convert to ms
    ibi_vals = epoch_df["IBI"].dropna().values * 1000.0
    rmssd, sdnn, pnn50 = hrv_from_ibi(ibi_vals)

    # Accelerometer magnitude
    acc = epoch_df[["ACC_X", "ACC_Y", "ACC_Z"]].values.astype(np.float32)
    mag = np.sqrt(np.sum(acc ** 2, axis=1))
    accel_mean = float(np.mean(mag))
    accel_std  = float(np.std(mag))

    # Zero-crossing rate on Z axis
    z = acc[:, 2]
    zcr = float(np.sum(np.diff(np.sign(z)) != 0) / max(len(z) - 1, 1))

    # EDA and temperature
    eda_mean  = float(epoch_df["EDA"].mean())
    temp_mean = float(epoch_df["TEMP"].mean())

    return [hr_mean, hr_std, hr_min, hr_max,
            rmssd, sdnn, pnn50,
            accel_mean, accel_std, zcr,
            eda_mean, temp_mean]


def majority_label(epoch_df: pd.DataFrame) -> int:
    """Return mapped label for the epoch using the most frequent Sleep_Stage value."""
    counts = epoch_df["Sleep_Stage"].value_counts()
    if counts.empty:
        return -1
    for stage, _ in counts.items():
        mapped = STAGE_MAP.get(str(stage).strip(), -1)
        if mapped >= 0:
            return mapped
    return -1


def apnea_flag(epoch_df: pd.DataFrame) -> int:
    """1 if any apnea/hypopnea event occurred in this epoch, else 0."""
    cols = [c for c in ["Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events"]
            if c in epoch_df.columns]
    if not cols:
        return 0
    return int(epoch_df[cols].notna().any().any())


# ── Per-subject pipeline ──────────────────────────────────────────────────────

def process_subject(csv_path: str, out_dir: str):
    subject_id = os.path.basename(csv_path).replace("_whole_df.csv", "")
    out_path   = os.path.join(out_dir, f"{subject_id}.npz")

    df = pd.read_csv(csv_path, low_memory=False)

    # Forward-fill HR (sparse column)
    df["HR"]  = pd.to_numeric(df["HR"],  errors="coerce").ffill()
    df["IBI"] = pd.to_numeric(df["IBI"], errors="coerce")

    n_epochs = len(df) // SAMPLES_PER_EPOCH
    if n_epochs == 0:
        print(f"  [SKIP] {subject_id}: too short ({len(df)} samples)")
        return

    features, labels, apnea_flags = [], [], []

    for i in range(n_epochs):
        start = i * SAMPLES_PER_EPOCH
        end   = start + SAMPLES_PER_EPOCH
        epoch = df.iloc[start:end]

        label = majority_label(epoch)
        if label < 0:
            continue   # skip Preparation and unlabelled epochs

        feats = extract_epoch_features(epoch)
        features.append(feats)
        labels.append(label)
        apnea_flags.append(apnea_flag(epoch))

    if not features:
        print(f"  [SKIP] {subject_id}: no valid labelled epochs")
        return

    x = np.array(features,    dtype=np.float32)   # (n, 12)
    y = np.array(labels,       dtype=np.int32)     # (n,)
    a = np.array(apnea_flags,  dtype=np.int32)     # (n,)  bonus: apnea labels

    stage_counts = {v: int(np.sum(y == v)) for v in range(4)}
    apnea_count  = int(np.sum(a))

    np.savez(out_path, x=x, y=y, apnea=a)
    print(f"  [OK] {subject_id}: {len(y)} epochs  stages={stage_counts}  apnea_epochs={apnea_count}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare DREAMT v2.2.0 for WristSleepNet")
    parser.add_argument("--raw_dir", type=str, default="data/dreamt/raw")
    parser.add_argument("--out_dir", type=str, default="data/dreamt/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(args.raw_dir, "S*_whole_df.csv")))

    if not csv_files:
        print(f"No S*_whole_df.csv files found in {args.raw_dir}")
        print("Run download_dreamt.py first.")
        return

    print(f"Processing {len(csv_files)} subjects...")
    for f in csv_files:
        process_subject(f, args.out_dir)

    done = glob.glob(os.path.join(args.out_dir, "*.npz"))
    print(f"\nDone. {len(done)} subjects saved to {args.out_dir}/")
    print("Next:")
    print("  python wrist_model/train_wrist.py --data_dir data/dreamt/processed")


if __name__ == "__main__":
    main()
