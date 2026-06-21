"""
parsers/csv_generic.py

Parses a user-provided CSV file into per-30s epoch feature dicts.
Any wearable device that can export a CSV (e.g., Polar, Whoop, Samsung Health)
can be used by mapping its columns to our template.

Expected CSV format (download template from the app UI):

    timestamp,hr_mean,hrv_rmssd,accel_mean,temp_mean,spo2
    2024-01-15 22:00:00,62.1,45.3,0.02,36.5,97.0
    2024-01-15 22:00:30,61.8,48.0,0.01,36.5,97.1
    ...

Required columns : timestamp, hr_mean
Optional columns : hrv_rmssd, hrv_sdnn, hrv_pnn50,
                   hr_std, hr_min, hr_max,
                   accel_mean, accel_std, accel_zcr,
                   eda_mean, temp_mean, spo2

Rows are assumed to be spaced 30 seconds apart (epoch-level data).
If rows are at a different interval (e.g. 1 minute), they will be resampled.
"""

import csv
import os
from datetime import datetime, timedelta

EPOCH_SEC = 30

REQUIRED_COLS = {"timestamp", "hr_mean"}

OPTIONAL_DEFAULTS = {
    "hr_std":     2.0,
    "hr_min":     0.0,       # computed from hr_mean if missing
    "hr_max":     0.0,       # computed from hr_mean if missing
    "hrv_rmssd":  30.0,
    "hrv_sdnn":   27.0,
    "hrv_pnn50":  0.0,
    "accel_mean": 0.0,
    "accel_std":  0.0,
    "accel_zcr":  0.0,
    "eda_mean":   0.0,
    "temp_mean":  0.0,
}


def _parse_dt(s: str) -> datetime:
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp: {s!r}")


def parse(csv_path: str) -> list[dict]:
    """
    Parse a generic wearable CSV into epoch dicts.

    Returns:
        list of epoch dicts matching FEATURE_KEYS.

    Raises:
        ValueError: if required columns are missing or file is empty.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - headers
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Found: {headers}. "
                f"Download the template from the app for the correct format."
            )
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty (no data rows).")

    # Parse timestamps and detect interval
    parsed = []
    for row in rows:
        try:
            ts = _parse_dt(row["timestamp"])
            hr = float(row["hr_mean"])
            parsed.append((ts, hr, row))
        except (ValueError, KeyError):
            continue

    if not parsed:
        raise ValueError("No valid rows could be parsed from the CSV.")

    # Detect interval between rows
    if len(parsed) > 1:
        interval_sec = (parsed[1][0] - parsed[0][0]).total_seconds()
    else:
        interval_sec = EPOCH_SEC

    epochs = []
    for i, (ts, hr, row) in enumerate(parsed):
        hr_std = float(row.get("hr_std", OPTIONAL_DEFAULTS["hr_std"]) or OPTIONAL_DEFAULTS["hr_std"])
        hr_min = float(row.get("hr_min", 0) or 0) or hr * 0.94
        hr_max = float(row.get("hr_max", 0) or 0) or hr * 1.06

        def opt(key):
            v = row.get(key)
            try:
                return float(v) if v not in (None, "", "N/A") else OPTIONAL_DEFAULTS[key]
            except (ValueError, TypeError):
                return OPTIONAL_DEFAULTS[key]

        epoch = {
            "timestamp":  ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hr_mean":    hr,
            "hr_std":     hr_std,
            "hr_min":     hr_min,
            "hr_max":     hr_max,
            "hrv_rmssd":  opt("hrv_rmssd"),
            "hrv_sdnn":   opt("hrv_sdnn"),
            "hrv_pnn50":  opt("hrv_pnn50"),
            "accel_mean": opt("accel_mean"),
            "accel_std":  opt("accel_std"),
            "accel_zcr":  opt("accel_zcr"),
            "eda_mean":   opt("eda_mean"),
            "temp_mean":  opt("temp_mean"),
        }
        epochs.append(epoch)

        # If rows are at > 30s intervals (e.g. 60s), duplicate to fill 30s epochs
        if interval_sec > EPOCH_SEC and i < len(parsed) - 1:
            n_fill = int(interval_sec / EPOCH_SEC) - 1
            for j in range(1, n_fill + 1):
                fill_epoch = epoch.copy()
                fill_epoch["timestamp"] = (ts + timedelta(seconds=j * EPOCH_SEC)).strftime("%Y-%m-%d %H:%M:%S")
                epochs.append(fill_epoch)

    return epochs


def write_template(out_path: str, n_rows: int = 5):
    """Write a filled example CSV template for users to download."""
    headers = [
        "timestamp", "hr_mean", "hr_std", "hr_min", "hr_max",
        "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
        "accel_mean", "accel_std", "accel_zcr",
        "eda_mean", "temp_mean",
    ]
    from datetime import datetime
    base_ts = datetime(2024, 1, 15, 22, 0, 0)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for i in range(n_rows):
            ts = base_ts + timedelta(seconds=i * EPOCH_SEC)
            writer.writerow({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "hr_mean": 63.0,
                "hr_std": 2.1,
                "hr_min": 59.0,
                "hr_max": 67.0,
                "hrv_rmssd": 44.5,
                "hrv_sdnn": 40.1,
                "hrv_pnn50": 18.3,
                "accel_mean": 0.02,
                "accel_std": 0.01,
                "accel_zcr": 0.05,
                "eda_mean": 0.3,
                "temp_mean": 36.4,
            })
