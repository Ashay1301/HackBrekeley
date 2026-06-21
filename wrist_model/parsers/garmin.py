"""
parsers/garmin.py

Parses a Garmin Connect export — either a .fit file or a Garmin Connect
JSON export (WELLNESS_SLEEP_DATA_*.json / SLEEP_*.json format).

How to export from Garmin Connect:
    garminconnect.garmin.com → Health Stats → Sleep → Export (.fit or .json)

Supports:
    - .fit files via the `fitparse` library
    - Garmin Connect JSON (from bulk export zip)
"""

import json
import os
from datetime import datetime, timedelta
import numpy as np

EPOCH_SEC = 30

GARMIN_SLEEP_STAGE_MAP = {
    0: 0,   # Awake   → Wake
    1: 1,   # Light   → Light
    2: 2,   # Deep    → Deep
    3: 3,   # REM     → REM
    4: 0,   # Unmeasurable → Wake (conservative)
}


# ── FIT file parser ────────────────────────────────────────────────────────────

def _parse_fit(fit_path: str) -> list[dict]:
    try:
        import fitparse
    except ImportError:
        raise ImportError("Install fitparse: pip install fitparse")

    fitfile = fitparse.FitFile(fit_path)
    sleep_levels = []
    hr_records = []

    for record in fitfile.get_messages():
        name = record.name
        if name == "sleep_level":
            data = {f.name: f.value for f in record}
            if "timestamp" in data and "sleep_level" in data:
                sleep_levels.append(data)
        elif name == "record":
            data = {f.name: f.value for f in record}
            if "timestamp" in data and "heart_rate" in data and data["heart_rate"]:
                hr_records.append(data)

    if not sleep_levels:
        raise ValueError("No sleep_level messages found in .fit file.")

    sleep_levels.sort(key=lambda r: r["timestamp"])
    hr_by_ts = {r["timestamp"]: r["heart_rate"] for r in hr_records}

    epochs = []
    for i, lvl in enumerate(sleep_levels):
        ts = lvl["timestamp"]
        stage_raw = int(lvl.get("sleep_level", 0))
        stage = GARMIN_SLEEP_STAGE_MAP.get(stage_raw, 0)
        next_ts = sleep_levels[i + 1]["timestamp"] if i + 1 < len(sleep_levels) else ts + timedelta(seconds=EPOCH_SEC)
        duration_sec = (next_ts - ts).total_seconds()
        n = max(1, int(np.ceil(duration_sec / EPOCH_SEC)))

        # Find closest HR
        hr = 65.0
        for offset in range(60):
            candidate = ts + timedelta(seconds=offset)
            if candidate in hr_by_ts:
                hr = float(hr_by_ts[candidate])
                break

        for j in range(n):
            epoch_ts = ts + timedelta(seconds=j * EPOCH_SEC)
            epochs.append({
                "timestamp":  epoch_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "hr_mean":    hr,
                "hr_std":     2.0,
                "hr_min":     hr * 0.94,
                "hr_max":     hr * 1.06,
                "hrv_rmssd":  30.0,
                "hrv_sdnn":   27.0,
                "hrv_pnn50":  0.0,
                "accel_mean": 0.05 if stage == 0 else 0.01,
                "accel_std":  0.02,
                "accel_zcr":  0.0,
                "eda_mean":   0.0,
                "temp_mean":  0.0,
            })
    return epochs


# ── Garmin Connect JSON parser ─────────────────────────────────────────────────

def _parse_json(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)

    # Garmin bulk export has two variants
    sleep_data = data if isinstance(data, dict) else {}
    if not sleep_data:
        raise ValueError("Unexpected Garmin JSON structure.")

    sleep_movement = sleep_data.get("sleepMovement", [])
    sleep_levels = sleep_data.get("sleepLevels", sleep_data.get("sleepStages", []))
    hr_data = sleep_data.get("wellnessEpochRespirationDataDTOList", [])

    start_ts_str = sleep_data.get("sleepStartTimestampLocal") or sleep_data.get("startTimeLocal", "")
    if not start_ts_str:
        raise ValueError("Could not find sleep start time in Garmin JSON.")

    try:
        start_ts = datetime.fromtimestamp(int(start_ts_str) / 1000)
    except (ValueError, TypeError):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                start_ts = datetime.strptime(str(start_ts_str), fmt)
                break
            except ValueError:
                pass

    duration_sec = sleep_data.get("sleepTimeSeconds", 0)
    if duration_sec == 0:
        duration_sec = int(sleep_data.get("durationInMs", 0) / 1000)

    n_epochs = max(1, duration_sec // EPOCH_SEC)

    hr_map = {}
    for entry in hr_data:
        t = entry.get("startGMT") or entry.get("timestamp", "")
        val = entry.get("respirationValue") or entry.get("heartRateValue")
        if t and val:
            hr_map[t[:16]] = float(val)

    def get_hr(epoch_ts):
        key = epoch_ts.strftime("%Y-%m-%dT%H:%M")
        return hr_map.get(key, 65.0)

    epochs = []
    for i in range(n_epochs):
        ts = start_ts + timedelta(seconds=i * EPOCH_SEC)
        hr = get_hr(ts)
        epochs.append({
            "timestamp":  ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hr_mean":    hr,
            "hr_std":     2.0,
            "hr_min":     hr * 0.94,
            "hr_max":     hr * 1.06,
            "hrv_rmssd":  30.0,
            "hrv_sdnn":   27.0,
            "hrv_pnn50":  0.0,
            "accel_mean": 0.01,
            "accel_std":  0.01,
            "accel_zcr":  0.0,
            "eda_mean":   0.0,
            "temp_mean":  0.0,
        })
    return epochs


# ── Public entry point ─────────────────────────────────────────────────────────

def parse(path: str) -> list[dict]:
    """
    Auto-detect Garmin export format (.fit or .json) and return epoch list.

    Returns:
        list of epoch dicts matching FEATURE_KEYS.
    """
    if path.endswith(".fit"):
        return _parse_fit(path)
    elif path.endswith(".json"):
        return _parse_json(path)
    else:
        raise ValueError(f"Unsupported Garmin file format: {path}. Expected .fit or .json")
