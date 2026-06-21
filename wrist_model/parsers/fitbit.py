"""
parsers/fitbit.py

Parses a Fitbit data export JSON file.

How to export from Fitbit:
    fitbit.com → Settings → Data Export → Request Data → Download zip
    Relevant files:
      sleep/sleep-YYYY-MM-DD.json       (sleep stage timeline)
      heart-rate/heart_rate-YYYY-MM-DD.json  (per-minute HR)

Pass the path to a single sleep JSON or a directory containing multiple nights.
"""

import json
import os
import glob
from datetime import datetime, timedelta
import numpy as np

EPOCH_SEC = 30

STAGE_MAP = {
    "wake": 0, "restless": 0,
    "light": 1, "deep": 2, "rem": 3,
}


def _parse_dt(s: str) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised datetime: {s}")


def _load_hr_for_date(hr_dir: str, date_str: str) -> dict:
    """Load per-minute HR from heart_rate-YYYY-MM-DD.json for a given date."""
    hr_path = os.path.join(hr_dir, f"heart_rate-{date_str}.json")
    if not os.path.exists(hr_path):
        return {}
    with open(hr_path) as f:
        data = json.load(f)
    minute_hr = {}
    for entry in data.get("activities-heart-intraday", {}).get("dataset", []):
        try:
            t = datetime.strptime(f"{date_str} {entry['time']}", "%Y-%m-%d %H:%M:%S")
            minute_hr[t] = float(entry["value"])
        except (KeyError, ValueError):
            pass
    return minute_hr


def _expand_sleep_stages(levels_data: list) -> list:
    """Expand Fitbit levels data into per-30s epoch stage codes."""
    epochs = []
    for segment in levels_data:
        try:
            seg_start = _parse_dt(segment["dateTime"])
            duration_sec = int(segment["seconds"])
            stage_name = segment["level"].lower()
            stage_code = STAGE_MAP.get(stage_name, 0)
            n = max(1, int(np.ceil(duration_sec / EPOCH_SEC)))
            for i in range(n):
                epochs.append((seg_start + timedelta(seconds=i * EPOCH_SEC), stage_code))
        except (KeyError, ValueError):
            pass
    return epochs


def _get_hr_for_epoch(ts: datetime, minute_hr: dict, fallback=65.0) -> float:
    """Return HR for the minute containing ts."""
    t_min = ts.replace(second=0, microsecond=0)
    return minute_hr.get(t_min, fallback)


def parse(path: str, hr_dir: str = None) -> list[dict]:
    """
    Parse Fitbit sleep JSON (single file or directory of nightly files).

    Args:
        path:   Path to a sleep-YYYY-MM-DD.json file or directory of them.
        hr_dir: Optional directory containing heart_rate-YYYY-MM-DD.json files.
                If None, HR values default to 65 bpm.

    Returns:
        list of epoch dicts matching FEATURE_KEYS.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "sleep-*.json")))
        if not files:
            raise FileNotFoundError(f"No sleep-*.json files found in {path}")
        # Use the file covering the longest sleep session
        best_file, best_dur = files[0], 0
        for f in files:
            with open(f) as fp:
                d = json.load(fp)
            dur = sum(s.get("minutesAsleep", 0) for s in d if isinstance(s, dict))
            if dur > best_dur:
                best_file, best_dur = f, dur
        path = best_file

    with open(path) as f:
        raw = json.load(f)

    # Fitbit export can be a list or a dict with "sleep" key
    sessions = raw if isinstance(raw, list) else raw.get("sleep", [raw])
    # Pick the main sleep session (isMainSleep=true or longest)
    main = next((s for s in sessions if s.get("isMainSleep")), None)
    if main is None and sessions:
        main = max(sessions, key=lambda s: s.get("minutesAsleep", 0))
    if main is None:
        raise ValueError("No valid sleep session found in Fitbit export.")

    start_str = main.get("startTime", "")
    date_str = start_str[:10]
    minute_hr = {}
    if hr_dir:
        minute_hr = _load_hr_for_date(hr_dir, date_str)

    levels_data = main.get("levels", {}).get("data", [])
    if not levels_data:
        raise ValueError("No levels/data in Fitbit sleep JSON. Export may be classic (non-stage) format.")

    stage_epochs = _expand_sleep_stages(levels_data)
    if not stage_epochs:
        raise ValueError("Could not expand sleep stage data into epochs.")

    epochs = []
    for ts, stage_code in stage_epochs:
        hr = _get_hr_for_epoch(ts, minute_hr)
        epochs.append({
            "timestamp":  ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hr_mean":    hr,
            "hr_std":     2.5,
            "hr_min":     hr * 0.93,
            "hr_max":     hr * 1.07,
            "hrv_rmssd":  30.0,
            "hrv_sdnn":   27.0,
            "hrv_pnn50":  0.0,
            "accel_mean": 0.05 if stage_code == 0 else 0.01,
            "accel_std":  0.02,
            "accel_zcr":  0.0,
            "eda_mean":   0.0,
            "temp_mean":  0.0,
            # Device-reported stage for reference (not used by model)
            "_device_stage": stage_code,
        })
    return epochs
