"""
parsers/apple_health.py

Parses an Apple Health export.xml file.

How to export from iPhone:
    Health app → Profile picture → Export All Health Data → Share as .zip
    Unzip → export.xml

Extracts per-30s epochs covering the longest contiguous sleep window found,
with HR, HRV, SpO2, and movement (step count proxy).
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np


RECORD_TYPES = {
    "hr":   "HKQuantityTypeIdentifierHeartRate",
    "hrv":  "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "spo2": "HKQuantityTypeIdentifierOxygenSaturation",
    "steps":"HKQuantityTypeIdentifierStepCount",
    "sleep":"HKCategoryTypeIdentifierSleepAnalysis",
}

EPOCH_SEC = 30


def _parse_dt(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised datetime format: {s}")


def _load_records(xml_path: str) -> dict:
    """Stream-parse export.xml and bucket records by type."""
    data = defaultdict(list)
    context = ET.iterparse(xml_path, events=("end",))
    for _, elem in context:
        if elem.tag != "Record":
            elem.clear()
            continue
        rtype = elem.get("type", "")
        for key, hk_type in RECORD_TYPES.items():
            if rtype == hk_type:
                try:
                    start = _parse_dt(elem.get("startDate"))
                    end = _parse_dt(elem.get("endDate"))
                    value = elem.get("value", "0")
                    data[key].append((start, end, float(value)))
                except (ValueError, TypeError):
                    pass
        elem.clear()
    return data


def _find_sleep_window(sleep_records):
    """Return (window_start, window_end) for the longest sleep block."""
    if not sleep_records:
        return None, None
    sorted_recs = sorted(sleep_records, key=lambda r: r[0])
    best_start, best_end = sorted_recs[0][0], sorted_recs[0][1]
    cur_start, cur_end = best_start, best_end
    for start, end, _ in sorted_recs[1:]:
        if start <= cur_end + timedelta(minutes=10):
            cur_end = max(cur_end, end)
        else:
            if (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start, cur_end = start, end
    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end
    return best_start, best_end


def _bucket_to_epochs(records, window_start, n_epochs, agg="mean"):
    """Place point-in-time records into 30s epoch bins."""
    bins = [[] for _ in range(n_epochs)]
    for start, end, val in records:
        idx = int((start - window_start).total_seconds() / EPOCH_SEC)
        if 0 <= idx < n_epochs:
            bins[idx].append(val)
    result = []
    for b in bins:
        if b:
            result.append(float(np.mean(b)))
        else:
            result.append(None)
    return result


def _fill_none(values, default=0.0):
    """Forward/backward fill None values; use default if all None."""
    arr = list(values)
    last = default
    for i, v in enumerate(arr):
        if v is not None:
            last = v
        else:
            arr[i] = last
    return arr


def parse(xml_path: str) -> list[dict]:
    """
    Parse an Apple Health export.xml.

    Returns:
        list of epoch dicts with keys matching wrist_model/inference.py FEATURE_KEYS.
    """
    data = _load_records(xml_path)

    window_start, window_end = _find_sleep_window(data.get("sleep", []))
    if window_start is None:
        raise ValueError("No sleep records found in export.xml")

    n_epochs = max(1, int((window_end - window_start).total_seconds() / EPOCH_SEC))

    hr_bins   = _fill_none(_bucket_to_epochs(data.get("hr", []),    window_start, n_epochs), 65.0)
    hrv_bins  = _fill_none(_bucket_to_epochs(data.get("hrv", []),   window_start, n_epochs), 30.0)
    spo2_bins = _fill_none(_bucket_to_epochs(data.get("spo2", []),  window_start, n_epochs), 97.0)
    step_bins = _fill_none(_bucket_to_epochs(data.get("steps", []), window_start, n_epochs), 0.0)

    epochs = []
    for i in range(n_epochs):
        ts = window_start + timedelta(seconds=i * EPOCH_SEC)
        hr = hr_bins[i]
        epochs.append({
            "timestamp":  ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hr_mean":    hr,
            "hr_std":     2.0,     # Apple Health gives aggregated HR, no intra-epoch std
            "hr_min":     hr * 0.95,
            "hr_max":     hr * 1.05,
            "hrv_rmssd":  hrv_bins[i],
            "hrv_sdnn":   hrv_bins[i] * 0.9,
            "hrv_pnn50":  0.0,
            "accel_mean": step_bins[i] / 30.0,  # rough activity proxy
            "accel_std":  0.1,
            "accel_zcr":  0.0,
            "eda_mean":   0.0,
            "temp_mean":  0.0,
            "spo2":       spo2_bins[i],
        })
    return epochs
