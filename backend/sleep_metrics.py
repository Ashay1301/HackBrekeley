"""
AASM-correct sleep metric calculations.
Used by both EEG and wrist inference pipelines.
"""

import numpy as np
from datetime import datetime, timedelta


# ── Metric helpers ──────────────────────────────────────────────────────────────

def sleep_latency(predictions: np.ndarray, epoch_sec: int = 30) -> float:
    """Time (minutes) from recording start to first 3 consecutive non-Wake epochs (AASM)."""
    preds = list(predictions)
    n = len(preds)
    for i in range(n - 2):
        if preds[i] != 0 and preds[i + 1] != 0 and preds[i + 2] != 0:
            return round((i * epoch_sec) / 60.0, 1)
    return round((n * epoch_sec) / 60.0, 1)


def count_awakenings(predictions: np.ndarray, onset_idx: int) -> int:
    """
    Awakenings after sleep onset: W→non-W transitions where W lasts ≥2 epochs (AASM).
    """
    preds = list(predictions)
    awakenings = 0
    i = onset_idx + 1
    while i < len(preds):
        if preds[i] == 0:
            # Count consecutive wake epochs
            j = i
            while j < len(preds) and preds[j] == 0:
                j += 1
            wake_run = j - i
            if wake_run >= 2 and j < len(preds):
                awakenings += 1
            i = j
        else:
            i += 1
    return awakenings


def quality_score(efficiency: float, latency_min: float, rem_pct: float,
                  deep_pct: float, awakenings: int) -> int:
    """
    0–100 sleep quality score (higher = better).
    Weights: efficiency 40%, latency 25%, REM 20%, Deep 15%.
    Penalises excessive awakenings (−3 pts each beyond 1).
    """
    eff_score   = min(efficiency, 100) * 0.40
    lat_score   = max(0, (30 - min(latency_min, 60)) / 30) * 100 * 0.25
    rem_score   = min(rem_pct / 25 * 100, 100) * 0.20   # ideal ≥25%
    deep_score  = min(deep_pct / 20 * 100, 100) * 0.15  # ideal ≥20%
    penalty     = max(0, awakenings - 1) * 3
    return max(0, min(100, round(eff_score + lat_score + rem_score + deep_score - penalty)))


def grade(score: int) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    return "D"


# ── Full analysis ───────────────────────────────────────────────────────────────

def analyze_predictions(
    predictions: np.ndarray,
    probs: np.ndarray,
    start_dt,
    epoch_sec: int = 30,
    class_names: dict = None,
) -> dict:
    """
    Build the full sleep report dict from raw model output.

    predictions : (n,)  integer class indices
    probs       : (n, n_classes)  softmax probabilities
    start_dt    : datetime of recording start
    class_names : {0: "W", 1: "N1", ...} or {0: "Wake", ...}
    """
    if class_names is None:
        class_names = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    n = len(predictions)
    n_classes = probs.shape[1]

    # ── Onset index
    onset_idx = None
    p = list(predictions)
    for i in range(n - 2):
        if p[i] != 0 and p[i + 1] != 0 and p[i + 2] != 0:
            onset_idx = i
            break

    lat_min = round((onset_idx * epoch_sec) / 60.0, 1) if onset_idx is not None else round((n * epoch_sec) / 60.0, 1)
    awakenings = count_awakenings(predictions, onset_idx) if onset_idx is not None else 0

    # ── Stage counts & times
    counts = {v: int(np.sum(predictions == v)) for v in range(n_classes)}
    times_min = {class_names[v]: round(counts[v] * epoch_sec / 60.0, 1) for v in range(n_classes)}
    pcts = {class_names[v]: round(counts[v] / n * 100, 1) for v in range(n_classes)}

    # ── Sleep efficiency
    total_sleep_epochs = sum(counts[v] for v in range(n_classes) if v != 0)
    efficiency = round(total_sleep_epochs / n * 100, 1)

    # ── Quality score
    rem_key = next((k for k, v in class_names.items() if "REM" in v.upper() or v.upper() == "REM"), None)
    deep_key = next((k for k, v in class_names.items() if "N3" in v.upper() or "DEEP" in v.upper()), None)
    rem_pct_val  = pcts.get(class_names[rem_key], 0.0) if rem_key is not None else 0.0
    deep_pct_val = pcts.get(class_names[deep_key], 0.0) if deep_key is not None else 0.0
    q_score = quality_score(efficiency, lat_min, rem_pct_val, deep_pct_val, awakenings)

    # ── Epoch-by-epoch
    epoch_details = []
    for i, (pred, prob_row) in enumerate(zip(predictions, probs)):
        ts = start_dt + timedelta(seconds=i * epoch_sec)
        epoch_details.append({
            "epoch_number":       i + 1,
            "timestamp":          ts.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_stage_code": int(pred),
            "predicted_stage_name": class_names.get(int(pred), "Unknown"),
            "confidence":         round(float(np.max(prob_row)), 3),
            "probabilities": {
                class_names.get(j, str(j)): round(float(prob_row[j]), 3)
                for j in range(n_classes)
            },
        })

    return {
        "sleep_summary": {
            "quality_score":           q_score,
            "quality_grade":           grade(q_score),
            "efficiency_pct":          efficiency,
            "latency_min":             lat_min,
            "awakenings":              awakenings,
            "total_recording_min":     round(n * epoch_sec / 60.0, 1),
            "total_sleep_min":         round(total_sleep_epochs * epoch_sec / 60.0, 1),
            "time_in_stage_min":       times_min,
            "pct_in_stage":            pcts,
        },
        "epoch_by_epoch_data": epoch_details,
    }
