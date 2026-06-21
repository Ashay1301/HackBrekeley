"""
Psychiatric risk screener — combines sleep biomarkers, voice biomarkers,
and multi-night trend to produce an evidence-informed risk indicator.

This is a WELLNESS TOOL, not a clinical diagnostic instrument.
Signals are derived from published correlations between acoustic/sleep
features and mood disorder risk; they are NOT equivalent to a PHQ-9 or
clinical assessment.

Risk levels: none | low | moderate | high
Each triggered signal = 1 point toward the composite score.
"""

from typing import Optional


_AASM = {
    "efficiency_good":  85.0,
    "rem_ideal_pct":    20.0,
    "rem_low_pct":      15.0,
    "latency_ok_min":   20.0,
    "latency_bad_min":  30.0,
    "awakenings_ok":    2,
    "total_sleep_ok_min": 360,   # 6 hours
}

_VOICE_THRESHOLDS = {
    "fatigue_high":       65,
    "cognitive_load_high": 65,
    "speaking_rate_low":  100,   # WPM — bradyphrenia marker
    "hnr_low_db":         10.0,
    "f0_low_hz":          120.0, # rough cross-gender floor; below this → vocal monotony
}


def _level(score: int) -> str:
    if score == 0:
        return "none"
    if score <= 2:
        return "low"
    if score <= 4:
        return "moderate"
    return "high"


def _build_baseline(history_nights: list) -> dict:
    """
    Return per-metric baseline values.
    Uses personal rolling average when ≥3 nights available,
    otherwise returns AASM population norms.
    """
    if len(history_nights) >= 3:
        def _avg(key, sub=None):
            vals = []
            for n in history_nights:
                v = n.get(key) if sub is None else (n.get(sub) or {}).get(key)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            return round(sum(vals) / len(vals), 1) if vals else None

        return {
            "source":       "personal_7night",
            "efficiency":   _avg("efficiency_pct") or _AASM["efficiency_good"],
            "rem_pct":      _avg("rem_pct")         or _AASM["rem_ideal_pct"],
            "latency":      _avg("latency_min")     or _AASM["latency_ok_min"],
            "awakenings":   _avg("awakenings")      or float(_AASM["awakenings_ok"]),
            "quality":      _avg("quality_score"),
        }

    return {
        "source":     "aasm_norms",
        "efficiency": _AASM["efficiency_good"],
        "rem_pct":    _AASM["rem_ideal_pct"],
        "latency":    _AASM["latency_ok_min"],
        "awakenings": float(_AASM["awakenings_ok"]),
        "quality":    None,
    }


def _trend_declining(history_nights: list) -> bool:
    """True if quality score declined for 3+ consecutive nights."""
    scores = []
    for n in sorted(history_nights, key=lambda x: x.get("date", "")):
        q = n.get("quality_score")
        if q is not None:
            try:
                scores.append(float(q))
            except (TypeError, ValueError):
                pass
    if len(scores) < 3:
        return False
    # Check last 3 entries
    tail = scores[-3:]
    return tail[0] > tail[1] > tail[2]


def compute_risk(
    sleep_summary: dict,
    voice_result: Optional[dict],
    history_nights: list,
) -> dict:
    """
    Compute psychiatric risk indicators from sleep + voice + trend data.

    Parameters
    ----------
    sleep_summary   : dict from sleep analysis (quality_score, efficiency_pct, etc.)
    voice_result    : dict from /api/voice-check (scores + features), or None
    history_nights  : list of lightweight night dicts from /api/history

    Returns
    -------
    dict with risk_level, risk_score, signals, baseline, recommendation, wellness_note
    """
    signals: list[str] = []
    baseline = _build_baseline(history_nights)

    efficiency  = float(sleep_summary.get("efficiency_pct", 100))
    latency     = float(sleep_summary.get("latency_min", 0))
    awakenings  = int(sleep_summary.get("awakenings", 0))
    total_sleep = float(sleep_summary.get("total_sleep_min", 480))
    pcts        = sleep_summary.get("pct_in_stage", {})
    rem_pct     = next((float(v) for k, v in pcts.items() if k.upper() == "REM"), 0.0)

    # ── Sleep signals ────────────────────────────────────────────────────────
    if rem_pct > 0 and rem_pct < _AASM["rem_low_pct"]:
        signals.append(f"REM sleep low ({rem_pct:.0f}% vs normal 20–25%) — associated with depression and mood disorders")

    if efficiency < _AASM["efficiency_good"]:
        signals.append(f"Sleep efficiency reduced ({efficiency:.0f}% vs normal >85%)")

    if latency > _AASM["latency_bad_min"]:
        signals.append(f"Sleep onset delayed ({latency:.0f} min) — onset insomnia correlates with anxiety/depression")

    if awakenings >= _AASM["awakenings_ok"] + 1:
        signals.append(f"Fragmented sleep ({awakenings} awakenings) — night-time waking pattern seen in mood disorders")

    if total_sleep < _AASM["total_sleep_ok_min"]:
        signals.append(f"Short sleep ({total_sleep/60:.1f}h) — chronic sleep restriction is a depression risk factor")

    if _trend_declining(history_nights):
        signals.append("Sleep quality declining over 3+ consecutive nights — sustained dip may indicate early mood episode")

    # ── Baseline deviation flags (personal history only) ─────────────────────
    if baseline["source"] == "personal_7night":
        def _flag_below(current, base_val, label, threshold_pct=20):
            if base_val and base_val > 0:
                delta = (base_val - current) / base_val * 100
                if delta > threshold_pct:
                    signals.append(
                        f"{label} dropped {delta:.0f}% below your 7-night average "
                        f"({current:.0f} vs baseline {base_val:.0f})"
                    )

        _flag_below(efficiency,  baseline["efficiency"], "Sleep efficiency")
        if rem_pct > 0:
            _flag_below(rem_pct, baseline["rem_pct"], "REM sleep %")

    # ── Voice signals ────────────────────────────────────────────────────────
    if voice_result:
        scores   = voice_result.get("scores", {})
        features = voice_result.get("features", {})

        fatigue       = scores.get("fatigue", 0)
        cognitive     = scores.get("cognitive_load", 0)
        speaking_rate = features.get("speaking_rate_wpm", 0)
        hnr           = features.get("hnr_db", 20)
        f0_mean       = features.get("f0_mean_hz", 0)

        if fatigue > _VOICE_THRESHOLDS["fatigue_high"]:
            signals.append(
                f"Voice fatigue elevated ({fatigue}/100) — acoustic pattern consistent with psychomotor slowing"
            )

        if cognitive > _VOICE_THRESHOLDS["cognitive_load_high"]:
            signals.append(
                f"Cognitive load indicator high ({cognitive}/100) — reduced verbal fluency is associated with mood episodes"
            )

        if speaking_rate > 0 and speaking_rate < _VOICE_THRESHOLDS["speaking_rate_low"]:
            signals.append(
                f"Speaking rate slow ({speaking_rate:.0f} WPM vs normal 130–160) — bradyphrenia is a validated depression marker"
            )

        if hnr < _VOICE_THRESHOLDS["hnr_low_db"]:
            signals.append(
                f"Voice clarity reduced (HNR {hnr:.1f} dB) — roughness/breathiness pattern associated with fatigue and low mood"
            )

        if f0_mean > 0 and f0_mean < _VOICE_THRESHOLDS["f0_low_hz"]:
            signals.append(
                f"Vocal pitch low ({f0_mean:.0f} Hz) — reduced fundamental frequency correlates with flat affect in depression"
            )

    # ── Score + level ────────────────────────────────────────────────────────
    score = len(signals)
    level = _level(score)

    # ── Narrative ────────────────────────────────────────────────────────────
    if level == "none":
        recommendation = "No psychiatric risk signals detected. Sleep and voice biomarkers are within expected ranges."
        wellness_note  = "Your sleep and voice patterns look healthy. Keep up the good work."

    elif level == "low":
        recommendation = (
            f"{score} mild signal(s) noted. Monitor over the next 5–7 nights. "
            "If signals persist or accumulate, consider a PHQ-2 screening conversation."
        )
        wellness_note = (
            "A couple of minor wellness signals were detected. "
            "These can reflect temporary stress or poor sleep hygiene rather than a clinical concern. "
            "If you notice these patterns persisting, it's worth a conversation with your doctor."
        )

    elif level == "moderate":
        recommendation = (
            f"{score} moderate signals detected across sleep and/or voice biomarkers. "
            "Recommend PHQ-9 screening at next visit. "
            "Key indicators: " + "; ".join(signals[:3]) + "."
        )
        wellness_note = (
            "Your data shows several patterns worth discussing with a healthcare provider. "
            "This doesn't mean something is wrong — but a check-in could be helpful, "
            "especially if you've been feeling low or overwhelmed lately."
        )

    else:  # high
        recommendation = (
            f"{score} high-risk signals detected. Immediate follow-up recommended. "
            "Sleep fragmentation, voice biomarker changes, and/or multi-night trend decline "
            "are consistent with a mood episode or significant psychiatric distress. "
            "Administer PHQ-9 and consider referral."
        )
        wellness_note = (
            "Your data shows a pattern that your doctor should know about. "
            "Please consider reaching out to a healthcare provider or a trusted person soon. "
            "If you're in distress, the 988 Suicide & Crisis Lifeline is available 24/7."
        )

    return {
        "risk_level":      level,
        "risk_score":      score,
        "signals":         signals,
        "baseline":        baseline,
        "recommendation":  recommendation,
        "wellness_note":   wellness_note,
    }
