"""
Rule-based sleep disorder risk screener.
Runs on the sleep_summary dict returned by the inference pipeline.
Returns a list of risk flags — empty list means no concerns flagged.

Thresholds are conservative (favour recall over precision) and include
"talk to a doctor" disclaimers. This is a screener, not a diagnosis.
"""

_NORMS = {
    "rem_ideal_pct":   20.0,
    "deep_ideal_pct":  15.0,
    "efficiency_good": 85.0,
    "latency_ok_min":  20.0,
    "latency_bad_min": 30.0,
}


def screen(summary: dict) -> list:
    flags = []

    efficiency  = float(summary.get("efficiency_pct", 100))
    latency     = float(summary.get("latency_min", 0))
    awakenings  = int(summary.get("awakenings", 0))
    total_sleep = float(summary.get("total_sleep_min", 480))
    pcts        = summary.get("pct_in_stage", {})

    rem_pct  = next((v for k, v in pcts.items() if k.upper() == "REM"), 0.0)
    deep_pct = next((v for k, v in pcts.items() if k.upper() in ("N3", "DEEP")), 0.0)

    # ── Sleep Apnea proxy ────────────────────────────────────────────────────
    if awakenings >= 5:
        flags.append({
            "condition":      "Sleep Apnea Risk",
            "risk_level":     "high",
            "indicators":     [
                f"{awakenings} awakenings (normal: ≤2)",
                f"{efficiency}% sleep efficiency",
            ],
            "recommendation": (
                "Frequent awakenings with low efficiency are associated with "
                "sleep-disordered breathing. If you snore, gasp, or feel "
                "unrefreshed after a full night, speak with a doctor about "
                "a formal sleep study (polysomnography)."
            ),
        })
    elif awakenings >= 3 and efficiency < 80:
        flags.append({
            "condition":      "Sleep Apnea Risk",
            "risk_level":     "moderate",
            "indicators":     [
                f"{awakenings} awakenings",
                f"{efficiency}% sleep efficiency",
            ],
            "recommendation": (
                "Mildly fragmented sleep. Common causes include sleep apnea, "
                "restless legs, or environmental disruption. Track this over "
                "several nights — if it persists, discuss with your doctor."
            ),
        })

    # ── Insomnia ─────────────────────────────────────────────────────────────
    if latency > _NORMS["latency_bad_min"] and efficiency < _NORMS["efficiency_good"]:
        flags.append({
            "condition":      "Insomnia",
            "risk_level":     "high",
            "indicators":     [
                f"{latency} min to fall asleep (normal: <20 min)",
                f"{efficiency}% sleep efficiency (normal: >85%)",
            ],
            "recommendation": (
                "Long sleep onset combined with poor efficiency is a hallmark "
                "of insomnia disorder. Cognitive Behavioural Therapy for "
                "Insomnia (CBT-I) is the first-line evidence-based treatment — "
                "more effective than medication long-term."
            ),
        })
    elif latency > _NORMS["latency_ok_min"] or efficiency < _NORMS["efficiency_good"]:
        flags.append({
            "condition":      "Insomnia",
            "risk_level":     "moderate",
            "indicators":     [
                f"{latency} min to fall asleep",
                f"{efficiency}% sleep efficiency",
            ],
            "recommendation": (
                "Mild difficulty falling or staying asleep. Consistent "
                "sleep/wake times, avoiding screens 1 hour before bed, and "
                "keeping the bedroom cool (65–68°F) are the highest-evidence "
                "behavioural changes."
            ),
        })

    # ── REM Suppression ───────────────────────────────────────────────────────
    if rem_pct > 0:
        if rem_pct < 8:
            flags.append({
                "condition":      "REM Suppression",
                "risk_level":     "high",
                "indicators":     [f"{rem_pct}% REM sleep (normal: 20–25%)"],
                "recommendation": (
                    "Very low REM is commonly caused by alcohol within 3 hours "
                    "of bedtime, certain antidepressants (SSRIs, SNRIs), "
                    "benzodiazepines, or untreated sleep apnea. It affects "
                    "memory consolidation and emotional regulation. Worth "
                    "discussing with your doctor."
                ),
            })
        elif rem_pct < 15:
            flags.append({
                "condition":      "REM Suppression",
                "risk_level":     "moderate",
                "indicators":     [f"{rem_pct}% REM sleep (normal: 20–25%)"],
                "recommendation": (
                    "Below-average REM. Most common cause: alcohol or cannabis "
                    "within a few hours of bedtime. Also affected by irregular "
                    "sleep schedules — REM is disproportionately concentrated "
                    "in the last third of a full night."
                ),
            })

    # ── Short Sleep ───────────────────────────────────────────────────────────
    if total_sleep < 300:
        flags.append({
            "condition":      "Short Sleep",
            "risk_level":     "high",
            "indicators":     [f"{round(total_sleep / 60, 1)} hours total sleep"],
            "recommendation": (
                "Less than 5 hours of sleep significantly impairs cognitive "
                "function, immune response, and metabolic health. The CDC "
                "recommends 7–9 hours for adults. Chronic short sleep is "
                "associated with increased cardiovascular and diabetes risk."
            ),
        })
    elif total_sleep < 360:
        flags.append({
            "condition":      "Short Sleep",
            "risk_level":     "moderate",
            "indicators":     [f"{round(total_sleep / 60, 1)} hours total sleep"],
            "recommendation": (
                "Slightly below the recommended 7–9 hours. Even one hour of "
                "sleep debt measurably reduces next-day attention and reaction "
                "time. Prioritise an earlier bedtime over a later wake time."
            ),
        })

    return flags
