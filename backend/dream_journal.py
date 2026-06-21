"""
Dream journal storage and REM correlation via Claude.

Optionally cross-references the morning voice health check (fatigue/stress/cognitive
load scores) to give a complete picture: what you dreamed → when REM occurred →
how your voice sounded this morning → what it all means together.
"""

import json
import os
from typing import Optional

import anthropic


def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")
    return anthropic.Anthropic(api_key=api_key)


def _extract_rem_context(analysis: dict) -> dict:
    summary = analysis.get("sleep_summary", {})
    epochs  = analysis.get("epoch_by_epoch_data", [])
    pcts    = summary.get("pct_in_stage", {})

    rem_pct = next((v for k, v in pcts.items() if k.upper() == "REM"), 0.0)
    rem_min = next(
        (v for k, v in summary.get("time_in_stage_min", {}).items() if k.upper() == "REM"),
        0.0,
    )

    # Find consecutive REM runs in the epoch list
    rem_runs = []
    if epochs:
        rem_code = next(
            (ep.get("predicted_stage_code") for ep in epochs
             if ep.get("predicted_stage_name", "").upper() == "REM"),
            None,
        )
        if rem_code is not None:
            i = 0
            while i < len(epochs):
                if epochs[i].get("predicted_stage_code") == rem_code:
                    start = epochs[i].get("timestamp", "")
                    j = i
                    while j < len(epochs) and epochs[j].get("predicted_stage_code") == rem_code:
                        j += 1
                    rem_runs.append({
                        "start":        start,
                        "duration_min": round((j - i) * 0.5, 1),
                    })
                    i = j
                else:
                    i += 1

    return {
        "rem_pct":          rem_pct,
        "rem_min":          rem_min,
        "n_rem_runs":       len(rem_runs),
        "first_rem_at":     rem_runs[0]["start"] if rem_runs else "unknown",
        "last_rem_at":      rem_runs[-1]["start"] if rem_runs else "unknown",
        "longest_rem_min":  max((r["duration_min"] for r in rem_runs), default=0),
        "quality_score":    summary.get("quality_score", "?"),
        "efficiency_pct":   summary.get("efficiency_pct", "?"),
        "recording_start":  analysis.get("metadata", {}).get("recording_start", ""),
    }


def analyze_dream(dream_text: str, analysis: dict, voice_result: Optional[dict] = None) -> str:
    """
    Send dream description + REM context (+ optional voice biomarkers) to Claude.

    Parameters
    ----------
    dream_text   : what the user remembers dreaming
    analysis     : full sleep analysis dict from /api/analyze
    voice_result : optional voice-check result dict (scores + features) from that morning
    """
    rem = _extract_rem_context(analysis)

    rem_block = (
        f"REM sleep: {rem['rem_pct']}% of total recording ({rem['rem_min']} minutes)\n"
        f"Distinct REM periods: {rem['n_rem_runs']}\n"
        f"First REM began: {rem['first_rem_at']}\n"
        f"Longest single REM period: {rem['longest_rem_min']} minutes\n"
        f"Last REM period started: {rem['last_rem_at']}\n"
        f"Overall sleep quality: {rem['quality_score']}/100, efficiency {rem['efficiency_pct']}%"
    )

    voice_block = ""
    if voice_result and voice_result.get("scores"):
        scores = voice_result["scores"]
        feats  = voice_result.get("features", {})
        voice_block = (
            f"\nMorning voice health check (recorded after waking):\n"
            f"- Fatigue score: {scores.get('fatigue', '?')}/100\n"
            f"- Stress score: {scores.get('stress', '?')}/100\n"
            f"- Cognitive load: {scores.get('cognitive_load', '?')}/100\n"
            f"- Speaking rate: {feats.get('speaking_rate_wpm', '?')} WPM\n"
            f"- HNR (voice clarity): {feats.get('hnr_db', '?')} dB\n"
            f"- Jitter: {feats.get('jitter_pct', '?')}%"
        )

    voice_instruction = ""
    if voice_block:
        voice_instruction = (
            "3. Connect their morning voice biomarkers to the dream and REM data. "
            "High fatigue after vivid dreaming often means the REM was restorative but the "
            "brain is still consolidating — or that the REM was fragmented and didn't restore well. "
            "High stress in the voice after an emotional dream is physiologically expected. "
            "Be specific about which scores align or contradict expectations.\n"
        )
    else:
        voice_instruction = (
            "3. Offer one insight or takeaway about their REM health and what they "
            "could do to improve dream quality and REM depth.\n"
        )

    prompt = (
        f'The user described their dream:\n\n"{dream_text}"\n\n'
        f"Sleep data from last night:\n{rem_block}\n"
        f"{voice_block}\n\n"
        "Write a 2–3 paragraph response that:\n"
        "1. Comments on the dream's characteristics (vivid vs. mundane, emotional vs. neutral, "
        "narrative richness) and what those features correlate with in sleep science.\n"
        "2. Connects the dream to the actual REM data — timing, number of periods, longest run. "
        "Note whether the characteristics match expectations (e.g. vivid emotional dreams "
        "typically occur in the final, longest REM period of the night).\n"
        + voice_instruction +
        "Be warm, curious, and evidence-based. Avoid pseudo-scientific dream interpretation."
    )

    response = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=550,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def save_dream(session_id: str, dream_text: str, analysis_text: str) -> None:
    from backend import session as session_store
    payload = json.dumps({"dream": dream_text, "analysis": analysis_text})
    r = session_store._redis()
    if r:
        r.setex(f"dream:{session_id}", session_store.TTL, payload)
    else:
        session_store._memory_store[f"dream:{session_id}"] = payload


def load_dream(session_id: str) -> Optional[dict]:
    from backend import session as session_store
    r = session_store._redis()
    raw = (r.get(f"dream:{session_id}") if r
           else session_store._memory_store.get(f"dream:{session_id}"))
    return json.loads(raw) if raw else None
