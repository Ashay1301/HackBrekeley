"""
Claude-powered sleep analysis chatbot.
"""

import json
import os

import anthropic

CLIENT = None


def _client() -> anthropic.Anthropic:
    global CLIENT
    if CLIENT is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
        CLIENT = anthropic.Anthropic(api_key=api_key)
    return CLIENT


def _system_prompt(analysis: dict) -> str:
    summary = analysis.get("sleep_summary", {})
    meta    = analysis.get("metadata", {})
    atype   = meta.get("analysis_type", "unknown")

    stage_times = summary.get("time_in_stage_min", {})
    stage_pcts  = summary.get("pct_in_stage", {})

    context = f"""You are a friendly, knowledgeable sleep health analyst. You help users understand
their sleep data clearly and actionably. Never diagnose medical conditions, but explain
patterns and offer evidence-based lifestyle recommendations.

The user's sleep data ({atype.upper()} analysis):
- Quality score: {summary.get('quality_score', 'N/A')}/100 (grade {summary.get('quality_grade', '?')})
- Sleep efficiency: {summary.get('efficiency_pct', 'N/A')}%
- Sleep latency: {summary.get('latency_min', 'N/A')} minutes
- Awakenings: {summary.get('awakenings', 'N/A')}
- Total recording: {summary.get('total_recording_min', 'N/A')} minutes
- Total sleep: {summary.get('total_sleep_min', 'N/A')} minutes
- Time in each stage (minutes): {json.dumps(stage_times)}
- Stage percentages: {json.dumps(stage_pcts)}

Keep answers concise (2–4 paragraphs max). Use plain language. If a pattern might indicate
a sleep disorder, note it gently and suggest seeing a doctor — do not diagnose."""

    return context


def chat(message: str, analysis: dict) -> str:
    """Send a user message with the analysis as context; return the assistant reply."""
    response = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=_system_prompt(analysis),
        messages=[{"role": "user", "content": message}],
    )
    return response.content[0].text


def generate_smart_questions(analysis: dict) -> list[str]:
    """Generate 3 questions tailored to this user's specific sleep data."""
    summary = analysis.get("sleep_summary", {})
    pcts    = summary.get("pct_in_stage", {})
    score   = summary.get("quality_score", 50)
    lat     = summary.get("latency_min", 0)
    awk     = summary.get("awakenings", 0)

    questions = []

    # Question 1: always about overall quality
    if score < 55:
        questions.append("My sleep quality score is low — what are the biggest factors dragging it down?")
    elif score < 75:
        questions.append("My sleep score is okay but not great — what would push it to excellent?")
    else:
        questions.append("My sleep looks good overall — any tips to maintain this consistently?")

    # Question 2: about a specific metric that stands out
    rem_pct = next((v for k, v in pcts.items() if "rem" in k.lower() or k.upper() == "REM"), 0)
    deep_pct = next((v for k, v in pcts.items() if "n3" in k.lower() or "deep" in k.lower()), 0)
    if rem_pct < 15:
        questions.append(f"My REM sleep was only {rem_pct}% — is that low, and why does it matter?")
    elif deep_pct < 10:
        questions.append(f"My deep (N3) sleep was only {deep_pct}% — what affects deep sleep quality?")
    elif awk > 3:
        questions.append(f"I had {awk} awakenings during the night — what could be causing this?")
    else:
        questions.append("What's the ideal breakdown of sleep stages for someone my age?")

    # Question 3: actionable
    if lat > 20:
        questions.append(f"It took me {lat} minutes to fall asleep — what are the best evidence-based ways to reduce sleep latency?")
    else:
        questions.append("What time should I go to bed tonight to optimise my sleep?")

    return questions


def generate_weekly_summary(nights: list) -> str:
    rows = []
    for n in sorted(nights, key=lambda x: x.get("date", "")):
        rows.append(
            f"  {n.get('date','?')}: score={n.get('quality_score','?')},"
            f" efficiency={n.get('efficiency_pct','?')}%,"
            f" latency={n.get('latency_min','?')}min,"
            f" awakenings={n.get('awakenings','?')},"
            f" REM={n.get('rem_pct','?')}%"
        )
    prompt = (
        "Here is the user's sleep data over the past several nights:\n\n"
        + "\n".join(rows)
        + "\n\nWrite a 2–3 paragraph narrative weekly sleep summary. "
        "Cover: the overall trend (improving, declining, or stable), "
        "the best and worst night and what might explain the difference, "
        "and one specific, actionable recommendation. "
        "Be warm, specific, and evidence-based. Plain language only."
    )
    response = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
