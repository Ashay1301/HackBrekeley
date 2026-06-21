"""
Autonomous sleep monitoring agent (Fetch AI uAgents).

Run:  bin/python agents/sleep_monitor.py

Every CHECK_INTERVAL_SECONDS (default 86400 = 24 h) the agent:
  1. Loads the last 8 sleep sessions from Redis
  2. Compares tonight vs the 7-night rolling baseline
  3. Sends a SleepAlert if any metric deviates > 1.5 SD in a bad direction

Register on Agentverse to make the agent discoverable:
  https://agentverse.ai
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from uagents import Agent, Context, Model


# ── Message models ───────────────────────────────────────────────────────────

class SleepAlert(Model):
    user_id:     str
    date:        str
    metric:      str
    tonight_val: float
    baseline:    float
    deviation:   float
    message:     str


class SleepStatusRequest(Model):
    user_id: str


class SleepStatusResponse(Model):
    session_id:    str
    quality_score: int
    quality_grade: str
    efficiency:    float
    summary:       str


# ── Agent definition ─────────────────────────────────────────────────────────

AGENT_SEED     = os.environ.get("AGENT_SEED", "sleepsense_monitor_default_seed_2026")
AGENT_PORT     = int(os.environ.get("AGENT_PORT", 8001))
ALERT_ADDRESS  = os.environ.get("ALERT_AGENT_ADDRESS", "")
USER_ID        = os.environ.get("SLEEPSENSE_USER_ID", "default")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL_SECONDS", 86400))

agent = Agent(
    name="sleepsense-monitor",
    seed=AGENT_SEED,
    port=AGENT_PORT,
    endpoint=[f"http://localhost:{AGENT_PORT}/submit"],
)


# ── Redis helpers ────────────────────────────────────────────────────────────

def _get_redis():
    url = os.environ.get("REDIS_URL") or os.environ.get("REDIS_PRIVATE_URL")
    if not url:
        return None
    import redis as redis_lib
    return redis_lib.from_url(url, decode_responses=True)


def _load_recent_sessions(n: int = 8) -> list:
    r = _get_redis()
    if not r:
        return []

    session_ids = r.zrevrange("sessions:index", 0, n - 1)
    sessions = []
    for sid in session_ids:
        raw = r.get(f"analysis:{sid}")
        if not raw:
            continue
        data    = json.loads(raw)
        summary = data.get("sleep_summary", {})
        meta    = data.get("metadata", {})
        rem_pct = summary.get("pct_in_stage", {})
        sessions.append({
            "session_id":    sid,
            "date":          str(meta.get("recording_start", ""))[:10],
            "quality_score": float(summary.get("quality_score", 0)),
            "efficiency":    float(summary.get("efficiency_pct", 0)),
            "rem_pct":       float(rem_pct.get("REM", 0) if isinstance(rem_pct, dict) else 0),
            "awakenings":    float(summary.get("awakenings", 0)),
            "latency_min":   float(summary.get("latency_min", 0)),
        })
    return sessions


# ── Anomaly detection ────────────────────────────────────────────────────────

def _detect_anomalies(sessions: list) -> list:
    """
    Compare tonight (sessions[0]) against 7-night baseline (sessions[1:]).
    Returns anomalies for metrics deviating > 1.5 SD in a bad direction.
    """
    import numpy as np
    if len(sessions) < 3:
        return []

    tonight  = sessions[0]
    baseline = sessions[1:]
    metrics  = ["quality_score", "efficiency", "rem_pct", "awakenings", "latency_min"]
    anomalies = []

    for m in metrics:
        vals = [s[m] for s in baseline if s.get(m) is not None]
        if not vals:
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        if std < 1e-6:
            continue

        tonight_val = float(tonight.get(m, mean))
        z = (tonight_val - mean) / std

        bad = (
            (m in ("quality_score", "efficiency", "rem_pct") and z < -1.5)
            or (m in ("awakenings", "latency_min") and z > 1.5)
        )
        if bad:
            anomalies.append({
                "metric":      m,
                "tonight_val": round(tonight_val, 1),
                "baseline":    round(mean, 1),
                "deviation":   round(z, 2),
            })

    return anomalies


def _write_alert_message(anomalies: list, tonight: dict) -> str:
    label = {
        "quality_score": "Quality score",
        "efficiency":    "Sleep efficiency",
        "rem_pct":       "REM sleep",
        "awakenings":    "Awakenings",
        "latency_min":   "Sleep latency",
    }
    lines = [
        f"SleepSense Alert — {tonight.get('date', 'today')}",
        f"Quality score tonight: {tonight.get('quality_score', '?')}",
        "Anomalies vs. 7-night baseline:",
    ]
    for a in anomalies:
        m = a["metric"]
        lines.append(
            f"  • {label.get(m, m)}: {a['tonight_val']} "
            f"(baseline {a['baseline']}, {abs(a['deviation']):.1f} SD)"
        )
    return "\n".join(lines)


# ── Agent behaviour ──────────────────────────────────────────────────────────

@agent.on_interval(period=float(CHECK_INTERVAL))
async def daily_sleep_check(ctx: Context):
    ctx.logger.info("Running daily sleep check...")
    sessions = _load_recent_sessions(n=8)

    if not sessions:
        ctx.logger.info("No sessions in Redis yet. Waiting for first analysis.")
        return

    tonight   = sessions[0]
    anomalies = _detect_anomalies(sessions)

    if not anomalies:
        ctx.logger.info(
            f"Sleep check OK — score {tonight.get('quality_score')}, "
            "no anomalies vs. 7-night baseline."
        )
        return

    message = _write_alert_message(anomalies, tonight)
    ctx.logger.warning(f"Anomalies detected:\n{message}")

    if ALERT_ADDRESS:
        alert = SleepAlert(
            user_id     = USER_ID,
            date        = tonight.get("date", ""),
            metric      = anomalies[0]["metric"],
            tonight_val = anomalies[0]["tonight_val"],
            baseline    = anomalies[0]["baseline"],
            deviation   = anomalies[0]["deviation"],
            message     = message,
        )
        await ctx.send(ALERT_ADDRESS, alert)
        ctx.logger.info(f"Alert sent to {ALERT_ADDRESS}")


@agent.on_message(model=SleepStatusRequest)
async def handle_status_request(ctx: Context, sender: str, msg: SleepStatusRequest):
    """Respond to another agent asking for the latest sleep status."""
    sessions = _load_recent_sessions(n=1)
    if not sessions:
        await ctx.send(sender, SleepStatusResponse(
            session_id="", quality_score=0, quality_grade="?",
            efficiency=0.0, summary="No sleep data recorded yet.",
        ))
        return

    latest = sessions[0]
    score  = int(latest["quality_score"])
    grade  = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
    await ctx.send(sender, SleepStatusResponse(
        session_id    = latest["session_id"],
        quality_score = score,
        quality_grade = grade,
        efficiency    = latest["efficiency"],
        summary       = (
            f"Last night: score {score}, efficiency {latest['efficiency']}%, "
            f"{int(latest['awakenings'])} awakenings."
        ),
    ))


if __name__ == "__main__":
    print(f"Agent address : {agent.address}")
    print(f"Check interval: every {CHECK_INTERVAL}s ({CHECK_INTERVAL // 3600}h)")
    print(f"Alert address : {ALERT_ADDRESS or '(none — logging to stdout only)'}")
    agent.run()
