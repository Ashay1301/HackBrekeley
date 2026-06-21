"""
Google Health (Fit) OAuth 2.0 + sleep API client.

Covers Fitbit (syncs to Google Fit), Wear OS, Samsung Health,
and any Google Fit compatible app — one OAuth flow for all.

Sleep segment values from Google Fit:
  1 = Awake in bed   → Wake
  2 = Undifferentiated sleep → Light
  3 = Out of bed     → Wake
  4 = Light sleep (N1/N2) → Light
  5 = Deep sleep (N3)     → Deep
  6 = REM                 → REM
"""

import asyncio
import os
import secrets
import time
from datetime import datetime, timedelta, timezone

import httpx

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get(
    "GOOGLE_REDIRECT_URI", "http://localhost:8000/api/health/callback"
)

_AUTH_URL    = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL   = "https://oauth2.googleapis.com/token"
_HEALTH_BASE = "https://health.googleapis.com/v4/users/me"

SCOPES = " ".join([
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
])

_STAGE_MAP = {1: "Wake", 2: "Light", 3: "Wake", 4: "Light", 5: "Deep", 6: "REM"}
_HYPNO_MAP = {"Wake": "W", "Light": "N2", "Deep": "N3", "REM": "REM"}

# In-memory stores (Redis upgrade path: same pattern as session.py)
_tokens:   dict[str, dict]           = {}
_states:   dict[str, str]            = {}   # state → nonce (CSRF protection)
_inflight: dict[str, asyncio.Future] = {}   # single-flight refresh guard


# ── OAuth flow ───────────────────────────────────────────────────────────────

def get_auth_url() -> tuple[str, str]:
    """Return (authorization_url, state). Redirect the browser here."""
    if not GOOGLE_CLIENT_ID:
        raise ValueError("GOOGLE_CLIENT_ID not configured")

    state = secrets.token_urlsafe(16)
    _states[state] = state

    params = (
        f"client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope={SCOPES.replace(' ', '%20')}"
        f"&state={state}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return f"{_AUTH_URL}?{params}", state


async def exchange_code(code: str, state: str) -> dict:
    """Exchange authorization code for tokens. Call from /callback."""
    if state not in _states:
        raise ValueError("Invalid or expired OAuth state")
    _states.pop(state)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  GOOGLE_REDIRECT_URI,
                "grant_type":    "authorization_code",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    token = {
        "access_token":  data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "expires_at":    time.time() + data.get("expires_in", 3600),
    }
    _tokens["default"] = token
    return token


async def _do_refresh(token: dict) -> dict:
    if not token.get("refresh_token"):
        raise ValueError("No refresh token — user must re-authorise.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "refresh_token": token["refresh_token"],
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "grant_type":    "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    new_token = {
        "access_token":  data["access_token"],
        "refresh_token": token["refresh_token"],   # refresh tokens don't rotate on Google
        "expires_at":    time.time() + data.get("expires_in", 3600),
    }
    _tokens["default"] = new_token
    return new_token


async def get_valid_token(user_id: str = "default") -> str:
    """Return a live access token, refreshing automatically if needed."""
    token = _tokens.get(user_id)
    if not token:
        raise ValueError("Google Health not connected. Please authorise first.")

    if time.time() < token["expires_at"] - 60:
        return token["access_token"]

    if user_id in _inflight:
        new = await _inflight[user_id]
        return new["access_token"]

    loop = asyncio.get_event_loop()
    fut  = loop.create_future()
    _inflight[user_id] = fut
    try:
        new = await _do_refresh(token)
        fut.set_result(new)
    except Exception as exc:
        fut.set_exception(exc)
        raise
    finally:
        _inflight.pop(user_id, None)

    return new["access_token"]


def is_connected(user_id: str = "default") -> bool:
    return user_id in _tokens


# ── Google Fit API ───────────────────────────────────────────────────────────

async def get_profile(user_id: str = "default") -> dict:
    token = await get_valid_token(user_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
    return resp.json()


async def get_sleep_by_date(date: str, user_id: str = "default") -> dict:
    """
    Fetch sleep data for YYYY-MM-DD via Google Health API v4.
    Uses sleep.interval.end_time filter — sleeps whose end time falls
    within a ±12h window around the target date are returned.
    """
    token = await get_valid_token(user_id)

    day_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    window_start = day_start - timedelta(hours=12)
    window_end   = day_start + timedelta(hours=36)

    filter_str = (
        f'sleep.interval.end_time >= "{window_start.strftime("%Y-%m-%dT%H:%M:%SZ")}"'
        f' AND sleep.interval.end_time < "{window_end.strftime("%Y-%m-%dT%H:%M:%SZ")}"'
    )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{_HEALTH_BASE}/dataTypes/sleep/dataPoints",
            headers={"Authorization": f"Bearer {token}"},
            params={"filter": filter_str, "pageSize": 5},
        )
        resp.raise_for_status()
    return resp.json()


# ── Convert to our sleep_summary format ─────────────────────────────────────

def parse_fitbit_export_json(raw) -> dict:
    """
    Normalise Google Takeout / Fitbit export format → the dict that
    fitbit_client.fitbit_sleep_to_summary() expects: {"sleep": [...]}.

    The Takeout file is either:
      - A list  → [{"logId": ..., "levels": {...}}, ...]
      - A dict  → {"sleep": [...]}  (already correct)
    """
    if isinstance(raw, list):
        return {"sleep": raw}
    if isinstance(raw, dict):
        if "sleep" in raw:
            return raw
        # single log entry
        if "logId" in raw or "levels" in raw:
            return {"sleep": [raw]}
    return {"sleep": []}


def google_fit_sleep_to_summary(data: dict) -> dict:
    """
    Map Google Health API v4 dataPoints/sleep response → sleep_summary dict
    used by the frontend, agent query, and session store.

    Response shape:
      {"dataPoints": [{"sleep": {"sleepSummary": {...}, "sleepStages": [...]},
                       "interval": {"startTime": "...", "endTime": "..."}}]}
    """
    points = data.get("dataPoints", [])
    if not points:
        return {}

    # Pick the longest sleep session
    def _dur(p):
        iv = p.get("sleep", {}).get("interval", {})
        try:
            from datetime import datetime
            s = datetime.fromisoformat(iv.get("startTime", "").replace("Z", "+00:00"))
            e = datetime.fromisoformat(iv.get("endTime",   "").replace("Z", "+00:00"))
            return (e - s).total_seconds()
        except Exception:
            return 0

    point = max(points, key=_dur)
    sleep = point.get("sleep", {})
    iv    = sleep.get("interval", {})

    # Parse interval times
    from datetime import datetime as _dt
    def _parse(ts):
        try:
            return _dt.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

    t_start = _parse(iv.get("startTime", ""))
    t_end   = _parse(iv.get("endTime",   ""))
    time_in_bed = round((t_end - t_start).total_seconds() / 60) if t_start and t_end else 0

    # Stage mapping — Google Health API stage names
    gh_stage_map = {
        "AWAKE": "Wake", "LIGHT": "Light", "DEEP": "Deep", "REM": "REM",
        "SLEEP": "Light",  # undifferentiated fallback
    }
    hypno_map = {"Wake": "W", "Light": "N2", "Deep": "N3", "REM": "REM"}

    stage_secs = {"Wake": 0.0, "Light": 0.0, "Deep": 0.0, "REM": 0.0}
    epochs: list[str] = []

    stages = sorted(
        sleep.get("stages", []),
        key=lambda s: s.get("startTime", ""),
    )
    for seg in stages:
        seg_start = _parse(seg.get("startTime", ""))
        seg_end   = _parse(seg.get("endTime",   ""))
        dur_s = max(0.0, (seg_end - seg_start).total_seconds()) if seg_start and seg_end else 0.0
        stage = gh_stage_map.get(seg.get("type", "").upper(), "Light")
        stage_secs[stage] += dur_s
        epochs.extend([hypno_map[stage]] * max(1, round(dur_s / 30)))

    # Fall back to sleepSummary fields when no stage segments returned
    summary = sleep.get("sleepSummary", {})
    if not stages and summary:
        stage_secs["Deep"]  = summary.get("deepSleepDurationMinutes", 0) * 60
        stage_secs["Light"] = summary.get("lightSleepDurationMinutes", 0) * 60
        stage_secs["REM"]   = summary.get("remSleepDurationMinutes", 0) * 60
        stage_secs["Wake"]  = summary.get("awakeDurationMinutes", 0) * 60

    stage_mins = {k: round(v / 60) for k, v in stage_secs.items()}
    total_sleep = stage_mins["Light"] + stage_mins["Deep"] + stage_mins["REM"]

    # If no stage data at all, derive from interval
    if total_sleep == 0 and time_in_bed > 0:
        total_sleep = round(time_in_bed * 0.92)
        stage_mins  = {"Light": total_sleep, "Deep": 0, "REM": 0, "Wake": time_in_bed - total_sleep}

    latency    = summary.get("sleepOnsetLatencyMinutes", 0)
    efficiency = round(total_sleep / time_in_bed * 100, 1) if time_in_bed > 0 else 0
    awakenings = sum(1 for i in range(1, len(epochs)) if epochs[i] == "W" and epochs[i-1] != "W")

    eff_score  = min(efficiency / 90 * 40, 40)
    rem_score  = min(stage_mins["REM"]  / 90 * 25, 25)
    deep_score = min(stage_mins["Deep"] / 60 * 25, 25)
    lat_score  = max(0, 10 - latency / 3)
    quality    = round(min(eff_score + rem_score + deep_score + lat_score, 100))
    grade      = "A" if quality >= 85 else "B" if quality >= 70 else "C" if quality >= 55 else "D"

    pct = {}
    if total_sleep > 0:
        pct = {k: round(stage_mins[k] / total_sleep * 100, 1) for k in ("Deep", "Light", "REM")}
    if time_in_bed > 0:
        pct["Wake"] = round(stage_mins["Wake"] / time_in_bed * 100, 1)

    return {
        "quality_score":     quality,
        "quality_grade":     grade,
        "total_sleep_min":   total_sleep,
        "efficiency_pct":    efficiency,
        "latency_min":       latency,
        "awakenings":        awakenings,
        "time_in_bed_min":   time_in_bed,
        "pct_in_stage":      pct,
        "time_in_stage_min": stage_mins,
        "epochs":            epochs,
        "source":            "google_health",
    }
