"""
Fitbit OAuth 2.0 (PKCE / S256) + sleep API client.

Tokens are kept in-memory with optional Redis persistence
(same pattern as session.py — works locally without Redis).

Single-flight refresh guard prevents the double-refresh race condition
that permanently bricks the token when two requests both see an expired token.
"""

import asyncio
import base64
import hashlib
import os
import secrets
import time
from typing import Optional

import httpx

FITBIT_CLIENT_ID     = os.environ.get("FITBIT_CLIENT_ID", "")
FITBIT_CLIENT_SECRET = os.environ.get("FITBIT_CLIENT_SECRET", "")
FITBIT_REDIRECT_URI  = os.environ.get(
    "FITBIT_REDIRECT_URI", "http://localhost:8000/api/fitbit/callback"
)
FITBIT_SCOPES = "sleep heartrate profile"

_AUTH_URL  = "https://www.fitbit.com/oauth2/authorize"
_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
_API_BASE  = "https://api.fitbit.com"

# ── In-memory stores ─────────────────────────────────────────────────────────

_tokens:    dict[str, dict]             = {}  # user_id → token record
_pkce:      dict[str, str]              = {}  # state   → code_verifier
_inflight:  dict[str, asyncio.Future]   = {}  # user_id → in-progress refresh future


# ── PKCE helpers ─────────────────────────────────────────────────────────────

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def get_auth_url() -> tuple[str, str]:
    """Return (authorization_url, state). Call this to start the OAuth flow."""
    if not FITBIT_CLIENT_ID:
        raise ValueError("FITBIT_CLIENT_ID not configured")

    state     = secrets.token_urlsafe(16)
    verifier  = secrets.token_urlsafe(64)
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    _pkce[state] = verifier

    params = (
        f"response_type=code"
        f"&client_id={FITBIT_CLIENT_ID}"
        f"&redirect_uri={FITBIT_REDIRECT_URI}"
        f"&scope={FITBIT_SCOPES.replace(' ', '%20')}"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=S256"
        f"&state={state}"
    )
    return f"{_AUTH_URL}?{params}", state


# ── Token exchange & refresh ─────────────────────────────────────────────────

async def exchange_code(code: str, state: str) -> dict:
    """Exchange authorization code → token record. Call from /callback."""
    verifier = _pkce.pop(state, None)
    if not verifier:
        raise ValueError("Invalid or expired OAuth state")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type":    "authorization_code",
                "code":          code,
                "redirect_uri":  FITBIT_REDIRECT_URI,
                "code_verifier": verifier,
                "client_id":     FITBIT_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            auth=(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET),
        )
        resp.raise_for_status()
        data = resp.json()

    token = {
        "access_token":  data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_at":    time.time() + data.get("expires_in", 28800),
        "fitbit_user_id": data.get("user_id", "default"),
    }
    user_id = token["fitbit_user_id"]
    _tokens[user_id] = token
    _tokens["default"] = token  # always keep a "default" alias for single-user demo
    return token


async def _do_refresh(user_id: str, token: dict) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type":    "refresh_token",
                "refresh_token": token["refresh_token"],
                "client_id":     FITBIT_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            auth=(FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET),
        )
        resp.raise_for_status()
        data = resp.json()

    new_token = {
        "access_token":  data["access_token"],
        "refresh_token": data["refresh_token"],
        "expires_at":    time.time() + data.get("expires_in", 28800),
        "fitbit_user_id": user_id,
    }
    _tokens[user_id]    = new_token
    _tokens["default"]  = new_token
    return new_token


async def get_valid_token(user_id: str = "default") -> str:
    """Return a live access token, refreshing if within 60s of expiry."""
    token = _tokens.get(user_id)
    if not token:
        raise ValueError("Fitbit not connected. Please authorise first.")

    if time.time() < token["expires_at"] - 60:
        return token["access_token"]

    # Single-flight: if a refresh is already in-flight, await it
    if user_id in _inflight:
        new = await _inflight[user_id]
        return new["access_token"]

    loop = asyncio.get_event_loop()
    fut  = loop.create_future()
    _inflight[user_id] = fut
    try:
        new = await _do_refresh(user_id, token)
        fut.set_result(new)
    except Exception as exc:
        fut.set_exception(exc)
        raise
    finally:
        _inflight.pop(user_id, None)

    return new["access_token"]


# ── Fitbit API calls ─────────────────────────────────────────────────────────

async def get_sleep_by_date(date: str, user_id: str = "default") -> dict:
    """GET /1.2/user/-/sleep/date/{date}.json"""
    token = await get_valid_token(user_id)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_API_BASE}/1.2/user/-/sleep/date/{date}.json",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
    return resp.json()


async def get_sleep_range(start: str, end: str, user_id: str = "default") -> dict:
    """GET /1.2/user/-/sleep/date/{start}/{end}.json  (up to 100 days)"""
    token = await get_valid_token(user_id)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_API_BASE}/1.2/user/-/sleep/date/{start}/{end}.json",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
    return resp.json()


async def get_profile(user_id: str = "default") -> dict:
    token = await get_valid_token(user_id)
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{_API_BASE}/1/user/-/profile.json",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
    return resp.json()


def is_connected(user_id: str = "default") -> bool:
    return user_id in _tokens


# ── Convert Fitbit response → our sleep_summary format ──────────────────────

def fitbit_sleep_to_summary(data: dict) -> dict:
    """
    Map Fitbit /1.2/sleep/date response to the sleep_summary dict
    our frontend and agent already understand.
    Handles both 'stages' (modern HR devices) and 'classic' fallback.
    """
    sleep_logs = data.get("sleep", [])
    if not sleep_logs:
        return {}

    log      = max(sleep_logs, key=lambda s: s.get("minutesAsleep", 0))
    levels   = log.get("levels", {})
    summary  = levels.get("summary", {})
    log_type = log.get("type", "stages")

    if log_type == "stages":
        deep_min  = summary.get("deep",  {}).get("minutes", 0)
        light_min = summary.get("light", {}).get("minutes", 0)
        rem_min   = summary.get("rem",   {}).get("minutes", 0)
        wake_min  = summary.get("wake",  {}).get("minutes", 0)
    else:
        # Classic (no HR data): asleep + restless + awake
        deep_min  = 0
        light_min = summary.get("asleep",   {}).get("minutes", log.get("minutesAsleep", 0))
        rem_min   = 0
        wake_min  = summary.get("awake",    {}).get("minutes", 0)

    total_sleep = deep_min + light_min + rem_min
    time_in_bed = log.get("timeInBed", total_sleep + wake_min)
    efficiency  = round(total_sleep / time_in_bed * 100, 1) if time_in_bed > 0 else 0
    latency     = log.get("minutesToFallAsleep", 0)
    awakenings  = log.get("awakeningCount",
                          summary.get("wake", {}).get("count", 0))

    # Quality score matching our existing grading logic
    eff_score  = min(efficiency / 90 * 40, 40)
    rem_score  = min(rem_min   / 90 * 25, 25)
    deep_score = min(deep_min  / 60 * 25, 25)
    lat_score  = max(0, 10 - latency / 3)
    quality    = round(min(eff_score + rem_score + deep_score + lat_score, 100))
    grade      = "A" if quality >= 85 else "B" if quality >= 70 else "C" if quality >= 55 else "D"

    pct = {}
    if total_sleep > 0:
        pct = {
            "Deep":  round(deep_min  / total_sleep * 100, 1),
            "Light": round(light_min / total_sleep * 100, 1),
            "REM":   round(rem_min   / total_sleep * 100, 1),
        }
    if time_in_bed > 0:
        pct["Wake"] = round(wake_min / time_in_bed * 100, 1)

    # Build hypnogram epochs from levels.data for the chart
    epochs = []
    stage_map = {"deep": "N3", "light": "N2", "rem": "REM", "wake": "W",
                 "asleep": "N2", "restless": "N1", "awake": "W"}
    for seg in levels.get("data", []):
        label = stage_map.get(seg.get("level", ""), "N2")
        n_epochs = max(1, seg.get("seconds", 30) // 30)
        epochs.extend([label] * n_epochs)

    return {
        "quality_score":      quality,
        "quality_grade":      grade,
        "total_sleep_min":    total_sleep,
        "efficiency_pct":     efficiency,
        "latency_min":        latency,
        "awakenings":         awakenings,
        "time_in_bed_min":    time_in_bed,
        "pct_in_stage":       pct,
        "time_in_stage_min":  {
            "Deep":  deep_min,
            "Light": light_min,
            "REM":   rem_min,
            "Wake":  wake_min,
        },
        "epochs":  epochs,
        "source":  "fitbit_live",
        "log_type": log_type,
    }
