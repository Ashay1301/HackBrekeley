"""
Redis-backed session store.
Falls back silently to an in-memory dict if REDIS_URL is not set,
so the app works locally without Redis.
"""
import json
import os
import time
from typing import Optional

_redis_client = None
_memory_store: dict = {}
TTL = 60 * 60 * 24 * 7  # 7 days


def _redis():
    global _redis_client
    if _redis_client is None:
        url = os.environ.get("REDIS_URL") or os.environ.get("REDIS_PRIVATE_URL")
        if url:
            import redis
            _redis_client = redis.from_url(url, decode_responses=True)
    return _redis_client


def save_analysis(session_id: str, data: dict) -> None:
    r, payload = _redis(), json.dumps(data)
    if r:
        r.setex(f"analysis:{session_id}", TTL, payload)
        r.zadd("sessions:index", {session_id: time.time()})
        r.expire("sessions:index", TTL)
    else:
        _memory_store[f"analysis:{session_id}"] = payload
        _memory_store.setdefault("sessions:index", [])
        idx = _memory_store["sessions:index"]
        if session_id not in idx:
            idx.append(session_id)


def load_analysis(session_id: str) -> Optional[dict]:
    r = _redis()
    raw = (r.get(f"analysis:{session_id}") if r
           else _memory_store.get(f"analysis:{session_id}"))
    return json.loads(raw) if raw else None


def append_message(session_id: str, role: str, text: str) -> None:
    r   = _redis()
    key = f"chat:{session_id}"
    msg = json.dumps({"role": role, "text": text})
    if r:
        r.rpush(key, msg)
        r.expire(key, TTL)
    else:
        _memory_store.setdefault(key, []).append(msg)


def load_messages(session_id: str) -> list:
    r   = _redis()
    key = f"chat:{session_id}"
    items = r.lrange(key, 0, 49) if r else _memory_store.get(key, [])
    return [json.loads(i) for i in items]


def get_recent_session_ids(n: int = 10) -> list:
    r = _redis()
    if r:
        return r.zrevrange("sessions:index", 0, n - 1)
    idx = _memory_store.get("sessions:index", [])
    return list(reversed(idx[-n:]))


def list_sessions(limit: int = 8) -> list:
    """Alias for get_recent_session_ids — used by the ASI:One agent route."""
    return get_recent_session_ids(n=limit)


def save_voice_result(session_id: str, result: dict) -> None:
    r, payload = _redis(), json.dumps(result)
    if r:
        r.setex(f"voice:{session_id}", TTL, payload)
    else:
        _memory_store[f"voice:{session_id}"] = payload


def load_voice_result(session_id: str) -> Optional[dict]:
    r = _redis()
    raw = (r.get(f"voice:{session_id}") if r
           else _memory_store.get(f"voice:{session_id}"))
    return json.loads(raw) if raw else None


# ── Token revocation (for server-side logout) ─────────────────────────────────

def revoke_token(jti: str, ttl: int = TTL) -> None:
    """Blacklist a JWT by its jti claim so decode_token rejects it immediately."""
    r = _redis()
    key = f"revoked_jti:{jti}"
    if r:
        r.setex(key, ttl, "1")
    else:
        _memory_store[key] = str(time.time() + ttl)


def is_token_revoked(jti: str) -> bool:
    r = _redis()
    key = f"revoked_jti:{jti}"
    if r:
        return r.exists(key) > 0
    entry = _memory_store.get(key)
    if entry is None:
        return False
    if float(entry) < time.time():
        del _memory_store[key]
        return False
    return True
