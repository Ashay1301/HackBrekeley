"""
SQLite / PostgreSQL persistence layer for SleepSense AI.

Driver is selected at startup:
  DATABASE_URL starts with "postgres" → psycopg2 (Railway PostgreSQL add-on)
  anything else                       → sqlite3  (local dev default)

Tables
------
users              — accounts (patient + provider roles)
sleep_sessions     — one row per sleep analysis
voice_results      — one row per voice-check result
dream_entries      — dream journal entries
provider_patients  — many-to-many provider↔patient roster
auth_tokens        — email-verification and password-reset tokens
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("DB_PATH", os.path.join(ROOT, "sleepsense.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")

_USE_PG = DATABASE_URL.startswith("postgres")


# ── Connection management ─────────────────────────────────────────────────────

@contextmanager
def _conn():
    if _USE_PG:
        import psycopg2
        import psycopg2.extras
        con = psycopg2.connect(DATABASE_URL)
        con.autocommit = False
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
    else:
        con = sqlite3.connect(DB_PATH, check_same_thread=False)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()


def _fetchone(cur, pg: bool):
    row = cur.fetchone()
    if row is None:
        return None
    if pg:
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    return dict(row)


def _fetchall(cur, pg: bool):
    rows = cur.fetchall()
    if pg:
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in rows]
    return [dict(r) for r in rows]


def _ph(pg: bool) -> str:
    """Return the correct placeholder: %s for psycopg2, ? for sqlite3."""
    return "%s" if pg else "?"


def _migrate(con, pg: bool):
    """Apply additive migrations to existing tables (idempotent)."""
    cur = con.cursor()
    if pg:
        cur.execute("""
            ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE
        """)
    else:
        # SQLite doesn't support IF NOT EXISTS on ALTER TABLE — check first
        cur.execute("PRAGMA table_info(users)")
        cols = {r[1] for r in cur.fetchall()}
        if "email_verified" not in cols:
            cur.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")


def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    with _conn() as con:
        cur = con.cursor()
        if _USE_PG:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                name          TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'patient',
                email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                created_at    TEXT NOT NULL
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sleep_sessions (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                data       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS voice_results (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_id TEXT,
                data       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS dream_entries (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_id    TEXT,
                dream_text    TEXT NOT NULL,
                analysis_text TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS provider_patients (
                provider_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                patient_id  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (provider_id, patient_id)
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS auth_tokens (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                type       TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used       BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TEXT NOT NULL
            )""")
        else:
            cur.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id             TEXT PRIMARY KEY,
                email          TEXT UNIQUE NOT NULL,
                name           TEXT NOT NULL,
                password_hash  TEXT NOT NULL,
                role           TEXT NOT NULL DEFAULT 'patient',
                email_verified INTEGER NOT NULL DEFAULT 0,
                created_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sleep_sessions (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                data       TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS voice_results (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                session_id TEXT,
                data       TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS dream_entries (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                session_id    TEXT,
                dream_text    TEXT NOT NULL,
                analysis_text TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS provider_patients (
                provider_id TEXT NOT NULL,
                patient_id  TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (provider_id, patient_id),
                FOREIGN KEY (provider_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (patient_id)  REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS auth_tokens (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                type       TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """)
        _migrate(con, _USE_PG)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── User CRUD ─────────────────────────────────────────────────────────────────

def create_user(email: str, name: str, password_hash: str, role: str = "patient") -> dict:
    uid = str(uuid.uuid4())
    ph  = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            f"INSERT INTO users (id, email, name, password_hash, role, email_verified, created_at) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{'FALSE' if _USE_PG else '0'},{ph})",
            (uid, email.lower().strip(), name, password_hash, role, _now()),
        )
    return {"id": uid, "email": email, "name": name, "role": role, "email_verified": False}


def get_user_by_email(email: str) -> dict | None:
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM users WHERE email = {ph}", (email.lower().strip(),))
        return _fetchone(cur, _USE_PG)


def get_user_by_id(user_id: str) -> dict | None:
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM users WHERE id = {ph}", (user_id,))
        return _fetchone(cur, _USE_PG)


def mark_email_verified(user_id: str):
    ph = _ph(_USE_PG)
    val = "TRUE" if _USE_PG else "1"
    with _conn() as con:
        con.cursor().execute(
            f"UPDATE users SET email_verified = {val} WHERE id = {ph}", (user_id,)
        )


def update_password(user_id: str, new_hash: str):
    ph = _ph(_USE_PG)
    with _conn() as con:
        con.cursor().execute(
            f"UPDATE users SET password_hash = {ph} WHERE id = {ph}", (new_hash, user_id)
        )


# ── Auth tokens (email verify + password reset) ───────────────────────────────

def save_auth_token(user_id: str, token_hash: str, token_type: str, expires_at: str):
    tid = str(uuid.uuid4())
    ph  = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        # Invalidate any existing unused tokens of the same type for this user
        cur.execute(
            f"UPDATE auth_tokens SET used = {'TRUE' if _USE_PG else '1'} "
            f"WHERE user_id = {ph} AND type = {ph} AND used = {'FALSE' if _USE_PG else '0'}",
            (user_id, token_type),
        )
        cur.execute(
            f"INSERT INTO auth_tokens (id, user_id, token_hash, type, expires_at, used, created_at) "
            f"VALUES ({ph},{ph},{ph},{ph},{ph},{'FALSE' if _USE_PG else '0'},{ph})",
            (tid, user_id, token_hash, token_type, expires_at, _now()),
        )


def consume_auth_token(token_hash: str, token_type: str) -> dict | None:
    """
    Returns the matching auth_token row if valid (unused + not expired), then marks it used.
    Returns None if not found, already used, or expired.
    """
    ph  = _ph(_USE_PG)
    now = _now()
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT * FROM auth_tokens WHERE token_hash = {ph} AND type = {ph} "
            f"AND used = {'FALSE' if _USE_PG else '0'} AND expires_at > {ph}",
            (token_hash, token_type, now),
        )
        row = _fetchone(cur, _USE_PG)
        if row:
            cur.execute(
                f"UPDATE auth_tokens SET used = {'TRUE' if _USE_PG else '1'} WHERE id = {ph}",
                (row["id"],),
            )
    return row


# ── Sleep session CRUD ────────────────────────────────────────────────────────

def save_sleep_session(session_id: str, user_id: str, data: dict):
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        if _USE_PG:
            cur.execute(
                f"INSERT INTO sleep_sessions (id, user_id, data, created_at) VALUES ({ph},{ph},{ph},{ph}) "
                f"ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data",
                (session_id, user_id, json.dumps(data), _now()),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO sleep_sessions (id, user_id, data, created_at) VALUES (?,?,?,?)",
                (session_id, user_id, json.dumps(data), _now()),
            )


def get_sleep_session(session_id: str) -> dict | None:
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(f"SELECT * FROM sleep_sessions WHERE id = {ph}", (session_id,))
        row = _fetchone(cur, _USE_PG)
    if not row:
        return None
    row["data"] = json.loads(row["data"])
    return row


def get_sessions_for_user(user_id: str, limit: int = 10) -> list[dict]:
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT * FROM sleep_sessions WHERE user_id = {ph} ORDER BY created_at DESC LIMIT {ph}",
            (user_id, limit),
        )
        rows = _fetchall(cur, _USE_PG)
    for r in rows:
        r["data"] = json.loads(r["data"])
    return rows


# ── Voice result CRUD ─────────────────────────────────────────────────────────

def save_voice_result(user_id: str, session_id: str | None, data: dict):
    rid = str(uuid.uuid4())
    ph  = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        if _USE_PG:
            cur.execute(
                f"INSERT INTO voice_results (id, user_id, session_id, data, created_at) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph}) ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data",
                (rid, user_id, session_id, json.dumps(data), _now()),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO voice_results (id, user_id, session_id, data, created_at) VALUES (?,?,?,?,?)",
                (rid, user_id, session_id, json.dumps(data), _now()),
            )


def get_latest_voice_result(user_id: str) -> dict | None:
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT data FROM voice_results WHERE user_id = {ph} ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )
        row = _fetchone(cur, _USE_PG)
    return json.loads(row["data"]) if row else None


def get_voice_result_for_session(user_id: str, session_id: str) -> dict | None:
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT data FROM voice_results WHERE user_id = {ph} AND session_id = {ph} "
            f"ORDER BY created_at DESC LIMIT 1",
            (user_id, session_id),
        )
        row = _fetchone(cur, _USE_PG)
    return json.loads(row["data"]) if row else None


# ── Dream entry CRUD ──────────────────────────────────────────────────────────

def save_dream(user_id: str, session_id: str | None, dream_text: str, analysis_text: str):
    did = str(uuid.uuid4())
    ph  = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        if _USE_PG:
            cur.execute(
                f"INSERT INTO dream_entries (id, user_id, session_id, dream_text, analysis_text, created_at) "
                f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph}) ON CONFLICT (id) DO UPDATE SET analysis_text = EXCLUDED.analysis_text",
                (did, user_id, session_id, dream_text, analysis_text, _now()),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO dream_entries (id, user_id, session_id, dream_text, analysis_text, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (did, user_id, session_id, dream_text, analysis_text, _now()),
            )


def get_dream_for_session(user_id: str, session_id: str) -> dict | None:
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT dream_text, analysis_text FROM dream_entries "
            f"WHERE user_id = {ph} AND session_id = {ph} ORDER BY created_at DESC LIMIT 1",
            (user_id, session_id),
        )
        row = _fetchone(cur, _USE_PG)
    return {"dream": row["dream_text"], "analysis": row["analysis_text"]} if row else None


# ── Provider–patient links ────────────────────────────────────────────────────

def add_patient_to_provider(provider_id: str, patient_id: str):
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        if _USE_PG:
            cur.execute(
                f"INSERT INTO provider_patients (provider_id, patient_id, created_at) "
                f"VALUES ({ph},{ph},{ph}) ON CONFLICT DO NOTHING",
                (provider_id, patient_id, _now()),
            )
        else:
            cur.execute(
                "INSERT OR IGNORE INTO provider_patients (provider_id, patient_id, created_at) VALUES (?,?,?)",
                (provider_id, patient_id, _now()),
            )


def get_patients_for_provider(provider_id: str) -> list[dict]:
    ph = _ph(_USE_PG)
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT u.id, u.email, u.name, u.created_at "
            f"FROM users u JOIN provider_patients pp ON u.id = pp.patient_id "
            f"WHERE pp.provider_id = {ph} ORDER BY pp.created_at DESC",
            (provider_id,),
        )
        return _fetchall(cur, _USE_PG)


def get_all_patients() -> list[dict]:
    with _conn() as con:
        cur = con.cursor()
        cur.execute(
            "SELECT id, email, name, created_at FROM users WHERE role = 'patient' ORDER BY created_at DESC"
        )
        return _fetchall(cur, _USE_PG)
