"""
SleepSense AI — FastAPI backend
"""

import json
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.eeg_inference import EEGPredictor
from backend.wrist_inference import WristPredictor
from backend import chat as chat_module
from backend import session as session_store
from backend.arize_logger import log_prediction
from backend.voice_analysis import analyze_voice
from backend import fitbit_client as fitbit
from backend import google_health_client as ghealth
from backend.disorder_screener import screen as disorder_screen
from backend import dream_journal as dream_module
from backend import psychiatric_risk as psych_risk
from backend import db, auth as auth_module
from backend import email_client as email_mod

# ── Sentry (optional — only activates when SENTRY_DSN is set) ───────────────
_sentry_dsn = os.environ.get("SENTRY_DSN")
if _sentry_dsn:
    try:
        import sentry_sdk as _sentry
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        _sentry.init(
            dsn=_sentry_dsn,
            integrations=[FastApiIntegration()],
            traces_sample_rate=0.2,
            environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
        )
    except Exception as e:
        print(f"[startup] Sentry init failed (non-fatal): {e}")

# ── Model singletons ────────────────────────────────────────────────────────────

eeg_predictor  = EEGPredictor()
wrist_predictor = WristPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise SQLite tables (no-op if already created)
    try:
        db.init_db()
    except Exception as e:
        print(f"[startup] DB init failed: {e}")

    # Load models at startup
    try:
        eeg_predictor.load()
    except Exception as e:
        print(f"[startup] EEG model load failed: {e}")

    try:
        wrist_predictor.load()
    except Exception as e:
        print(f"[startup] Wrist model load failed (OK if not trained yet): {e}")

    yield  # app runs here


app = FastAPI(title="SleepSense AI", lifespan=lifespan)

_ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ──────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
    _limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _RATE_LIMIT = True
except ImportError:
    _RATE_LIMIT = False
    _limiter = None

# ── Wearable parsers ────────────────────────────────────────────────────────────

def _parse_wearable(filename: str, file_bytes: bytes) -> tuple[list[dict], str]:
    """Returns (epoch_list, source_label)."""
    import wrist_model.parsers.apple_health as apple
    import wrist_model.parsers.fitbit as fitbit
    import wrist_model.parsers.garmin as garmin
    import wrist_model.parsers.csv_generic as csv_gen

    name = filename.lower()
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(filename)[1] or ".bin", delete=False
    ) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if name.endswith(".xml"):
            return apple.parse(tmp_path), "Apple Health"
        elif name.endswith(".fit"):
            return garmin.parse(tmp_path), "Garmin (.fit)"
        elif name.endswith(".json"):
            # Try Fitbit first, fall back to Garmin JSON
            try:
                return fitbit.parse(tmp_path), "Fitbit"
            except Exception:
                return garmin.parse(tmp_path), "Garmin (.json)"
        elif name.endswith(".csv"):
            return csv_gen.parse(tmp_path), "CSV"
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    finally:
        os.unlink(tmp_path)


# ── Auth routes ─────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    role: str = "patient"  # "patient" | "provider"


def _rate_limit(limit_str: str):
    """Decorator that applies slowapi rate limit when available, is a no-op otherwise."""
    def decorator(func):
        if _RATE_LIMIT and _limiter:
            return _limiter.limit(limit_str)(func)
        return func
    return decorator


def _sentiment_from_claude(text: str, api_key: str) -> dict:
    """Return {"label": "positive|neutral|negative", "score": 0-100} for the given text."""
    if not text or not text.strip():
        return {"label": "neutral", "score": 50}
    try:
        import anthropic, json as _json
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": (
                    "Analyse the sentiment of this text and reply with ONLY valid JSON "
                    'in the form {"label":"positive|neutral|negative","score":0-100} '
                    "where score = positivity (0 very negative, 50 neutral, 100 very positive).\n\n"
                    f'Text: """{text[:800]}"""'
                ),
            }],
        )
        raw = resp.content[0].text.strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return _json.loads(raw)
    except Exception:
        return {"label": "neutral", "score": 50}


@app.post("/api/auth/register")
@_rate_limit("5/minute")
async def register(req: RegisterRequest, request: "Request"):
    if req.role not in ("patient", "provider"):
        raise HTTPException(status_code=400, detail="role must be 'patient' or 'provider'")
    existing = db.get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with that email already exists.")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    pw_hash = auth_module.hash_password(req.password)
    user = db.create_user(req.email, req.name, pw_hash, req.role)
    # Send verification email
    _send_verify_email(user["id"], req.email, req.name)
    token = auth_module.create_access_token(user["id"], user["role"])
    return {"access_token": token, "token_type": "bearer", "user": user}


@app.post("/api/auth/login")
@_rate_limit("10/minute")
async def login(request: "Request", form: OAuth2PasswordRequestForm = Depends()):
    user = db.get_user_by_email(form.username)
    if not user or not auth_module.verify_password(form.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.get("email_verified"):
        raise HTTPException(status_code=403, detail="check_email")
    token = auth_module.create_access_token(user["id"], user["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {k: user[k] for k in ("id", "email", "name", "role")},
    }


@app.get("/api/auth/me")
async def me(current_user: dict = Depends(auth_module.get_current_user)):
    return {k: current_user[k] for k in ("id", "email", "name", "role", "created_at")}


@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(auth_module.get_current_user),
                 token: Optional[str] = Depends(auth_module.oauth2_scheme)):
    if token:
        payload = auth_module.decode_token(token)
        jti = payload.get("jti")
        if jti:
            from backend import session as _sess
            _sess.revoke_token(jti, ttl=auth_module.ACCESS_TOKEN_EXPIRE_SECONDS)
    return {"detail": "Logged out."}


@app.post("/api/auth/resend-verification")
@_rate_limit("3/minute")
async def resend_verification(req: dict, request: "Request"):
    email = req.get("email", "").lower().strip()
    user  = db.get_user_by_email(email)
    # Always return 200 to avoid email enumeration
    if user and not user.get("email_verified"):
        _send_verify_email(user["id"], user["email"], user["name"])
    return {"detail": "If that account exists and is unverified, a new email has been sent."}


@app.get("/api/auth/verify-email")
async def verify_email(token: str = ""):
    import hashlib
    from fastapi.responses import RedirectResponse
    if not token:
        return RedirectResponse(url="/?verified=error")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    row = db.consume_auth_token(token_hash, "verify")
    if not row:
        return RedirectResponse(url="/?verified=error")
    db.mark_email_verified(row["user_id"])
    return RedirectResponse(url="/?verified=1")


@app.post("/api/auth/forgot-password")
@_rate_limit("3/minute")
async def forgot_password(req: dict, request: "Request"):
    email = req.get("email", "").lower().strip()
    user  = db.get_user_by_email(email)
    if user:
        _send_reset_email(user["id"], user["email"], user["name"])
    # Always 200 — no enumeration
    return {"detail": "If that email is registered, a reset link has been sent."}


@app.post("/api/auth/reset-password")
async def reset_password(req: dict):
    import hashlib
    token    = req.get("token", "")
    new_pass = req.get("password", "")
    if not token or len(new_pass) < 8:
        raise HTTPException(status_code=400, detail="Token and password (min 8 chars) are required.")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    row = db.consume_auth_token(token_hash, "reset")
    if not row:
        raise HTTPException(status_code=400, detail="Reset link is invalid or has expired.")
    db.update_password(row["user_id"], auth_module.hash_password(new_pass))
    return {"detail": "Password updated. You can now log in."}


# ── Email helpers ─────────────────────────────────────────────────────────────

def _send_verify_email(user_id: str, email: str, name: str):
    import hashlib, secrets as _sec
    from datetime import timedelta
    token      = _sec.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires    = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    db.save_auth_token(user_id, token_hash, "verify", expires)
    email_mod.send_verification_email(email, name, token)


def _send_reset_email(user_id: str, email: str, name: str):
    import hashlib, secrets as _sec
    from datetime import timedelta
    token      = _sec.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires    = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    db.save_auth_token(user_id, token_hash, "reset", expires)
    email_mod.send_password_reset_email(email, name, token)


# ── API endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    fname = file.filename or ""
    ext = os.path.splitext(fname)[1].lower()

    try:
        if ext == ".edf":
            if eeg_predictor.model is None:
                raise HTTPException(
                    status_code=503,
                    detail="EEG model unavailable — TensorFlow not installed on this server.",
                )
            result = eeg_predictor.predict(data)
        elif ext in (".xml", ".json", ".fit", ".csv"):
            if not wrist_predictor.available and ext == ".json":
                # Rule-based fallback for Fitbit JSON when ML model unavailable
                try:
                    raw_json = json.loads(data)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON file.")
                wrapped = ghealth.parse_fitbit_export_json(raw_json)
                sleep_summary = fitbit.fitbit_sleep_to_summary(wrapped)
                if not sleep_summary:
                    raise HTTPException(status_code=422, detail="No sleep data found in this Fitbit JSON.")
                sleep_logs = wrapped.get("sleep") or [{}]
                date_str = sleep_logs[0].get("dateOfSleep", "")
                result = {
                    "source":        "fitbit_export",
                    "date":          date_str,
                    "sleep_summary": sleep_summary,
                    "hypnogram":     sleep_summary.pop("epochs", []),
                    "metadata":      {"recording_start": date_str, "source": "fitbit_export"},
                }
            elif not wrist_predictor.available:
                raise HTTPException(
                    status_code=503,
                    detail=f"Wrist model unavailable (TensorFlow not installed). "
                           f"Upload a Fitbit .json for rule-based analysis, or sync via Google Health.",
                )
            else:
                epochs, source = _parse_wearable(fname, data)
                if not epochs:
                    raise HTTPException(status_code=400, detail="No sleep epochs found in uploaded file.")
                result = wrist_predictor.predict(epochs, source=source)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Upload .edf, .xml, .json, .fit, or .csv.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result["disorder_flags"] = disorder_screen(result.get("sleep_summary", {}))

    # Persist session
    sid = result.get("session_id") or str(uuid.uuid4())
    result["session_id"] = sid
    session_store.save_analysis(sid, result)
    log_prediction(sid, result)

    return JSONResponse(result)


class ChatRequest(BaseModel):
    message: str
    analysis: dict


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        reply = chat_module.chat(req.message, req.analysis)
        return {"reply": reply}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/smart-questions")
async def smart_questions(session_id: str = ""):
    # Used when the frontend wants pre-generated questions after analysis.
    # The analysis JSON must be passed as query param or we rely on client to call
    # this by posting analysis. Simple version: return generic defaults.
    return {
        "questions": [
            "How was my sleep quality overall?",
            "What does my REM percentage tell you?",
            "How can I improve my deep sleep?",
        ]
    }


@app.post("/api/smart-questions")
async def smart_questions_post(analysis: dict):
    try:
        questions = chat_module.generate_smart_questions(analysis)
        return {"questions": questions}
    except Exception as e:
        return {"questions": [
            "How was my sleep quality overall?",
            "What does my REM percentage tell you?",
            "How can I improve my deep sleep?",
        ]}


@app.get("/api/demo")
async def demo():
    """Return a pre-analysed demo sleep report."""
    demo_paths = [
        os.path.join(ROOT, "demo_sleep_report.json"),
        os.path.join(ROOT, "patient4001_sleep_report.json"),
        os.path.join(ROOT, "patient_sleep_report.json"),
    ]
    for p in demo_paths:
        if os.path.exists(p):
            with open(p) as f:
                raw = json.load(f)
            result = _normalise_demo(raw)
            # Save so /api/psychiatric-risk and session restore work on demo
            session_store.save_analysis(result["session_id"], result)
            return result
    raise HTTPException(status_code=404, detail="Demo file not found.")


def _normalise_demo(raw: dict) -> dict:
    """Convert old predict_for_llm.py schema to our API schema."""
    if "sleep_summary" in raw and "quality_score" in raw.get("sleep_summary", {}):
        raw.setdefault("session_id", "demo-session")
        return raw

    # Old format: has "metadata", "sleep_summary" with different keys, "epoch_by_epoch_data"
    old_summary = raw.get("sleep_summary", {})
    stage_times = old_summary.get("time_in_stage_minutes", {})
    stage_pcts  = old_summary.get("percentage_in_stage", {})
    total_min   = old_summary.get("total_recording_time_minutes", 480)
    awakenings  = old_summary.get("awakenings_after_onset", 0)
    lat         = old_summary.get("sleep_latency_minutes", 0)

    # Build normalised epoch list with confidence
    epochs = []
    import random
    random.seed(42)
    for ep in raw.get("epoch_by_epoch_data", []):
        code = ep.get("predicted_stage_code", 0)
        conf = round(0.6 + random.random() * 0.35, 3)
        stages = ["W", "N1", "N2", "N3", "REM"]
        probs = [0.05] * 5
        probs[code] = conf
        rest = (1 - conf) / 4
        for j in range(5):
            if j != code:
                probs[j] = rest
        ep["confidence"] = conf
        ep["probabilities"] = {s: round(probs[i], 3) for i, s in enumerate(stages)}
        epochs.append(ep)

    # Compute efficiency
    wake_min = stage_times.get("W", 0) or 0
    total_sleep = total_min - wake_min
    eff = round(total_sleep / total_min * 100, 1) if total_min > 0 else 0

    rem_pct  = stage_pcts.get("REM", 0)
    deep_pct = stage_pcts.get("N3", 0)

    from backend.sleep_metrics import quality_score as qs, grade as grd
    lat_f = float(lat) if lat != "N/A" else 0.0
    q = qs(eff, lat_f, rem_pct, deep_pct, awakenings)

    return {
        "session_id": "demo-session",
        "metadata": {
            "analysis_type": "eeg",
            **raw.get("metadata", {}),
        },
        "sleep_summary": {
            "quality_score": q,
            "quality_grade":  grd(q),
            "efficiency_pct": eff,
            "latency_min":    lat_f,
            "awakenings":     awakenings,
            "total_recording_min": total_min,
            "total_sleep_min": round(total_sleep, 1),
            "time_in_stage_min": stage_times,
            "pct_in_stage":      stage_pcts,
        },
        "epoch_by_epoch_data": epochs,
        "disorder_flags": disorder_screen({
            "efficiency_pct": eff,
            "latency_min": lat_f,
            "awakenings": awakenings,
            "total_sleep_min": round(total_sleep, 1),
            "pct_in_stage": stage_pcts,
        }),
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    data = session_store.load_analysis(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    data["chat_history"] = session_store.load_messages(session_id)
    return JSONResponse(data)


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Receive audio blob (webm/wav), return transcript via Deepgram."""
    dg_key = os.environ.get("DEEPGRAM_API_KEY")
    if not dg_key:
        raise HTTPException(status_code=503, detail="Deepgram API key not configured.")
    audio_bytes = await file.read()
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
        dg = DeepgramClient(dg_key)
        options = PrerecordedOptions(
            model="nova-2", language="en-US", punctuate=True, smart_format=True,
        )
        response = dg.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_bytes, "mimetype": file.content_type or "audio/webm"}, options,
        )
        transcript = response.results.channels[0].alternatives[0].transcript
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.post("/api/tts")
async def text_to_speech(req: dict):
    """Convert text to speech MP3 via Deepgram Aura."""
    dg_key = os.environ.get("DEEPGRAM_API_KEY")
    if not dg_key:
        raise HTTPException(status_code=503, detail="Deepgram API key not configured.")
    text = req.get("text", "")[:500]
    try:
        from deepgram import DeepgramClient, SpeakOptions
        from fastapi.responses import Response as FastAPIResponse
        dg = DeepgramClient(dg_key)
        options = SpeakOptions(model="aura-asteria-en")
        response = dg.speak.v("1").stream({"text": text}, options)
        audio_data = b"".join(response.stream)
        return FastAPIResponse(content=audio_data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@app.post("/api/voice-check")
async def voice_check(file: UploadFile = File(...), session_id: str = ""):
    """
    Receive audio blob → extract vocal biomarkers → score fatigue/stress/cognitive load.
    If session_id is provided, Claude cross-references last night's sleep data.
    """
    audio_bytes = await file.read()

    # 1. Transcribe with Deepgram for speaking rate
    transcript = ""
    dg_key = os.environ.get("DEEPGRAM_API_KEY")
    if dg_key:
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
            dg = DeepgramClient(dg_key)
            opts = PrerecordedOptions(model="nova-2", language="en-US",
                                      punctuate=True, smart_format=True)
            resp = dg.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio_bytes, "mimetype": file.content_type or "audio/webm"}, opts)
            transcript = resp.results.channels[0].alternatives[0].transcript
        except Exception:
            pass

    # 2. Extract voice biomarkers
    try:
        result = analyze_voice(audio_bytes, transcript=transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {e}")

    # 3. Claude interpretation — cross-reference with sleep if session provided
    interpretation = ""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            sleep_context = ""
            if session_id:
                analysis = session_store.load_analysis(session_id)
                if analysis:
                    s = analysis.get("sleep_summary", {})
                    sleep_context = (
                        f"Last night's sleep: quality score {s.get('quality_score', '?')}/100 "
                        f"({s.get('quality_grade', '?')}), "
                        f"efficiency {s.get('efficiency_pct', '?')}%, "
                        f"awakenings {s.get('awakenings', '?')}."
                    )
            scores = result["scores"]
            feats  = result["features"]
            prompt = (
                f"The user just completed a 30-second voice recording. "
                f"Acoustic biomarker analysis results:\n"
                f"- Fatigue score: {scores['fatigue']}/100\n"
                f"- Stress score: {scores['stress']}/100\n"
                f"- Cognitive load: {scores['cognitive_load']}/100\n"
                f"- Speaking rate: {feats['speaking_rate_wpm']} WPM (normal: 130–160)\n"
                f"- Vocal jitter: {feats['jitter_pct']}% (normal: 0.5–1.0%)\n"
                f"- HNR: {feats['hnr_db']} dB (normal: 15–25 dB)\n"
                f"- F0 mean: {feats['f0_mean_hz']} Hz"
                + (f"\n\n{sleep_context}" if sleep_context else "")
                + "\n\nWrite 2 short paragraphs: (1) what the voice scores reveal about their "
                "current physical/cognitive state, (2) if sleep data is available, how last "
                "night's sleep explains today's vocal patterns. Be specific, warm, and practical."
            )
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}],
            )
            interpretation = resp.content[0].text
        except Exception:
            pass

    result["interpretation"] = interpretation

    # Sentiment analysis on transcript
    if api_key and result.get("transcript", "").strip():
        result["sentiment"] = _sentiment_from_claude(result["transcript"], api_key)
    else:
        result["sentiment"] = {"label": "neutral", "score": 50}

    # Persist voice result so the ASI:One agent can retrieve it
    sid = session_id or "latest-voice"
    session_store.save_voice_result(sid, result)

    return JSONResponse(result)


# ── Fitbit OAuth + Live Sync ─────────────────────────────────────────────────

@app.get("/api/fitbit/auth")
async def fitbit_auth():
    """Redirect the browser to Fitbit's OAuth consent screen."""
    if not fitbit.FITBIT_CLIENT_ID:
        raise HTTPException(status_code=503, detail="FITBIT_CLIENT_ID not configured.")
    from fastapi.responses import RedirectResponse
    auth_url, _ = fitbit.get_auth_url()
    return RedirectResponse(url=auth_url)


@app.get("/api/fitbit/callback")
async def fitbit_callback(code: str = "", state: str = "", error: str = ""):
    """Handle Fitbit OAuth callback. Exchanges code → tokens then redirects to frontend."""
    from fastapi.responses import RedirectResponse
    if error:
        return RedirectResponse(url=f"/?fitbit=error&reason={error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state.")
    try:
        token = await fitbit.exchange_code(code, state)
        uid = token.get("fitbit_user_id", "default")
        return RedirectResponse(url=f"/?fitbit=connected&uid={uid}")
    except Exception as e:
        return RedirectResponse(url=f"/?fitbit=error&reason={str(e)[:80]}")


@app.get("/api/fitbit/status")
async def fitbit_status():
    """Return whether Fitbit is connected and basic profile info."""
    if not fitbit.is_connected():
        return {"connected": False}
    try:
        profile = await fitbit.get_profile()
        user = profile.get("user", {})
        return {
            "connected":   True,
            "display_name": user.get("displayName", ""),
            "avatar":       user.get("avatar150", ""),
            "member_since": user.get("memberSince", ""),
        }
    except Exception:
        return {"connected": True, "display_name": "", "avatar": ""}


@app.post("/api/fitbit/sync")
async def fitbit_sync(req: dict = {}):
    """
    Fetch latest sleep from Fitbit API, convert to our format, and save as a session.
    Optional body: {"date": "YYYY-MM-DD"}  (defaults to today).
    """
    from datetime import date as dt_date
    target_date = req.get("date") or str(dt_date.today())
    if not fitbit.is_connected():
        raise HTTPException(status_code=401, detail="Fitbit not connected. Visit /api/fitbit/auth first.")
    try:
        raw = await fitbit.get_sleep_by_date(target_date)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"Fitbit API error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    sleep_summary = fitbit.fitbit_sleep_to_summary(raw)
    if not sleep_summary:
        return {"message": f"No sleep data found for {target_date}", "session_id": None}

    # Build a full analysis object matching our existing session format
    sid = str(uuid.uuid4())
    result = {
        "session_id":    sid,
        "source":        "fitbit_live",
        "date":          target_date,
        "sleep_summary": sleep_summary,
        "hypnogram":     sleep_summary.pop("epochs", []),
        "metadata":      {"recording_start": target_date, "source": "fitbit_live"},
    }
    session_store.save_analysis(sid, result)
    log_prediction(sid, sleep_summary)
    return result


# ── Google Health OAuth + Live Sync ─────────────────────────────────────────
# Covers Fitbit (via Google Fit sync), Wear OS, Samsung Health, any Google Fit app.

@app.get("/api/health/auth")
async def health_auth():
    """Redirect to Google OAuth consent screen."""
    if not ghealth.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="GOOGLE_CLIENT_ID not configured.")
    auth_url, _ = ghealth.get_auth_url()
    return RedirectResponse(url=auth_url)


@app.get("/api/health/callback")
async def health_callback(code: str = "", state: str = "", error: str = ""):
    """Handle Google OAuth callback → store tokens → redirect to frontend."""
    if error:
        return RedirectResponse(url=f"/?health=error&reason={error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state.")
    try:
        await ghealth.exchange_code(code, state)
        return RedirectResponse(url="/?health=connected")
    except Exception as e:
        import traceback
        from urllib.parse import quote
        err = str(e) or type(e).__name__
        print(f"[health/callback] ERROR: {err}\n{traceback.format_exc()}")
        return RedirectResponse(url=f"/?health=error&reason={quote(err[:120])}")


@app.get("/api/health/status")
async def health_status():
    """Return connection status + Google profile info."""
    if not ghealth.is_connected():
        return {"connected": False}
    try:
        profile = await ghealth.get_profile()
        return {
            "connected":    True,
            "display_name": profile.get("name", ""),
            "email":        profile.get("email", ""),
            "avatar":       profile.get("picture", ""),
        }
    except Exception:
        return {"connected": True, "display_name": "", "email": "", "avatar": ""}


@app.post("/api/health/sync")
async def health_sync(req: dict = {}):
    """
    Fetch latest sleep from Google Fit, convert to our format, save as a session.
    Optional body: {"date": "YYYY-MM-DD"}  (defaults to today).
    """
    from datetime import date as dt_date
    target_date = req.get("date") or str(dt_date.today())

    if not ghealth.is_connected():
        raise HTTPException(
            status_code=401,
            detail="Google Health not connected. Visit /api/health/auth first."
        )
    # Try requested date then walk back up to 7 days to find most recent night with data
    from datetime import date as _dt, timedelta as _td
    sleep_summary = None
    found_date = None
    dates_to_try = [target_date] + [
        str(_dt.today() - _td(days=i)) for i in range(1, 8)
        if str(_dt.today() - _td(days=i)) != target_date
    ]
    for d in dates_to_try:
        try:
            raw = await ghealth.get_sleep_by_date(d)
            sleep_summary = ghealth.google_fit_sleep_to_summary(raw)
            if sleep_summary:
                found_date = d
                break
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"Google Fit API error: {e.response.text[:200]}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    if not sleep_summary:
        return {"message": "No sleep data found in the last 7 days. "
                           "Make sure your Fitbit has synced to Google Fit recently.", "session_id": None}
    target_date = found_date

    sid = str(uuid.uuid4())
    result = {
        "session_id":    sid,
        "source":        "google_health",
        "date":          target_date,
        "sleep_summary": sleep_summary,
        "hypnogram":     sleep_summary.pop("epochs", []),
        "metadata":      {"recording_start": target_date, "source": "google_health"},
    }
    session_store.save_analysis(sid, result)
    log_prediction(sid, sleep_summary)
    return result


@app.get("/api/health/weekly")
async def health_weekly(days: int = 7):
    """Return sleep summaries for up to `days` most recent nights from Google Health."""
    from datetime import datetime as _dt2, timedelta as _td2
    if not ghealth.is_connected():
        raise HTTPException(status_code=401, detail="Google Health not connected. Visit /api/health/auth first.")

    results = []
    today = _dt2.today()
    for i in range(days):
        date_str = str((today - _td2(days=i)).date())
        try:
            raw = await ghealth.get_sleep_by_date(date_str)
            summary = ghealth.google_fit_sleep_to_summary(raw)
            if summary:
                results.append({"date": date_str, "sleep_summary": summary})
        except Exception:
            pass  # skip days with no data or errors

    results.sort(key=lambda x: x["date"])
    return {"days": results}


@app.post("/api/health/upload")
async def health_upload_fitbit_export(file: UploadFile = File(...)):
    """
    Accept a Fitbit sleep export JSON (from Google Takeout:
    Google Health/Global Export Data/sleep-YYYY-MM-DD.json)
    and load it as a session — no OAuth needed for the data itself.
    """
    data = await file.read()
    try:
        raw = json.loads(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

    wrapped = ghealth.parse_fitbit_export_json(raw)
    sleep_summary = fitbit.fitbit_sleep_to_summary(wrapped)
    if not sleep_summary:
        raise HTTPException(
            status_code=422,
            detail="No sleep data found in file. "
                   "Upload a sleep-YYYY-MM-DD.json from Google Takeout → "
                   "Google Health → Global Export Data."
        )

    sid = str(uuid.uuid4())
    date_str = wrapped["sleep"][0].get("dateOfSleep", str(__import__("datetime").date.today()))
    result = {
        "session_id":    sid,
        "source":        "fitbit_export",
        "date":          date_str,
        "sleep_summary": sleep_summary,
        "hypnogram":     sleep_summary.pop("epochs", []),
        "metadata":      {"recording_start": date_str, "source": "fitbit_export"},
        "disorder_flags": disorder_screen(sleep_summary),
    }
    session_store.save_analysis(sid, result)
    log_prediction(sid, sleep_summary)
    return result


@app.get("/api/history")
async def get_history(limit: int = 10):
    """Return the last `limit` analyses as lightweight night summaries for trend display."""
    ids = session_store.list_sessions(limit=limit)
    nights = []
    for sid in ids:
        data = session_store.load_analysis(sid)
        if not data:
            continue
        s = data.get("sleep_summary", {})
        m = data.get("metadata", {})
        pcts = s.get("pct_in_stage", {})
        rem_pct = next((v for k, v in pcts.items() if k.upper() == "REM"), None)
        deep_pct = next((v for k, v in pcts.items() if k.upper() in ("N3", "DEEP")), None)
        nights.append({
            "session_id":     sid,
            "date":           str(m.get("recording_start", ""))[:10],
            "analysis_type":  m.get("analysis_type", ""),
            "quality_score":  s.get("quality_score"),
            "quality_grade":  s.get("quality_grade"),
            "efficiency_pct": s.get("efficiency_pct"),
            "latency_min":    s.get("latency_min"),
            "awakenings":     s.get("awakenings"),
            "rem_pct":        rem_pct,
            "deep_pct":       deep_pct,
            "total_sleep_min": s.get("total_sleep_min"),
        })
    return {"nights": nights}


class WeeklySummaryRequest(BaseModel):
    nights: list


@app.post("/api/weekly-summary")
async def weekly_summary(req: WeeklySummaryRequest):
    if not req.nights:
        raise HTTPException(status_code=400, detail="No nights provided.")
    try:
        text = chat_module.generate_weekly_summary(req.nights)
        return {"summary": text}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DreamRequest(BaseModel):
    session_id: str
    text: str


@app.post("/api/dream")
async def submit_dream(req: DreamRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Dream text cannot be empty.")
    analysis = session_store.load_analysis(req.session_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Session not found — analyse a file first.")
    voice_result = session_store.load_voice_result(req.session_id)
    try:
        correlation = dream_module.analyze_dream(req.text, analysis, voice_result=voice_result)
        dream_module.save_dream(req.session_id, req.text, correlation)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        sentiment = _sentiment_from_claude(req.text, api_key) if api_key else {"label": "neutral", "score": 50}
        return {"analysis": correlation, "sentiment": sentiment}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dream/{session_id}")
async def get_dream(session_id: str):
    data = dream_module.load_dream(session_id)
    if not data:
        return {"dream": None, "analysis": None}
    return data


@app.post("/api/agent/query")
async def agent_query(req: dict):
    """
    Natural-language sleep query endpoint for the ASI:One connector agent.
    Accepts {"user_id": str, "query": str} and returns {"summary": str}.

    Intent routing:
      - latest / last night / how did I sleep → most recent session
      - weekly / trend / debt / consistency   → weekly summary across sessions
      - voice / fatigue / stress              → last voice-check result
      - fallback                              → Claude with demo data
    """
    user_id = req.get("user_id", "")
    query   = req.get("query", "").lower().strip()

    if not query:
        return {"summary": "Please ask a question about your sleep, e.g. 'How did I sleep last night?'"}

    # ── Classify intent ──────────────────────────────────────────────────────
    WEEKLY_KEYWORDS  = {"week", "trend", "debt", "7 day", "seven day", "regularity",
                        "consistent", "average", "history", "this week"}
    VOICE_KEYWORDS   = {"fatigue", "stress", "cognitive", "voice", "tired", "energy"}
    LATEST_KEYWORDS  = {"last night", "tonight", "yesterday", "latest", "recent",
                        "how did i sleep", "sleep score", "efficiency", "rem", "deep",
                        "awakenings", "latency", "quality", "hypnogram"}

    is_weekly = any(k in query for k in WEEKLY_KEYWORDS)
    is_voice  = any(k in query for k in VOICE_KEYWORDS)

    # ── Load data ────────────────────────────────────────────────────────────
    all_sessions = session_store.list_sessions(limit=8)

    if not all_sessions:
        # Fall back to demo data
        demo_path = os.path.join(ROOT, "demo_sleep_report.json")
        if os.path.exists(demo_path):
            with open(demo_path) as f:
                analysis = json.load(f)
        else:
            return {"summary": "No sleep data found. Please upload a sleep file first at sleepsense.ai."}
    else:
        analysis = session_store.load_analysis(all_sessions[0])

    # ── Build context for Claude ─────────────────────────────────────────────
    summary = analysis.get("sleep_summary", {}) if analysis else {}
    stage_pcts = summary.get("pct_in_stage", {})
    stage_mins = summary.get("time_in_stage_min", {})

    if is_weekly and len(all_sessions) > 1:
        scores = []
        efficiencies = []
        for sid in all_sessions[:7]:
            s = session_store.load_analysis(sid)
            if s:
                sm = s.get("sleep_summary", {})
                scores.append(sm.get("quality_score", 0))
                efficiencies.append(sm.get("efficiency_pct", 0))
        weekly_ctx = (
            f"7-night sleep scores: {scores}\n"
            f"7-night efficiencies: {efficiencies}\n"
            f"Average score: {round(sum(scores)/len(scores), 1) if scores else 'N/A'}\n"
            f"Sleep debt estimate: {max(0, round((8*60 - (summary.get('total_sleep_min', 480)))*len(scores)/60, 1))}h over {len(scores)} nights\n"
        )
        context_block = weekly_ctx
    elif is_voice:
        voice_result = session_store.load_voice_result(all_sessions[0]) if all_sessions else None
        if voice_result:
            scores_v = voice_result.get("scores", {})
            context_block = (
                f"Voice health check scores — "
                f"Fatigue: {scores_v.get('fatigue', '?')}/100, "
                f"Stress: {scores_v.get('stress', '?')}/100, "
                f"Cognitive Load: {scores_v.get('cognitive_load', '?')}/100\n"
                f"Cross-referenced with last night: score {summary.get('quality_score', '?')}/100, "
                f"efficiency {summary.get('efficiency_pct', '?')}%"
            )
        else:
            context_block = (
                f"No voice check recorded. Last sleep: score {summary.get('quality_score', '?')}/100, "
                f"efficiency {summary.get('efficiency_pct', '?')}%."
            )
    else:
        context_block = (
            f"Last night's sleep:\n"
            f"- Quality score: {summary.get('quality_score', 'N/A')}/100 (grade {summary.get('quality_grade', '?')})\n"
            f"- Sleep efficiency: {summary.get('efficiency_pct', 'N/A')}%\n"
            f"- Total sleep: {summary.get('total_sleep_min', 'N/A')} min\n"
            f"- Sleep latency: {summary.get('latency_min', 'N/A')} min\n"
            f"- Awakenings: {summary.get('awakenings', 'N/A')}\n"
            f"- Stage breakdown (min): {json.dumps(stage_mins)}\n"
            f"- Stage percentages: {json.dumps(stage_pcts)}\n"
        )

    # ── Generate response — ASI:One preferred, Claude fallback ─────────────
    SYSTEM_PROMPT = (
        "You are SleepSense AI, a concise sleep health assistant integrated with ASI:One. "
        "Answer the user's question using the sleep data provided. "
        "Be direct and conversational — 2–3 sentences max. "
        "Highlight the single most important insight. "
        "Do not diagnose medical conditions."
    )
    user_message = f"Sleep data:\n{context_block}\n\nQuestion: {req.get('query', '')}"

    asi1_key = os.environ.get("ASI1_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if asi1_key:
        # ASI:One — OpenAI-compatible API
        try:
            resp = await httpx.AsyncClient(timeout=20).post(
                "https://api.asi1.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {asi1_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": "asi1-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    "max_tokens": 400,
                },
            )
            resp.raise_for_status()
            summary_text = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            summary_text = f"Sleep data summary: {context_block}"

    elif anthropic_key:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            summary_text = msg.content[0].text
        except Exception:
            summary_text = f"Sleep data summary: {context_block}"

    else:
        summary_text = f"Sleep data: {context_block}"

    return {"summary": summary_text}


@app.get("/api/psychiatric-risk/{session_id}")
async def get_psychiatric_risk(session_id: str):
    """
    Compute psychiatric risk indicators for a session by combining
    sleep biomarkers, voice biomarkers, and multi-night trend.
    """
    analysis = session_store.load_analysis(session_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Session not found.")

    sleep_summary = analysis.get("sleep_summary", {})
    voice_result  = session_store.load_voice_result(session_id)

    # Load recent history for baseline/trend (exclude the current session)
    ids = session_store.get_recent_session_ids(n=8)
    history_nights = []
    for sid in ids:
        if sid == session_id:
            continue
        d = session_store.load_analysis(sid)
        if d:
            s = d.get("sleep_summary", {})
            m = d.get("metadata", {})
            history_nights.append({
                "session_id":     sid,
                "date":           m.get("recording_start", "")[:10],
                "quality_score":  s.get("quality_score"),
                "efficiency_pct": s.get("efficiency_pct"),
                "rem_pct":        next((v for k, v in s.get("pct_in_stage", {}).items()
                                        if k.upper() == "REM"), None),
                "latency_min":    s.get("latency_min"),
                "awakenings":     s.get("awakenings"),
            })

    risk = psych_risk.compute_risk(sleep_summary, voice_result, history_nights)
    return JSONResponse(risk)


_DEMO_PATIENTS = [
    {
        "id":            "demo-healthy",
        "name":          "Alex M.",
        "last_date":     "2026-06-20",
        "quality_score": 86,
        "quality_grade": "A",
        "risk_level":    "none",
        "risk_score":    0,
        "signals":       [],
        "trend_7d":      [84, 87, 85, 88, 86, 89, 86],
        "efficiency":    91.2,
        "rem_pct":       22.1,
        "voice_fatigue": 28,
    },
    {
        "id":            "demo-moderate",
        "name":          "Jordan K.",
        "last_date":     "2026-06-19",
        "quality_score": 61,
        "quality_grade": "C",
        "risk_level":    "moderate",
        "risk_score":    4,
        "signals":       [
            "REM sleep low (14% vs normal 20–25%)",
            "Sleep quality declining over 3+ consecutive nights",
            "Voice fatigue elevated (72/100)",
            "Sleep efficiency reduced (78%)",
        ],
        "trend_7d":      [74, 71, 68, 65, 63, 62, 61],
        "efficiency":    78.0,
        "rem_pct":       14.0,
        "voice_fatigue": 72,
    },
    {
        "id":            "demo-high",
        "name":          "Riley T.",
        "last_date":     "2026-06-20",
        "quality_score": 48,
        "quality_grade": "D",
        "risk_level":    "high",
        "risk_score":    6,
        "signals":       [
            "REM sleep critically low (9% vs normal 20–25%)",
            "Short sleep (4.8h) — chronic sleep restriction",
            "Fragmented sleep (5 awakenings)",
            "Voice fatigue elevated (78/100)",
            "Cognitive load indicator high (71/100)",
            "Speaking rate slow (87 WPM)",
        ],
        "trend_7d":      [55, 52, 50, 51, 48, 47, 48],
        "efficiency":    74.5,
        "rem_pct":       9.0,
        "voice_fatigue": 78,
    },
]


@app.get("/api/provider/patients")
async def provider_patients():
    """
    Return patient list for the provider dashboard.
    Includes the real user's latest data + 3 seeded demo patients.
    """
    patients = []

    # Real user — build from latest session
    ids = session_store.get_recent_session_ids(n=8)
    if ids:
        latest_id = ids[0]
        latest    = session_store.load_analysis(latest_id)
        if latest:
            s = latest.get("sleep_summary", {})
            m = latest.get("metadata", {})
            voice = session_store.load_voice_result(latest_id)

            history_nights = []
            for sid in ids[1:]:
                d = session_store.load_analysis(sid)
                if d:
                    ds = d.get("sleep_summary", {})
                    dm = d.get("metadata", {})
                    history_nights.append({
                        "session_id":     sid,
                        "date":           dm.get("recording_start", "")[:10],
                        "quality_score":  ds.get("quality_score"),
                        "efficiency_pct": ds.get("efficiency_pct"),
                        "rem_pct":        next((v for k, v in
                                                ds.get("pct_in_stage", {}).items()
                                                if k.upper() == "REM"), None),
                        "latency_min":    ds.get("latency_min"),
                        "awakenings":     ds.get("awakenings"),
                    })

            risk = psych_risk.compute_risk(s, voice, history_nights)

            trend_7d = [d.get("quality_score") for d in history_nights
                        if d.get("quality_score") is not None]
            trend_7d.append(s.get("quality_score"))
            trend_7d = [x for x in trend_7d if x is not None][-7:]

            patients.append({
                "id":            latest_id,
                "name":          "You (live data)",
                "last_date":     m.get("recording_start", "")[:10],
                "quality_score": s.get("quality_score"),
                "quality_grade": s.get("quality_grade"),
                "risk_level":    risk["risk_level"],
                "risk_score":    risk["risk_score"],
                "signals":       risk["signals"],
                "trend_7d":      trend_7d,
                "efficiency":    s.get("efficiency_pct"),
                "rem_pct":       next((v for k, v in s.get("pct_in_stage", {}).items()
                                       if k.upper() == "REM"), None),
                "voice_fatigue": (voice or {}).get("scores", {}).get("fatigue"),
                "session_id":    latest_id,
            })

    patients.extend(_DEMO_PATIENTS)
    return {"patients": patients}


@app.get("/provider")
async def provider_page():
    """Serve the provider dashboard HTML."""
    provider_path = os.path.join(ROOT, "frontend", "provider.html")
    if not os.path.exists(provider_path):
        raise HTTPException(status_code=404, detail="Provider dashboard not found.")
    return FileResponse(provider_path)


@app.get("/api/csv-template")
async def csv_template():
    """Download a CSV template for generic wearable data."""
    import wrist_model.parsers.csv_generic as csv_gen
    tmp = tempfile.mktemp(suffix=".csv")
    csv_gen.write_template(tmp)
    return FileResponse(tmp, media_type="text/csv", filename="sleep_data_template.csv")


@app.get("/api/status")
async def status():
    return {
        "eeg_model_ready":   eeg_predictor.model is not None,
        "wrist_model_ready": wrist_predictor.available,
    }


# ── Serve frontend ──────────────────────────────────────────────────────────────
# Mount at /static so API routes at /api/* are never shadowed.
# The root "/" and any SPA path fall back to index.html.

FRONTEND_DIR = os.path.join(ROOT, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        # Don't catch /api/* — those are handled by the routes above
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404)
        index = os.path.join(FRONTEND_DIR, "index.html")
        return FileResponse(index)
