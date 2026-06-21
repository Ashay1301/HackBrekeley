Ready for review
Select text to add comments on the plan
Plan: SleepSense AI — Sponsor Integrations (HackBerkeley)
Context
SleepSense AI is a complete, running sleep analysis app (FastAPI + dark-mode SPA). The core pipeline is done: EEG/.edf → TinySleepNet → hypnogram + metrics + Claude chatbot. The goal now is to integrate 4 hackathon sponsors to turn this from a demo into a production app: Sentry (monitoring), Redis (persistence), Deepgram (voice), Arize (ML observability).

Implement in order — each is independent and adds to the previous.

Integration 1: Sentry — Error Monitoring (30 min)
Impact: Zero-effort credibility. Engineers judging will immediately notice.

backend/requirements.txt — add:
sentry-sdk[fastapi]>=2.0.0
backend/main.py — add at the top (after imports, before app = FastAPI(...)):
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

_sentry_dsn = os.environ.get("SENTRY_DSN")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.2,
        environment=os.environ.get("RAILWAY_ENVIRONMENT", "development"),
    )
frontend/index.html — add in <head> before closing tag:
<script>
  const _sentryDsn = window.__SENTRY_DSN__;
  if (_sentryDsn) {
    // Sentry Browser SDK loaded lazily
    import("https://browser.sentry-cdn.com/7.x.x/bundle.min.js").catch(() => {});
  }
</script>
Actually, since the app uses a strict CSP and no external scripts, just skip the browser SDK. The FastAPI backend SDK is sufficient — it will catch all API errors.

Environment variables needed:
SENTRY_DSN — from Sentry dashboard (Create Project → Python → DSN)
Test:
SENTRY_DSN=https://xxx@sentry.io/yyy bin/python -m uvicorn backend.main:app --port 8000
# POST /api/analyze with a bad file → error should appear in Sentry dashboard
Integration 2: Redis — Session Persistence + Multi-Night History (2 hours)
Impact: Users get a shareable URL to their results. Chat history survives page refresh. A returning user sees: "Last time (June 18): efficiency 82%, tonight: 71%" — that's a product.

backend/requirements.txt — add:
redis>=5.0.0
New file: backend/session.py
"""
Redis-backed session storage for sleep analyses and chat history.
Falls back to in-memory dict if Redis is unavailable.
"""
import json
import os
from typing import Optional

_redis_client = None
_memory_store: dict = {}  # fallback when Redis unavailable

def _redis():
    global _redis_client
    if _redis_client is None:
        url = os.environ.get("REDIS_URL") or os.environ.get("REDIS_PRIVATE_URL")
        if url:
            import redis
            _redis_client = redis.from_url(url, decode_responses=True)
    return _redis_client

TTL = 60 * 60 * 24 * 7  # 7 days


def save_analysis(session_id: str, data: dict) -> None:
    r = _redis()
    payload = json.dumps(data)
    if r:
        r.setex(f"analysis:{session_id}", TTL, payload)
    else:
        _memory_store[f"analysis:{session_id}"] = payload


def load_analysis(session_id: str) -> Optional[dict]:
    r = _redis()
    raw = r.get(f"analysis:{session_id}") if r else _memory_store.get(f"analysis:{session_id}")
    return json.loads(raw) if raw else None


def append_message(session_id: str, role: str, text: str) -> None:
    r = _redis()
    key = f"chat:{session_id}"
    msg = json.dumps({"role": role, "text": text})
    if r:
        r.rpush(key, msg)
        r.expire(key, TTL)
    else:
        _memory_store.setdefault(key, [])
        _memory_store[key].append(msg)


def load_messages(session_id: str) -> list[dict]:
    r = _redis()
    key = f"chat:{session_id}"
    if r:
        items = r.lrange(key, 0, 49)  # last 50 messages
    else:
        items = _memory_store.get(key, [])
    return [json.loads(i) for i in items]


def get_all_analyses(session_id: str) -> list[dict]:
    """Return all analyses stored under keys matching this user's session pattern."""
    r = _redis()
    if r:
        keys = r.keys(f"analysis:*")
        results = []
        for k in sorted(keys)[-10:]:  # last 10 sessions
            raw = r.get(k)
            if raw:
                d = json.loads(raw)
                d["_session_id"] = k.split(":", 1)[1]
                results.append(d)
        return results
    return []
backend/main.py — changes:
Import at top:

from backend import session as session_store
In POST /api/analyze — after result = ... and before return JSONResponse(result):

    sid = result.get("session_id") or str(uuid.uuid4())
    result["session_id"] = sid
    session_store.save_analysis(sid, result)
Add import uuid at top if not already there.

In POST /api/chat — after getting reply:

    session_store.append_message(req.analysis.get("session_id", ""), "user", req.message)
    session_store.append_message(req.analysis.get("session_id", ""), "ai", reply)
New route — GET /api/session/{session_id}:

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    data = session_store.load_analysis(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    data["chat_history"] = session_store.load_messages(session_id)
    return JSONResponse(data)
New route — GET /api/history:

@app.get("/api/history")
async def get_history():
    """Return last 10 analyses (for multi-night trend view)."""
    analyses = session_store.get_all_analyses("")
    return {"analyses": analyses}
frontend/index.html — JS changes:
After receiving analysis result, store session ID:

// After successful /api/analyze response:
localStorage.setItem('sleepSessionId', result.session_id);
history.replaceState({}, '', `?session=${result.session_id}`);

// On page load, restore prior session:
const urlParams = new URLSearchParams(window.location.search);
const savedSession = urlParams.get('session') || localStorage.getItem('sleepSessionId');
if (savedSession) {
    fetch(`/api/session/${savedSession}`)
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data) { renderResults(data); restoreChat(data.chat_history); } });
}
Add a "Multi-Night History" button in the header that calls /api/history and shows a comparison table: date, score, efficiency, REM%, latency.

Environment variables needed:
REDIS_URL — Railway auto-injects this when you add Redis plugin
Integration 3: Deepgram — Voice Chat (2 hours)
Impact: The WOW demo moment. Speak a question, hear the AI answer.

backend/requirements.txt — add:
deepgram-sdk>=3.0.0
backend/main.py — new routes:
Import:

from deepgram import DeepgramClient, PrerecordedOptions
Route — POST /api/transcribe:

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Receive audio blob (webm/wav), return transcript."""
    dg_key = os.environ.get("DEEPGRAM_API_KEY")
    if not dg_key:
        raise HTTPException(status_code=503, detail="Deepgram API key not configured.")
    
    audio_bytes = await file.read()
    try:
        dg = DeepgramClient(dg_key)
        options = PrerecordedOptions(
            model="nova-2",
            language="en-US",
            punctuate=True,
            smart_format=True,
        )
        response = dg.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_bytes, "mimetype": file.content_type or "audio/webm"},
            options,
        )
        transcript = response.results.channels[0].alternatives[0].transcript
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
Route — POST /api/tts (text-to-speech):

@app.post("/api/tts")
async def text_to_speech(req: dict):
    """Convert AI reply to speech audio."""
    dg_key = os.environ.get("DEEPGRAM_API_KEY")
    if not dg_key:
        raise HTTPException(status_code=503, detail="Deepgram API key not configured.")
    
    text = req.get("text", "")[:500]  # cap at 500 chars
    try:
        dg = DeepgramClient(dg_key)
        from deepgram import SpeakOptions
        options = SpeakOptions(model="aura-asteria-en")
        response = dg.speak.v("1").stream({"text": text}, options)
        audio_data = b"".join(response.stream)
        from fastapi.responses import Response
        return Response(content=audio_data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")
frontend/index.html — UI changes:
Add mic button next to chat input (in the chat form area):

<button id="micBtn" type="button" title="Hold to speak" style="...">🎤</button>
<button id="ttsToggle" type="button" title="Toggle voice replies" style="...">🔊</button>
Add JS for recording:

let mediaRecorder, audioChunks = [], isRecording = false;
let ttsEnabled = false;

document.getElementById('ttsToggle').addEventListener('click', () => {
    ttsEnabled = !ttsEnabled;
    document.getElementById('ttsToggle').style.opacity = ttsEnabled ? '1' : '0.4';
});

document.getElementById('micBtn').addEventListener('mousedown', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const fd = new FormData();
        fd.append('file', blob, 'audio.webm');
        const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
        const { transcript } = await res.json();
        if (transcript) {
            document.getElementById('chatInput').value = transcript;
            document.getElementById('chatForm').dispatchEvent(new Event('submit'));
        }
        stream.getTracks().forEach(t => t.stop());
    };
    mediaRecorder.start();
    isRecording = true;
    document.getElementById('micBtn').style.background = '#e74c3c';
});

document.getElementById('micBtn').addEventListener('mouseup', () => {
    if (isRecording && mediaRecorder) { mediaRecorder.stop(); isRecording = false; }
    document.getElementById('micBtn').style.background = '';
});

// After receiving AI reply in chat, if ttsEnabled:
async function speakReply(text) {
    if (!ttsEnabled) return;
    const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    new Audio(url).play();
}
// Call speakReply(aiReplyText) after appending AI message to chat
Environment variables needed:
DEEPGRAM_API_KEY — from console.deepgram.com
Integration 4: Arize — ML Observability (1 hour)
Impact: Shows judges a live dashboard of model confidence, stage distribution, and prediction quality across all app users. Answers "how do you know the model works in production?"

backend/requirements.txt — add:
arize>=7.0.0
New file: backend/arize_logger.py
"""
Logs sleep stage predictions to Arize for ML observability.
Gracefully no-ops if ARIZE_API_KEY / ARIZE_SPACE_KEY not set.
"""
import os
import numpy as np
from datetime import datetime

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key   = os.environ.get("ARIZE_API_KEY")
        space_key = os.environ.get("ARIZE_SPACE_KEY")
        if api_key and space_key:
            from arize.api import Client
            _client = Client(space_key=space_key, api_key=api_key)
    return _client


def log_prediction(session_id: str, analysis: dict) -> None:
    """Log a completed sleep analysis to Arize Phoenix."""
    client = _get_client()
    if not client:
        return  # silently skip if not configured

    try:
        from arize.utils.types import ModelTypes, Environments, Schema
        import pandas as pd

        summary = analysis.get("sleep_summary", {})
        meta    = analysis.get("metadata", {})
        epochs  = analysis.get("epoch_by_epoch_data", [])

        # Log aggregate metrics as a single "prediction" record
        features = {
            "analysis_type":  meta.get("analysis_type", "unknown"),
            "n_epochs":       meta.get("n_epochs", len(epochs)),
            "efficiency_pct": summary.get("efficiency_pct", 0),
            "latency_min":    summary.get("latency_min", 0),
            "awakenings":     summary.get("awakenings", 0),
            "rem_pct":        summary.get("pct_in_stage", {}).get("REM", 0),
            "deep_pct":       summary.get("pct_in_stage", {}).get("N3", 0),
        }
        pred_label = summary.get("quality_grade", "?")
        pred_score = float(summary.get("quality_score", 0)) / 100.0

        schema = Schema(
            prediction_id_column_name="id",
            prediction_label_column_name="grade",
            prediction_score_column_name="score",
            feature_column_names=list(features.keys()),
        )
        df = pd.DataFrame([{
            "id":    session_id,
            "grade": pred_label,
            "score": pred_score,
            **features,
        }])

        client.log(
            dataframe=df,
            schema=schema,
            model_id="sleepsense-ai",
            model_version="1.0",
            model_type=ModelTypes.SCORE_CATEGORICAL,
            environment=Environments.PRODUCTION,
        )
    except Exception as e:
        print(f"[Arize] Logging failed (non-fatal): {e}")
backend/main.py — in POST /api/analyze, after session_store.save_analysis(...):
    from backend.arize_logger import log_prediction
    log_prediction(result["session_id"], result)
Environment variables needed:
ARIZE_API_KEY — from app.arize.com → Settings → API Keys
ARIZE_SPACE_KEY — same page
Environment Variables Summary
Add all of these to Railway dashboard (Settings → Variables):

Variable	Source	Required?
ANTHROPIC_API_KEY	console.anthropic.com	Yes (chat)
SENTRY_DSN	sentry.io → project → DSN	Optional
REDIS_URL	Railway auto-injects after adding Redis plugin	Optional
DEEPGRAM_API_KEY	console.deepgram.com	Optional
ARIZE_API_KEY	app.arize.com	Optional
ARIZE_SPACE_KEY	app.arize.com	Optional
File Changes Summary
File	Action	What changes
backend/requirements.txt	Edit	Add 4 packages
backend/main.py	Edit	Import Sentry init, add 4 new routes, call session+arize logging
backend/session.py	Create	Full Redis session module with memory fallback
backend/arize_logger.py	Create	Arize prediction logger
frontend/index.html	Edit	Mic button, TTS toggle, session restore on load, history button
Implementation Order
Sentry — requirements.txt + 5 lines in main.py. Test: deploy, trigger a bad request, check Sentry.
Redis — Create session.py, update main.py routes, update frontend. Test: analyze → refresh page → results restored.
Deepgram — Add 2 routes to main.py, add mic button + JS to frontend. Test: hold mic, speak, see transcript auto-fill.
Arize — Create arize_logger.py, 2-line hook in main.py. Test: analyze → check Arize dashboard for logged prediction.
Verification
# 1. Start server with all keys
ANTHROPIC_API_KEY=... SENTRY_DSN=... REDIS_URL=redis://localhost:6379 \
DEEPGRAM_API_KEY=... ARIZE_API_KEY=... ARIZE_SPACE_KEY=... \
bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 2. Demo mode test
curl http://localhost:8000/api/demo   # should return JSON with session_id

# 3. Session persistence test
SID=$(curl -s http://localhost:8000/api/demo | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
curl http://localhost:8000/api/session/$SID   # should return same data + empty chat_history

# 4. Voice test — open browser, go to http://localhost:8000, hold mic button, speak
# 5. Verify Arize dashboard shows a new prediction row after /api/analyze
# 6. Verify Sentry shows breadcrumbs for each request