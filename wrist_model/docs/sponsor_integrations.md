# SleepSense AI — Sponsor Integrations

Implement in order. Each integration is independent.

**Current state**: FastAPI backend + dark-mode SPA running at `localhost:8000`.
Core pipeline is complete: EEG → TinySleepNet → hypnogram + metrics + Claude chatbot.

---

## Build order

| # | Integration | Time | Impact |
|---|-------------|------|--------|
| 1 | Sentry | 30 min | Error monitoring — instant production credibility |
| 2 | Redis | 2 hrs | Session persistence, shareable URLs, multi-night history |
| 3 | Deepgram | 2 hrs | Voice chat — speak questions, hear AI answers |
| 4 | Arize | 1 hr | ML observability dashboard for every prediction |

---

## 1. Sentry — Error Monitoring

### `backend/requirements.txt` — add

```
sentry-sdk[fastapi]>=2.0.0
```

### `backend/main.py` — add after imports, before `app = FastAPI(...)`

```python
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
```

### Env var

| Variable | Where to get it |
|----------|----------------|
| `SENTRY_DSN` | sentry.io → Create Project → Python → DSN |

### Verify

```bash
# Trigger a bad request, check sentry.io for the error event
curl -X POST http://localhost:8000/api/analyze -F "file=@/dev/null"
```

---

## 2. Redis — Session Persistence

### `backend/requirements.txt` — add

```
redis>=5.0.0
```

### Create `backend/session.py`

```python
"""
Redis-backed session store. Falls back silently to in-memory dict if
REDIS_URL is not set, so the app works locally without Redis.
"""
import json
import os
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
        _memory_store.setdefault(key, []).append(msg)


def load_messages(session_id: str) -> list[dict]:
    r = _redis()
    key = f"chat:{session_id}"
    items = r.lrange(key, 0, 49) if r else _memory_store.get(key, [])
    return [json.loads(i) for i in items]
```

### `backend/main.py` — changes

**Add import at top:**

```python
import uuid
from backend import session as session_store
```

**In `POST /api/analyze`**, after `result = ...`, before `return JSONResponse(result)`:

```python
    sid = result.get("session_id") or str(uuid.uuid4())
    result["session_id"] = sid
    session_store.save_analysis(sid, result)
```

**In `POST /api/chat`**, after getting `reply`:

```python
    session_store.append_message(req.analysis.get("session_id", ""), "user", req.message)
    session_store.append_message(req.analysis.get("session_id", ""), "ai", reply)
```

**New route — add anywhere before the SPA catch-all:**

```python
@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    data = session_store.load_analysis(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    data["chat_history"] = session_store.load_messages(session_id)
    return JSONResponse(data)
```

### `frontend/index.html` — JS changes

After receiving a successful `/api/analyze` response, store and expose the session:

```javascript
// After successful /api/analyze response, inside the .then(result => ...) block:
localStorage.setItem('sleepSessionId', result.session_id);
history.replaceState({}, '', `?session=${result.session_id}`);
```

On page load, restore a prior session if one is in the URL or localStorage:

```javascript
// At the bottom of the existing DOMContentLoaded or window.onload handler:
const urlSession = new URLSearchParams(window.location.search).get('session');
const savedSession = urlSession || localStorage.getItem('sleepSessionId');
if (savedSession) {
    fetch(`/api/session/${savedSession}`)
        .then(r => r.ok ? r.json() : null)
        .then(data => {
            if (data) {
                renderResults(data);
                if (data.chat_history) {
                    data.chat_history.forEach(msg => appendChatMessage(msg.role, msg.text));
                }
            }
        });
}
```

### Env var

| Variable | Where to get it |
|----------|----------------|
| `REDIS_URL` | Railway: add the Redis plugin — it's injected automatically |

### Verify

```bash
# Analyze any file, then restore by session ID
SID=$(curl -s http://localhost:8000/api/demo | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
curl http://localhost:8000/api/session/$SID
# → returns analysis JSON + {"chat_history": []}
```

---

## 3. Deepgram — Voice Chat

### `backend/requirements.txt` — add

```
deepgram-sdk>=3.0.0
```

### `backend/main.py` — two new routes

Add after the existing `/api/chat` route:

```python
@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Receive audio blob (webm/wav), return transcript."""
    dg_key = os.environ.get("DEEPGRAM_API_KEY")
    if not dg_key:
        raise HTTPException(status_code=503, detail="Deepgram API key not configured.")

    from deepgram import DeepgramClient, PrerecordedOptions

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


@app.post("/api/tts")
async def text_to_speech(req: dict):
    """Convert text to speech audio (MP3)."""
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
```

### `frontend/index.html` — mic button + JS

**HTML** — add inside the chat input form, next to the send button:

```html
<button id="micBtn" type="button" title="Hold to speak" aria-label="Hold to record voice question">🎤</button>
<button id="ttsToggle" type="button" title="Toggle voice replies" aria-label="Toggle voice replies" style="opacity: 0.4">🔊</button>
```

**JS** — add to the script section:

```javascript
let mediaRecorder, audioChunks = [], isRecording = false;
let ttsEnabled = false;

document.getElementById('ttsToggle').addEventListener('click', () => {
    ttsEnabled = !ttsEnabled;
    document.getElementById('ttsToggle').style.opacity = ttsEnabled ? '1' : '0.4';
});

document.getElementById('micBtn').addEventListener('mousedown', async () => {
    try {
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
    } catch (err) {
        console.error('Microphone access denied:', err);
    }
});

document.getElementById('micBtn').addEventListener('mouseup', () => {
    if (isRecording && mediaRecorder) {
        mediaRecorder.stop();
        isRecording = false;
    }
    document.getElementById('micBtn').style.background = '';
});

async function speakReply(text) {
    if (!ttsEnabled) return;
    try {
        const res = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        if (!res.ok) return;
        const blob = await res.blob();
        new Audio(URL.createObjectURL(blob)).play();
    } catch (e) {
        console.warn('TTS failed:', e);
    }
}
// Call speakReply(replyText) after appending each AI message to the chat panel
```

### Env var

| Variable | Where to get it |
|----------|----------------|
| `DEEPGRAM_API_KEY` | console.deepgram.com → Create API Key |

### Verify

Open `http://localhost:8000`, hold the mic button, speak a question, release — the transcript auto-fills and submits. If TTS is toggled on, the AI reply plays as audio.

---

## 4. Arize — ML Observability

### `backend/requirements.txt` — add

```
arize>=7.0.0
```

### Create `backend/arize_logger.py`

```python
"""
Logs each sleep analysis to Arize Phoenix for ML observability.
Gracefully no-ops if ARIZE_API_KEY / ARIZE_SPACE_KEY are not set.
"""
import os

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
    """Log a completed sleep analysis as a scored prediction."""
    client = _get_client()
    if not client:
        return  # silently skip if not configured

    try:
        import pandas as pd
        from arize.utils.types import ModelTypes, Environments, Schema

        summary = analysis.get("sleep_summary", {})
        meta    = analysis.get("metadata", {})

        features = {
            "analysis_type":  meta.get("analysis_type", "unknown"),
            "n_epochs":       meta.get("n_epochs", 0),
            "efficiency_pct": summary.get("efficiency_pct", 0),
            "latency_min":    summary.get("latency_min", 0),
            "awakenings":     summary.get("awakenings", 0),
            "rem_pct":        summary.get("pct_in_stage", {}).get("REM", 0),
            "deep_pct":       summary.get("pct_in_stage", {}).get("N3", 0),
        }

        schema = Schema(
            prediction_id_column_name="id",
            prediction_label_column_name="grade",
            prediction_score_column_name="score",
            feature_column_names=list(features.keys()),
        )

        df = pd.DataFrame([{
            "id":    session_id,
            "grade": summary.get("quality_grade", "?"),
            "score": float(summary.get("quality_score", 0)) / 100.0,
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
```

### `backend/main.py` — hook into `/api/analyze`

After `session_store.save_analysis(sid, result)`:

```python
    from backend.arize_logger import log_prediction
    log_prediction(result["session_id"], result)
```

### Env vars

| Variable | Where to get it |
|----------|----------------|
| `ARIZE_API_KEY` | app.arize.com → Settings → API Keys |
| `ARIZE_SPACE_KEY` | app.arize.com → Settings → API Keys (same page) |

### Verify

After running an analysis, open `app.arize.com` → model `sleepsense-ai` — one prediction row should appear with efficiency, latency, awakenings, and quality grade.

---

## All environment variables

Add to Railway dashboard → Settings → Variables:

| Variable | Required | Source |
|----------|----------|--------|
| `ANTHROPIC_API_KEY` | **Yes** | console.anthropic.com |
| `SENTRY_DSN` | Optional | sentry.io → Project → DSN |
| `REDIS_URL` | Optional | Railway Redis plugin (auto-injected) |
| `DEEPGRAM_API_KEY` | Optional | console.deepgram.com |
| `ARIZE_API_KEY` | Optional | app.arize.com → Settings |
| `ARIZE_SPACE_KEY` | Optional | app.arize.com → Settings |

All optional integrations degrade gracefully — the app runs without them.

---

## Files to create / edit

| File | Action | What changes |
|------|--------|--------------|
| `backend/requirements.txt` | Edit | Add 4 packages |
| `backend/main.py` | Edit | Sentry init, 3 new routes, session + arize hooks in /analyze and /chat |
| `backend/session.py` | **Create** | Redis session module with in-memory fallback |
| `backend/arize_logger.py` | **Create** | Arize prediction logger |
| `frontend/index.html` | Edit | Mic button, TTS toggle, session restore on page load |

---

## End-to-end test

```bash
# Start with all keys
ANTHROPIC_API_KEY=... \
SENTRY_DSN=... \
REDIS_URL=redis://localhost:6379 \
DEEPGRAM_API_KEY=... \
ARIZE_API_KEY=... \
ARIZE_SPACE_KEY=... \
bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 1. Session persistence
SID=$(curl -s http://localhost:8000/api/demo | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
curl http://localhost:8000/api/session/$SID
# → full analysis JSON + chat_history array

# 2. Error monitoring — trigger a 500, check sentry.io
curl -X POST http://localhost:8000/api/analyze -F "file=@/dev/null"

# 3. Voice — open browser at localhost:8000, hold mic, speak, release
# 4. Arize — check app.arize.com → model sleepsense-ai after any analysis
```
