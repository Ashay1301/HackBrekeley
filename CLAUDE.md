# SleepSense AI — Hackathon Project

Built for HackBerkeley. FastAPI backend + dark-mode SPA for sleep stage classification, voice health analysis, and autonomous monitoring.

---

## Project Overview

**SleepSense AI** classifies sleep stages from EEG or wearable data, generates an AI-powered report via Claude, and adds two standalone features: Voice Health Check (acoustic biomarker scoring) and an Autonomous Health Agent (Fetch AI uAgents).

---

## Directory Structure

```
Hackathon/
├── backend/           FastAPI app
│   ├── main.py        All API routes
│   ├── chat.py        Claude claude-sonnet-4-6 chatbot
│   ├── eeg_inference.py  TinySleepNet predictor
│   ├── wrist_inference.py  WristSleepNet predictor
│   ├── sleep_metrics.py   AASM-compliant metrics
│   ├── voice_analysis.py  Acoustic biomarker extraction
│   ├── session.py     Redis session store (in-memory fallback)
│   ├── arize_logger.py    Arize ML observability
│   └── requirements.txt
├── frontend/
│   └── index.html     Dark-mode SPA (Chart.js, no framework)
├── agents/
│   └── sleep_monitor.py   Fetch AI uAgent — daily anomaly check
├── wrist_model/
│   ├── model_wrist.py     1D-CNN + BiLSTM architecture
│   ├── inference.py       WristPredictor class
│   ├── checkpoints/best_model/   ckpt-5 (33.9% MF1, 14 subjects)
│   ├── parsers/           apple_health, fitbit, garmin, csv_generic
│   ├── config/dreamt.py   DREAMT dataset config
│   └── docs/              feature_plan.md, project.md, sponsor_integrations.md
├── final_model_45F1/best_model/  TinySleepNet EEG checkpoint (ckpt-10, 45% MF1)
├── Procfile           web + agent processes
├── railway.toml       Railway deployment config
└── demo_sleep_report.json
```

---

## Models

### TinySleepNet (EEG)
- Architecture: ResNet CNN + BiLSTM
- Classes: 5 (W / N1 / N2 / N3 / REM)
- MF1: **45%** on Sleep-EDF
- Checkpoint: `final_model_45F1/best_model/` — `ckpt-10`
- Input: `.edf` PSG recordings

### WristSleepNet (Wearable)
- Architecture: 1D-CNN + BiLSTM
- Classes: 4 (Wake / Light / Deep / REM)
- MF1: **33.9%** on DREAMT v2.2.0 (14 subjects)
- Checkpoint: `wrist_model/checkpoints/best_model/` — `ckpt-5`
- Input: `.xml` Apple Health / `.json` Fitbit / `.fit` Garmin / `.csv` generic
- **Note**: Limited by only 14 subjects. Retraining with 30+ subjects expected to reach 45–55% MF1.
- seq_length=20 padding: all inference inputs padded to multiple of 20 before BiLSTM reshape

---

## API Routes

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/analyze` | Upload sleep file → hypnogram + metrics. Session auto-saved. |
| POST | `/api/chat` | Claude chatbot with full sleep summary as system context |
| POST | `/api/smart-questions` | Generate 3 Claude-suggested questions for the analysis |
| GET  | `/api/demo` | Load pre-analysed demo report |
| GET  | `/api/session/{id}` | Restore analysis + chat history by session ID |
| POST | `/api/transcribe` | Audio blob → transcript via Deepgram nova-2 |
| POST | `/api/tts` | Text → speech MP3 via Deepgram Aura |
| POST | `/api/voice-check` | Audio blob → fatigue/stress/cognitive load scores + Claude narrative |
| GET  | `/api/csv-template` | Download CSV template for generic wearable data |
| GET  | `/api/status` | Model readiness check |

---

## Voice Health Check

`POST /api/voice-check` — receives a 20–30s audio blob (WebM from browser MediaRecorder).

**Pipeline:**
1. Deepgram transcription for speaking rate (optional; falls back gracefully)
2. `backend/voice_analysis.py` extracts 8 acoustic features via scipy + librosa:
   - F0 mean/std, jitter %, shimmer %, HNR (dB), spectral centroid, RMS energy, ZCR
3. Three 0–100 scores: **Fatigue**, **Stress**, **Cognitive Load**
4. Claude writes 2-paragraph interpretation cross-referencing last night's sleep if `session_id` provided

**Scoring logic** (from vocal fatigue literature):
- Fatigue: high jitter + shimmer + low HNR + low energy + slow speech
- Stress: elevated F0 + high F0 variability + bright spectral centroid + high energy
- Cognitive load: slow speech + low ZCR + high jitter

---

## Autonomous Health Agent (Fetch AI)

`agents/sleep_monitor.py` — Fetch AI uAgents process.

- Runs `@agent.on_interval(period=CHECK_INTERVAL_SECONDS)` (default 86400 = 24h)
- Loads last 8 sessions from Redis via `_load_recent_sessions()`
- Compares tonight vs 7-night baseline using z-score; flags > 1.5 SD bad deviations
- Sends `SleepAlert` model to `ALERT_AGENT_ADDRESS` env var (or logs to stdout)
- Also handles `SleepStatusRequest` from other agents

```bash
# Run with 60s interval for testing
CHECK_INTERVAL_SECONDS=60 bin/python agents/sleep_monitor.py
# → Agent address: agent1qxxxxxxxx
```

---

## Sponsor Integrations (all graceful no-ops when keys not set)

| Sponsor | Integration | Env Var |
|---------|-------------|---------|
| Anthropic | Claude chatbot + voice interpretation | `ANTHROPIC_API_KEY` |
| Fetch AI / ASI:One | ASI:One LLM for agent queries + Agentverse Mailbox | `ASI1_API_KEY` + `AGENTVERSE_API_KEY` |
| Deepgram | Transcription (`/api/transcribe`) + TTS (`/api/tts`) | `DEEPGRAM_API_KEY` |
| Fitbit | Live OAuth sleep sync (`/api/fitbit/*`) | `FITBIT_CLIENT_ID` + `FITBIT_CLIENT_SECRET` + `FITBIT_REDIRECT_URI` |
| Redis | Session persistence + agent baseline | `REDIS_URL` |
| Sentry | Error monitoring (FastAPI integration) | `SENTRY_DSN` |
| Arize | ML observability — logs every prediction | `ARIZE_API_KEY` + `ARIZE_SPACE_KEY` |

---

## Running Locally

```bash
# 1. Create virtualenv and install deps
python -m venv bin
bin/pip install -r backend/requirements.txt

# 2. Run server
ANTHROPIC_API_KEY=sk-ant-... \
bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 3. (Optional) Run agent in a second terminal
REDIS_URL=redis://localhost:6379 \
CHECK_INTERVAL_SECONDS=60 \
bin/python agents/sleep_monitor.py

# 4. Open http://localhost:8000 — click "Try Demo" to test without uploading a file
```

---

## Deployment (Railway)

```toml
# railway.toml already configured
[build]
builder = "nixpacks"
[deploy]
startCommand = "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/api/status"
```

Set these in Railway → Settings → Variables:
- `ANTHROPIC_API_KEY` (required — Claude for frontend chat + voice)
- `ASI1_API_KEY` (required — ASI:One LLM for agent queries)
- `AGENTVERSE_API_KEY` (required — Fetch AI Mailbox for ASI:One discovery)
- `DEEPGRAM_API_KEY` (required — transcription + TTS)
- `FITBIT_CLIENT_ID`, `FITBIT_CLIENT_SECRET` (required for Fitbit live sync)
- `FITBIT_REDIRECT_URI` = `https://<your-railway-url>/api/fitbit/callback`
- `NODE_SERVICE_URL` = `https://<your-railway-url>` (agent → FastAPI URL)
- `REDIS_URL`, `SENTRY_DSN`, `ARIZE_API_KEY`, `ARIZE_SPACE_KEY` (all optional)

---

## Key Technical Decisions

- **seq_length=20 padding**: WristSleepNet BiLSTM requires input length to be a multiple of 20. `wrist_model/inference.py` pads epochs accordingly.
- **No top-level sentry import**: `sentry_sdk` is imported inside a try/except block in `main.py` so the server starts cleanly without the package installed.
- **Deepgram lazy imports**: All `deepgram` imports are deferred inside route functions and wrapped in try/except so the server runs without the SDK.
- **Session store fallback**: `backend/session.py` uses an in-memory dict if `REDIS_URL` is not set — the app is fully functional locally without Redis.
- **Voice analysis without Deepgram**: Speaking rate defaults to 0 WPM; `_score_cognitive_load` uses a 0.5 fallback for the rate component. All other acoustic features still computed.

---

## Known Limitations

- Wrist model (33.9% MF1) is limited by 14 training subjects. Download more from PhysioNet DREAMT v2.2.0 and retrain:
  ```bash
  bin/python wrist_model/download_dreamt.py --username ashaypanchal --workers 1 --n_subjects 30
  bin/python wrist_model/prepare_dreamt.py --raw_dir data/dreamt/raw --out_dir data/dreamt/processed
  bin/python wrist_model/train_wrist.py
  ```
- EEG model expects Sleep-EDF formatted EDF files.
- Voice analysis is heuristic (normalised acoustic features against literature baselines), not trained on a clinical fatigue dataset.

---

## Previous Session Summary

This project was built across multiple sessions:

1. **Session 1–2**: TinySleepNet EEG pipeline, WristSleepNet training on DREAMT, FastAPI SPA, Claude chatbot
2. **Session 3**: Sponsor integrations (Sentry, Redis, Deepgram, Arize), sleep disorder screener design, dream journal design, multi-night dashboard design
3. **Session 4 (this session)**: Implemented Voice Health Check + Fetch AI Autonomous Agent. Created `backend/voice_analysis.py`, `agents/sleep_monitor.py`, `backend/session.py`, `backend/arize_logger.py`. Added `/api/voice-check`, `/api/transcribe`, `/api/tts`, `/api/session/{id}` routes. Added Voice Check UI section to `frontend/index.html`.

### Pending features (designed but not yet implemented — see `wrist_model/docs/feature_plan.md`):
- Sleep Disorder Screener (`backend/disorder_screener.py`) — AASM-based rule engine
- Multi-Night Trend Dashboard — Chart.js trend lines + Claude weekly narrative
- Dream Journal + REM Correlation — Claude correlates dream content with actual REM timing
