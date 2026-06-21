# SleepSense AI — Complete Project Documentation

SleepSense AI classifies sleep stages from medical EEG or consumer wearable data, then lets users explore their results through a Claude-powered chatbot. It runs as a single FastAPI service that serves both the API and the dark-mode SPA frontend.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Directory Structure](#2-directory-structure)
3. [API Reference](#3-api-reference)
4. [EEG Pipeline](#4-eeg-pipeline)
5. [Wrist Pipeline](#5-wrist-pipeline)
6. [Sleep Metrics Engine](#6-sleep-metrics-engine)
7. [Claude Chatbot](#7-claude-chatbot)
8. [Wearable Data Parsers](#8-wearable-data-parsers)
9. [Model Architectures](#9-model-architectures)
10. [Training](#10-training)
11. [API Response Schema](#11-api-response-schema)
12. [Environment Variables](#12-environment-variables)
13. [Local Development](#13-local-development)
14. [Deployment (Railway)](#14-deployment-railway)
15. [Known Limitations](#15-known-limitations)

---

## 1. Architecture Overview

```
Browser
  │
  │  GET /           → frontend/index.html (SPA)
  │  POST /api/analyze  → file upload
  │  POST /api/chat     → Claude chatbot
  │  GET  /api/demo     → pre-built demo report
  │
  ▼
FastAPI  (backend/main.py)
  │
  ├── .edf file  ──────────────────────► EEGPredictor
  │                                        └─ TinySleepNet (ResNet+BiLSTM, 5-class)
  │                                             W / N1 / N2 / N3 / REM
  │
  ├── .xml / .json / .fit / .csv ──────► WristPredictor
  │     │                                  └─ WristSleepNet (1D-CNN+BiLSTM, 4-class)
  │     └─ Parser (Apple / Fitbit /             Wake / Light / Deep / REM
  │                Garmin / CSV)
  │
  ├── any result ──────────────────────► sleep_metrics.analyze_predictions()
  │                                        └─ AASM scoring, quality grade, hypnogram
  │
  └── user message + analysis ─────────► Claude claude-sonnet-4-6
                                           └─ personalised sleep insights
```

---

## 2. Directory Structure

```
sleepMaster/
│
├── backend/                    ← FastAPI application
│   ├── __init__.py
│   ├── main.py                 ← All routes, startup, SPA serving
│   ├── eeg_inference.py        ← EEGPredictor class
│   ├── wrist_inference.py      ← WristPredictor class
│   ├── sleep_metrics.py        ← AASM metric calculations
│   ├── chat.py                 ← Claude API integration
│   └── requirements.txt
│
├── frontend/
│   └── index.html              ← Full dark-mode SPA (Chart.js, vanilla JS)
│
├── wrist_model/
│   ├── model_wrist.py          ← WristSleepNet architecture
│   ├── train_wrist.py          ← Training script
│   ├── prepare_dreamt.py       ← Feature extraction from DREAMT CSVs
│   ├── download_dreamt.py      ← PhysioNet downloader
│   ├── checkpoints/
│   │   └── best_model/         ← Saved wrist model checkpoint
│   ├── config/
│   │   └── dreamt.py           ← Hyperparameters
│   ├── parsers/
│   │   ├── apple_health.py     ← Apple Health .xml
│   │   ├── fitbit.py           ← Fitbit .json
│   │   ├── garmin.py           ← Garmin .fit / .json
│   │   └── csv_generic.py      ← Generic CSV + template writer
│   └── docs/
│       ├── project.md          ← This file
│       └── sponsor_integrations.md
│
├── model.py                    ← TinySleepNet architecture
├── config/
│   └── sleepedf.py             ← EEG training hyperparameters
├── final_model_45F1/
│   └── best_model/             ← EEG checkpoint (45% Macro F1)
│
├── data/
│   ├── sleepedf/               ← SleepEDF-20 raw + processed
│   └── dreamt/
│       ├── raw/                ← DREAMT CSVs (S002_whole_df.csv …)
│       └── processed/          ← Per-subject .npz files
│
├── demo_sleep_report.json      ← Synthetic demo (509 epochs, score 72/B)
├── create_demo.py              ← Generates demo_sleep_report.json
├── Procfile                    ← Railway process definition
└── railway.toml                ← Railway deployment config
```

---

## 3. API Reference

Base URL: `http://localhost:8000` (local) or Railway public URL (production).

---

### POST `/api/analyze`

Upload a sleep data file for analysis.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | binary | `.edf`, `.xml`, `.json`, `.fit`, or `.csv` |

**Routing logic:**

| Extension | Pipeline | Model |
|-----------|----------|-------|
| `.edf` | EEG | TinySleepNet (5-class) |
| `.xml` | Wrist → Apple Health parser | WristSleepNet (4-class) |
| `.json` | Wrist → Fitbit (fallback Garmin) | WristSleepNet (4-class) |
| `.fit` | Wrist → Garmin FIT parser | WristSleepNet (4-class) |
| `.csv` | Wrist → Generic CSV parser | WristSleepNet (4-class) |

**Response:** See [API Response Schema](#11-api-response-schema).

**Errors:**

| Code | Reason |
|------|--------|
| 400 | Unsupported file type or no epochs found |
| 503 | Wrist model not trained yet |
| 500 | Internal inference error |

---

### POST `/api/chat`

Ask the Claude chatbot a question about a sleep analysis.

**Request:**
```json
{
  "message": "Why did I wake up so many times?",
  "analysis": { ...full analysis JSON from /api/analyze... }
}
```

**Response:**
```json
{ "reply": "Based on your 5 awakenings and 71% efficiency..." }
```

**Errors:** 503 if `ANTHROPIC_API_KEY` not set.

---

### POST `/api/smart-questions`

Generate 3 analysis-specific questions for the chat panel.

**Request:** Full analysis dict (same shape as `/api/analyze` response).

**Response:**
```json
{
  "questions": [
    "My sleep quality score is low — what are the biggest factors dragging it down?",
    "My REM sleep was only 12% — is that low, and why does it matter?",
    "It took me 28 minutes to fall asleep — what are the best evidence-based ways to reduce sleep latency?"
  ]
}
```

---

### GET `/api/demo`

Returns a pre-built synthetic sleep report. No file upload needed. Used for the "Try Demo" button.

**Response:** See [API Response Schema](#11-api-response-schema).

The demo report has:
- 509 epochs (~4.3 hours)
- 4 AASM sleep cycles (N1→N2→N3→REM pattern)
- Quality score 72 / Grade B
- Efficiency 87.2%, latency 15 min, 3 awakenings

---

### GET `/api/csv-template`

Download a filled CSV template showing the expected columns for generic wearable data.

**Response:** `text/csv` file with 5 example rows.

---

### GET `/api/status`

Health check. Returns model readiness. Used by Railway's healthcheck.

**Response:**
```json
{
  "eeg_model_ready": true,
  "wrist_model_ready": false
}
```

---

### GET `/` and `GET /{path}`

Serves `frontend/index.html` for all non-API paths (SPA catch-all). Paths starting with `api/` return 404.

---

## 4. EEG Pipeline

**File:** `backend/eeg_inference.py`

### Input

Raw EDF file bytes (`.edf`). Any PSG recording with at least one EEG channel.

### Channel Selection

Channels are tried in this priority order:
1. `EEG Fpz-Cz`
2. `EEG Pz-Oz`
3. Any channel named `EEG` (case-insensitive)
4. First channel in the file

### Signal Processing

1. Read channel at native sampling rate
2. Resample to 100 Hz if needed
3. Segment into 30-second epochs → each epoch is **3000 samples**
4. Pad total epoch count to the nearest multiple of `seq_length=20` (required by BiLSTM reshape)
5. Reshape to model input: `(n_epochs, 3000, 1, 1)`

### Inference

```python
logits = model(input_data, training=False)   # (n_epochs, 5)
probs  = tf.nn.softmax(logits).numpy()       # (n_epochs, 5)
preds  = np.argmax(probs, axis=1)            # (n_epochs,)
```

Padding epochs are trimmed before returning.

### Class Map

| Code | Stage |
|------|-------|
| 0 | W (Wake) |
| 1 | N1 |
| 2 | N2 |
| 3 | N3 (Deep) |
| 4 | REM |

### Checkpoint

`final_model_45F1/best_model/` — 45% Macro F1 on SleepEDF-20 (20 subjects, LOSO cross-validation).

---

## 5. Wrist Pipeline

**File:** `backend/wrist_inference.py`

### Input

A list of epoch dicts, each with 12 feature keys (produced by a parser):

```python
{
    "timestamp":  "2024-01-15 22:30:00",
    "hr_mean":    63.2,
    "hr_std":     4.1,
    "hr_min":     55.0,
    "hr_max":     72.0,
    "hrv_rmssd":  42.0,
    "hrv_sdnn":   38.5,
    "hrv_pnn50":  18.2,
    "accel_mean": 0.023,
    "accel_std":  0.008,
    "accel_zcr":  0.12,
    "eda_mean":   0.0,   # unavailable on most wearables — set 0
    "temp_mean":  0.0,   # unavailable on most wearables — set 0
}
```

### Processing

1. Stack 12 features per epoch into array `(n_epochs, 12)`
2. Pad to multiple of `seq_length=20` with zeros
3. Reshape: `(n_epochs, 12, 1)` — 12 features treated as a 1D signal
4. Model outputs `(n_epochs, 4)` logits
5. Trim padding, apply softmax

### Class Map

| Code | Stage |
|------|-------|
| 0 | Wake |
| 1 | Light (N1 + N2 merged) |
| 2 | Deep (N3) |
| 3 | REM |

### Checkpoint

`wrist_model/checkpoints/best_model/` — trained on DREAMT v2.2.0 (Empatica E4 wristband).
Current: 33.9% Macro F1 on 14 subjects. Expected ~45–55% with 30+ subjects.

---

## 6. Sleep Metrics Engine

**File:** `backend/sleep_metrics.py`

All metric definitions follow AASM (American Academy of Sleep Medicine) standards.

### Functions

#### `sleep_latency(predictions, epoch_sec=30) -> float`

Time from lights-out to sleep onset, in minutes.

Sleep onset = first index `i` where `predictions[i], predictions[i+1], predictions[i+2]` are all non-Wake (code ≠ 0).

Returns total recording duration if no onset found.

#### `count_awakenings(predictions, onset_idx) -> int`

Number of awakenings after sleep onset.

An awakening = a run of ≥2 consecutive Wake epochs that is preceded and followed by sleep. Single-epoch Wake blips are not counted.

#### `quality_score(efficiency, latency_min, rem_pct, deep_pct, awakenings) -> int`

Weighted 0–100 score:

| Component | Weight | Ideal |
|-----------|--------|-------|
| Sleep efficiency | 40% | 100% |
| Sleep latency | 25% | ≤0 min (max bonus), 0 at 30+ min |
| REM % | 20% | ≥25% |
| Deep (N3) % | 15% | ≥20% |
| Awakening penalty | −3 pts each | 1 or fewer |

```
score = efficiency×0.40 + latency_component×0.25 + rem_component×0.20 + deep_component×0.15 − max(0, awakenings−1)×3
```

#### `grade(score) -> str`

| Score | Grade |
|-------|-------|
| ≥85 | A |
| 70–84 | B |
| 55–69 | C |
| <55 | D |

#### `analyze_predictions(predictions, probs, start_dt, epoch_sec, class_names) -> dict`

Master function. Calls all of the above and assembles the full API response body.

Inputs:
- `predictions`: `np.ndarray` shape `(n,)` — integer class codes
- `probs`: `np.ndarray` shape `(n, n_classes)` — softmax probabilities
- `start_dt`: `datetime` — recording start time
- `epoch_sec`: seconds per epoch (30)
- `class_names`: `dict[int, str]` mapping code → label

---

## 7. Claude Chatbot

**File:** `backend/chat.py`

### Model

`claude-sonnet-4-6` via Anthropic SDK. Max 600 tokens per reply.

### System Prompt

Constructed dynamically from the analysis JSON. Includes:
- Quality score and grade
- Efficiency, latency, awakenings
- Total recording and sleep minutes
- Time in each stage (minutes and %)
- Analysis type (eeg / wrist)

Persona: friendly sleep health analyst. Never diagnoses. Offers evidence-based lifestyle recommendations. Answers in 2–4 paragraphs.

### Smart Questions

`generate_smart_questions(analysis) -> list[str]` produces 3 questions tuned to the specific result:

1. **Quality question** — varies based on score bucket (<55, 55–75, ≥75)
2. **Metric question** — targets the worst metric: low REM, low Deep, high awakenings, or generic
3. **Actionable question** — targets latency if >20 min, otherwise bedtime optimisation

---

## 8. Wearable Data Parsers

All parsers live in `wrist_model/parsers/` and return:
```python
list[dict]   # one dict per 30-second epoch, 12 feature keys + "timestamp"
```

Missing values default to physiologically neutral values (HR=65, HRV=30, accel=0, etc.).

---

### Apple Health (`apple_health.py`)

**Input file:** `export.xml` from Apple Health app → Export All Health Data → unzip.

**Extracted records:**
- `HKQuantityTypeIdentifierHeartRate` → hr_mean, hr_std, hr_min, hr_max
- `HKQuantityTypeIdentifierHeartRateVariabilitySDNN` → hrv_sdnn (hrv_rmssd approximated)
- `HKQuantityTypeIdentifierStepCount` → accel proxy
- `HKCategoryTypeIdentifierSleepAnalysis` → defines epoch time window

**Notes:**
- Finds the longest contiguous sleep window in the file
- Apple Health does not export EDA or skin temperature → always 0
- `accel_zcr` estimated from step count density

---

### Fitbit (`fitbit.py`)

**Input file:** `sleep-YYYY-MM-DD.json` from fitbit.com → Data Export.

**Fitbit stage mapping:**

| Fitbit stage | Code |
|-------------|------|
| wake / restless | 0 (Wake) |
| light | 1 (Light) |
| deep | 2 (Deep) |
| rem | 3 (REM) |

**Notes:**
- Classic format (pre-2017, no stage data) raises `ValueError`
- HR loaded from matching `heart_rate-YYYY-MM-DD.json` if present
- HRV estimated from HR variability between adjacent minutes

---

### Garmin (`garmin.py`)

**Input files:**
- Binary `.fit` from Garmin device
- `.json` from Garmin Connect web export

**FIT binary:** reads `sleep_level` messages (stage codes 0–4) and `record` messages (HR).

**JSON:** reads `sleepLevels` or `sleepStages`, `sleepMovement`, and respiration data. Timestamps parsed as Unix milliseconds or ISO strings.

**Notes:**
- Garmin JSON field names vary by device model; parser handles multiple schema variants
- HRV always defaulted to 30/27 (Garmin exports do not include per-epoch HRV)

---

### Generic CSV (`csv_generic.py`)

**Input file:** Any CSV with at minimum a `timestamp` column and a `hr_mean` column.

**Supported timestamp formats:**
```
YYYY-MM-DD HH:MM:SS
YYYY-MM-DDTHH:MM:SS
YYYY/MM/DD HH:MM:SS
MM/DD/YYYY HH:MM:SS
```

**Auto-resampling:** If rows are spaced >30s apart, the parser duplicates rows to ensure 30s epoch granularity.

**Template:** `GET /api/csv-template` returns a filled example with all 12 feature columns.

---

## 9. Model Architectures

### TinySleepNet (EEG)

**File:** `model.py`

Input shape: `(batch × seq_length, 3000, 1, 1)` — 30s EEG at 100 Hz.

```
Input (3000, 1, 1)
│
├── Conv2D(128, kernel=(50,1), stride=(6,1)) → BN → ReLU → MaxPool(8,1) → Dropout(0.5)
│
├── ResBlock 1:
│   ├── Conv2D(128, k=7) → BN → ReLU → Conv2D(128, k=7) → BN
│   └── Add(skip) → ReLU
│
├── ResBlock 2:
│   ├── Conv2D(128, k=7) → BN → ReLU → Conv2D(128, k=7) → BN
│   └── Add(skip) → ReLU
│
├── MaxPool(4,1) → Flatten → Dropout(0.5)
│
│   Reshape to (batch, seq_length=20, features)
│
├── Bidirectional(LSTM(128, return_sequences=True)) → Dropout(0.5)
│
│   Reshape back to (batch×seq_length, 256)
│
└── Dense(5)  ← logits for W / N1 / N2 / N3 / REM
```

Parameters: ~2M. L2 weight decay: 1e-5.

---

### WristSleepNet (Wrist)

**File:** `wrist_model/model_wrist.py`

Input shape: `(batch × seq_length, 12, 1)` — 12 engineered features per epoch.

```
Input (12, 1)
│
├── Conv1D(64, k=3, same) → BN → ReLU
├── Conv1D(128, k=3, same) → BN → ReLU → MaxPool(2) → Dropout(0.5)
│
├── Flatten
│
│   Reshape to (batch, seq_length=20, features)
│
├── Bidirectional(LSTM(64, return_sequences=True)) → Dropout(0.5)
│
│   Reshape back to (batch×seq_length, 128)
│
├── Dense(64, relu) → Dropout(0.5)
│
└── Dense(4)  ← logits for Wake / Light / Deep / REM
```

Parameters: ~200K. L2 weight decay: 1e-5.

---

## 10. Training

### EEG Model

**Dataset:** SleepEDF-20 (PhysioNet, 20 subjects, Cassette study)
**Preprocessing:** `prepare_sleepedf.py` → `data/sleepedf/`
**Script:** `train_final.py`

**Key config (`config/sleepedf.py`):**

| Parameter | Value |
|-----------|-------|
| Sampling rate | 100 Hz |
| Input size | 3000 (30s × 100Hz) |
| seq_length | 20 |
| batch_size | 15 subjects |
| n_epochs | 200 |
| learning_rate | 1e-5 |
| early_stopping | 50 epochs no improvement |
| n_classes | 5 |

**Loss:** SparseCategoricalFocalLoss (γ=2.0)
**Optimiser:** Adam + WarmupCosineDecay LR schedule
**Class weights:** Balanced per training fold

**Best result:** 45% Macro F1 (checkpoint `final_model_45F1`). This is competitive with published TinySleepNet results on this dataset.

---

### Wrist Model

**Dataset:** DREAMT v2.2.0 (PhysioNet, Empatica E4 wristband)
**Preprocessing:** `wrist_model/prepare_dreamt.py` → `data/dreamt/processed/`
**Script:** `wrist_model/train_wrist.py`

**Key config (`wrist_model/config/dreamt.py`):**

| Parameter | Value |
|-----------|-------|
| Input features | 12 (engineered) |
| seq_length | 20 |
| batch_size | 15 |
| n_epochs | 150 |
| learning_rate | 1e-4 |
| warmup_epochs | 5 |
| early_stopping | 30 epochs no improvement |
| n_classes | 4 |

**Label mapping (5-class AASM → 4-class):**

| AASM | Code | Merged as |
|------|------|-----------|
| Wake | 0 | Wake |
| N1 | 1 | Light |
| N2 | 2 | Light |
| N3 | 3 | Deep |
| REM | 4 | REM |

**Run training:**
```bash
bin/python wrist_model/train_wrist.py \
  --data_dir data/dreamt/processed \
  --output_dir wrist_model/checkpoints
```

**Current performance:** 33.9% Macro F1 on 14 subjects. Expected 45–55% with 30+ subjects (download more when PhysioNet unblocks).

**Download more data:**
```bash
bin/python wrist_model/download_dreamt.py --username ashaypanchal --workers 1 --n_subjects 30
```

**Feature extraction after download:**
```bash
bin/python wrist_model/prepare_dreamt.py \
  --raw_dir data/dreamt/raw \
  --out_dir data/dreamt/processed
```

---

## 11. API Response Schema

Both `/api/analyze` and `/api/demo` return this shape:

```json
{
  "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",

  "metadata": {
    "analysis_type": "eeg",
    "channel": "EEG Fpz-Cz",
    "sampling_rate_hz": 100.0,
    "n_epochs": 509,
    "recording_start": "2023-10-12 22:15:00"
  },

  "sleep_summary": {
    "quality_score": 72,
    "quality_grade": "B",
    "efficiency_pct": 87.2,
    "latency_min": 15.0,
    "awakenings": 3,
    "total_recording_min": 254.5,
    "total_sleep_min": 221.9,
    "time_in_stage_min": {
      "W": 32.6,
      "N1": 18.0,
      "N2": 96.5,
      "N3": 44.0,
      "REM": 63.4
    },
    "pct_in_stage": {
      "W": 12.8,
      "N1": 7.1,
      "N2": 37.9,
      "N3": 17.3,
      "REM": 24.9
    }
  },

  "epoch_by_epoch_data": [
    {
      "epoch_number": 1,
      "timestamp": "2023-10-12 22:15:00",
      "predicted_stage_code": 0,
      "predicted_stage_name": "W",
      "confidence": 0.923,
      "probabilities": {
        "W": 0.923,
        "N1": 0.041,
        "N2": 0.022,
        "N3": 0.008,
        "REM": 0.006
      }
    }
  ]
}
```

**Wrist analysis difference:** `metadata.analysis_type = "wrist"`, `metadata.source = "Apple Health"` (or Fitbit / Garmin / CSV), stage names are `Wake / Light / Deep / REM` instead of `W / N1 / N2 / N3 / REM`.

---

## 12. Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | **Yes** | Claude API key. Chat endpoint returns 503 without it. |
| `PORT` | Railway only | Injected by Railway. Uvicorn binds to `$PORT`. |
| `SENTRY_DSN` | Optional | Sentry error tracking DSN. App runs normally without it. |
| `REDIS_URL` | Optional | Redis connection string. Sessions fall back to in-memory without it. |
| `DEEPGRAM_API_KEY` | Optional | Deepgram API key for voice transcription + TTS. |
| `ARIZE_API_KEY` | Optional | Arize prediction logging. |
| `ARIZE_SPACE_KEY` | Optional | Arize space key (same settings page as API key). |
| `RAILWAY_ENVIRONMENT` | Railway only | Passed to Sentry as environment name. |

---

## 13. Local Development

### Prerequisites

- Python 3.11 (the repo ships a conda env at `bin/python`)
- TensorFlow (metal plugin for Apple Silicon or standard for Linux)

### Start the server

```bash
# With Claude chat enabled:
ANTHROPIC_API_KEY=your_key bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Without chat (EEG + wrist analysis still work):
bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.

### Verify models are loaded

```bash
curl http://localhost:8000/api/status
# {"eeg_model_ready": true, "wrist_model_ready": true}
```

### Test with demo data

```bash
curl http://localhost:8000/api/demo | python3 -m json.tool | head -40
```

### Install backend dependencies

```bash
bin/python -m pip install -r backend/requirements.txt
```

### VS Code interpreter

The repo includes `.vscode/settings.json` pointing to `bin/python`. Open the project folder in VS Code and it picks up the right interpreter automatically.

---

## 14. Deployment (Railway)

### `railway.toml`

```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/api/status"
healthcheckTimeout = 60
restartPolicyType = "on_failure"
```

### `Procfile`

```
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

### Steps

1. Push code to GitHub (include `final_model_45F1/best_model/` and `wrist_model/checkpoints/best_model/`)
2. `railway link` → select project
3. Set env vars in Railway dashboard → Settings → Variables
4. `railway up`
5. Railway runs healthcheck at `/api/status` — goes green once both models load

### Model checkpoint sizes

| Checkpoint | Approx size |
|------------|-------------|
| `final_model_45F1/best_model/` | ~30 MB |
| `wrist_model/checkpoints/best_model/` | ~5 MB |

Both fit well under Railway's 100 MB repo limit.

---

## 15. Known Limitations

| Area | Limitation | Workaround / Fix |
|------|-----------|-----------------|
| Wrist model accuracy | 33.9% MF1 on 14 subjects | Download 30+ subjects from PhysioNet, retrain |
| EEG channel support | Only single-channel EEG (Fpz-Cz preferred) | Multi-channel fusion not implemented |
| EDF recording length | Very long recordings (>12h) may be slow | First run allocates TF graph; subsequent calls are fast |
| Fitbit classic format | Pre-2017 Fitbit export with no stage data → error | User must re-export using Fitbit's stage-aware API |
| Garmin HRV | Always defaulted to 30/27 | Garmin Connect does not export per-epoch HRV in standard exports |
| Apple Health accel | `accel_zcr` is a step-count proxy, not raw accelerometry | Full accel requires raw sensor export (HealthKit) |
| Chat context | No conversation history — each `/api/chat` call is stateless | See `sponsor_integrations.md` §2 (Redis) for multi-turn memory |
| PhysioNet downloads | IP rate-limited; server does not support HTTP byte-range resume | Use 1 worker, wait 1–2 hours after IP block, then resume |
| seq_length padding | Inputs must be padded to multiples of 20 before BiLSTM inference | Already implemented in both EEGPredictor and WristPredictor |
