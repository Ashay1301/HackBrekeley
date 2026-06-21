# Plan: Voice Health Check + Autonomous Monitoring Agent

## Context

Two new features that go beyond sleep analysis and stand alone as their own demos.

**Voice Health Check** — Record 30 seconds of speech. Extract acoustic biomarkers (F0, jitter, shimmer, HNR, speaking rate) using `scipy` + `librosa`. Score fatigue, stress, and cognitive load. Claude cross-references with last night's sleep to explain the connection. Uses Deepgram for transcription.

**Autonomous Health Agent** — A Fetch AI `uAgents` process that runs on a daily timer, loads the user's latest sleep session from Redis, compares against their personal baseline, and sends an alert when something is anomalous. Runs alongside the FastAPI server as a separate process.

**Environment:** `scipy 1.16.1` already installed. `librosa` and `uagents` need to be installed.

---

## Feature 1: Voice Health Check

### How it works

1. Frontend: user clicks "Voice Check", records 30s of the rainbow passage (standardised text)
2. `POST /api/voice-check` receives the audio blob
3. Backend extracts 8 acoustic features with `scipy` + `librosa`
4. Features are normalised into three 0–100 scores: Fatigue, Stress, Cognitive Load
5. Claude reads the scores + last night's sleep summary and writes a 2-paragraph interpretation

The tie-back to sleep data is what makes this distinctive — "your vocal fatigue score is 71, which aligns with your fragmented sleep last night (score 68/C)."

---

### `backend/requirements.txt` — add

```
librosa>=0.10.0
soundfile>=0.12.0
```

---

### Create `backend/voice_analysis.py`

```python
"""
Voice biomarker extraction for fatigue / stress / cognitive load scoring.
Uses scipy (already installed) + librosa for audio feature extraction.

Reference features:
  - F0 (fundamental frequency): lower + less variable → fatigue, depression
  - Jitter: cycle-to-cycle F0 perturbation → vocal stress / tremor
  - Shimmer: amplitude perturbation → fatigue, respiratory issues  
  - HNR (harmonics-to-noise ratio): lower → hoarseness, fatigue
  - Speaking rate (WPM from transcript): slower → cognitive fatigue
  - Spectral centroid: higher → alertness
  - RMS energy: lower → fatigue / low arousal
  - Zero crossing rate: correlates with voice clarity
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Optional


# ── Low-level feature extractors ────────────────────────────────────────────

def _load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load raw audio bytes → (samples float32, sample_rate)."""
    import io
    import soundfile as sf
    buf = io.BytesIO(audio_bytes)
    samples, sr = sf.read(buf, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)   # stereo → mono
    return samples, sr


def _f0_series(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimate fundamental frequency using autocorrelation.
    Returns array of F0 values (Hz) for voiced frames; 0 for unvoiced.
    """
    frame_len = int(sr * 0.04)   # 40ms frames
    hop       = int(sr * 0.01)   # 10ms hop
    f0_values = []

    for start in range(0, len(samples) - frame_len, hop):
        frame = samples[start:start + frame_len]
        frame -= frame.mean()

        # Autocorrelation
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]

        # Find first peak after the global minimum
        min_lag = int(sr / 500)   # max F0 = 500 Hz
        max_lag = int(sr / 60)    # min F0 = 60 Hz
        if max_lag > len(corr):
            f0_values.append(0)
            continue

        sub = corr[min_lag:max_lag]
        if len(sub) == 0 or corr[0] == 0:
            f0_values.append(0)
            continue

        # Peak prominence threshold: voiced frames have strong autocorrelation
        peak_idx = np.argmax(sub) + min_lag
        if corr[peak_idx] / corr[0] > 0.3:
            f0_values.append(sr / peak_idx)
        else:
            f0_values.append(0)

    voiced = np.array([f for f in f0_values if f > 0])
    return voiced


def _jitter(f0_series: np.ndarray) -> float:
    """Relative jitter: mean absolute F0 difference / mean F0."""
    if len(f0_series) < 2:
        return 0.0
    diffs = np.abs(np.diff(f0_series))
    return float(np.mean(diffs) / np.mean(f0_series)) * 100   # percent


def _shimmer(samples: np.ndarray, sr: int, f0_series: np.ndarray) -> float:
    """Approximate shimmer from short-time RMS variation."""
    frame_len = int(sr * 0.04)
    hop       = int(sr * 0.01)
    rms_vals  = []
    for start in range(0, len(samples) - frame_len, hop):
        frame = samples[start:start + frame_len]
        rms_vals.append(np.sqrt(np.mean(frame ** 2)))
    rms_vals = np.array(rms_vals)
    if len(rms_vals) < 2:
        return 0.0
    diffs = np.abs(np.diff(rms_vals))
    mean_rms = np.mean(rms_vals)
    return float(np.mean(diffs) / mean_rms) * 100 if mean_rms > 0 else 0.0


def _hnr(samples: np.ndarray, sr: int) -> float:
    """
    Harmonics-to-noise ratio via autocorrelation peak.
    Higher = cleaner voice. Typical healthy range: 15–25 dB.
    """
    frame_len = int(sr * 0.04)
    hnr_vals  = []
    for start in range(0, len(samples) - frame_len, frame_len):
        frame = samples[start:start + frame_len]
        frame -= frame.mean()
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        if corr[0] == 0:
            continue
        min_lag = int(sr / 400)
        max_lag = int(sr / 75)
        if max_lag >= len(corr):
            continue
        r0  = corr[0]
        rp  = np.max(corr[min_lag:max_lag])
        ratio = rp / r0
        if 0 < ratio < 1:
            hnr_vals.append(10 * np.log10(ratio / (1 - ratio + 1e-9)))
    return float(np.median(hnr_vals)) if hnr_vals else 10.0


def _spectral_centroid(samples: np.ndarray, sr: int) -> float:
    """Mean spectral centroid (Hz). Higher → brighter / more alert voice."""
    freqs, times, Sxx = scipy_signal.spectrogram(samples, sr, nperseg=512)
    if Sxx.sum() == 0:
        return 0.0
    centroids = np.sum(freqs[:, None] * Sxx, axis=0) / (Sxx.sum(axis=0) + 1e-9)
    return float(np.mean(centroids))


def _rms_energy(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2)))


def _zero_crossing_rate(samples: np.ndarray) -> float:
    signs = np.sign(samples)
    return float(np.mean(np.abs(np.diff(signs)) > 0))


# ── Scoring ──────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _score_fatigue(jitter: float, shimmer: float, hnr: float,
                   rms: float, speaking_rate_wpm: float) -> int:
    """
    Higher score → more fatigue.
    Reference baselines from published vocal fatigue literature.
    """
    # Jitter: healthy ~0.5–1.0%, fatigued >1.5%
    j_score = _clamp((jitter - 0.5) / 2.0, 0, 1)
    # Shimmer: healthy ~2–4%, fatigued >5%
    s_score = _clamp((shimmer - 2.0) / 5.0, 0, 1)
    # HNR: healthy >18dB, fatigued <12dB
    h_score = _clamp((18 - hnr) / 12.0, 0, 1)
    # Energy: fatigued voice is quieter
    e_score = _clamp((0.05 - rms) / 0.05, 0, 1)
    # Speaking rate: healthy ~130–160 WPM, fatigued <100 WPM
    r_score = _clamp((130 - speaking_rate_wpm) / 80.0, 0, 1) if speaking_rate_wpm > 0 else 0.5

    weighted = j_score * 0.25 + s_score * 0.20 + h_score * 0.25 + e_score * 0.15 + r_score * 0.15
    return round(_clamp(weighted * 100, 0, 100))


def _score_stress(f0_mean: float, f0_std: float, spectral_centroid: float, rms: float) -> int:
    """
    Stress raises mean F0 and energy. High variability also indicates stress.
    Approximate baselines: male F0 ~120Hz, female ~220Hz — we use z-score relative to neutral.
    """
    # Elevated F0 (stress lifts pitch)
    if f0_mean > 0:
        baseline = 160.0   # rough cross-gender midpoint
        f0_stress = _clamp((f0_mean - baseline) / 120.0, -0.5, 1.0)
    else:
        f0_stress = 0.2

    # High F0 variability (anxious speech is erratic)
    if f0_mean > 0:
        cv = f0_std / f0_mean
        var_stress = _clamp((cv - 0.05) / 0.2, 0, 1)
    else:
        var_stress = 0.2

    # Bright spectral centroid (tense vocal tract)
    sc_stress = _clamp((spectral_centroid - 1500) / 1500, 0, 1)

    # High energy (vocal effort under stress)
    e_stress = _clamp((rms - 0.05) / 0.1, 0, 1)

    weighted = f0_stress * 0.30 + var_stress * 0.25 + sc_stress * 0.25 + e_stress * 0.20
    return round(_clamp(weighted * 100, 0, 100))


def _score_cognitive_load(speaking_rate_wpm: float, zcr: float, jitter: float) -> int:
    """
    High cognitive load → slower, more hesitant speech with more disfluencies.
    """
    # Very slow = high load (below 90 WPM)
    r_load = _clamp((110 - speaking_rate_wpm) / 80.0, 0, 1) if speaking_rate_wpm > 0 else 0.5
    # Low ZCR = less articulation
    z_load = _clamp((0.15 - zcr) / 0.12, 0, 1)
    # High jitter = hesitant
    j_load = _clamp((jitter - 0.8) / 2.0, 0, 1)

    weighted = r_load * 0.50 + z_load * 0.25 + j_load * 0.25
    return round(_clamp(weighted * 100, 0, 100))


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_voice(audio_bytes: bytes, transcript: str = "") -> dict:
    """
    Full voice biomarker pipeline.

    Parameters
    ----------
    audio_bytes : raw audio (webm/wav/ogg — soundfile handles the codec)
    transcript  : Deepgram transcript used for speaking rate

    Returns
    -------
    dict with scores (0–100), raw features, and a features_summary dict
    """
    samples, sr = _load_audio(audio_bytes)
    duration_sec = len(samples) / sr

    # Raw features
    f0     = _f0_series(samples, sr)
    f0_mean = float(np.mean(f0)) if len(f0) > 0 else 0.0
    f0_std  = float(np.std(f0))  if len(f0) > 0 else 0.0
    jitter  = _jitter(f0)
    shimmer = _shimmer(samples, sr, f0)
    hnr     = _hnr(samples, sr)
    sc      = _spectral_centroid(samples, sr)
    rms     = _rms_energy(samples)
    zcr     = _zero_crossing_rate(samples)

    # Speaking rate from transcript
    word_count = len(transcript.split()) if transcript else 0
    wpm = (word_count / duration_sec * 60) if duration_sec > 0 and word_count > 0 else 0.0

    # Scores
    fatigue       = _score_fatigue(jitter, shimmer, hnr, rms, wpm)
    stress        = _score_stress(f0_mean, f0_std, sc, rms)
    cognitive_load = _score_cognitive_load(wpm, zcr, jitter)

    return {
        "scores": {
            "fatigue":        fatigue,
            "stress":         stress,
            "cognitive_load": cognitive_load,
        },
        "features": {
            "f0_mean_hz":       round(f0_mean, 1),
            "f0_std_hz":        round(f0_std, 1),
            "jitter_pct":       round(jitter, 3),
            "shimmer_pct":      round(shimmer, 3),
            "hnr_db":           round(hnr, 1),
            "spectral_centroid_hz": round(sc, 0),
            "rms_energy":       round(rms, 5),
            "zero_crossing_rate": round(zcr, 4),
            "speaking_rate_wpm": round(wpm, 1),
            "duration_sec":     round(duration_sec, 1),
        },
        "transcript": transcript,
    }
```

---

### `backend/main.py` — one new route

Add import:
```python
from backend.voice_analysis import analyze_voice
```

Add route (after `/api/transcribe`):

```python
@app.post("/api/voice-check")
async def voice_check(file: UploadFile = File(...), session_id: str = ""):
    """
    Receive audio blob → extract vocal biomarkers → score fatigue/stress/cognitive load.
    If session_id is provided, Claude cross-references last night's sleep data.
    """
    audio_bytes = await file.read()

    # 1. Transcribe with Deepgram (for speaking rate + transcript)
    transcript = ""
    dg_key = os.environ.get("DEEPGRAM_API_KEY")
    if dg_key:
        try:
            from deepgram import DeepgramClient, PrerecordedOptions
            dg = DeepgramClient(dg_key)
            opts = PrerecordedOptions(model="nova-2", language="en-US",
                                      punctuate=True, smart_format=True)
            resp = dg.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio_bytes,
                 "mimetype": file.content_type or "audio/webm"}, opts)
            transcript = resp.results.channels[0].alternatives[0].transcript
        except Exception:
            pass   # transcript stays empty; speaking rate will be 0

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
            from backend import chat as chat_module
            sleep_context = ""
            if session_id:
                from backend import session as session_store
                analysis = session_store.load_analysis(session_id)
                if analysis:
                    s = analysis.get("sleep_summary", {})
                    sleep_context = (
                        f"Last night's sleep: quality score {s.get('quality_score','?')}/100 "
                        f"({s.get('quality_grade','?')}), "
                        f"efficiency {s.get('efficiency_pct','?')}%, "
                        f"awakenings {s.get('awakenings','?')}."
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
                f"- F0 mean: {feats['f0_mean_hz']} Hz\n"
                + (f"\n{sleep_context}" if sleep_context else "")
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
    return JSONResponse(result)
```

---

### `frontend/index.html` — Voice Check tab

**Add tab button next to the existing nav (or as a standalone section):**

```html
<!-- Voice check section — shown always, not just after analysis -->
<div id="voiceSection" style="margin-top: 40px; padding-top: 28px; border-top: 1px solid #253347;">
    <h3 style="color:#fff; font-size:16px; margin-bottom:6px;">🎙️ Voice Health Check</h3>
    <p style="font-size:13px; color:#6E8099; margin-bottom:16px;">
        Read the following sentence aloud for 20–30 seconds. We'll analyse your voice for
        signs of fatigue, stress, and cognitive load.
    </p>

    <div style="
        background: rgba(77,255,210,0.05); border: 1px solid #253347;
        border-radius: 2px; padding: 14px 16px; margin-bottom: 16px;
        font-style: italic; font-size: 14px; color: #D4DDE8; line-height: 1.7;
    ">
        "When the sunlight strikes raindrops in the air, they act as a prism and form a
        rainbow. The rainbow is a division of white light into many beautiful colors.
        These take the shape of a long round arch, with its path high above, and its
        two ends apparently beyond the horizon."
    </div>

    <div style="display:flex; gap:12px; align-items:center; margin-bottom:20px;">
        <button id="voiceRecordBtn" style="
            padding: 10px 20px; background: #172032; border: 1px solid #4DFFD2;
            color: #4DFFD2; border-radius: 2px; cursor: pointer; font-size: 14px;
            font-weight: 600;
        ">⏺ Start Recording</button>
        <span id="voiceTimer" style="font-size:13px; color:#6E8099;"></span>
    </div>

    <div id="voiceResults" style="display:none;">
        <!-- Score gauges -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;">
            <div id="fatigueGauge"  class="voice-gauge"></div>
            <div id="stressGauge"   class="voice-gauge"></div>
            <div id="cogLoadGauge"  class="voice-gauge"></div>
        </div>
        <!-- Claude interpretation -->
        <div id="voiceInterpretation" style="
            font-size:14px; color:#D4DDE8; line-height:1.7;
            background: rgba(77,255,210,0.04); border: 1px solid #253347;
            border-radius: 2px; padding: 16px;
        "></div>
    </div>
</div>
```

**CSS for gauges (add to `<style>`):**
```css
.voice-gauge {
    background: #172032;
    border: 1px solid #253347;
    border-radius: 2px;
    padding: 16px;
    text-align: center;
}
.voice-gauge .gauge-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6E8099; margin-bottom: 8px;
}
.voice-gauge .gauge-value {
    font-size: 36px; font-weight: 700; line-height: 1;
}
.voice-gauge .gauge-bar {
    height: 4px; background: #253347; border-radius: 2px; margin-top: 8px;
}
.voice-gauge .gauge-fill {
    height: 100%; border-radius: 2px; transition: width 0.6s ease;
}
```

**JS — recording + display:**
```javascript
let voiceRecorder, voiceChunks = [], voiceTimerInterval;
let isVoiceRecording = false;

document.getElementById('voiceRecordBtn').addEventListener('click', async () => {
    if (!isVoiceRecording) {
        // Start recording
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        voiceChunks = [];
        voiceRecorder = new MediaRecorder(stream);
        voiceRecorder.ondataavailable = e => voiceChunks.push(e.data);
        voiceRecorder.onstop = () => submitVoiceCheck(stream);
        voiceRecorder.start();
        isVoiceRecording = true;

        document.getElementById('voiceRecordBtn').textContent = '⏹ Stop & Analyse';
        document.getElementById('voiceRecordBtn').style.borderColor = '#FF7059';
        document.getElementById('voiceRecordBtn').style.color = '#FF7059';

        // Timer
        let secs = 0;
        voiceTimerInterval = setInterval(() => {
            secs++;
            document.getElementById('voiceTimer').textContent = `${secs}s recorded`;
            if (secs >= 60) { voiceRecorder.stop(); }  // auto-stop at 60s
        }, 1000);
    } else {
        voiceRecorder.stop();
    }
});

async function submitVoiceCheck(stream) {
    clearInterval(voiceTimerInterval);
    isVoiceRecording = false;
    stream.getTracks().forEach(t => t.stop());

    document.getElementById('voiceRecordBtn').textContent = 'Analysing…';
    document.getElementById('voiceRecordBtn').disabled = true;

    const blob = new Blob(voiceChunks, { type: 'audio/webm' });
    const fd = new FormData();
    fd.append('file', blob, 'voice.webm');

    // Include session_id if one exists
    const sessionId = localStorage.getItem('sleepSessionId') || '';
    const url = `/api/voice-check${sessionId ? `?session_id=${sessionId}` : ''}`;

    try {
        const res = await fetch(url, { method: 'POST', body: fd });
        const data = await res.json();
        renderVoiceResults(data);
    } catch (e) {
        alert('Voice analysis failed.');
    } finally {
        document.getElementById('voiceRecordBtn').textContent = '⏺ Record Again';
        document.getElementById('voiceRecordBtn').disabled = false;
        document.getElementById('voiceRecordBtn').style.borderColor = '#4DFFD2';
        document.getElementById('voiceRecordBtn').style.color = '#4DFFD2';
        document.getElementById('voiceTimer').textContent = '';
    }
}

function renderVoiceResults(data) {
    const scores = data.scores || {};
    const colorFor = v => v > 65 ? '#FF7059' : v > 40 ? '#FFB347' : '#4DFFD2';

    const gauges = [
        { id: 'fatigueGauge',  label: 'Fatigue',        key: 'fatigue' },
        { id: 'stressGauge',   label: 'Stress',          key: 'stress' },
        { id: 'cogLoadGauge',  label: 'Cognitive Load',  key: 'cognitive_load' },
    ];

    gauges.forEach(g => {
        const val = scores[g.key] || 0;
        const color = colorFor(val);
        document.getElementById(g.id).innerHTML = `
            <div class="gauge-label">${g.label}</div>
            <div class="gauge-value" style="color:${color}">${val}</div>
            <div style="font-size:11px; color:#6E8099; margin-top:4px;">/100</div>
            <div class="gauge-bar">
                <div class="gauge-fill" style="width:${val}%; background:${color};"></div>
            </div>
        `;
    });

    if (data.interpretation) {
        document.getElementById('voiceInterpretation').textContent = data.interpretation;
    }
    document.getElementById('voiceResults').style.display = 'block';
}
```

---

### Verify

```bash
# Install new deps
bin/python -m pip install librosa soundfile

# Start server
ANTHROPIC_API_KEY=... DEEPGRAM_API_KEY=... \
bin/python -m uvicorn backend.main:app --port 8000

# Test with a WAV file (or use the browser UI)
curl -X POST http://localhost:8000/api/voice-check \
  -F "file=@sample_speech.wav" \
  -F "session_id=demo-session"
# → {"scores": {"fatigue": 43, "stress": 31, "cognitive_load": 38}, ...}
```

---

## Feature 2: Autonomous Health Agent (Fetch AI)

A `uAgents` process running alongside the FastAPI server. Every 24 hours it loads the user's
latest sleep session, compares to their rolling 7-night baseline, and fires an alert if anything
is anomalous. Runs as a separate process: `bin/python agents/sleep_monitor.py`.

---

### `backend/requirements.txt` — add

```
uagents>=0.13.0
```

---

### Create `agents/sleep_monitor.py`

```python
"""
Autonomous sleep monitoring agent (Fetch AI uAgents).

Run: bin/python agents/sleep_monitor.py

The agent:
  1. Every 24 hours: loads latest sleep session from Redis
  2. Computes 7-night rolling baseline (avg score, efficiency, REM%)
  3. If tonight deviates >2 SD from baseline on any metric → fires alert
  4. Sends alert to registered alert_address (set via env var ALERT_AGENT_ADDRESS)
     or logs to stdout if not configured.

Register on Agentverse to make it discoverable:
  https://agentverse.ai
"""

import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from uagents import Agent, Context, Model


# ── Message models ───────────────────────────────────────────────────────────

class SleepAlert(Model):
    """Sent when a significant sleep anomaly is detected."""
    user_id:     str
    date:        str
    metric:      str          # which metric deviated
    tonight_val: float
    baseline:    float
    deviation:   float        # in standard deviations
    message:     str          # Claude-written narrative


class SleepStatusRequest(Model):
    """Another agent can request current sleep status."""
    user_id: str


class SleepStatusResponse(Model):
    session_id:    str
    quality_score: int
    quality_grade: str
    efficiency:    float
    summary:       str


# ── Agent definition ─────────────────────────────────────────────────────────

agent = Agent(
    name="sleepsense-monitor",
    seed=os.environ.get("AGENT_SEED", "sleepsense_monitor_default_seed_2026"),
    port=int(os.environ.get("AGENT_PORT", 8001)),
    endpoint=[f"http://localhost:{os.environ.get('AGENT_PORT', 8001)}/submit"],
)

ALERT_ADDRESS = os.environ.get("ALERT_AGENT_ADDRESS", "")
USER_ID       = os.environ.get("SLEEPSENSE_USER_ID", "default")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL_SECONDS", 86400))   # 24h default


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_redis():
    """Get Redis client (same connection logic as backend/session.py)."""
    url = os.environ.get("REDIS_URL") or os.environ.get("REDIS_PRIVATE_URL")
    if not url:
        return None
    import redis
    return redis.from_url(url, decode_responses=True)


def _load_recent_sessions(n: int = 8) -> list[dict]:
    """Load the n most recent session summaries from Redis."""
    r = _get_redis()
    if not r:
        return []
    session_ids = r.zrevrange("sessions:index", 0, n - 1)
    sessions = []
    for sid in session_ids:
        raw = r.get(f"analysis:{sid}")
        if raw:
            data = json.loads(raw)
            summary = data.get("sleep_summary", {})
            meta    = data.get("metadata", {})
            sessions.append({
                "session_id":    sid,
                "date":          meta.get("recording_start", "")[:10],
                "quality_score": summary.get("quality_score", 0),
                "efficiency":    summary.get("efficiency_pct", 0),
                "rem_pct":       (summary.get("pct_in_stage", {}).get("REM")
                                  or summary.get("pct_in_stage", {}).get("REM", 0)),
                "awakenings":    summary.get("awakenings", 0),
                "latency_min":   summary.get("latency_min", 0),
            })
    return sessions


def _detect_anomalies(sessions: list[dict]) -> list[dict]:
    """
    Compare tonight (sessions[0]) against 7-night baseline (sessions[1:8]).
    Returns list of anomaly dicts for metrics that deviate > 1.5 SD.
    """
    import numpy as np
    if len(sessions) < 3:
        return []   # need at least 3 nights to establish baseline

    tonight  = sessions[0]
    baseline = sessions[1:]

    metrics = ["quality_score", "efficiency", "rem_pct", "awakenings", "latency_min"]
    anomalies = []

    for m in metrics:
        vals = [s[m] for s in baseline if s.get(m) is not None]
        if not vals:
            continue
        mean = np.mean(vals)
        std  = np.std(vals)
        if std < 1e-6:
            continue
        tonight_val = tonight.get(m, mean)
        z = (tonight_val - mean) / std

        # Flag if deviated more than 1.5 SD in a bad direction
        bad = (
            (m in ("quality_score", "efficiency", "rem_pct") and z < -1.5)
            or (m in ("awakenings", "latency_min") and z > 1.5)
        )
        if bad:
            anomalies.append({
                "metric":      m,
                "tonight_val": round(float(tonight_val), 1),
                "baseline":    round(float(mean), 1),
                "deviation":   round(float(z), 2),
            })

    return anomalies


def _write_alert_message(anomalies: list[dict], tonight: dict) -> str:
    """Build a brief alert narrative (no Claude call — keeps agent lightweight)."""
    lines = [f"SleepSense Alert — {tonight.get('date', 'today')}"]
    lines.append(f"Quality score: {tonight.get('quality_score', '?')}")
    lines.append("Anomalies detected:")
    label = {
        "quality_score": "Quality score",
        "efficiency":    "Sleep efficiency",
        "rem_pct":       "REM sleep",
        "awakenings":    "Awakenings",
        "latency_min":   "Sleep latency",
    }
    for a in anomalies:
        m = a["metric"]
        lines.append(
            f"  • {label.get(m, m)}: {a['tonight_val']} "
            f"(baseline {a['baseline']}, {abs(a['deviation']):.1f}SD deviation)"
        )
    return "\n".join(lines)


# ── Agent behaviour ──────────────────────────────────────────────────────────

@agent.on_interval(period=float(CHECK_INTERVAL))
async def daily_sleep_check(ctx: Context):
    """Run nightly anomaly check."""
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
            f"no anomalies vs. 7-night baseline."
        )
        return

    message = _write_alert_message(anomalies, tonight)
    ctx.logger.warning(f"Anomalies detected:\n{message}")

    # Send to alert agent if configured
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
    """Respond to another agent asking for current sleep status."""
    sessions = _load_recent_sessions(n=1)
    if not sessions:
        await ctx.send(sender, SleepStatusResponse(
            session_id="", quality_score=0, quality_grade="?",
            efficiency=0, summary="No sleep data recorded yet.",
        ))
        return

    latest = sessions[0]
    await ctx.send(sender, SleepStatusResponse(
        session_id    = latest["session_id"],
        quality_score = latest["quality_score"],
        quality_grade = "A" if latest["quality_score"] >= 85 else
                        "B" if latest["quality_score"] >= 70 else
                        "C" if latest["quality_score"] >= 55 else "D",
        efficiency    = latest["efficiency"],
        summary       = (
            f"Last night: score {latest['quality_score']}, "
            f"efficiency {latest['efficiency']}%, "
            f"{latest['awakenings']} awakenings."
        ),
    ))


if __name__ == "__main__":
    print(f"Agent address: {agent.address}")
    print(f"Checking every {CHECK_INTERVAL}s ({CHECK_INTERVAL//3600}h)")
    agent.run()
```

---

### `Procfile` — add agent process

```
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
agent: bin/python agents/sleep_monitor.py
```

On Railway, both run in the same service. Locally, run in separate terminals.

---

### Run locally (two terminals)

```bash
# Terminal 1 — FastAPI
ANTHROPIC_API_KEY=... REDIS_URL=redis://localhost:6379 \
bin/python -m uvicorn backend.main:app --port 8000

# Terminal 2 — Agent
REDIS_URL=redis://localhost:6379 \
CHECK_INTERVAL_SECONDS=60 \
bin/python agents/sleep_monitor.py
# → Agent address: agent1qxxxxxxxxxxxxxxxx
# → Running daily sleep check... (after 60s)
```

---

### Register on Agentverse (optional but impressive for demo)

1. Go to `agentverse.ai` → Create Agent → paste the agent address printed on startup
2. Add name + description: "SleepSense Monitor — nightly sleep anomaly detection"
3. The agent is now discoverable by other agents in the Fetch AI ecosystem

---

## All new files

| File | Action | Purpose |
|------|--------|---------|
| `backend/voice_analysis.py` | **Create** | Acoustic biomarker extraction + 3 health scores |
| `agents/sleep_monitor.py` | **Create** | Fetch AI uAgent — daily anomaly check |
| `backend/requirements.txt` | Edit | Add librosa, soundfile, uagents |
| `backend/main.py` | Edit | Add `POST /api/voice-check` route |
| `frontend/index.html` | Edit | Voice Check section with recording UI + score gauges |
| `Procfile` | Edit | Add agent process |

---

## New API routes

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/voice-check` | Audio blob → fatigue/stress/cognitive load scores + Claude narrative |

---

## Verify end-to-end

```bash
# Install
bin/python -m pip install librosa soundfile uagents

# Voice check (test with any WAV/WebM)
curl -X POST http://localhost:8000/api/voice-check \
  -F "file=@any_speech.wav"
# → {"scores": {"fatigue": N, "stress": N, "cognitive_load": N},
#    "features": {...}, "interpretation": "..."}

# Agent anomaly check (with 2+ sessions in Redis, then wait 60s)
CHECK_INTERVAL_SECONDS=60 REDIS_URL=redis://localhost:6379 \
bin/python agents/sleep_monitor.py
# → "Running daily sleep check..."
# → "Sleep check OK" or "Anomalies detected: ..."
```
