"""
Voice biomarker extraction for fatigue / stress / cognitive load scoring.
Uses scipy (already installed) + librosa for audio feature extraction.

Features extracted:
  F0 (fundamental frequency): lower + less variable → fatigue, depression
  Jitter: cycle-to-cycle F0 perturbation → vocal stress / tremor
  Shimmer: amplitude perturbation → fatigue, respiratory issues
  HNR (harmonics-to-noise ratio): lower → hoarseness, fatigue
  Speaking rate (WPM from transcript): slower → cognitive fatigue
  Spectral centroid: higher → alertness
  RMS energy: lower → fatigue / low arousal
  Zero crossing rate: correlates with voice clarity
"""

import numpy as np
from scipy import signal as scipy_signal


# ── Low-level feature extractors ────────────────────────────────────────────

def _load_audio(audio_bytes: bytes) -> tuple:
    """Load raw audio bytes → (samples float32, sample_rate).
    Tries soundfile first (fast, handles WAV/FLAC/OGG); falls back to
    PyAV for browser-native formats (WebM/Opus, MP4/AAC, etc.)."""
    import io
    import soundfile as sf

    buf = io.BytesIO(audio_bytes)
    try:
        samples, sr = sf.read(buf, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return samples, int(sr)
    except Exception:
        pass

    # PyAV fallback — handles WebM/Opus from browser MediaRecorder
    import av
    all_samples = []
    target_sr = 16000
    buf.seek(0)
    with av.open(buf) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        actual_sr = stream.codec_context.sample_rate or target_sr
        for frame in container.decode(stream):
            arr = frame.to_ndarray()  # shape: (channels, samples) or (samples,)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            all_samples.append(arr.astype(np.float32))

    if not all_samples:
        raise ValueError("No audio frames decoded.")

    samples = np.concatenate(all_samples)
    # Normalise int PCM to float [-1, 1] if needed
    if samples.max() > 1.0:
        samples = samples / 32768.0
    return samples, actual_sr


def _f0_series(samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Estimate fundamental frequency using autocorrelation.
    Returns array of F0 values (Hz) for voiced frames; unvoiced frames are excluded.
    """
    frame_len = int(sr * 0.04)   # 40 ms frames
    hop       = int(sr * 0.01)   # 10 ms hop
    f0_values = []

    for start in range(0, len(samples) - frame_len, hop):
        frame = samples[start:start + frame_len]
        frame = frame - frame.mean()

        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]

        min_lag = int(sr / 500)   # max F0 = 500 Hz
        max_lag = int(sr / 60)    # min F0 =  60 Hz
        if max_lag >= len(corr) or corr[0] == 0:
            continue

        sub = corr[min_lag:max_lag]
        if len(sub) == 0:
            continue

        peak_idx = np.argmax(sub) + min_lag
        if corr[peak_idx] / (corr[0] + 1e-9) > 0.3:
            f0_values.append(sr / peak_idx)

    return np.array(f0_values) if f0_values else np.array([0.0])


def _jitter(f0_series: np.ndarray) -> float:
    """Relative jitter: mean absolute F0 difference / mean F0 (percent)."""
    if len(f0_series) < 2 or np.mean(f0_series) == 0:
        return 0.0
    diffs = np.abs(np.diff(f0_series))
    return float(np.mean(diffs) / np.mean(f0_series)) * 100


def _shimmer(samples: np.ndarray, sr: int) -> float:
    """Approximate shimmer from short-time RMS amplitude variation (percent)."""
    frame_len = int(sr * 0.04)
    hop       = int(sr * 0.01)
    rms_vals  = []
    for start in range(0, len(samples) - frame_len, hop):
        frame = samples[start:start + frame_len]
        rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))

    if len(rms_vals) < 2:
        return 0.0
    rms_arr  = np.array(rms_vals)
    mean_rms = np.mean(rms_arr)
    if mean_rms < 1e-9:
        return 0.0
    return float(np.mean(np.abs(np.diff(rms_arr))) / mean_rms) * 100


def _hnr(samples: np.ndarray, sr: int) -> float:
    """
    Harmonics-to-noise ratio via autocorrelation peak.
    Higher = cleaner voice. Typical healthy range: 15–25 dB.
    """
    frame_len = int(sr * 0.04)
    hnr_vals  = []
    for start in range(0, len(samples) - frame_len, frame_len):
        frame = samples[start:start + frame_len]
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]
        if corr[0] < 1e-9:
            continue
        min_lag = int(sr / 400)
        max_lag = int(sr / 75)
        if max_lag >= len(corr):
            continue
        r0 = corr[0]
        rp = float(np.max(corr[min_lag:max_lag]))
        ratio = rp / r0
        if 0 < ratio < 1:
            hnr_vals.append(10 * np.log10(ratio / (1 - ratio + 1e-9)))

    return float(np.median(hnr_vals)) if hnr_vals else 10.0


def _spectral_centroid(samples: np.ndarray, sr: int) -> float:
    """Mean spectral centroid (Hz). Higher → brighter / more alert voice."""
    freqs, _, Sxx = scipy_signal.spectrogram(samples, sr, nperseg=512)
    col_sums = Sxx.sum(axis=0)
    if col_sums.sum() == 0:
        return 0.0
    centroids = np.sum(freqs[:, None] * Sxx, axis=0) / (col_sums + 1e-9)
    return float(np.mean(centroids))


def _rms_energy(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2)))


def _zero_crossing_rate(samples: np.ndarray) -> float:
    signs = np.sign(samples)
    return float(np.mean(np.abs(np.diff(signs)) > 0))


# ── Scoring helpers ──────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _score_fatigue(jitter: float, shimmer: float, hnr: float,
                   rms: float, speaking_rate_wpm: float) -> int:
    """Higher score → more fatigue. Reference baselines from vocal fatigue literature."""
    j_score = _clamp((jitter - 0.5) / 2.0, 0, 1)          # healthy ~0.5–1%, fatigued >1.5%
    s_score = _clamp((shimmer - 2.0) / 5.0, 0, 1)          # healthy ~2–4%, fatigued >5%
    h_score = _clamp((18 - hnr) / 12.0, 0, 1)              # healthy >18 dB, fatigued <12 dB
    e_score = _clamp((0.05 - rms) / 0.05, 0, 1)            # fatigued voice is quieter
    r_score = _clamp((130 - speaking_rate_wpm) / 80.0, 0, 1) if speaking_rate_wpm > 0 else 0.5

    weighted = (j_score * 0.25 + s_score * 0.20 + h_score * 0.25
                + e_score * 0.15 + r_score * 0.15)
    return round(_clamp(weighted * 100, 0, 100))


def _score_stress(f0_mean: float, f0_std: float,
                  spectral_centroid: float, rms: float) -> int:
    """Stress raises mean F0, variability, and energy."""
    baseline = 160.0   # rough cross-gender F0 midpoint
    f0_stress  = _clamp((f0_mean - baseline) / 120.0, -0.5, 1.0) if f0_mean > 0 else 0.2
    var_stress = _clamp((f0_std / f0_mean - 0.05) / 0.2, 0, 1) if f0_mean > 0 else 0.2
    sc_stress  = _clamp((spectral_centroid - 1500) / 1500, 0, 1)
    e_stress   = _clamp((rms - 0.05) / 0.1, 0, 1)

    weighted = (f0_stress * 0.30 + var_stress * 0.25
                + sc_stress * 0.25 + e_stress * 0.20)
    return round(_clamp(weighted * 100, 0, 100))


def _score_cognitive_load(speaking_rate_wpm: float, zcr: float, jitter: float) -> int:
    """High cognitive load → slower, more hesitant speech."""
    r_load = _clamp((110 - speaking_rate_wpm) / 80.0, 0, 1) if speaking_rate_wpm > 0 else 0.5
    z_load = _clamp((0.15 - zcr) / 0.12, 0, 1)
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
    transcript  : optional text transcript used to compute speaking rate (WPM)

    Returns
    -------
    dict with keys: scores, features, transcript
    """
    samples, sr = _load_audio(audio_bytes)
    duration_sec = len(samples) / sr

    f0      = _f0_series(samples, sr)
    f0_mean = float(np.mean(f0)) if len(f0) > 0 else 0.0
    f0_std  = float(np.std(f0))  if len(f0) > 0 else 0.0
    jitter  = _jitter(f0)
    shimmer = _shimmer(samples, sr)
    hnr     = _hnr(samples, sr)
    sc      = _spectral_centroid(samples, sr)
    rms     = _rms_energy(samples)
    zcr     = _zero_crossing_rate(samples)

    word_count = len(transcript.split()) if transcript.strip() else 0
    wpm = (word_count / duration_sec * 60) if duration_sec > 0 and word_count > 0 else 0.0

    fatigue        = _score_fatigue(jitter, shimmer, hnr, rms, wpm)
    stress         = _score_stress(f0_mean, f0_std, sc, rms)
    cognitive_load = _score_cognitive_load(wpm, zcr, jitter)

    return {
        "scores": {
            "fatigue":        fatigue,
            "stress":         stress,
            "cognitive_load": cognitive_load,
        },
        "features": {
            "f0_mean_hz":           round(f0_mean, 1),
            "f0_std_hz":            round(f0_std, 1),
            "jitter_pct":           round(jitter, 3),
            "shimmer_pct":          round(shimmer, 3),
            "hnr_db":               round(hnr, 1),
            "spectral_centroid_hz": round(sc, 0),
            "rms_energy":           round(rms, 5),
            "zero_crossing_rate":   round(zcr, 4),
            "speaking_rate_wpm":    round(wpm, 1),
            "duration_sec":         round(duration_sec, 1),
        },
        "transcript": transcript,
    }
