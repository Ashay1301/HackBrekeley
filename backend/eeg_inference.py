"""
EEG inference — wraps TinySleepNet for FastAPI use.
Loads the model once at startup; call predict(edf_bytes) per request.
"""

import importlib
import io
import os
import sys
import uuid
from datetime import datetime

import numpy as np

# TensorFlow is optional — server starts cleanly without it; EEG route returns 503
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ModuleNotFoundError:
    tf = None
    _TF_AVAILABLE = False

# Ensure project root is on path when imported from backend/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from model import TinySleepNet
    from sleepstage import class_dict as EEG_CLASS_DICT
    EEG_CLASS_NAMES = {k: v for k, v in EEG_CLASS_DICT.items() if k <= 4}
except Exception:
    TinySleepNet = None
    EEG_CLASS_NAMES = {}

from backend.sleep_metrics import analyze_predictions
EPOCH_SEC = 30


class EEGPredictor:
    def __init__(self):
        self.model = None
        self.config = None

    def load(self, config_path: str = None, checkpoint_dir: str = None):
        if not _TF_AVAILABLE:
            print("[EEG] TensorFlow not installed — EEG inference disabled. Install tensorflow or tensorflow-macos.")
            return
        if config_path is None:
            config_path = os.path.join(ROOT, "config", "sleepedf.py")
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(ROOT, "final_model_45F1", "best_model")

        spec = importlib.util.spec_from_file_location("eeg_config", config_path)
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        self.config = cfg_mod.train

        self.model = TinySleepNet(config=self.config)

        ckpt = tf.train.Checkpoint(model=self.model)
        mgr  = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)
        if not mgr.latest_checkpoint:
            raise FileNotFoundError(f"No EEG checkpoint in {checkpoint_dir}")
        ckpt.restore(mgr.latest_checkpoint).expect_partial()
        print(f"[EEG] Loaded checkpoint: {mgr.latest_checkpoint}")

    def predict(self, edf_bytes: bytes) -> dict:
        import pyedflib
        import tempfile

        # pyedflib requires a file path; write to a temp file
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(edf_bytes)
            tmp_path = tmp.name

        try:
            return self._predict_from_path(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _predict_from_path(self, edf_path: str) -> dict:
        import pyedflib

        with pyedflib.EdfReader(edf_path) as f:
            labels = f.getSignalLabels()
            # Try preferred channel, fall back to first EEG channel
            channel_prefs = ["EEG Fpz-Cz", "EEG Pz-Oz", "EEG", "eeg"]
            channel_idx = -1
            for pref in channel_prefs:
                for i, lbl in enumerate(labels):
                    if pref.lower() in lbl.lower():
                        channel_idx = i
                        break
                if channel_idx != -1:
                    break
            if channel_idx == -1:
                channel_idx = 0  # fall back to first channel

            sr = f.getSampleFrequency(channel_idx)
            start_dt = f.getStartdatetime()
            signal = f.readSignal(channel_idx)

        spe = int(EPOCH_SEC * sr)
        n_epochs = len(signal) // spe
        if n_epochs == 0:
            raise ValueError("EDF file is too short — no complete 30s epochs found.")

        data = signal[: n_epochs * spe].reshape(n_epochs, spe)

        # Pad to a multiple of seq_length so the BiLSTM reshape is clean
        seq_len = self.config["seq_length"]
        pad = (seq_len - n_epochs % seq_len) % seq_len
        if pad:
            data = np.vstack([data, np.zeros((pad, spe), dtype=data.dtype)])

        model_input = data[:, :, np.newaxis, np.newaxis].astype(np.float32)

        logits = self.model(model_input, training=False)
        probs  = tf.nn.softmax(logits).numpy()[:n_epochs]
        preds  = np.argmax(probs, axis=1)

        report = analyze_predictions(preds, probs, start_dt, EPOCH_SEC, EEG_CLASS_NAMES)
        report["metadata"] = {
            "analysis_type":   "eeg",
            "channel":         labels[channel_idx],
            "sampling_rate_hz": int(sr),
            "n_epochs":        n_epochs,
            "recording_start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
        report["session_id"] = str(uuid.uuid4())
        return report
