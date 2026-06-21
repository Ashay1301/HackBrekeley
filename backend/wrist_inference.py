"""
Wrist inference — wraps WristSleepNet for FastAPI use.
Accepts parsed epoch list from any wearable parser.
"""

import os
import sys
import uuid
from datetime import datetime

import numpy as np

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ModuleNotFoundError:
    tf = None
    _TF_AVAILABLE = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from wrist_model.model_wrist import WristSleepNet
    from wrist_model.config.dreamt import params as WRIST_PARAMS
except Exception:
    WristSleepNet = None
    WRIST_PARAMS = {"seq_length": 20}

from backend.sleep_metrics import analyze_predictions

WRIST_CLASS_NAMES = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
FEATURE_KEYS = [
    "hr_mean", "hr_std", "hr_min", "hr_max",
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
    "accel_mean", "accel_std", "accel_zcr",
    "eda_mean", "temp_mean",
]
EPOCH_SEC = 30


class WristPredictor:
    def __init__(self):
        self.model = None
        self.available = False

    def load(self, checkpoint_dir: str = None):
        if not _TF_AVAILABLE:
            print("[Wrist] TensorFlow not installed — wrist model disabled.")
            return
        if checkpoint_dir is None:
            base = os.path.join(ROOT, "wrist_model", "checkpoints")
            checkpoint_dir = os.path.join(base, "best_model") if os.path.exists(os.path.join(base, "best_model", "checkpoint")) else base

        if not os.path.exists(checkpoint_dir):
            print("[Wrist] No checkpoint directory found — wrist model not available.")
            return

        self.model = WristSleepNet(config=WRIST_PARAMS)
        ckpt = tf.train.Checkpoint(model=self.model)
        mgr  = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)
        if not mgr.latest_checkpoint:
            print(f"[Wrist] No checkpoint in {checkpoint_dir} — wrist model not available.")
            return

        ckpt.restore(mgr.latest_checkpoint).expect_partial()
        self.available = True
        print(f"[Wrist] Loaded checkpoint: {mgr.latest_checkpoint}")

    def predict(self, epochs: list[dict], source: str = "wearable") -> dict:
        if not self.available:
            raise RuntimeError("Wrist model not loaded. Train it first with train_wrist.py.")

        features = []
        for ep in epochs:
            row = [float(ep.get(k, 0.0) or 0.0) for k in FEATURE_KEYS]
            features.append(row)

        x = np.array(features, dtype=np.float32)   # (n, 12)
        seq_len = WRIST_PARAMS["seq_length"]

        # Pad to a multiple of seq_len
        n = len(x)
        pad = (seq_len - n % seq_len) % seq_len
        if pad:
            x = np.vstack([x, np.zeros((pad, 12), dtype=np.float32)])

        x_input = x.reshape(-1, 12, 1)  # (batch*seq_len, 12, 1)
        logits = self.model(x_input, training=False)
        probs  = tf.nn.softmax(logits).numpy()
        preds  = np.argmax(probs, axis=1)

        # Trim padding
        preds = preds[:n]
        probs = probs[:n]

        # Parse start datetime from first epoch
        try:
            start_dt = datetime.strptime(epochs[0]["timestamp"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            start_dt = datetime(2024, 1, 1, 22, 0, 0)

        report = analyze_predictions(preds, probs, start_dt, EPOCH_SEC, WRIST_CLASS_NAMES)
        report["metadata"] = {
            "analysis_type": "wrist",
            "source":        source,
            "n_epochs":      n,
            "recording_start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
        report["session_id"] = str(uuid.uuid4())
        return report
