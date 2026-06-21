"""
inference.py

Loads a trained WristSleepNet checkpoint and runs prediction on
pre-parsed wearable data (output of any parser in wrist_model/parsers/).

Input  : list of epoch dicts with numeric feature fields
Output : same JSON schema as predict_for_llm.py (4-class labels)
"""

import os
import numpy as np
import tensorflow as tf

from wrist_model.model_wrist import WristSleepNet
from wrist_model.config.dreamt import predict as config, CLASS_NAMES


FEATURE_KEYS = [
    "hr_mean", "hr_std", "hr_min", "hr_max",
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
    "accel_mean", "accel_std", "accel_zcr",
    "eda_mean", "temp_mean",
]


class WristPredictor:
    def __init__(self, checkpoint_dir: str):
        self.model = WristSleepNet(config=config)
        ckpt = tf.train.Checkpoint(model=self.model)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)
        if not manager.latest_checkpoint:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
        ckpt.restore(manager.latest_checkpoint).expect_partial()

    def predict(self, epochs: list[dict]) -> dict:
        """
        Args:
            epochs: list of dicts, each with keys matching FEATURE_KEYS
                    (output of any parser in wrist_model/parsers/)

        Returns:
            dict with keys: predictions, probabilities, sleep_summary, epoch_by_epoch_data
        """
        features = self._epochs_to_features(epochs)
        features_tf = tf.constant(features, dtype=tf.float32)
        logits = self.model(features_tf, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        preds = np.argmax(probs, axis=1)

        epoch_details = []
        for i, (epoch, pred, prob) in enumerate(zip(epochs, preds, probs)):
            epoch_details.append({
                "epoch_number": i + 1,
                "timestamp": epoch.get("timestamp", ""),
                "predicted_stage_code": int(pred),
                "predicted_stage_name": CLASS_NAMES[int(pred)],
                "confidence": round(float(np.max(prob)), 3),
                "probabilities": {
                    CLASS_NAMES[j]: round(float(prob[j]), 3) for j in range(4)
                },
            })

        summary = self._analyze(preds, epochs)

        return {
            "model_type": "wrist",
            "n_classes": 4,
            "class_names": list(CLASS_NAMES.values()),
            "sleep_summary": summary,
            "epoch_by_epoch_data": epoch_details,
        }

    def _epochs_to_features(self, epochs: list[dict]) -> np.ndarray:
        """Convert list of epoch dicts → (N, 12) float32 array."""
        rows = []
        for ep in epochs:
            row = [float(ep.get(k, 0.0)) for k in FEATURE_KEYS]
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def _analyze(self, preds: np.ndarray, epochs: list[dict]) -> dict:
        """Compute sleep summary statistics from predictions."""
        n = len(preds)
        epoch_sec = 30
        total_min = round(n * epoch_sec / 60, 2)

        stage_counts = {name: int(np.sum(preds == code)) for code, name in CLASS_NAMES.items()}
        stage_pct = {name: round(cnt / n * 100, 2) for name, cnt in stage_counts.items()}
        stage_min = {name: round(cnt * epoch_sec / 60, 2) for name, cnt in stage_counts.items()}

        # Sleep latency: first epoch of Light/Deep/REM after 3 consecutive non-Wake
        sleep_latency_min = "N/A"
        consecutive = 0
        for i, p in enumerate(preds):
            if p != 0:
                consecutive += 1
                if consecutive >= 3:
                    onset_epoch = i - 2
                    sleep_latency_min = round(onset_epoch * epoch_sec / 60, 2)
                    break
            else:
                consecutive = 0

        # Sleep efficiency
        sleep_epochs = np.sum(preds != 0)
        sleep_efficiency = round(sleep_epochs / n * 100, 2) if n > 0 else 0.0

        # Awakenings after sleep onset (≥2 consecutive Wake epochs)
        awakenings = 0
        if sleep_latency_min != "N/A":
            onset_idx = int(sleep_latency_min * 60 / epoch_sec)
            i = onset_idx + 1
            while i < n:
                if preds[i] == 0:
                    run_len = 1
                    while i + run_len < n and preds[i + run_len] == 0:
                        run_len += 1
                    if run_len >= 2:
                        awakenings += 1
                    i += run_len
                else:
                    i += 1

        # Sleep quality score (0–100)
        eff_score = min(sleep_efficiency, 100) * 0.40
        rem_pct = stage_pct.get("REM", 0)
        rem_score = min(rem_pct / 25 * 100, 100) * 0.25
        deep_pct = stage_pct.get("Deep", 0)
        deep_score = min(deep_pct / 20 * 100, 100) * 0.20
        latency_score = 0.0
        if sleep_latency_min != "N/A":
            latency_score = max(0, 1 - sleep_latency_min / 30) * 100 * 0.15
        quality_score = round(eff_score + rem_score + deep_score + latency_score, 1)

        return {
            "total_recording_time_minutes": total_min,
            "sleep_latency_minutes": sleep_latency_min,
            "sleep_efficiency_percent": sleep_efficiency,
            "awakenings_after_onset": awakenings,
            "sleep_quality_score": quality_score,
            "time_in_stage_minutes": stage_min,
            "percentage_in_stage": stage_pct,
        }
