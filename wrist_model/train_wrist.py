"""
train_wrist.py

Trains WristSleepNet on DREAMT features (12 features/epoch, 4-class output).
Architecture and training loop mirror train_final.py from the EEG pipeline.

Usage:
    python wrist_model/train_wrist.py \
        --data_dir data/dreamt/processed \
        --output_dir wrist_model/checkpoints \
        --log_file wrist_model/checkpoints/training.log
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import tensorflow as tf
import sklearn.metrics as skmetrics
from sklearn.utils import class_weight

# Add project root to path so we can import from wrist_model/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrist_model.model_wrist import WristSleepNet
from wrist_model.config.dreamt import train as config, CLASS_NAMES

import logging
tf.get_logger().setLevel(logging.ERROR)


# ── LR schedule (identical to train_final.py) ────────────────────────────────

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate, decay_steps - warmup_steps, alpha=alpha
        )

    def __call__(self, step):
        warmup = lambda: self.initial_learning_rate * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
        )
        decay = lambda: self.cosine_decay(step - self.warmup_steps)
        return tf.cond(step < self.warmup_steps, warmup, decay)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }


# ── Focal loss (identical to train_final.py) ─────────────────────────────────

class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        probs = tf.nn.softmax(y_pred, axis=-1) if self.from_logits else y_pred
        probs = tf.clip_by_value(probs, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        y_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(probs)[-1])
        p_t = tf.reduce_sum(probs * y_one_hot, axis=-1)
        return tf.math.pow(1.0 - p_t, self.gamma) * ce


# ── Data augmentation ─────────────────────────────────────────────────────────

def augment_features(x_batch):
    """Add small Gaussian noise and random scaling to feature vectors."""
    noise = tf.random.normal(shape=tf.shape(x_batch), mean=0.0, stddev=0.02)
    scale = tf.random.uniform(shape=(tf.shape(x_batch)[0], 1), minval=0.95, maxval=1.05)
    return x_batch * scale + noise


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_subjects(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}. Run prepare_dreamt.py first.")
    all_x, all_y = [], []
    for f in files:
        d = np.load(f)
        all_x.append(d["x"].astype(np.float32))   # (n_epochs, 12)
        all_y.append(d["y"].astype(np.int32))
    return all_x, all_y


def iterate_minibatches(x_list, y_list, batch_size, seq_length, shuffle=True):
    """Yield (x_batch, y_batch, weights) where batch is batch_size*seq_length epochs."""
    n = len(x_list)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    n_loops = int(np.ceil(n / batch_size))

    for loop in range(n_loops):
        start = loop * batch_size
        sel = indices[start:start + batch_size]
        seqs_x = [x_list[i] for i in sel]
        seqs_y = [y_list[i] for i in sel]

        # Find max length in this batch
        max_len = max(len(s) for s in seqs_x)
        n_chunks = int(np.ceil(max_len / seq_length))
        n_features = x_list[0].shape[1]

        for chunk in range(n_chunks):
            cs = chunk * seq_length
            ce = (chunk + 1) * seq_length
            bx = np.zeros((len(sel), seq_length, n_features), dtype=np.float32)
            by = np.zeros((len(sel), seq_length), dtype=np.int32)
            bw = np.zeros((len(sel), seq_length), dtype=np.float32)

            for s_idx, (sx, sy) in enumerate(zip(seqs_x, seqs_y)):
                chunk_x = sx[cs:ce]
                chunk_y = sy[cs:ce]
                bx[s_idx, :len(chunk_x)] = chunk_x
                by[s_idx, :len(chunk_y)] = chunk_y
                bw[s_idx, :len(chunk_x)] = 1.0

            yield (
                bx.reshape(-1, n_features),
                by.reshape(-1),
                bw.reshape(-1),
            )


# ── Main training loop ────────────────────────────────────────────────────────

def train(data_dir, output_dir, log_file, random_seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # Simple console + file logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger(__name__)

    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    all_x, all_y = load_all_subjects(data_dir)
    n_subjects = len(all_x)
    n_valid = max(1, int(n_subjects * 0.1))

    val_idx = np.random.choice(n_subjects, n_valid, replace=False)
    train_idx = np.setdiff1d(np.arange(n_subjects), val_idx)

    train_x = [all_x[i] for i in train_idx]
    train_y = [all_y[i] for i in train_idx]
    valid_x = [all_x[i] for i in val_idx]
    valid_y = [all_y[i] for i in val_idx]

    logger.info(f"Train subjects: {len(train_x)}, Validation subjects: {len(valid_x)}")

    flat_y = np.concatenate(train_y)
    classes = np.unique(flat_y)
    weights = class_weight.compute_class_weight("balanced", classes=classes, y=flat_y)
    weights = np.clip(weights, 0.5, 5.0)
    logger.info(f"Class weights: { {CLASS_NAMES[i]: round(w, 3) for i, w in enumerate(weights)} }")

    steps_per_epoch = int(np.ceil(len(train_x) / config["batch_size"]))
    total_steps = steps_per_epoch * config["n_epochs"]
    warmup_steps = steps_per_epoch * config.get("warmup_epochs", 5)

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=config["learning_rate"],
        decay_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    model = WristSleepNet(config=config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=config["clip_grad_value"])
    loss_fn = SparseCategoricalFocalLoss(from_logits=True)

    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_loss_metric = tf.keras.metrics.Mean()
    valid_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    best_ckpt_dir = os.path.join(output_dir, "best_model")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, best_ckpt_dir, max_to_keep=1)

    best_mf1 = -1.0
    last_improve = 0
    best_report = ""

    for epoch in range(config["n_epochs"]):
        t0 = time.time()
        train_loss_metric.reset_state()
        train_acc_metric.reset_state()
        valid_loss_metric.reset_state()
        valid_acc_metric.reset_state()

        for x_b, y_b, w_b in iterate_minibatches(train_x, train_y, config["batch_size"], config["seq_length"]):
            x_b_tf = tf.constant(x_b)
            if config.get("use_augmentation"):
                x_b_tf = augment_features(x_b_tf)
            with tf.GradientTape() as tape:
                logits = model(x_b_tf, training=True)
                sample_w = tf.gather(tf.constant(weights, dtype=tf.float32), y_b) * w_b
                loss = loss_fn(y_b, logits, sample_weight=sample_w)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_metric(loss)
            train_acc_metric(y_b, logits, sample_weight=w_b)

        all_preds, all_true = [], []
        for x_b, y_b, w_b in iterate_minibatches(valid_x, valid_y, config["batch_size"], config["seq_length"], shuffle=False):
            logits = model(tf.constant(x_b), training=False)
            loss = loss_fn(y_b, logits, sample_weight=w_b)
            valid_loss_metric(loss)
            valid_acc_metric(y_b, logits, sample_weight=w_b)
            all_preds.extend(tf.argmax(logits, axis=1).numpy())
            all_true.extend(y_b)

        # Filter out padded zeros (weight==0 means padding, but we can't easily here — use all)
        mf1 = skmetrics.f1_score(all_true, all_preds, average="macro", zero_division=0)
        dur = time.time() - t0

        logger.info(
            f"[Epoch {epoch+1}/{config['n_epochs']}] ({dur:.1f}s) "
            f"Train Loss: {train_loss_metric.result():.4f}, Train Acc: {train_acc_metric.result()*100:.1f}% | "
            f"Valid Loss: {valid_loss_metric.result():.4f}, Valid Acc: {valid_acc_metric.result()*100:.1f}%, MF1: {mf1*100:.1f}%"
        )

        if mf1 > best_mf1:
            best_mf1 = mf1
            last_improve = epoch
            ckpt_manager.save()
            logger.info(f"MF1 improved to {best_mf1*100:.1f}%. Saved checkpoint.")
            best_report = skmetrics.classification_report(
                all_true, all_preds,
                labels=list(range(4)),
                target_names=[CLASS_NAMES[i] for i in range(4)],
                zero_division=0,
            )

        if (epoch + 1) % config.get("evaluate_span", 25) == 0:
            report = skmetrics.classification_report(
                all_true, all_preds,
                labels=list(range(4)),
                target_names=[CLASS_NAMES[i] for i in range(4)],
                zero_division=0,
            )
            logger.info(f"--- Epoch {epoch+1} Report ---\n{report}")

        if epoch - last_improve >= config["no_improve_epochs"]:
            logger.info(f"No improvement for {config['no_improve_epochs']} epochs. Early stopping.")
            break

    logger.info(f"Training complete. Best Macro F1: {best_mf1*100:.1f}%")
    if best_report:
        logger.info(f"Best model classification report:\n{best_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dreamt/processed")
    parser.add_argument("--output_dir", type=str, default="wrist_model/checkpoints")
    parser.add_argument("--log_file", type=str, default="wrist_model/checkpoints/training.log")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.log_file, args.seed)
