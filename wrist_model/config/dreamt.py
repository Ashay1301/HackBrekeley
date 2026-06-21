# Config for WristSleepNet trained on DREAMT dataset.
# 4-class output: Wake(0) / Light(1) / Deep(2) / REM(3)
# Label mapping from AASM 5-class: N1+N2 -> Light, N3 -> Deep

params = {
    # Training
    "n_epochs": 150,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "evaluate_span": 25,
    "checkpoint_span": 25,

    # Early stopping
    "no_improve_epochs": 30,

    # Model
    "input_size": 12,           # number of features per epoch (HR x4, HRV x3, accel x3, temp, EDA)
    "n_classes": 4,             # Wake / Light / Deep / REM
    "l2_weight_decay": 1e-5,
    "warmup_epochs": 5,

    # Dataset
    "dataset": "dreamt",
    "data_dir": "./data/dreamt/processed",

    # Sequence
    "seq_length": 20,
    "batch_size": 15,

    # Augmentation
    "use_augmentation": True,
}

train = params.copy()
train.update({
    "seq_length": 20,
    "batch_size": 15,
})

predict = params.copy()
predict.update({
    "seq_length": 1,
    "batch_size": 1,
})

# Label mapping: DREAMT raw labels -> 4-class
LABEL_MAP = {
    0: 0,   # Wake  -> Wake
    1: 1,   # N1    -> Light
    2: 1,   # N2    -> Light
    3: 2,   # N3    -> Deep
    4: 3,   # REM   -> REM
    5: -1,  # Prep/Movement -> ignore
}

CLASS_NAMES = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
