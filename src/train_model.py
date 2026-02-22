import os
import sys
import json
import time
import numpy as np

# ── TensorFlow guard ──────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        TimeDistributed, Conv1D, BatchNormalization, MaxPooling1D,
        Flatten, LSTM, Dense, Dropout
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
except ImportError:
    sys.exit("TensorFlow is not installed.  Run:  pip install tensorflow")

# ── sklearn guard ─────────────────────────────────────────────────────────────
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    sys.exit("scikit-learn is not installed.  Run:  pip install scikit-learn")

# ─────────────────────────── PATHS / CONFIG ───────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT, "data", "processed")
MODEL_DIR  = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "labels.json")
REPORT     = os.path.join(MODEL_DIR, "training_report.txt")

SEQ_LEN = 30
N_FEAT  = 63

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────── DATA LOADING ────────────────────────────────────
def load_data():
    """
    Scans DATA_DIR for .npy files, validates shape (N, 30, 63) and minimum
    sample count, then concatenates into one X / y pair.

    Returns:
        X      – np.ndarray  shape (total_samples, SEQ_LEN, N_FEAT)
        y      – np.ndarray  shape (total_samples,)  label strings
        labels – sorted list of unique gesture strings
    """
    X_list, y_list = [], []

    npy_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
    if not npy_files:
        sys.exit(f"No .npy files found in {DATA_DIR}. Run collect_data.py first.")

    for fname in sorted(npy_files):
        gesture = os.path.splitext(fname)[0]
        arr = np.load(os.path.join(DATA_DIR, fname))

        if arr.ndim != 3 or arr.shape[1] != SEQ_LEN or arr.shape[2] != N_FEAT:
            print(f"  [skip] {fname}: wrong shape {arr.shape}")
            continue
        if len(arr) < 5:
            print(f"  [skip] {fname}: only {len(arr)} samples (need ≥ 5)")
            continue

        X_list.append(arr.astype(np.float32))
        y_list.append(np.array([gesture] * len(arr)))
        print(f"  [load] {fname}: {len(arr)} sequences")

    if not X_list:
        sys.exit("No valid data loaded.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    labels = sorted(set(y.tolist()))

    print(f"\nTotal sequences: {len(X)}  |  Classes: {len(labels)}")
    return X, y, labels


# ─────────────────────────── AUGMENTATION ────────────────────────────────────
def augment(X, y_cat, factor=2):
    """
    Duplicate training data `factor-1` times with small perturbations:
      - Gaussian noise  (std = 0.005)
      - Random per-sample scale  uniform(0.92, 1.08)

    Returns augmented X and y_cat (concatenated with originals).
    """
    aug_X = [X]
    aug_y = [y_cat]

    for _ in range(factor - 1):
        noise  = np.random.normal(0, 0.005, X.shape).astype(np.float32)
        scales = np.random.uniform(0.92, 1.08, (len(X), 1, 1)).astype(np.float32)
        aug_X.append((X + noise) * scales)
        aug_y.append(y_cat)

    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


# ─────────────────────────── MODEL DEFINITION ────────────────────────────────
def build_model(n_classes: int) -> tf.keras.Model:
    """CNN-LSTM hybrid for sequence gesture recognition."""
    model = Sequential([
        # ── CNN feature extractor (applied per time-step) ──
        Input(shape=(SEQ_LEN, N_FEAT)),
        TimeDistributed(Conv1D(64, 3, activation="relu", padding="same")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(MaxPooling1D(2)),
        TimeDistributed(Conv1D(128, 3, activation="relu", padding="same")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Flatten()),
        # ── Temporal modelling ──
        LSTM(128, return_sequences=True,  dropout=0.2, recurrent_dropout=0.1),
        LSTM(64,  return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        # ── Classifier head ──
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.35),
        Dense(64, activation="relu"),
        Dropout(0.25),
        Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────── TRAINING ────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Sign-Language CNN-LSTM Trainer")
    print("=" * 60)

    # ── Load ──
    X, y_raw, labels = load_data()

    # ── Encode labels ──
    le = LabelEncoder()
    le.fit(labels)
    y_enc = le.transform(y_raw)
    y_cat = to_categorical(y_enc, num_classes=len(labels))

    # ── Split ──
    X_train, X_test, y_train, y_test, y_enc_train, y_enc_test = train_test_split(
        X, y_cat, y_enc,
        test_size=0.15,
        stratify=y_enc,
        random_state=42
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Augment training set ──
    X_train, y_train = augment(X_train, y_train, factor=2)
    print(f"After augmentation — Train: {len(X_train)}")

    # ── Build ──
    model = build_model(n_classes=len(labels))
    model.summary()

    # ── Callbacks ──
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    # ── Fit ──
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed/60:.1f} min.")

    # ── Evaluate ──
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_enc  = np.argmax(y_pred_prob, axis=1)
    y_pred_lbl  = le.inverse_transform(y_pred_enc)
    y_true_lbl  = le.inverse_transform(y_enc_test)

    report = classification_report(y_true_lbl, y_pred_lbl, zero_division=0)
    cm     = confusion_matrix(y_enc_test, y_pred_enc)

    print("\n── Classification Report ──")
    print(report)
    print("── Confusion Matrix ──")
    print(cm)

    # ── Save labels ──
    with open(LABEL_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    # ── Save text report ──
    with open(REPORT, "w") as f:
        f.write(f"Training duration : {elapsed/60:.1f} min\n")
        f.write(f"Train samples     : {len(X_train)}\n")
        f.write(f"Test  samples     : {len(X_test)}\n\n")
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix\n")
        f.write("=" * 50 + "\n")
        f.write(np.array2string(cm) + "\n")

    print("\n── Saved files ──")
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Labels : {LABEL_PATH}")
    print(f"  Report : {REPORT}")


if __name__ == "__main__":
    train()
