"""
Emotion Recognition from Speech
================================

Objective
---------
Recognize human emotions (e.g. happy, angry, sad, neutral, fearful, disgust,
surprised, calm) from speech audio.

Approach
--------
- Feature extraction : MFCCs (Mel-Frequency Cepstral Coefficients), plus
                        optional delta/delta-delta, chroma, and mel-spectrogram
                        features for a richer representation.
- Model              : CNN + LSTM (a CRNN-style architecture) — the CNN
                        learns local spectro-temporal patterns, the LSTM
                        models how those patterns evolve over time.
- Datasets           : RAVDESS, TESS, EMO-DB (Berlin Database of Emotional
                        Speech). The loader below is written for RAVDESS's
                        filename convention by default and documents how to
                        adapt it to TESS / EMO-DB.

Pipeline stages
---------------
    1. Data loading & label parsing (dataset-specific filename conventions)
    2. Audio preprocessing & feature extraction (MFCC + deltas)
    3. Dataset assembly (padding/truncating to a fixed length, train/test split)
    4. CNN-LSTM model definition
    5. Training
    6. Evaluation
    7. Single-file inference helper

Requirements
------------
    pip install librosa soundfile numpy scikit-learn tensorflow

Usage
-----
    python speech_emotion_recognition.py --data-dir /path/to/RAVDESS --dataset ravdess --epochs 30
    python speech_emotion_recognition.py --data-dir /path/to/TESS    --dataset tess    --epochs 30
    python speech_emotion_recognition.py --data-dir /path/to/EMODB   --dataset emodb   --epochs 30
"""

import os
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Emotion label maps per dataset
# ---------------------------------------------------------------------------

# RAVDESS: filenames look like 03-01-06-01-02-01-12.wav
# The 3rd number (index 2) is the emotion code.
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# TESS: filenames/folders contain the emotion word directly, e.g.
# "OAF_back_angry.wav" or folder "YAF_happy"
TESS_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

# EMO-DB: filenames encode emotion as a single letter, e.g. "03a01Fa.wav"
# where the 6th character is the emotion code.
EMODB_EMOTION_MAP = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}


# ---------------------------------------------------------------------------
# 1. Data loading & label parsing
# ---------------------------------------------------------------------------
def collect_files(data_dir, dataset):
    """
    Walk `data_dir` and return a list of (filepath, emotion_label) tuples.
    Handles the three supported dataset naming conventions.
    """
    samples = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith((".wav", ".flac", ".ogg")):
                continue
            fpath = os.path.join(root, fname)

            if dataset == "ravdess":
                parts = fname.split("-")
                if len(parts) < 3:
                    continue
                code = parts[2]
                label = RAVDESS_EMOTION_MAP.get(code)

            elif dataset == "tess":
                lower = fname.lower()
                label = next((e for e in TESS_EMOTIONS if e in lower), None)

            elif dataset == "emodb":
                if len(fname) < 6:
                    continue
                code = fname[5].upper()
                label = EMODB_EMOTION_MAP.get(code)

            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            if label is not None:
                samples.append((fpath, label))

    if not samples:
        raise RuntimeError(
            f"No labeled audio files found under {data_dir} for dataset={dataset}. "
            "Check the directory path and folder structure."
        )
    return samples


# ---------------------------------------------------------------------------
# 2. Audio preprocessing & feature extraction
# ---------------------------------------------------------------------------
def extract_features(filepath, sr=22050, n_mfcc=40, max_len=174):
    """
    Load an audio file and extract a fixed-size MFCC feature matrix.

    Returns an array of shape (n_mfcc * 3, max_len):
        - MFCCs
        - delta MFCCs (first derivative, captures how coefficients change)
        - delta-delta MFCCs (second derivative)
    stacked together, then padded/truncated along the time axis to `max_len`
    frames so every example has identical shape for batching.
    """
    import librosa

    y, sr = librosa.load(filepath, sr=sr)
    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, delta, delta2], axis=0)  # (n_mfcc*3, T)

    # Pad or truncate along time axis to a fixed length
    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    else:
        features = features[:, :max_len]

    return features.astype("float32")


# ---------------------------------------------------------------------------
# 3. Dataset assembly
# ---------------------------------------------------------------------------
def build_dataset(data_dir, dataset, n_mfcc=40, max_len=174, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    samples = collect_files(data_dir, dataset)
    print(f"Found {len(samples)} labeled audio files.")

    X, y_labels = [], []
    for i, (fpath, label) in enumerate(samples):
        try:
            feats = extract_features(fpath, n_mfcc=n_mfcc, max_len=max_len)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            continue
        X.append(feats)
        y_labels.append(label)
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(samples)}")

    X = np.stack(X)                      # (N, n_mfcc*3, max_len)
    X = np.expand_dims(X, -1)            # (N, n_mfcc*3, max_len, 1) for Conv2D

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)  # integer-encode emotion labels

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize using train-set statistics
    mean, std = x_train.mean(), x_train.std()
    x_train = (x_train - mean) / (std + 1e-8)
    x_test = (x_test - mean) / (std + 1e-8)

    return (x_train, y_train), (x_test, y_test), encoder, (mean, std)


# ---------------------------------------------------------------------------
# 4. CNN-LSTM model definition
# ---------------------------------------------------------------------------
def build_cnn_lstm(input_shape, num_classes):
    """
    input_shape: (n_mfcc*3, max_len, 1)

    CNN blocks extract local time-frequency patterns; the result is reshaped
    into a sequence of feature vectors over time, which an LSTM then models
    to capture how the emotional expression evolves across the utterance.
    """
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Collapse the frequency axis, keep time as the sequence dimension
    # x shape here: (batch, freq', time', channels)
    shape = x.shape
    x = layers.Permute((2, 1, 3))(x)                       # (batch, time', freq', channels)
    x = layers.Reshape((shape[2], shape[1] * shape[3]))(x)  # (batch, time', features)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
def train_model(model, x_train, y_train, x_test, y_test, epochs=30, batch_size=32):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    return history


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, x_test, y_test, encoder):
    from sklearn.metrics import classification_report

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return loss, acc


# ---------------------------------------------------------------------------
# 7. Single-file inference helper
# ---------------------------------------------------------------------------
def predict_emotion(model, filepath, encoder, mean, std, n_mfcc=40, max_len=174):
    feats = extract_features(filepath, n_mfcc=n_mfcc, max_len=max_len)
    feats = (feats - mean) / (std + 1e-8)
    feats = feats.reshape(1, feats.shape[0], feats.shape[1], 1)

    probs = model.predict(feats, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = encoder.inverse_transform([idx])[0]
    return label, float(probs[idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition (MFCC + CNN-LSTM)")
    parser.add_argument("--data-dir", required=True, help="Root folder of the dataset")
    parser.add_argument("--dataset", choices=["ravdess", "tess", "emodb"], default="ravdess")
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=174,
                         help="Fixed number of time frames per clip (pad/truncate)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-path", default="speech_emotion_cnn_lstm.h5")
    args = parser.parse_args()

    print(f"Building dataset from {args.data_dir} ({args.dataset}) ...")
    (x_train, y_train), (x_test, y_test), encoder, (mean, std) = build_dataset(
        args.data_dir, args.dataset, n_mfcc=args.n_mfcc, max_len=args.max_len
    )

    print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    model = build_cnn_lstm(input_shape=x_train.shape[1:], num_classes=len(encoder.classes_))
    model.summary()

    train_model(model, x_train, y_train, x_test, y_test,
                epochs=args.epochs, batch_size=args.batch_size)

    evaluate_model(model, x_test, y_test, encoder)

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
