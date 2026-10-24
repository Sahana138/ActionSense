"""
train_model.py

Loads the CNN feature sequences produced by extract_features.py, runs
k-fold cross-validation across a couple of hyperparameter configurations,
picks the best config, trains a final model on all data, and saves it.

ALSO builds a proper open-set rejection mechanism: for each trained class,
it computes a "centroid" (average internal embedding) from the training
data, plus a distance threshold based on how spread out that class's own
training examples are. At prediction time, a new video's embedding is
compared against these centroids - if it's too far from every one of
them, it gets flagged as "not one of the trained actions," regardless of
what the softmax layer confidently guesses. This generalizes to genuinely
unfamiliar inputs, not just a fixed list of "other" categories.

Run:
    python src/train_model.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATA_DIR = "data_processed/features"
MODEL_DIR = "models"
SEQUENCE_LENGTH = 20
FEATURES = 1280   # MobileNetV2 output dimension
N_FOLDS = 5

HYPERPARAM_CONFIGS = [
    {"lstm1": 64, "lstm2": 32, "dropout": 0.3, "lr": 1e-3},
    {"lstm1": 128, "lstm2": 64, "dropout": 0.4, "lr": 5e-4},
]


def load_dataset():
    X, y = [], []
    classes = sorted(os.listdir(DATA_DIR))
    for action in classes:
        action_dir = os.path.join(DATA_DIR, action)
        if not os.path.isdir(action_dir):
            continue
        for fname in os.listdir(action_dir):
            if fname.endswith(".npy"):
                seq = np.load(os.path.join(action_dir, fname))
                X.append(seq)
                y.append(action)
    return np.array(X, dtype=np.float32), np.array(y)


def build_model(num_classes, config):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, FEATURES)),
        tf.keras.layers.LSTM(config["lstm1"], return_sequences=True),
        tf.keras.layers.LSTM(config["lstm2"]),
        tf.keras.layers.Dense(32, activation="relu", name="embedding_layer"),
        tf.keras.layers.Dropout(config["dropout"]),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def cross_validate(X, y, num_classes, config):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = build_model(num_classes, config)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
        model.fit(
            X[train_idx], y[train_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=40, batch_size=8, callbacks=[early_stop], verbose=0,
        )
        _, acc = model.evaluate(X[val_idx], y[val_idx], verbose=0)
        print(f"    Fold {fold_idx}/{N_FOLDS}: val_accuracy = {acc:.3f}")
        fold_accuracies.append(acc)
    return float(np.mean(fold_accuracies))


def compute_open_set_profile(final_model, X, y, encoder):
    """Build per-class centroids + distance thresholds from the trained
    model's embedding layer (the Dense(32) layer just before the final
    softmax). This is what powers open-set rejection at inference time."""
    # Rebuild an explicit functional graph up to the embedding layer,
    # reusing the SAME (already-trained) layer objects/weights - this
    # avoids a Keras quirk where a Sequential model's `.input` isn't
    # available until it's used inside an explicit functional call.
    inp = tf.keras.Input(shape=(SEQUENCE_LENGTH, FEATURES))
    x = inp
    embedding_output = None
    for layer in final_model.layers:
        x = layer(x)
        if layer.name == "embedding_layer":
            embedding_output = x
    embedding_model = tf.keras.Model(inputs=inp, outputs=embedding_output)

    embeddings = embedding_model.predict(X, verbose=0)  # (N, 32)

    profile = {}
    for class_idx, class_name in enumerate(encoder.classes_):
        mask = (y == class_idx)
        class_embeddings = embeddings[mask]
        centroid = class_embeddings.mean(axis=0)
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        # Threshold: how far this class's OWN training examples spread out,
        # with a margin so we don't reject legitimate but slightly unusual
        # examples of a real trained action.
        threshold = float(distances.mean() + 2.5 * distances.std() + 1e-6)
        profile[class_name] = {
            "centroid": centroid.tolist(),
            "threshold": threshold,
        }
        print(f"  {class_name}: mean_dist={distances.mean():.3f}, "
              f"std={distances.std():.3f}, threshold={threshold:.3f}")

    return profile


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading CNN feature dataset...")
    X, y_raw = load_dataset()
    if len(X) == 0:
        raise RuntimeError(f"No .npy sequences found in '{DATA_DIR}'. Run extract_features.py first.")
    print(f"Loaded {len(X)} sequences across {len(set(y_raw))} classes.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    print("\nRunning cross-validated hyperparameter search...")
    best_config = None
    best_score = -1.0
    for i, config in enumerate(HYPERPARAM_CONFIGS, start=1):
        print(f"\nConfig {i}/{len(HYPERPARAM_CONFIGS)}: {config}")
        mean_acc = cross_validate(X, y, num_classes, config)
        print(f"  -> mean CV accuracy: {mean_acc:.3f}")
        if mean_acc > best_score:
            best_score = mean_acc
            best_config = config

    print(f"\nBest config: {best_config} (mean CV accuracy: {best_score:.3f})")

    print("\nTraining final model on the full dataset...")
    final_model = build_model(num_classes, best_config)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True
    )
    final_model.fit(
        X, y, epochs=25, batch_size=8, callbacks=[early_stop], verbose=1,
    )

    # Held-out style report, for a human-readable breakdown
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    report_model = build_model(num_classes, best_config)
    report_model.fit(X_train, y_train, epochs=40, batch_size=8, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(
                          monitor="loss", patience=8, restore_best_weights=True)])
    y_pred = np.argmax(report_model.predict(X_test, verbose=0), axis=1)

    print("\nClassification report (held-out split, for reference):")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(num_classes), encoder.classes_, rotation=90)
    plt.yticks(range(num_classes), encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))

    # --- Build the open-set profile (centroids + thresholds) from the
    #     FINAL model trained on ALL data - this is what gets deployed ---
    print("\nComputing open-set rejection profile (per-class centroids + thresholds)...")
    open_set_profile = compute_open_set_profile(final_model, X, y, encoder)

    # --- Save everything ---
    final_model.save(os.path.join(MODEL_DIR, "action_recognition_model.keras"))
    with open(os.path.join(MODEL_DIR, "labels.json"), "w") as f:
        json.dump(list(encoder.classes_), f)
    with open(os.path.join(MODEL_DIR, "best_config.json"), "w") as f:
        json.dump(best_config, f)
    with open(os.path.join(MODEL_DIR, "open_set_profile.json"), "w") as f:
        json.dump(open_set_profile, f)

    print(f"\nModel, labels, best config, and open-set profile saved in '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()