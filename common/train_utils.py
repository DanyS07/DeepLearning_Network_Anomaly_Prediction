"""
Shared training code for all models, so every model is trained, validated
and saved in exactly the same way.

Cross-validation:   run_cv(name, fit)
    For each fold k: fit on the fold's train runs (deep models: early
    stopping on the validation runs), then predict BOTH
        validation runs -> results/predictions/<name>/val_fold<k>.csv
        test runs       -> results/predictions/<name>/fold<k>.csv
    evaluate.py uses the validation predictions to choose an alarm
    threshold and the test predictions to score it.

Final model:  trained on ALL main-machine runs, saved to models/saved/
    (used by the demo). If other-machine runs exist it also predicts them
    -> results/predictions/<name>/external.csv

A "fit" function has the signature
    fit(d, tag) -> (predict_fn, info)
where predict_fn(X) returns class probabilities (n, 7).
"""

from pathlib import Path
import os
import random
import time

import numpy as np
import pandas as pd

from common.fold_data import (load_fold, load_external, load_full_main,
                              n_folds, CLASS_NAMES)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

N_CLASSES = len(CLASS_NAMES)
SEED = 42

META_KEEP = ["run_id", "scenario", "elapsed_sec", "target", "event_id",
             "lead_time_sec", "benign_excursion", "phase"]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ============================================================
# CLASS WEIGHTS
# ============================================================

def class_weights(y):
    """Square-root balanced weights (see thesis notes).

    Fully balanced weights would give each anomaly window ~100x the weight
    of a normal window and produce many false alarms. The square root keeps
    rare classes important without that. Mean weight per window = 1.
    """
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = np.zeros(N_CLASSES)
    present = counts > 0
    w[present] = np.sqrt(counts[present].max() / counts[present])
    w /= (w[y]).mean()
    return {i: float(w[i]) for i in range(N_CLASSES)}


# ============================================================
# SAVING PREDICTIONS
# ============================================================

def save_predictions(name, tag, meta, probs):
    out = meta[META_KEEP].copy()
    out["pred"] = probs.argmax(axis=1)
    for c in range(N_CLASSES):
        out[f"p_{CLASS_NAMES[c]}"] = probs[:, c]
    d = PRED_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"{tag}.csv", index=False)
    return out


def quick_scores(y, pred):
    from sklearn.metrics import f1_score
    present = sorted(set(np.unique(y)))     # classes that occur in y
    macro = f1_score(y, pred, average="macro", labels=present,
                     zero_division=0)
    anomaly = f1_score(y > 0, pred > 0, zero_division=0)
    return macro, anomaly


# ============================================================
# CROSS-VALIDATION DRIVER
# ============================================================

def run_cv(name, fit, final=True):
    set_seed()
    log = []
    print(f"\n===== {name}: {n_folds()}-fold cross-validation =====")
    for k in range(n_folds()):
        t0 = time.time()
        d = load_fold(k)
        predict, info = fit(d, f"fold{k}")
        probs = predict(d["X_test"])
        save_predictions(name, f"fold{k}", d["meta_test"], probs)
        save_predictions(name, f"val_fold{k}", d["meta_validation"],
                         predict(d["X_validation"]))
        macro, anom = quick_scores(d["y_test"], probs.argmax(1))
        info = {"fold": k, "macro_f1": round(macro, 4),
                "anomaly_f1": round(anom, 4),
                "seconds": round(time.time() - t0, 1), **(info or {})}
        log.append(info)
        extra = f"  epochs {info['epochs']}" if "epochs" in info else ""
        print(f"fold {k}: macro-F1 {macro:.3f}  any-anomaly F1 {anom:.3f}  "
              f"({info['seconds']} s){extra}")

    df = pd.DataFrame(log)
    print(f"{name}: macro-F1 {df.macro_f1.mean():.3f} +- "
          f"{df.macro_f1.std():.3f}")
    (PRED_DIR / name).mkdir(parents=True, exist_ok=True)
    df.to_csv(PRED_DIR / name / "cv_log.csv", index=False)

    if final:
        run_final(name, fit, log)
    return df


def run_final(name, fit, log):
    """Train on all main-machine runs; save it; predict external runs."""
    set_seed()
    full = load_full_main()
    epochs = [r["epochs"] for r in log if "epochs" in r]
    d = {"X_train": full["X"], "y_train": full["y"],
         "X_validation": None, "y_validation": None,
         "fixed_epochs": int(np.median(epochs)) if epochs else None,
         "save_as": name}
    print(f"training final {name} on all {len(full['y'])} windows ...")
    predict, _ = fit(d, "final")
    ext = load_external()
    if len(ext["y"]):
        probs = predict(ext["X"])
        save_predictions(name, "external", ext["meta"], probs)
        macro, anom = quick_scores(ext["y"], probs.argmax(1))
        print(f"final model on external runs: macro-F1 {macro:.3f}  "
              f"any-anomaly F1 {anom:.3f}")
    print(f"final model saved in {MODEL_DIR}")


# ============================================================
# KERAS TRAINING (shared by LSTM / GRU / TCN)
# ============================================================

N_SEEDS = 3          # each fold trains 3 networks and averages them


def keras_fit(build_model, max_epochs=100, batch_size=256, patience=15,
              lr=5e-4, n_seeds=N_SEEDS):
    """Returns a fit function for run_cv.

    Each fold trains n_seeds networks with different random seeds and
    averages their probabilities. One small network on ~12k windows is
    noisy; averaging makes the comparison between models fairer.
    """
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")     # hide harmless retracing warnings

    def fit_one(d, seed):
        set_seed(SEED + seed)
        tf.keras.backend.clear_session()
        X, y = d["X_train"], d["y_train"]
        cw = class_weights(y)
        model = build_model(X.shape[1:], N_CLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy",
                      weighted_metrics=["accuracy"])

        if d.get("X_validation") is not None:
            yv = d["y_validation"]
            # Weighted validation loss, so early stopping cares about the
            # rare classes the same way training does.
            wv = np.array([cw[c] for c in yv], np.float32)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience,
                    restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5,
                    min_lr=1e-5),
            ]
            hist = model.fit(X, y, validation_data=(d["X_validation"], yv, wv),
                             epochs=max_epochs, batch_size=batch_size,
                             class_weight=cw, callbacks=callbacks,
                             verbose=0, shuffle=True)
            best = int(np.argmin(hist.history["val_loss"])) + 1
        else:
            best = d["fixed_epochs"] or 30
            model.fit(X, y, epochs=best, batch_size=batch_size,
                      class_weight=cw, verbose=0, shuffle=True)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save(MODEL_DIR / f"{d['save_as']}_final_seed{seed}.keras")
        return model, best

    def fit(d, tag):
        models, epochs = [], []
        for s in range(n_seeds):
            m, e = fit_one(d, s)
            models.append(m)
            epochs.append(e)

        def predict(X):
            return np.mean([m.predict(X, batch_size=1024, verbose=0)
                            for m in models], axis=0)

        return predict, {"epochs": int(np.median(epochs))}

    return fit
