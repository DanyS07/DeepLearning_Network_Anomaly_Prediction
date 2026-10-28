"""
Load one cross-validation fold, already scaled, for training/evaluation.

    from common.fold_data import load_fold, load_external, CLASS_NAMES

    d = load_fold(0)
    d["X_train"], d["y_train"], d["meta_train"]       # train windows
    d["X_validation"], d["y_validation"], ...          # early stopping
    d["X_test"], d["y_test"], d["meta_test"]           # final scores

    e = load_external()   # other-machine runs, scaled with the scaler
                          # fitted on ALL main-machine runs ("full")

The scaler of each fold is fitted on that fold's TRAIN windows only
(Stage 6), so validation/test statistics never leak into training.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQ_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
FOLD_FILE = PROJECT_ROOT / "data" / "processed" / "folds" / "folds.json"
SCALER_FILE = PROJECT_ROOT / "data" / "processed" / "model_ready" / "scalers.json"

CLASS_NAMES = ["normal", "congestion", "high_latency", "packet_loss",
               "bandwidth_degradation", "queue_buildup", "traffic_burst"]

_cache = {}


def _load_all():
    if not _cache:
        _cache["X"] = np.load(SEQ_DIR / "X.npy", mmap_mode="r")
        _cache["y"] = np.load(SEQ_DIR / "y.npy")
        _cache["meta"] = pd.read_csv(SEQ_DIR / "metadata.csv")
        _cache["folds"] = json.loads(FOLD_FILE.read_text())
        _cache["scalers"] = json.loads(SCALER_FILE.read_text())
    return _cache


def _scale(X, scaler):
    mean = np.asarray(scaler["mean"], np.float32)
    std = np.asarray(scaler["std"], np.float32)
    return ((X - mean) / std).astype(np.float32)


def _take(run_ids, scaler):
    c = _load_all()
    mask = c["meta"]["run_id"].isin(run_ids).to_numpy()
    idx = np.flatnonzero(mask)
    X = _scale(np.asarray(c["X"][idx]), scaler)
    return X, c["y"][idx], c["meta"].iloc[idx].reset_index(drop=True)


def n_folds():
    return _load_all()["folds"]["n_folds"]


def load_fold(k):
    c = _load_all()
    fold = c["folds"]["folds"][str(k)]
    scaler = c["scalers"][str(k)]
    out = {}
    for part in ("train", "validation", "test"):
        X, y, m = _take(fold[part], scaler)
        out[f"X_{part}"], out[f"y_{part}"], out[f"meta_{part}"] = X, y, m
    return out


def load_external():
    c = _load_all()
    X, y, m = _take(c["folds"]["external"], c["scalers"]["full"])
    return {"X": X, "y": y, "meta": m}


def load_full_main():
    """All main-machine runs, scaled with the 'full' scaler (final model)."""
    c = _load_all()
    runs = sorted({r for f in c["folds"]["folds"].values()
                   for part in f.values() for r in part})
    X, y, m = _take(runs, c["scalers"]["full"])
    return {"X": X, "y": y, "meta": m}
