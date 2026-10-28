"""
Stage 7 - Model-ready validation. Run this before ANY training.

Checks every fold (and the external set) and stops with an error if
something is wrong. Prints PASS at the end if everything is fine.

  1. no run appears in two roles of the same fold (train/validation/test)
  2. every run of the main machine is tested exactly once over the folds
  3. external (other-machine) runs are never in a fold
  4. every class is present in every fold's TRAIN part
  5. scaled data is finite; train features have ~mean 0 / std 1
  6. targets are 0..6; normal runs only ever have target 0
  7. positive targets point 1..10 s into the future
  8. no metadata column is a feature
  9. windows really are LOOKBACK consecutive seconds of the same run
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from common.fold_data import (load_fold, load_external, n_folds,  # noqa: E402
                              CLASS_NAMES)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FOLDS = json.loads((PROCESSED_DIR / "folds" / "folds.json").read_text())
CONFIG = json.loads((PROCESSED_DIR / "sequences" / "sequence_config.json")
                    .read_text())

problems, warnings = [], []


def check(ok, msg):
    if not ok:
        problems.append(msg)


def main():
    features = CONFIG["features"]
    lookback = CONFIG["lookback"]
    bad = [f for f in features if f.startswith("injected_") or f in
           {"label", "phase", "episode_id", "benign_excursion", "target",
            "latency_ms", "elapsed_sec", "segment_id"}]
    check(not bad, f"metadata used as features: {bad}")

    tested = []
    for k in range(n_folds()):
        fold = FOLDS["folds"][str(k)]
        tr, va, te = (set(fold[p]) for p in ("train", "validation", "test"))
        check(not (tr & va or tr & te or va & te),
              f"fold {k}: a run has two roles")
        check(not (set(FOLDS["external"]) & (tr | va | te)),
              f"fold {k}: external run inside the fold")
        tested += fold["test"]

        d = load_fold(k)
        print(f"fold {k}:", end="")
        for part in ("train", "validation", "test"):
            X, y, m = d[f"X_{part}"], d[f"y_{part}"], d[f"meta_{part}"]
            check(X.ndim == 3 and X.shape[1:] == (lookback, len(features)),
                  f"fold {k} {part}: shape {X.shape}")
            check(len(X) == len(y) == len(m), f"fold {k} {part}: lengths")
            check(np.isfinite(X).all(), f"fold {k} {part}: NaN/inf")
            check(np.isin(y, np.arange(len(CLASS_NAMES))).all(),
                  f"fold {k} {part}: target outside 0..6")
            check(not ((m["scenario"] == "normal") & (y > 0)).any(),
                  f"fold {k} {part}: normal run with positive target")
            pos = m[y > 0]
            check(((pos["lead_time_sec"] > 0)
                   & (pos["lead_time_sec"] <= 10)).all(),
                  f"fold {k} {part}: lead time outside (0, 10]")
            counts = np.bincount(y, minlength=len(CLASS_NAMES))
            print(f"  {part} {len(y)}", end="")
            if part == "train":
                missing = [CLASS_NAMES[c] for c in range(len(CLASS_NAMES))
                           if counts[c] == 0]
                check(not missing, f"fold {k} train: no windows of {missing}")
                flat = X.reshape(-1, X.shape[-1])
                check(np.abs(flat.mean(0)).max() < 1e-2
                      and np.abs(flat.std(0)[flat.std(0) > 0] - 1).max() < 1e-2,
                      f"fold {k} train: not standardised")
            else:
                missing = [CLASS_NAMES[c] for c in range(1, len(CLASS_NAMES))
                           if counts[c] == 0]
                if missing:
                    warnings.append(f"fold {k} {part}: no {missing} windows")
        print()

    main_runs = {r for f in FOLDS["folds"].values()
                 for p in f.values() for r in p}
    check(len(tested) == len(set(tested)), "a run is tested twice")
    untested = main_runs - set(tested)
    if untested:
        warnings.append(f"runs never tested (extras, train only): "
                        f"{sorted(untested)}")

    if FOLDS["external"]:
        e = load_external()
        check(np.isfinite(e["X"]).all(), "external: NaN/inf")
        print(f"external: {len(e['y'])} windows from "
              f"{e['meta']['run_id'].nunique()} runs")

    # Window continuity spot-check against the labeled source file.
    src = pd.read_csv(PROCESSED_DIR / "network_telemetry_labeled.csv",
                      usecols=["run_id", "elapsed_sec", "segment_id"]
                      + features[:3])
    meta = pd.read_csv(PROCESSED_DIR / "sequences" / "metadata.csv")
    X_raw = np.load(PROCESSED_DIR / "sequences" / "X.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(meta), size=min(50, len(meta)), replace=False):
        r = meta.iloc[i]
        s = src[(src["run_id"] == r["run_id"])
                & (src["segment_id"] == r["segment_id"])
                & (src["elapsed_sec"] <= r["elapsed_sec"])].tail(lookback)
        check(len(s) == lookback
              and np.allclose(X_raw[i][:, :3], s[features[:3]].to_numpy(),
                              atol=1e-3),
              f"window {i} does not match its source rows")

    print()
    for w in warnings:
        print(f"[WARNING] {w}")
    if problems:
        print()
        for p in problems:
            print(f"[FAIL] {p}")
        raise SystemExit(f"{len(problems)} problem(s) - fix before training.")
    print("PASS: model-ready validation")


if __name__ == "__main__":
    main()
