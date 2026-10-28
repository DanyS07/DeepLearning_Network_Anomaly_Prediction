"""
Stage 6 - Feature scaling, fitted per fold on TRAIN windows only.

Input : data/processed/sequences/X.npy, metadata.csv    (Stage 4)
        data/processed/folds/folds.json                 (Stage 5)
Output: data/processed/model_ready/scalers.json
        results/scaling_report.txt

For every fold k: mean and std of each feature over the train windows of
that fold. Validation and test windows are scaled with the TRAIN numbers,
never their own, so nothing about them leaks into training.

Extra scaler "full": fitted on all main-machine runs. Used for the final
model and for the cross-machine (external) test.

Scaled arrays are not written to disk (5 folds x 3 parts would be ~500 MB).
common/fold_data.py applies the scaler when a fold is loaded.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

SEQ_DIR = PROCESSED_DIR / "sequences"
FOLD_FILE = PROCESSED_DIR / "folds" / "folds.json"
OUT_DIR = PROCESSED_DIR / "model_ready"
REPORT_FILE = RESULTS_DIR / "scaling_report.txt"

MIN_STD = 1e-6


def fit_scaler(X):
    """Mean/std per feature over all rows of all windows."""
    flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    const = std < MIN_STD
    std[const] = 1.0          # a constant feature is left unscaled
    return {"mean": mean.tolist(), "std": std.tolist(),
            "constant_features": np.flatnonzero(const).tolist()}


def main():
    for p in (SEQ_DIR / "X.npy", FOLD_FILE):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run Stages 4-5 first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(SEQ_DIR / "X.npy", mmap_mode="r")
    meta = pd.read_csv(SEQ_DIR / "metadata.csv")
    folds = json.loads(FOLD_FILE.read_text())
    features = json.loads((SEQ_DIR / "sequence_config.json").read_text())[
        "features"]

    scalers = {}
    lines = ["=" * 70, "FEATURE SCALING REPORT", "=" * 70, ""]

    for k, fold in folds["folds"].items():
        idx = np.flatnonzero(meta["run_id"].isin(fold["train"]).to_numpy())
        scalers[k] = fit_scaler(np.asarray(X[idx]))
        lines.append(f"fold {k}: fitted on {len(idx)} train windows "
                     f"({len(fold['train'])} runs)")

    all_main = sorted({r for f in folds["folds"].values()
                       for part in f.values() for r in part})
    idx = np.flatnonzero(meta["run_id"].isin(all_main).to_numpy())
    scalers["full"] = fit_scaler(np.asarray(X[idx]))
    lines.append(f"full  : fitted on {len(idx)} windows "
                 f"(all {len(all_main)} main-machine runs)")
    lines.append("")

    # How much the scaler moves between folds: large differences mean a
    # feature is dominated by a few runs.
    means = np.array([scalers[k]["mean"] for k in folds["folds"]])
    stds = np.array([scalers[k]["std"] for k in folds["folds"]])
    lines += ["FEATURE MEAN / STD ACROSS FOLDS (train only)", "-" * 70]
    lines.append(f"{'feature':28s} {'mean (min..max)':>26s} "
                 f"{'std (min..max)':>26s}")
    for j, f in enumerate(features):
        lines.append(f"{f:28s} {means[:, j].min():12.3f}..{means[:, j].max():<12.3f}"
                     f" {stds[:, j].min():12.3f}..{stds[:, j].max():<12.3f}")
    const = scalers["full"]["constant_features"]
    lines.append("")
    lines.append("Constant features: "
                 + (", ".join(features[i] for i in const) if const else "none"))

    (OUT_DIR / "scalers.json").write_text(json.dumps(
        {**scalers, "features": features}, indent=1))
    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print()
    print(f"Scalers: {OUT_DIR / 'scalers.json'}")


if __name__ == "__main__":
    main()
