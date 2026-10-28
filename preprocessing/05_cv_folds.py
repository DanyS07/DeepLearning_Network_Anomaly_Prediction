"""
Stage 5 - Cross-validation folds by run.

Input : data/processed/sequences/metadata.csv     (from Stage 4)
Output: data/processed/folds/folds.json            run lists per fold
        data/processed/folds/fold_assignment.csv   one row per run
        results/cv_folds_report.txt

Why folds instead of one split: with 5 runs per class, a single 3/1/1 split
tests each class on ONE run (2-3 onsets). One lucky or unlucky onset then
moves recall by 30-50 points. With 5 folds every run is tested exactly once
and results are reported as mean +- std over folds.

Rotation (per class, runs sorted by their number _1, _2, ...):
    slot = position % N_FOLDS          (run _1 -> slot 0, _6 -> slot 0, ...)
    fold k:  test       = runs in slot k
             validation = runs in slot (k + 1) % N_FOLDS
             train      = all other runs
With 5 runs per class: 1 test / 1 validation / 3 train per fold.
With 10 runs per class: 2 / 2 / 6.  With 15: 3 / 3 / 9.
Adding runs later needs NO code change - just rerun the pipeline.
No run is ever in two roles in the same fold.

Runs of other machines (e.g. UTM) are never in a fold. They are listed as
"external" and used only for the cross-machine test.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

META_FILE = PROCESSED_DIR / "sequences" / "metadata.csv"
OUT_DIR = PROCESSED_DIR / "folds"
REPORT_FILE = RESULTS_DIR / "cv_folds_report.txt"

MAIN_MACHINE = "vbox"   # folds are built from this machine's runs
N_FOLDS = 5


# ============================================================
# HELPERS
# ============================================================

def run_number(run_id):
    """'vbox/traffic_burst/traffic_3' -> 3 (runs without a number last)."""
    m = re.search(r"_(\d+)$", run_id)
    return int(m.group(1)) if m else 10 ** 6


def assign_positions(runs):
    """Slot 0..N_FOLDS-1 per class, assigned in run-number order."""
    runs = runs.copy()
    runs["number"] = runs["run_id"].map(run_number)
    runs = runs.sort_values(["scenario", "number", "run_id"])
    runs["position"] = runs.groupby("scenario").cumcount() % N_FOLDS
    return runs


def role(position, k):
    if position == k:
        return "test"
    if position == (k + 1) % N_FOLDS:
        return "validation"
    return "train"


# ============================================================
# MAIN
# ============================================================

def main():
    if not META_FILE.exists():
        raise FileNotFoundError(f"Run Stage 4 first: {META_FILE} missing.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_FILE)
    all_runs = meta[["run_id", "machine", "scenario"]].drop_duplicates()
    main_runs = assign_positions(all_runs[all_runs["machine"] == MAIN_MACHINE])
    external = sorted(all_runs.loc[all_runs["machine"] != MAIN_MACHINE,
                                   "run_id"])

    for k in range(N_FOLDS):
        main_runs[f"fold{k}"] = [role(p, k) for p in main_runs["position"]]

    folds = {}
    for k in range(N_FOLDS):
        col = main_runs[f"fold{k}"]
        folds[str(k)] = {r: sorted(main_runs.loc[col == r, "run_id"])
                         for r in ("train", "validation", "test")}
        # A run must have exactly one role in a fold.
        sets = [set(v) for v in folds[str(k)].values()]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    # Every run with a position is tested exactly once.
    tested = [r for f in folds.values() for r in f["test"]]
    assert len(tested) == len(set(tested))

    (OUT_DIR / "folds.json").write_text(json.dumps(
        {"main_machine": MAIN_MACHINE, "n_folds": N_FOLDS,
         "folds": folds, "external": external}, indent=2))
    main_runs.drop(columns="number").to_csv(
        OUT_DIR / "fold_assignment.csv", index=False)

    # ---------------- report ----------------
    lines = ["=" * 70, "CROSS-VALIDATION FOLDS", "=" * 70, ""]
    lines.append(f"Main machine : {MAIN_MACHINE}")
    lines.append(f"Folds        : {N_FOLDS}")
    lines.append(f"Runs in folds: {len(main_runs)}   "
                 f"external runs: {len(external)}")
    lines.append("")

    lines += ["RUNS PER CLASS (main machine)", "-" * 70]
    warn = []
    for sc, g in main_runs.groupby("scenario"):
        n = len(g)
        lines.append(f"{sc:24s} {n} runs")
        if n % N_FOLDS:
            warn.append(f"{sc}: {n} runs is not a multiple of {N_FOLDS} -> "
                        f"folds have unequal test sizes for this class.")
        if n < N_FOLDS:
            warn.append(f"{sc}: only {n} run(s) -> missing from the test set "
                        f"in {N_FOLDS - n} fold(s). Collect "
                        f"{N_FOLDS - n} more.")
    lines.append("")
    if warn:
        lines += ["WARNINGS", "-" * 70] + warn + [""]

    lines += ["WINDOWS PER FOLD (train / validation / test)", "-" * 70]
    role_of = main_runs.set_index("run_id")
    for k in range(N_FOLDS):
        r = meta["run_id"].map(role_of[f"fold{k}"])
        counts = r.value_counts()
        pos = meta[meta["target"] > 0]
        ev = pos.groupby(r.loc[pos.index])["event_id"].nunique()
        lines.append(f"fold {k}: windows "
                     f"{counts.get('train', 0):6d} / "
                     f"{counts.get('validation', 0):5d} / "
                     f"{counts.get('test', 0):5d}    onset events "
                     f"{ev.get('train', 0)} / {ev.get('validation', 0)} / "
                     f"{ev.get('test', 0)}")
    lines.append("")

    lines += ["TEST RUNS PER FOLD", "-" * 70]
    for k in range(N_FOLDS):
        lines.append(f"fold {k}: " + ", ".join(
            r.split("/")[-1] for r in folds[str(k)]["test"]))
    lines.append("")
    if external:
        lines += ["EXTERNAL (cross-machine test only)", "-" * 70]
        lines += external
    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print()
    print(f"Folds: {OUT_DIR / 'folds.json'}")


if __name__ == "__main__":
    main()
