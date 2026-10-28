"""
Stage 4 - Sequence (window) generation.

Input : data/processed/network_telemetry_labeled.csv   (from Stage 3)
        data/processed/feature_columns.json            (from Stage 2)
Output: data/processed/sequences/X.npy                 (N, LOOKBACK, F) float32
        data/processed/sequences/y.npy                 (N,) int64
        data/processed/sequences/metadata.csv          one row per window
        data/processed/sequences/sequence_config.json
        results/sequence_report.txt

One window per usable anchor (anchor_status == "used"):
    X = the LOOKBACK seconds ending AT the anchor (anchor included)
    y = the Stage 3 target (what starts in the next 10 s)

A window is built only if all LOOKBACK rows are in the same run AND the
same segment (no recording gap inside). Anchors too close to the start of
a run/segment are skipped and counted in the report.

X is NOT scaled here. Scaling is fitted per fold in Stage 6.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = PROCESSED_DIR / "network_telemetry_labeled.csv"
FEATURE_LIST_FILE = PROCESSED_DIR / "feature_columns.json"
OUT_DIR = PROCESSED_DIR / "sequences"
REPORT_FILE = RESULTS_DIR / "sequence_report.txt"

LOOKBACK = 30          # seconds of history per window (1 row = 1 s)
MAX_STEP_SEC = 1.5     # safety check inside a window (Stage 1 already
                       # splits segments at this threshold)

# Names that must never be model inputs. If any of them shows up in the
# feature list, something upstream is wrong.
FORBIDDEN = {"label", "anomaly_type", "target", "target_name", "phase",
             "episode_id", "benign_excursion", "segment_id", "elapsed_sec",
             "timestamp", "seed", "run_id", "machine", "scenario",
             "lead_time_sec", "future_onset_sec", "event_id",
             "anchor_status", "latency_ms", "baseline_latency_ms"}

META_COLUMNS = ["run_id", "machine", "scenario", "elapsed_sec", "segment_id",
                "phase", "episode_id", "benign_excursion", "target",
                "target_name", "event_id", "lead_time_sec",
                "future_onset_sec"]


# ============================================================
# MAIN
# ============================================================

def main():
    for p in (INPUT_FILE, FEATURE_LIST_FILE):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run Stages 1-3 first.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features = json.loads(FEATURE_LIST_FILE.read_text())
    leak = FORBIDDEN.intersection(features) | {
        f for f in features if f.startswith("injected_")}
    if leak:
        raise ValueError(f"Metadata in the feature list: {sorted(leak)}")

    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["run_id", "elapsed_sec"]).reset_index(drop=True)

    X_parts, meta_parts = [], []
    skipped_short = {}

    for (run_id, seg), s in df.groupby(["run_id", "segment_id"], sort=False):
        s = s.reset_index(drop=True)
        values = s[features].to_numpy(np.float32)
        t = s["elapsed_sec"].to_numpy()
        anchors = np.flatnonzero(s["anchor_status"].to_numpy() == "used")

        too_early = anchors[anchors < LOOKBACK - 1]
        for a in too_early:
            name = s.loc[a, "target_name"]
            skipped_short[name] = skipped_short.get(name, 0) + 1
        anchors = anchors[anchors >= LOOKBACK - 1]
        if len(anchors) == 0:
            continue

        # (n_anchors, LOOKBACK) row indices, then gather.
        idx = anchors[:, None] + np.arange(-LOOKBACK + 1, 1)[None, :]
        win = values[idx]
        steps = np.diff(t[idx], axis=1)
        if (steps > MAX_STEP_SEC).any() or (steps <= 0).any():
            raise ValueError(f"{run_id} segment {seg}: irregular step "
                             f"inside a window - Stage 1 gap split failed.")

        X_parts.append(win)
        meta_parts.append(s.loc[anchors, META_COLUMNS])

    X = np.concatenate(X_parts).astype(np.float32)
    meta = pd.concat(meta_parts, ignore_index=True)
    y = meta["target"].astype(np.int64).to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("NaN/inf inside windows.")

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    meta.to_csv(OUT_DIR / "metadata.csv", index=False)
    (OUT_DIR / "sequence_config.json").write_text(json.dumps({
        "lookback": LOOKBACK, "horizon_sec": 10, "features": features,
        "n_windows": int(len(X)),
    }, indent=2))

    # ---------------- report ----------------
    lines = ["=" * 70, "SEQUENCE GENERATION REPORT", "=" * 70, ""]
    lines.append(f"Lookback        : {LOOKBACK} s")
    lines.append(f"Features        : {len(features)}")
    lines.append(f"Windows         : {len(X)}   shape {X.shape}")
    lines.append(f"Size on disk    : {X.nbytes / 1e6:.1f} MB")
    lines.append("")
    lines += ["WINDOWS PER TARGET", "-" * 70]
    for k, v in meta["target_name"].value_counts().items():
        lines.append(f"{k:24s} {v}")
    lines.append("")
    lines += ["ANCHORS SKIPPED (fewer than LOOKBACK rows of history)",
              "-" * 70]
    if skipped_short:
        for k, v in sorted(skipped_short.items()):
            lines.append(f"{k:24s} {v}")
    else:
        lines.append("none")
    lines.append("")
    lines += ["EVENTS WITH AT LEAST ONE WINDOW, PER CLASS", "-" * 70]
    ev = meta[meta["event_id"].fillna("") != ""]
    for k, v in ev.groupby("target_name")["event_id"].nunique().items():
        lines.append(f"{k:24s} {v}")
    report = "\n".join(lines)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print()
    print(f"Sequences: {OUT_DIR}")


if __name__ == "__main__":
    main()
