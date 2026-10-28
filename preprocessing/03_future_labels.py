"""
Stage 3 - Future-onset targets.

Input : data/processed/network_telemetry_features.csv   (from Stage 2)
Output: data/processed/network_telemetry_labeled.csv
        results/future_labels_report.txt

Question asked at every anchor second t (a row where the network is
currently normal, label == 0):

    "Which anomaly class STARTS in the next H seconds, (t, t + H]?"

    target = that class    if an onset falls in the horizon
    target = normal (0)    if nothing starts in the horizon

Anchors that are NOT used (target stays empty):
  * label == 1           the anomaly is already happening; nothing to predict
  * cooldown             the network is still recovering from an episode
                         (windowed measurements still carry it)
  * horizon incomplete   t + H is past the end of the run, or the horizon
                         crosses a recording gap (an onset could be hidden)
  * inside an episode with no onset in the horizon (DROP_IN_EPISODE_NEGATIVES)
                         e.g. 25 s into a 40 s ramp: the network is already
                         degrading, so calling it "normal" would punish the
                         model for noticing early. These are "don't care".

Extra columns for event-level evaluation:
  event_id         one id per onset (run_id#k); all positive anchors that
                   point at the same onset share it
  lead_time_sec    onset time - anchor time (how early this anchor is)
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = PROCESSED_DIR / "network_telemetry_features.csv"
OUTPUT_FILE = PROCESSED_DIR / "network_telemetry_labeled.csv"
REPORT_FILE = RESULTS_DIR / "future_labels_report.txt"

HORIZON_SEC = 10.0
DROP_IN_EPISODE_NEGATIVES = True

CLASS_ID = {
    "normal": 0,
    "congestion": 1,
    "high_latency": 2,
    "packet_loss": 3,
    "bandwidth_degradation": 4,
    "queue_buildup": 5,
    "traffic_burst": 6,
}
CLASS_NAME = {v: k for k, v in CLASS_ID.items()}


# ============================================================
# PER-RUN LABELLING
# ============================================================

def label_run(s):
    """s: one run, sorted by elapsed_sec. Returns columns to add."""
    n = len(s)
    t = s["elapsed_sec"].to_numpy()
    y = s["label"].to_numpy().astype(int)
    seg = s["segment_id"].to_numpy()
    phase = s["phase"].astype(str).to_numpy()
    in_episode = s["episode_id"].to_numpy() > 0
    cls = CLASS_ID[str(s["scenario"].iloc[0])]

    # Onsets: label goes 0 -> 1. Computed over the whole run, so a
    # recording gap in the middle of an anomaly does not create a fake one.
    onset_idx = np.flatnonzero((y[1:] == 1) & (y[:-1] == 0)) + 1
    onset_t = t[onset_idx]

    # Last timestamp of each segment (horizon must end inside the segment).
    seg_end = pd.Series(t).groupby(seg).transform("max").to_numpy()

    target = np.full(n, np.nan)
    onset_time = np.full(n, np.nan)
    lead = np.full(n, np.nan)
    event_k = np.full(n, -1)
    reason = np.array(["used"] * n, dtype=object)

    for i in range(n):
        if y[i] == 1:
            reason[i] = "anomaly_now"
            continue
        if phase[i] == "cooldown":
            reason[i] = "cooldown"
            continue
        limit = t[i] + HORIZON_SEC
        if limit > seg_end[i]:
            reason[i] = "horizon_incomplete"
            continue

        hit = np.flatnonzero((onset_t > t[i]) & (onset_t <= limit))
        if len(hit):
            k = hit[0]
            target[i] = cls
            onset_time[i] = onset_t[k]
            lead[i] = onset_t[k] - t[i]
            event_k[i] = k
        elif DROP_IN_EPISODE_NEGATIVES and in_episode[i]:
            reason[i] = "in_episode_far_from_onset"
        else:
            target[i] = CLASS_ID["normal"]

    run_id = s["run_id"].iloc[0]
    event_id = np.where(event_k >= 0,
                        [f"{run_id}#{k + 1}" for k in event_k], "")
    return pd.DataFrame({
        "target": target,
        "target_name": [CLASS_NAME.get(int(v), "") if not np.isnan(v) else ""
                        for v in target],
        "future_onset_sec": onset_time,
        "lead_time_sec": lead,
        "event_id": event_id,
        "anchor_status": reason,
    }, index=s.index), len(onset_t)


# ============================================================
# REPORT
# ============================================================

def build_report(df, onsets_per_run):
    used = df[df["anchor_status"] == "used"]
    lines = ["=" * 70, "FUTURE-ONSET LABEL REPORT", "=" * 70, ""]
    lines.append(f"Horizon                    : {HORIZON_SEC:g} s")
    lines.append(f"Drop in-episode negatives  : {DROP_IN_EPISODE_NEGATIVES}")
    lines.append(f"Rows                       : {len(df)}")
    lines.append(f"Usable anchors             : {len(used)}")
    lines.append("")

    lines += ["ANCHOR STATUS", "-" * 70]
    for k, v in df["anchor_status"].value_counts().items():
        lines.append(f"{k:28s} {v}")
    lines.append("")

    lines += ["TARGETS (usable anchors)", "-" * 70]
    for k, v in used["target_name"].value_counts().items():
        lines.append(f"{k:24s} {v:7d}  ({100 * v / len(used):.1f}%)")
    lines.append("")

    lines += ["ONSET EVENTS PER CLASS (these drive event-level metrics)",
              "-" * 70]
    ev = used[used["event_id"] != ""]
    per_class = ev.groupby("target_name")["event_id"].nunique()
    for k, v in per_class.items():
        lines.append(f"{k:24s} {v} events")
    lines.append("")

    # Every onset should have positive anchors. If one has none, its
    # 10 s before the onset were unusable (gap, cooldown, run start).
    total_onsets = sum(onsets_per_run.values())
    lines.append(f"Onsets found in labels     : {total_onsets}")
    lines.append(f"Onsets with >=1 anchor     : {ev['event_id'].nunique()}")
    lines.append("")

    lines += ["POSITIVE ANCHORS PER EVENT (expect ~10 each)", "-" * 70]
    per_event = ev.groupby("event_id").size()
    lines.append(f"min {per_event.min()}, median {per_event.median():.0f}, "
                 f"max {per_event.max()}")
    few = per_event[per_event < 8]
    for e, c in few.items():
        lines.append(f"  only {c} anchors: {e}")
    lines.append("")
    lines.append("Inspect a few onsets by eye before Stage 4 "
                 "(see the check printed at the end of this script).")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Run Stage 2 first: {INPUT_FILE} missing.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["run_id", "elapsed_sec"]).reset_index(drop=True)

    unknown = set(df["scenario"].unique()) - set(CLASS_ID)
    if unknown:
        raise ValueError(f"Unknown scenario folder(s): {unknown}")

    parts, onsets_per_run = [], {}
    for run_id, s in df.groupby("run_id", sort=False):
        lab, n_on = label_run(s)
        parts.append(lab)
        onsets_per_run[run_id] = n_on
        if s["scenario"].iloc[0] != "normal" and n_on == 0:
            print(f"[WARNING] {run_id}: no onsets found")
    df = pd.concat([df, pd.concat(parts)], axis=1)

    # Normal runs must never produce a positive target.
    bad = df[(df["scenario"] == "normal") & (df["target"] > 0)]
    assert bad.empty, "normal run produced a positive target"
    # A positive target must always point at a future onset.
    pos = df[df["target"] > 0]
    assert ((pos["lead_time_sec"] > 0)
            & (pos["lead_time_sec"] <= HORIZON_SEC)).all()

    df.to_csv(OUTPUT_FILE, index=False)
    report = build_report(df, onsets_per_run)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)

    # Eyeball check: the seconds around the first onset of one run.
    ex = df[df["event_id"] != ""]["event_id"].iloc[0]
    onset = df.loc[df["event_id"] == ex, "future_onset_sec"].iloc[0]
    run = ex.split("#")[0]
    w = df[(df["run_id"] == run)
           & (df["elapsed_sec"].between(onset - 13, onset + 2))]
    print()
    print(f"Example around onset of {ex} (onset at {onset:.0f} s):")
    print(w[["elapsed_sec", "phase", "label", "anchor_status",
             "target_name", "lead_time_sec", "latency_rel",
             "queue_delay_ms"]].to_string(index=False))
    print()
    print(f"Labeled data: {OUTPUT_FILE}")
    print(f"Report      : {REPORT_FILE}")


if __name__ == "__main__":
    main()
