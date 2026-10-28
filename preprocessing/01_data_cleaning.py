"""
Stage 1 - Data cleaning (rewritten for collector v2.7 / v2.8 output).

Input : data/raw/<scenario>/<run>.csv + <run>.json      (VirtualBox)
        data/raw_utm/<scenario>/<run>.csv + <run>.json  (UTM, optional)
Output: data/processed/network_telemetry_clean.csv
        data/processed/run_manifest.csv
        results/telemetry_data_quality_report.txt

What this stage does
  * loads every run together with its JSON sidecar (seed, baseline latency)
  * gives every run a unique run_id = machine/scenario/run
  * converts all measurement columns to numbers
  * drops ONLY the first row of each run (its per-second rates are NaN by
    design); missing values in the middle of a run are KEPT and flagged,
    because deleting a row would create a silent hole in the time series
  * applies the traffic_burst label fix (label stayed 0 for ~3 s after each
    burst while the network was still disturbed)
  * marks recording gaps (actual_delta_t > 1.5 s) with a segment_id, so later
    stages never build a window across a gap

What it does NOT do: no scaling, no imputation, no future labels, no split.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# machine name -> raw folder. Missing folders are skipped.
RAW_DIRS = {
    "vbox": PROJECT_ROOT / "data" / "raw",
    "utm": PROJECT_ROOT / "data" / "raw_utm",
}

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_FILE = PROCESSED_DIR / "network_telemetry_clean.csv"
MANIFEST_FILE = PROCESSED_DIR / "run_manifest.csv"
REPORT_FILE = RESULTS_DIR / "telemetry_data_quality_report.txt"

# Runs rejected during collection (VM paused before an onset etc.).
# Better: move them out of data/raw. This list is a safety net.
EXCLUDED_RUNS = {
    # "vbox/bandwidth_degradation/bandwidth_degradation_2",
}

GAP_THRESHOLD_SEC = 1.5          # actual_delta_t above this = recording gap
BURST_QUEUE_BYTES = 64_000       # traffic_burst label fix thresholds
LAT_RISE_MS = 50.0


# ============================================================
# COLUMNS
# ============================================================

ID_COLUMNS = ["experiment_id", "scenario", "seed", "timestamp"]

TIMING_COLUMNS = ["elapsed_sec", "actual_delta_t", "sampling_delay_sec"]

# Metadata written by the collector. Kept for analysis and for later
# stages (cooldown skipping), but NEVER used as model inputs.
METADATA_COLUMNS = [
    "phase", "episode_id", "benign_excursion", "injected_intensity",
    "injected_bg_mbit", "injected_cross_mbit", "injected_delay_ms",
    "injected_loss_pct", "injected_rate_mbit", "injected_queue_target_bytes",
]

MEASUREMENT_COLUMNS = [
    "latency_ms", "latency_max_ms", "ping_replies", "ping_loss_pct",
    "rx_bytes", "tx_bytes", "rx_packets", "tx_packets",
    "rx_dropped", "tx_dropped", "rx_errors", "tx_errors",
    "rx_bytes_per_sec", "tx_bytes_per_sec",
    "rx_packets_per_sec", "tx_packets_per_sec", "tx_dropped_per_sec",
    "ingress_rx_bytes_per_sec", "ingress_rx_packets_per_sec",
    "qdisc_dropped", "qdisc_dropped_per_sec", "tbf_dropped_per_sec",
    "netem_dropped_per_sec",
    "qdisc_backlog_bytes", "qdisc_backlog_packets", "netem_backlog_packets",
]

LABEL_COLUMNS = ["label", "anomaly_type"]

EXPECTED_COLUMNS = (ID_COLUMNS + TIMING_COLUMNS + METADATA_COLUMNS
                    + MEASUREMENT_COLUMNS + LABEL_COLUMNS)

NUMERIC_COLUMNS = (["seed"] + TIMING_COLUMNS
                   + [c for c in METADATA_COLUMNS if c != "phase"]
                   + MEASUREMENT_COLUMNS + ["label"])

# Counters and rates that can never be negative.
NON_NEGATIVE_COLUMNS = [c for c in MEASUREMENT_COLUMNS]


# ============================================================
# LOADING
# ============================================================

def find_runs():
    runs = []
    for machine, raw_dir in RAW_DIRS.items():
        if not raw_dir.exists():
            continue
        for csv_path in sorted(raw_dir.glob("*/*.csv")):
            run_id = f"{machine}/{csv_path.parent.name}/{csv_path.stem}"
            runs.append((machine, csv_path, run_id))
    if not runs:
        raise FileNotFoundError(
            "No CSV files found in: "
            + ", ".join(str(p) for p in RAW_DIRS.values()))
    return runs


def load_sidecar(csv_path):
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(
            f"{csv_path.name}: JSON sidecar {json_path.name} is missing. "
            f"The baseline latency and seed come from it.")
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


# ============================================================
# PER-RUN CLEANING
# ============================================================

def validate_columns(df, name):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns: {', '.join(missing)}")


def convert_types(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce",
                                     utc=True)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("phase", "anomaly_type", "scenario", "experiment_id"):
        df[col] = df[col].astype("string")
    return df


def fix_traffic_burst_labels(df, baseline_ms):
    """Keep label=1 while the network is still disturbed after a burst.

    The collector switches the label off the instant the burst traffic stops,
    but the per-second counters, the queue and the ping RTT need 1-3 s to
    return to normal. Without this fix those disturbed rows look like
    'normal' and the model is punished for seeing them correctly.
    Returns the number of rows changed.
    """
    lab = df["label"].to_numpy().copy()
    dirty = ((df["qdisc_backlog_bytes"] > BURST_QUEUE_BYTES)
             | (df["tbf_dropped_per_sec"] > 0)
             | (df["latency_ms"] > baseline_ms + LAT_RISE_MS)).to_numpy()
    changed = 0
    for i in range(1, len(df)):
        if lab[i] == 0 and lab[i - 1] == 1 and dirty[i]:
            lab[i] = 1
            changed += 1
    df["label"] = lab
    df.loc[df["label"] == 1, "anomaly_type"] = "traffic_burst"
    return changed


def clean_run(machine, csv_path, run_id):
    df = pd.read_csv(csv_path)
    validate_columns(df, run_id)
    sidecar = load_sidecar(csv_path)
    baseline = sidecar.get("baseline_latency_ms")
    if baseline is None:
        raise ValueError(f"{run_id}: no baseline_latency_ms in the JSON.")

    df = convert_types(df)
    info = {"run_id": run_id, "machine": machine,
            "scenario": csv_path.parent.name, "run": csv_path.stem,
            "seed": sidecar.get("seed"),
            "baseline_latency_ms": baseline,
            "raw_rows": len(df),
            "finished_at": sidecar.get("finished_at"),
            "interrupted": sidecar.get("interrupted"),
            "collector_warnings": len(sidecar.get("warnings", [])),
            "stalls": len(sidecar.get("stalls", []))}

    # The folder name is the truth for the scenario.
    if (df["scenario"] != info["scenario"]).any():
        raise ValueError(f"{run_id}: scenario column does not match folder.")

    # 1. Rows with an unreadable timestamp (should be none).
    bad_ts = df["timestamp"].isna()
    info["invalid_timestamps"] = int(bad_ts.sum())
    df = df.loc[~bad_ts]

    # 2. Order by elapsed time inside the run, remove exact duplicates.
    df = df.sort_values("elapsed_sec")
    before = len(df)
    df = df.drop_duplicates(subset=["elapsed_sec"])
    info["duplicate_rows"] = before - len(df)

    # 3. Negative counters/rates are impossible -> NaN (flagged below).
    neg = (df[NON_NEGATIVE_COLUMNS] < 0)
    info["negative_values"] = int(neg.to_numpy().sum())
    df[NON_NEGATIVE_COLUMNS] = df[NON_NEGATIVE_COLUMNS].mask(neg)

    # 4. Drop only the first row: its rates are NaN because there is no
    #    previous sample. Everything else stays so the series has no holes.
    df = df.iloc[1:].copy()

    # 5. Flag (do not drop) rows with missing measurements. Example: every
    #    ping of a second lost during packet_loss -> latency_ms is NaN.
    #    Stage 2 fills these forward within the run and keeps this flag.
    df["missing_any"] = df[MEASUREMENT_COLUMNS].isna().any(axis=1).astype(int)
    info["rows_with_missing"] = int(df["missing_any"].sum())

    # 6. traffic_burst label fix.
    info["burst_label_rows_added"] = 0
    if info["scenario"] == "traffic_burst":
        info["burst_label_rows_added"] = fix_traffic_burst_labels(df, baseline)

    # 7. Recording gaps -> segment_id. Windows must stay inside a segment.
    gap = (df["actual_delta_t"] > GAP_THRESHOLD_SEC).fillna(False)
    df["segment_id"] = gap.cumsum().astype(int)
    info["gaps"] = int(gap.sum())
    info["max_delta_t"] = float(df["actual_delta_t"].max())

    # 8. Identity and baseline columns.
    df.insert(0, "run_id", run_id)
    df.insert(1, "machine", machine)
    df["baseline_latency_ms"] = baseline

    info["clean_rows"] = len(df)
    info["anomaly_rows"] = int((df["label"] == 1).sum())
    info["episodes"] = len(sidecar.get("schedule", {}).get("episodes", []))
    return df, info


# ============================================================
# REPORT
# ============================================================

def build_report(df, manifest):
    lines = ["=" * 70, "NETWORK TELEMETRY DATA QUALITY REPORT", "=" * 70, ""]
    lines.append(f"Runs loaded      : {len(manifest)}")
    lines.append(f"Raw rows         : {manifest['raw_rows'].sum()}")
    lines.append(f"Clean rows       : {len(df)}")
    lines.append(f"Rows dropped     : "
                 f"{manifest['raw_rows'].sum() - len(df)} "
                 f"(first row of each run + invalid/duplicate rows)")
    lines.append("")

    lines += ["RUNS PER MACHINE AND SCENARIO", "-" * 70]
    table = manifest.groupby(["machine", "scenario"]).size()
    for (m, s), n in table.items():
        lines.append(f"{m:5s} {s:24s} {n} runs")
    lines.append("")

    lines += ["TIMING", "-" * 70]
    dt = df["actual_delta_t"].dropna()
    lines.append(f"actual_delta_t median {dt.median():.3f} s, "
                 f"max {dt.max():.3f} s")
    gappy = manifest[manifest["gaps"] > 0]
    lines.append(f"Runs with recording gaps (> {GAP_THRESHOLD_SEC} s): "
                 f"{len(gappy)}")
    for _, r in gappy.iterrows():
        lines.append(f"  {r['run_id']}: {r['gaps']} gap(s), "
                     f"max {r['max_delta_t']:.1f} s")
    short = manifest[manifest["raw_rows"] < 890]
    lines.append(f"Short runs (< 890 rows): {len(short)}")
    for _, r in short.iterrows():
        lines.append(f"  {r['run_id']}: {r['raw_rows']} rows")
    lines.append("")

    lines += ["MISSING VALUES (rows kept, flagged in missing_any)", "-" * 70]
    miss = df[MEASUREMENT_COLUMNS].isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        lines.append("None.")
    for col, n in miss.items():
        lines.append(f"{col}: {n}")
    lines.append("")

    lines += ["LABEL FIX", "-" * 70]
    lines.append("traffic_burst rows switched 0 -> 1: "
                 f"{manifest['burst_label_rows_added'].sum()}")
    lines.append("")

    lines += ["BASELINE LATENCY PER MACHINE (ms)", "-" * 70]
    for m, g in manifest.groupby("machine"):
        lines.append(f"{m}: min {g['baseline_latency_ms'].min():.1f}, "
                     f"median {g['baseline_latency_ms'].median():.1f}, "
                     f"max {g['baseline_latency_ms'].max():.1f}")
    lines.append("")

    lines += ["LABELS", "-" * 70]
    for t, n in df["anomaly_type"].value_counts().items():
        lines.append(f"{t:24s} {n}")
    lines.append("")
    lines.append("Not done here: scaling, future labels, splitting.")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frames, infos = [], []
    for machine, csv_path, run_id in find_runs():
        if run_id in EXCLUDED_RUNS:
            print(f"[SKIP] {run_id} (excluded)")
            continue
        df, info = clean_run(machine, csv_path, run_id)
        frames.append(df)
        infos.append(info)
        print(f"[OK]   {run_id}: {info['clean_rows']} rows, "
              f"{info['gaps']} gap(s)")

    df = pd.concat(frames, ignore_index=True)
    manifest = pd.DataFrame(infos)

    # Same seed twice on one machine = same schedule twice (re-used run id).
    dup = manifest[manifest.duplicated(["machine", "seed"], keep=False)]
    if not dup.empty:
        raise ValueError("Repeated seed on one machine:\n"
                         + dup[["run_id", "seed"]].to_string())

    df.to_csv(OUTPUT_FILE, index=False)
    manifest.to_csv(MANIFEST_FILE, index=False)
    report = build_report(df, manifest)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print()
    print(report)
    print()
    print(f"Clean data : {OUTPUT_FILE}")
    print(f"Manifest   : {MANIFEST_FILE}")
    print(f"Report     : {REPORT_FILE}")


if __name__ == "__main__":
    main()
