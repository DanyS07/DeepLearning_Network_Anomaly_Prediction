"""
Stage 2 - Feature engineering.

Input : data/processed/network_telemetry_clean.csv   (from Stage 1)
Output: data/processed/network_telemetry_features.csv
        data/processed/feature_columns.json           (THE model inputs)
        results/feature_engineering_report.txt

Rules followed here
  * Only measured telemetry becomes a feature. Collector metadata
    (phase, episode_id, benign_excursion, injected_*) and the label are
    kept in the CSV for later stages but are never listed as features.
  * Latency is expressed relative to each run's own baseline
    (latency_rel), so the 40 ms vs 44 ms machine/day difference does not
    become a shortcut. Absolute latency is not a feature.
  * Every temporal feature (diff, rolling) is computed inside one
    run_id + segment_id, looking only backwards. Nothing crosses a
    recording gap and nothing uses future rows.
  * Missing measurements are forward-filled inside the segment; the
    Stage 1 flag missing_any is kept as a feature so the model knows.
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

INPUT_FILE = PROCESSED_DIR / "network_telemetry_clean.csv"
OUTPUT_FILE = PROCESSED_DIR / "network_telemetry_features.csv"
FEATURE_LIST_FILE = PROCESSED_DIR / "feature_columns.json"
REPORT_FILE = RESULTS_DIR / "feature_engineering_report.txt"

NOMINAL_BOTTLENECK_MBIT = 50.0   # the link's normal capacity (a constant,
                                 # NOT the injected rate, which is metadata)
ROLLING_WINDOWS = [5, 10]        # seconds (rows), backward-looking
EPS = 1e-9

GROUP = ["run_id", "segment_id"]

# Columns carried through unchanged for later stages - NOT features.
KEEP_COLUMNS = [
    "run_id", "machine", "scenario", "experiment_id", "seed", "timestamp",
    "elapsed_sec", "actual_delta_t", "segment_id", "baseline_latency_ms",
    "phase", "episode_id", "benign_excursion",
    "label", "anomaly_type",
]


# ============================================================
# STEP 1 - FILL MISSING MEASUREMENTS (inside a segment only)
# ============================================================

def fill_missing(df):
    # A second in which every ping was lost has no RTT. Carry the last
    # known value forward; missing_any (from Stage 1) records that it was
    # filled. At the very start of a segment there is nothing to carry,
    # so latency falls back to the baseline.
    for col in ["latency_ms", "latency_max_ms"]:
        df[col] = df.groupby(GROUP)[col].ffill()
        df[col] = df[col].fillna(df["baseline_latency_ms"])

    # Ping loss needs 20 judged probes, so it is NaN for the first ~12 s
    # of every run (always normal traffic). Before that: no loss seen.
    df["ping_loss_pct"] = df.groupby(GROUP)["ping_loss_pct"].ffill()
    df["ping_loss_pct"] = df["ping_loss_pct"].fillna(0.0)

    rate_cols = [
        "rx_bytes_per_sec", "tx_bytes_per_sec", "rx_packets_per_sec",
        "tx_packets_per_sec", "ingress_rx_bytes_per_sec",
        "ingress_rx_packets_per_sec", "tbf_dropped_per_sec",
        "netem_dropped_per_sec", "qdisc_backlog_bytes",
        "qdisc_backlog_packets", "ping_replies",
    ]
    for col in rate_cols:
        df[col] = df.groupby(GROUP)[col].ffill().fillna(0.0)
    return df


# ============================================================
# STEP 2 - PER-SECOND FEATURES
# ============================================================

def add_instant_features(df):
    f = pd.DataFrame(index=df.index)

    # Latency, relative to this run's baseline
    f["latency_rel"] = df["latency_ms"] - df["baseline_latency_ms"]
    f["latency_max_rel"] = df["latency_max_ms"] - df["baseline_latency_ms"]
    f["latency_spread"] = (df["latency_max_ms"] - df["latency_ms"]).clip(lower=0)

    # Ping
    f["ping_loss_pct"] = df["ping_loss_pct"]
    f["ping_replies"] = df["ping_replies"]

    # Load offered to the router, and what it actually delivered
    f["offered_mbit"] = df["ingress_rx_bytes_per_sec"] * 8 / 1e6
    f["delivered_mbit"] = df["tx_bytes_per_sec"] * 8 / 1e6
    f["offered_pps"] = df["ingress_rx_packets_per_sec"]
    f["utilization"] = f["delivered_mbit"] / NOMINAL_BOTTLENECK_MBIT
    f["load_ratio"] = f["offered_mbit"] / NOMINAL_BOTTLENECK_MBIT

    # Delivery shortfall: fraction of offered bytes not delivered this
    # second (queued or dropped). Clipped: a draining queue delivers more
    # than arrives, which would give large negative values.
    f["shortfall"] = (1 - df["tx_bytes_per_sec"]
                      / (df["ingress_rx_bytes_per_sec"] + EPS)).clip(-1, 1)
    f.loc[df["ingress_rx_bytes_per_sec"] <= 0, "shortfall"] = 0.0

    # Router queue
    f["queue_bytes"] = df["qdisc_backlog_bytes"]
    f["queue_packets"] = df["qdisc_backlog_packets"]
    f["queue_delay_ms"] = (df["qdisc_backlog_bytes"] * 8
                           / (NOMINAL_BOTTLENECK_MBIT * 1e6) * 1000)

    # Drops at the router
    pps = df["ingress_rx_packets_per_sec"]
    f["drop_ratio"] = np.where(pps > 0, df["tbf_dropped_per_sec"] / (pps + EPS), 0.0)
    f["loss_ratio"] = np.where(pps > 0, df["netem_dropped_per_sec"] / (pps + EPS), 0.0)
    f["drop_ratio"] = f["drop_ratio"].clip(0, 1)
    f["loss_ratio"] = f["loss_ratio"].clip(0, 1)

    # Data-quality flag from Stage 1
    f["missing_any"] = df["missing_any"].astype(int)
    return f


# ============================================================
# STEP 3 - TEMPORAL FEATURES (backward-looking, per segment)
# ============================================================

TREND_FEATURES = ["latency_rel", "queue_delay_ms", "offered_mbit",
                  "delivered_mbit", "shortfall", "drop_ratio", "loss_ratio"]
ROLLING_FEATURES = ["latency_rel", "queue_delay_ms", "offered_mbit",
                    "shortfall"]


def add_temporal_features(df, f):
    keys = [df[k] for k in GROUP]

    for col in TREND_FEATURES:
        f[f"{col}_diff"] = f[col].groupby(keys).diff().fillna(0.0)

    for w in ROLLING_WINDOWS:
        for col in ROLLING_FEATURES:
            roll = f[col].groupby(keys).rolling(w, min_periods=1)
            f[f"{col}_mean_{w}"] = roll.mean().reset_index(level=[0, 1], drop=True)
            f[f"{col}_std_{w}"] = (roll.std().reset_index(level=[0, 1], drop=True)
                                   .fillna(0.0))
    return f


# ============================================================
# STEP 4 - COMPRESS HEAVY-TAILED FEATURES
# ============================================================

# These range from ~0 in normal traffic to hundreds of ms / millions of
# bytes during an anomaly. Neural networks learn poorly from such ranges
# (a few huge values dominate the scaling), so they are compressed with
#     signed log:  sign(x) * log(1 + |x|)
# which keeps 0 at 0, keeps the sign of changes, and is monotonic (tree
# models such as the random forest are unaffected by it).
LOG_BASE_FEATURES = ["latency_rel", "latency_max_rel", "latency_spread",
                     "queue_bytes", "queue_packets", "queue_delay_ms",
                     "offered_pps"]


def compress_heavy_tails(f):
    cols = [c for c in f.columns
            if any(c == b or c.startswith(b + "_") for b in LOG_BASE_FEATURES)]
    f[cols] = np.sign(f[cols]) * np.log1p(np.abs(f[cols]))
    return f, cols


# ============================================================
# REPORT
# ============================================================

def build_report(out, features):
    lines = ["=" * 70, "FEATURE ENGINEERING REPORT", "=" * 70, ""]
    lines.append(f"Rows             : {len(out)}")
    lines.append(f"Runs             : {out['run_id'].nunique()}")
    lines.append(f"Segments         : {out.groupby(GROUP).ngroups}")
    lines.append(f"Model features   : {len(features)}")
    lines.append("")
    lines += ["FEATURES (mean / std over all rows)", "-" * 70]
    for c in features:
        lines.append(f"{c:28s} {out[c].mean():12.4f} {out[c].std():12.4f}")
    lines.append("")
    lines += ["CHECKS", "-" * 70]
    n_nan = int(out[features].isna().sum().sum())
    n_inf = int(np.isinf(out[features].to_numpy()).sum())
    lines.append(f"NaN in features : {n_nan}")
    lines.append(f"Inf in features : {n_inf}")
    const = [c for c in features if out[c].std() == 0]
    lines.append(f"Constant features: {const if const else 'none'}")
    lines.append("")
    lines += ["MEAN latency_rel DURING NORMAL ROWS, PER MACHINE", "-" * 70]
    normal = out[out["label"] == 0]
    for m, g in normal.groupby("machine"):
        lines.append(f"{m}: {g['latency_rel'].mean():.3f} (signed-log ms, "
                     f"should be close to 0)")
    lines.append("")
    lines.append("Metadata columns (phase, episode_id, benign_excursion) are "
                 "kept for later stages but are NOT model features.")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Run Stage 1 first: {INPUT_FILE} missing.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {df['run_id'].nunique()} runs.")

    df = df.sort_values(["run_id", "elapsed_sec"]).reset_index(drop=True)
    df = fill_missing(df)

    f = add_instant_features(df)
    f = add_temporal_features(df, f)
    f, logged = compress_heavy_tails(f)
    print(f"Signed-log applied to {len(logged)} heavy-tailed features.")
    features = list(f.columns)

    out = pd.concat([df[KEEP_COLUMNS], f], axis=1)

    bad = out[features].replace([np.inf, -np.inf], np.nan).isna().sum()
    if bad.sum() > 0:
        raise ValueError("NaN/inf left in features:\n" + str(bad[bad > 0]))

    out.to_csv(OUTPUT_FILE, index=False)
    FEATURE_LIST_FILE.write_text(json.dumps(features, indent=2))
    report = build_report(out, features)
    REPORT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print()
    print(f"Features     : {OUTPUT_FILE}")
    print(f"Feature list : {FEATURE_LIST_FILE}")
    print(f"Report       : {REPORT_FILE}")


if __name__ == "__main__":
    main()
