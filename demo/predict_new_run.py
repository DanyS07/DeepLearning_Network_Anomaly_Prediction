"""
Run the FINAL trained models on a NEW recording (never used in training)
and prepare it for the replay demo.

Typical use: a "mixed" run (one episode of every anomaly type) recorded with
    sudo python3 run_experiment.py --scenario mixed --run mixed_1
and copied to   data/demo_raw/mixed/mixed_1.csv  (+ mixed_1.json)

Run (from the project root):
    python demo/predict_new_run.py                     # every run in data/demo_raw
    python demo/predict_new_run.py data/demo_raw/mixed/mixed_1.csv

What it does, with the SAME code as the training pipeline:
    Stage 1  cleaning            (preprocessing/01_data_cleaning.py)
    Stage 2  features            (preprocessing/02_feature_engineering.py)
    windows  30 s, one for EVERY second (as a live system would)
    scaling  the "full" scaler   (fitted on all training runs only)
    models   every final model in models/saved/ (seed models averaged)
    alarm    threshold = median of the per-fold fa5 thresholds
    detector rule-based "what is wrong right now" (demo/detector.py)
    targets  future-onset labels as in Stage 3 (type taken per episode),
             used ONLY to score the run, never as model input

Output: results/demo_runs/<scenario>__<run>/
    signals.csv              Stage 1 rows + detector columns
    predictions_<model>.csv  probabilities for every second
    info.json                thresholds, onsets, episode order
    summary.txt              warned onsets, lead times, false alarms
"""

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "demo"))

DEMO_RAW = ROOT / "data" / "demo_raw"
OUT_ROOT = ROOT / "results" / "demo_runs"
SAVED = ROOT / "models" / "saved"
SCALERS = ROOT / "data" / "processed" / "model_ready" / "scalers.json"
SEQ_CONFIG = ROOT / "data" / "processed" / "sequences" / "sequence_config.json"
THRESH = ROOT / "results" / "evaluation" / "thresholds.csv"
HEADLINE_OP = "fa5"


def load_module(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stage1 = load_module("preprocessing/01_data_cleaning.py", "stage1")
stage2 = load_module("preprocessing/02_feature_engineering.py", "stage2")
stage3 = load_module("preprocessing/03_future_labels.py", "stage3")
from detector import detect                      # noqa: E402

CLASS_ID, CLASS_NAME = stage3.CLASS_ID, stage3.CLASS_NAME
CLASSES = [CLASS_NAME[i] for i in range(len(CLASS_NAME))]


# ============================================================
# STAGE 1 + 2 (same code as training)
# ============================================================

def fix_burst_labels_typed(df, baseline):
    """Stage 1 traffic_burst fix, applied only after traffic_burst rows
    (a mixed run also contains other anomaly types)."""
    lab = df["label"].to_numpy().copy()
    typ = df["anomaly_type"].astype(str).to_numpy().copy()
    dirty = ((df["qdisc_backlog_bytes"] > stage1.BURST_QUEUE_BYTES)
             | (df["tbf_dropped_per_sec"] > 0)
             | (df["latency_ms"] > baseline + stage1.LAT_RISE_MS)).to_numpy()
    changed = 0
    for i in range(1, len(df)):
        if (lab[i] == 0 and lab[i - 1] == 1 and typ[i - 1] == "traffic_burst"
                and dirty[i]):
            lab[i], typ[i] = 1, "traffic_burst"
            changed += 1
    df["label"], df["anomaly_type"] = lab, typ
    return changed


def clean_and_features(csv_path):
    csv_path = Path(csv_path)
    run_id = f"new/{csv_path.parent.name}/{csv_path.stem}"
    df, info = stage1.clean_run("new", csv_path, run_id)
    if info["scenario"] != "traffic_burst":     # clean_run did it otherwise
        info["burst_label_rows_added"] = fix_burst_labels_typed(
            df, info["baseline_latency_ms"])
    df = df.sort_values("elapsed_sec").reset_index(drop=True)
    clean = df.copy()

    df = stage2.fill_missing(df)
    f = stage2.add_instant_features(df)
    f = stage2.add_temporal_features(df, f)
    f, _ = stage2.compress_heavy_tails(f)
    return clean, df, f, info


# ============================================================
# FUTURE-ONSET TARGETS (Stage 3 rules, class taken per onset)
# ============================================================

def future_targets(df):
    t = df["elapsed_sec"].to_numpy()
    y = df["label"].to_numpy().astype(int)
    seg = df["segment_id"].to_numpy()
    phase = df["phase"].astype(str).to_numpy()
    in_ep = df["episode_id"].to_numpy() > 0
    typ = df["anomaly_type"].astype(str).to_numpy()
    onset_idx = np.flatnonzero((y[1:] == 1) & (y[:-1] == 0)) + 1
    onset_t, onset_cls = t[onset_idx], [CLASS_ID[typ[i]] for i in onset_idx]
    seg_end = pd.Series(t).groupby(seg).transform("max").to_numpy()
    H = stage3.HORIZON_SEC

    n = len(df)
    target, lead = np.full(n, np.nan), np.full(n, np.nan)
    event = np.array([""] * n, dtype=object)
    status = np.array(["used"] * n, dtype=object)
    run_id = df["run_id"].iloc[0]
    for i in range(n):
        if y[i] == 1:
            status[i] = "anomaly_now"
            continue
        if phase[i] == "cooldown":
            status[i] = "cooldown"
            continue
        if t[i] + H > seg_end[i]:
            status[i] = "horizon_incomplete"
            continue
        hit = np.flatnonzero((onset_t > t[i]) & (onset_t <= t[i] + H))
        if len(hit):
            k = hit[0]
            target[i], lead[i] = onset_cls[k], onset_t[k] - t[i]
            event[i] = f"{run_id}#{k + 1}"
        elif stage3.DROP_IN_EPISODE_NEGATIVES and in_ep[i]:
            status[i] = "in_episode_far_from_onset"
        else:
            target[i] = 0
    onsets = [{"t": float(onset_t[k]), "type": CLASSES[onset_cls[k]],
               "event_id": f"{run_id}#{k + 1}"} for k in range(len(onset_t))]
    return pd.DataFrame({"target": target, "lead_time_sec": lead,
                         "event_id": event, "anchor_status": status},
                        index=df.index), onsets


# ============================================================
# WINDOWS FOR EVERY SECOND
# ============================================================

def windows_every_second(df, f, features, lookback):
    X_parts, idx_parts = [], []
    values = f[features].to_numpy(np.float32)
    for _, rows in df.groupby("segment_id").indices.items():
        rows = np.sort(rows)
        if len(rows) < lookback:
            continue
        ends = np.arange(lookback - 1, len(rows))
        gather = rows[ends[:, None] + np.arange(-lookback + 1, 1)[None, :]]
        X_parts.append(values[gather])
        idx_parts.append(rows[ends])
    return np.concatenate(X_parts), np.concatenate(idx_parts)


# ============================================================
# MODELS
# ============================================================

def find_models():
    found = {}
    for p in sorted(SAVED.glob("*_final_seed*.keras")):
        found.setdefault(p.name.split("_final")[0], []).append(p)
    for p in sorted(SAVED.glob("*_final.joblib")):
        found[p.name.split("_final")[0]] = [p]
    return found


def predict(name, files, X):
    if files[0].suffix == ".joblib":
        import joblib
        summarise = load_module("models/01_classical_baselines.py",
                                "baselines").summarise
        model = joblib.load(files[0])
        probs = np.zeros((len(X), len(CLASSES)))
        probs[:, model.classes_] = model.predict_proba(summarise(X))
        return probs
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    out = []
    for p in files:
        m = tf.keras.models.load_model(p, compile=False)
        out.append(m.predict(X, batch_size=1024, verbose=0))
    return np.mean(out, axis=0)


def thresholds():
    if not THRESH.exists():
        return {}
    th = pd.read_csv(THRESH)
    if "op" in th:
        th = th[th["op"] == HEADLINE_OP]
    return th.groupby("model")["threshold"].median().to_dict()


# ============================================================
# MAIN
# ============================================================

def process(csv_path, models, taus, scaler, features, lookback, evaluate):
    clean, df, f, info = clean_and_features(csv_path)
    if list(f.columns) != features:
        raise ValueError("Feature list differs from the one the models were "
                         "trained with - rerun the pipeline / retrain.")
    tgt, onsets = future_targets(df)
    det = detect(clean)

    X, rows = windows_every_second(df, f, features, lookback)
    mean = np.asarray(scaler["mean"], np.float32)
    std = np.asarray(scaler["std"], np.float32)
    X = ((X - mean) / std).astype(np.float32)

    name = f"{info['scenario']}__{info['run']}"
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    clean.join(det).to_csv(out / "signals.csv", index=False)

    base = df.loc[rows, ["run_id", "scenario", "elapsed_sec",
                         "benign_excursion", "phase"]].join(tgt.loc[rows])
    base["detected_now"] = det.loc[rows, "detected"].to_numpy()

    sidecar = json.loads(Path(csv_path).with_suffix(".json").read_text())
    episodes = [{"id": e["id"], "type": e.get("type", info["scenario"]),
                 "start": e["start"], "end": e["end"]}
                for e in sidecar["schedule"]["episodes"]]

    lines = ["=" * 70, f"NEW RUN: {info['run_id']}", "=" * 70, "",
             f"rows {len(clean)}, windows {len(X)}, gaps {info['gaps']}, "
             f"baseline RTT {info['baseline_latency_ms']:.1f} ms", "",
             "Episode order: " + ", ".join(
                 f"{e['type']} @ {e['start']:.0f} s" for e in episodes), ""]
    lines += ["ONSETS (observed)", "-" * 70]
    for o in onsets:
        lines.append(f"  {o['t']:7.0f} s  {o['type']}")
    lines.append("")

    for model, files in models.items():
        probs = predict(model, files, X)
        p = base.copy()
        p["pred"] = probs.argmax(axis=1)
        for c in range(len(CLASSES)):
            p[f"p_{CLASSES[c]}"] = probs[:, c]
        p.to_csv(out / f"predictions_{model}.csv", index=False)

        tau = taus.get(model)
        scored = p[p["anchor_status"] == "used"].copy()
        scored = (evaluate.apply_threshold(scored, tau) if tau is not None
                  else scored)
        ev = evaluate.event_table(scored)
        fa = evaluate.false_alarm_stats(scored)
        lines += [f"MODEL {model}  (alarm threshold "
                  + (f"{tau:.4f}, {HEADLINE_OP})" if tau is not None
                     else "argmax - no thresholds.csv)"), "-" * 70]
        if len(ev):
            for _, r in ev.iterrows():
                lines.append(
                    f"  {r['event_id'].split('#')[-1]:>3s} {r['class']:24s} "
                    + (f"warned {r['lead_time_sec']:4.0f} s before"
                       if r["detected"] else
                       "missed" + (" (warned, wrong type)"
                                   if r["detected_any_anomaly"] else "")))
            lines.append(f"  detected {ev['detected'].sum()}/{len(ev)} "
                         f"onsets; mean lead "
                         f"{ev.loc[ev['detected'], 'lead_time_sec'].mean():.1f} s")
        lines.append(f"  false alarms/h on normal seconds: "
                     f"{fa['false_alarms_per_hour']:.1f}  "
                     f"({len(scored[scored['target'] == 0]) / 60:.0f} min "
                     f"of normal time scored)")
        lines.append("")
        print(f"  {model}: {ev['detected'].sum() if len(ev) else 0}/"
              f"{len(ev)} onsets warned")

    # detector check against the ground truth
    lab = clean["label"].to_numpy() == 1
    d = det["detected"].to_numpy() == 1
    typ_ok = (det["det_type"].to_numpy() == clean["anomaly_type"]
              .astype(str).to_numpy())[lab & d & (det["det_type"].to_numpy()
                                                  != "identifying")]
    # per detection: types shown in order (ignoring "identifying")
    n_det = n_changed = n_wrong = n_false = 0
    i, typ_shown = 0, det["det_type"].to_numpy()
    truth = clean["anomaly_type"].astype(str).to_numpy()
    while i < len(d):
        if not d[i]:
            i += 1
            continue
        j = i
        while j < len(d) and d[j]:
            j += 1
        seq = [k for k in pd.unique(typ_shown[i:j])
               if k not in ("identifying", "normal")]
        if lab[i:j].any():
            n_det += 1
            true_t = truth[i:j][lab[i:j]][0]
            n_changed += len(seq) > 1 and true_t in seq
            n_wrong += true_t not in seq
        elif clean["phase"].iloc[i] != "cooldown":
            n_false += 1
        i = j
    lines += ["RULE-BASED DETECTOR vs ground truth", "-" * 70,
              f"  anomaly seconds detected : {d[lab].mean():.3f}",
              f"  type correct (once shown): {typ_ok.mean():.3f}",
              f"  detections of anomalies  : {n_det}  (type changed "
              f"{n_changed}, wrong type {n_wrong})",
              f"  false detections         : {n_false}", ""]

    (out / "info.json").write_text(json.dumps({
        "run_id": info["run_id"], "scenario": info["scenario"],
        "baseline_latency_ms": info["baseline_latency_ms"],
        "thresholds": {m: taus.get(m) for m in models},
        "threshold_op": HEADLINE_OP, "onsets": onsets,
        "episodes": episodes, "models": list(models)}, indent=2))
    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved to {out}")


def main():
    paths = [Path(a) for a in sys.argv[1:]] or sorted(DEMO_RAW.glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No runs given and none in {DEMO_RAW}.")
    models = find_models()
    if not models:
        raise FileNotFoundError(f"No final models in {SAVED}. Train first.")
    scal = json.loads(SCALERS.read_text())
    features = scal["features"]
    lookback = json.loads(SEQ_CONFIG.read_text())["lookback"]
    taus = thresholds()
    evaluate = load_module("models/evaluate.py", "evaluate")
    print(f"models: {', '.join(models)}")
    for p in paths:
        print(f"\n>>> {p}")
        process(p, models, taus, scal["full"], features, lookback, evaluate)


if __name__ == "__main__":
    main()
