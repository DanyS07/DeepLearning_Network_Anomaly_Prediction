"""Separate reverse-domain experiment: train/calibrate on UTM, test on VirtualBox.

This script writes only to data/reverse_domain/, models/saved/reverse_domain/, and
results/reverse_domain/. Existing forward-domain outputs and source pipelines are unchanged.
"""

from pathlib import Path
import json
import random
import re
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent
SEQ_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
DATA_DIR = PROJECT_ROOT / "data" / "reverse_domain"
PRED_DIR = PROJECT_ROOT / "results" / "reverse_domain" / "predictions"
OUT_DIR = PROJECT_ROOT / "results" / "reverse_domain"
MODEL_DIR = PROJECT_ROOT / "models" / "saved" / "reverse_domain"
N_FOLD = 5
SEED = 42
CLASS_NAMES = ["normal", "congestion", "high_latency", "packet_loss",
               "bandwidth_degradation", "queue_buildup", "traffic_burst"]
N_CLASSES = len(CLASS_NAMES)


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def run_number(run_id):
    m = re.search(r"_(\d+)$", run_id)
    return int(m.group(1)) if m else 10 ** 6


def fit_scaler(X):
    flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    const = std < 1e-6
    std[const] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist(),
            "constant_features": np.flatnonzero(const).tolist()}


def scale(X, scaler):
    mean = np.asarray(scaler["mean"], np.float32)
    std = np.asarray(scaler["std"], np.float32)
    return ((X - mean) / std).astype(np.float32)


def false_alarm_stats(p, CLASS_NAMES=CLASS_NAMES):
    n = p[p["target"] == 0].sort_values(["run_id", "elapsed_sec"])
    alarms = 0
    for _, g in n.groupby("run_id"):
        a = g["pred"].to_numpy() > 0
        t = g["elapsed_sec"].to_numpy()
        cont = np.r_[False, a[:-1] & (np.diff(t) <= 1.5)]
        seg_start = a & ~cont
        seg_id = np.cumsum(seg_start)
        for sid in np.unique(seg_id[a]):
            m = a & (seg_id == sid)
            dur = t[m][-1] - t[m][0] + 1
            alarms += int(np.ceil(dur / 60.0))
    hours = len(n) / 3600
    return alarms / hours if hours else float("nan")


def apply_threshold(p, tau):
    q = p.copy()
    prob_cols = [f"p_{c}" for c in CLASS_NAMES]
    probs = q[prob_cols].to_numpy()
    score = 1.0 - probs[:, 0]
    best_anomaly = probs[:, 1:].argmax(axis=1) + 1
    q["pred"] = np.where(score >= tau, best_anomaly, 0)
    return q


def choose_threshold(val, target=5):
    grid = np.unique(np.concatenate([
        np.linspace(0.0, 0.9, 46), 1 - np.logspace(-1, -5, 60)]))
    for tau in grid:
        if false_alarm_stats(apply_threshold(val, tau)) <= target:
            return float(tau)
    return float(grid[-1])


def summarise_windows(X):
    return np.concatenate([X[:, -1, :], X.mean(axis=1),
                           X[:, -1, :] - X[:, 0, :]], axis=1)


def class_weights(y):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = np.zeros(N_CLASSES)
    present = counts > 0
    w[present] = np.sqrt(counts[present].max() / counts[present])
    w /= w[y].mean()
    return {i: float(w[i]) for i in range(N_CLASSES)}


def save_predictions(name, tag, meta, probs):
    out = meta[["run_id", "scenario", "elapsed_sec", "target", "event_id",
                "lead_time_sec", "benign_excursion", "phase"]].copy()
    out["pred"] = probs.argmax(axis=1)
    for c, name_c in enumerate(CLASS_NAMES):
        out[f"p_{name_c}"] = probs[:, c]
    d = PRED_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"{tag}.csv", index=False)
    return out


def evaluate_set(p):
    y = p["target"].to_numpy(dtype=int)
    pred = p["pred"].to_numpy(dtype=int)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, pred, labels=list(range(N_CLASSES)), zero_division=0)
    present = sup > 0
    ev_rows = []
    for eid, g in p[p["event_id"].notna()].groupby("event_id"):
        cls = int(g["target"].iloc[0])
        correct = g[g["pred"] == cls]
        ev_rows.append({"event_id": eid, "class": CLASS_NAMES[cls],
                        "windows": len(g), "detected": len(correct) > 0,
                        "lead_time_sec": correct["lead_time_sec"].max()
                        if len(correct) else 0.0})
    ev = pd.DataFrame(ev_rows)
    per_class = {
        CLASS_NAMES[c]: {"precision": pr[c], "recall": rc[c], "f1": f1[c],
                         "support": int(sup[c])}
        for c in range(N_CLASSES)
    }
    if len(ev):
        det = ev.groupby("class")["detected"].mean()
        lead = ev[ev["detected"]].groupby("class")["lead_time_sec"].mean()
        for c in CLASS_NAMES:
            if c in per_class:
                per_class[c]["event_detection"] = det.get(c, float("nan"))
                per_class[c]["lead_time_sec"] = lead.get(c, float("nan"))
    summary = {"macro_f1": float(f1[present].mean()),
               "event_detection_rate": float(ev["detected"].mean()) if len(ev) else float("nan"),
               "mean_lead_time_sec": float(ev.loc[ev["detected"], "lead_time_sec"].mean()) if len(ev) and ev["detected"].any() else float("nan"),
               "false_alarms_per_hour": false_alarm_stats(p), "events": len(ev)}
    return summary, per_class, ev


def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(SEQ_DIR / "X.npy", mmap_mode="r")
    y = np.load(SEQ_DIR / "y.npy")
    meta = pd.read_csv(SEQ_DIR / "metadata.csv")
    all_runs = meta[["run_id", "machine", "scenario"]].drop_duplicates()
    utm_runs = all_runs[all_runs["machine"] == "utm"].copy()
    utm_runs["number"] = utm_runs["run_id"].map(run_number)
    utm_runs = utm_runs.sort_values(["scenario", "number", "run_id"])
    utm_runs["position"] = utm_runs.groupby("scenario").cumcount() % N_FOLD
    external = sorted(all_runs.loc[all_runs["machine"] == "vbox", "run_id"])

    folds = {}
    for k in range(N_FOLD):
        role = utm_runs["position"].map(lambda p: "test" if p == k else
                                        ("validation" if p == (k + 1) % N_FOLD else "train"))
        folds[str(k)] = {r: sorted(utm_runs.loc[role == r, "run_id"].tolist())
                         for r in ("train", "validation", "test")}
    fold_doc = {"main_machine": "utm", "n_folds": N_FOLD,
                "folds": folds, "external": external}
    (DATA_DIR / "folds.json").write_text(json.dumps(fold_doc, indent=2), encoding="utf-8")

    scalers = {}
    features = json.loads((SEQ_DIR / "sequence_config.json").read_text())["features"]
    for k, fold in folds.items():
        idx = np.flatnonzero(meta["run_id"].isin(fold["train"]).to_numpy())
        scalers[k] = fit_scaler(np.asarray(X[idx]))
    all_utm = sorted(set(r for f in folds.values() for part in f.values() for r in part))
    idx = np.flatnonzero(meta["run_id"].isin(all_utm).to_numpy())
    scalers["full"] = fit_scaler(np.asarray(X[idx]))
    scalers["features"] = features
    (DATA_DIR / "scalers.json").write_text(json.dumps(scalers, indent=2), encoding="utf-8")
    return X, y, meta, fold_doc, scalers


def take(X, y, meta, run_ids, scaler):
    idx = np.flatnonzero(meta["run_id"].isin(run_ids).to_numpy())
    return scale(np.asarray(X[idx]), scaler), y[idx], meta.iloc[idx].reset_index(drop=True)


def train_sklearn(name, make_model, X, y, meta, folds, scalers):
    thresholds = []
    for k in range(N_FOLD):
        trX, tr_y, trm = take(X, y, meta, folds[str(k)]["train"], scalers[str(k)])
        vaX, vay, vam = take(X, y, meta, folds[str(k)]["validation"], scalers[str(k)])
        teX, tey, tem = take(X, y, meta, folds[str(k)]["test"], scalers[str(k)])
        model = make_model(class_weights(tr_y))
        model.fit(summarise_windows(trX), tr_y)

        def predict_probs(A):
            p = np.zeros((len(A), N_CLASSES))
            p[:, model.classes_] = model.predict_proba(summarise_windows(A))
            return p
        save_predictions(name, f"val_fold{k}", vam, predict_probs(vaX))
        save_predictions(name, f"fold{k}", tem, predict_probs(teX))
        thresholds.append(choose_threshold(pd.read_csv(PRED_DIR / name / f"val_fold{k}.csv")))

    fullX, fully, fullm = take(X, y, meta,
                               sorted(set(r for f in folds.values() for part in f.values() for r in part)),
                               scalers["full"])
    final = make_model(class_weights(fully))
    final.fit(summarise_windows(fullX), fully)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_DIR / f"{name}_final.joblib")

    def final_predict(A):
        p = np.zeros((len(A), N_CLASSES))
        p[:, final.classes_] = final.predict_proba(summarise_windows(A))
        return p
    return final_predict, thresholds


def train_deep(name, build_model, X, y, meta, folds, scalers):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for deep-model reverse experiments") from exc
    thresholds = []
    median_epochs = 30
    fold_epochs = []
    for k in range(N_FOLD):
        trX, tr_y, trm = take(X, y, meta, folds[str(k)]["train"], scalers[str(k)])
        vaX, vay, vam = take(X, y, meta, folds[str(k)]["validation"], scalers[str(k)])
        teX, tey, tem = take(X, y, meta, folds[str(k)]["test"], scalers[str(k)])
        seed_probs_val = []
        seed_probs_test = []
        seed_epochs = []
        cw = class_weights(tr_y)
        for seed in range(3):
            set_seed(SEED + seed)
            tf.keras.backend.clear_session()
            model = build_model(trX.shape[1:], N_CLASSES)
            model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                          loss="sparse_categorical_crossentropy")
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)]
            wv = np.array([cw[c] for c in vay], np.float32)
            hist = model.fit(trX, tr_y, validation_data=(vaX, vay, wv), epochs=100,
                             batch_size=256, class_weight=cw, callbacks=callbacks,
                             verbose=0, shuffle=True)
            seed_epochs.append(int(np.argmin(hist.history["val_loss"])) + 1)
            seed_probs_val.append(model.predict(vaX, verbose=0))
            seed_probs_test.append(model.predict(teX, verbose=0))
        best = int(np.median(seed_epochs))
        fold_epochs.append(best)
        vp = np.mean(seed_probs_val, axis=0)
        tp = np.mean(seed_probs_test, axis=0)
        valp = save_predictions(name, f"val_fold{k}", vam, vp)
        save_predictions(name, f"fold{k}", tem, tp)
        thresholds.append(choose_threshold(valp))

    median_epochs = int(np.median(fold_epochs)) if fold_epochs else 30
    all_utm = sorted(set(r for f in folds.values() for part in f.values() for r in part))
    fullX, fully, fullm = take(X, y, meta, all_utm, scalers["full"])
    seed_models = []
    cw = class_weights(fully)
    for seed in range(3):
        set_seed(SEED + seed)
        tf.keras.backend.clear_session()
        model = build_model(fullX.shape[1:], N_CLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                      loss="sparse_categorical_crossentropy")
        model.fit(fullX, fully, epochs=median_epochs, batch_size=256,
                  class_weight=cw, verbose=0, shuffle=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODEL_DIR / f"{name}_final_seed{seed}.keras")
        seed_models.append(model)

    def final_predict(A):
        return np.mean([m.predict(A, verbose=0) for m in seed_models], axis=0)
    return final_predict, thresholds


def main():
    set_seed()
    X, y, meta, folds, scalers = load_data()
    all_utm = sorted(set(r for f in folds["folds"].values() for part in f.values() for r in part))
    external = folds["external"]

    classical = []
    classical.append(("logistic_regression",
                      lambda cw: LogisticRegression(max_iter=3000, C=1.0, class_weight=cw)))
    classical.append(("random_forest",
                      lambda cw: RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                        class_weight=cw, n_jobs=-1,
                                                        random_state=42)))

    def build_lstm(input_shape, n_classes):
        import tensorflow as tf
        from tensorflow.keras import layers
        return tf.keras.Sequential([
            layers.Input(shape=input_shape), layers.LSTM(64), layers.Dropout(0.3),
            layers.Dense(64, activation="relu"), layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax")], name="lstm")

    def build_gru(input_shape, n_classes):
        import tensorflow as tf
        from tensorflow.keras import layers
        return tf.keras.Sequential([
            layers.Input(shape=input_shape), layers.GRU(64), layers.Dropout(0.3),
            layers.Dense(64, activation="relu"), layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax")], name="gru")

    def residual_block(x, dilation):
        import tensorflow as tf
        from tensorflow.keras import layers
        y = layers.Conv1D(64, 3, padding="causal", dilation_rate=dilation, activation="relu")(x)
        y = layers.Dropout(0.2)(y)
        y = layers.Conv1D(64, 3, padding="causal", dilation_rate=dilation, activation="relu")(y)
        if x.shape[-1] != 64:
            x = layers.Conv1D(64, 1)(x)
        return layers.Activation("relu")(layers.Add()([x, y]))

    def build_tcn(input_shape, n_classes):
        import tensorflow as tf
        from tensorflow.keras import layers
        inp = layers.Input(shape=input_shape)
        x = inp
        for d in [1, 2, 4, 8]:
            x = residual_block(x, d)
        x = layers.Cropping1D((input_shape[0] - 1, 0))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        return tf.keras.Model(inp, layers.Dense(n_classes, activation="softmax")(x), name="tcn")

    rows = []
    class_rows = []
    event_rows = []
    thresholds_rows = []
    for name, maker in classical:
        pred, thresholds = train_sklearn(name, maker, X, y, meta, folds["folds"], scalers)
        extX, exty, extm = take(X, y, meta, external, scalers["full"])
        extp = pred(extX)
        ext = save_predictions(name, "external", extm, extp)
        thresholds_rows.append({"model": name, "median_fa5_threshold": float(np.median(thresholds)), "thresholds": thresholds})
        for op in ("argmax", "fa5"):
            q = ext if op == "argmax" else apply_threshold(ext, float(np.median(thresholds)))
            s, pc, ev = evaluate_set(q)
            rows.append({"model": name, "op": op, **s})
            for c, v in pc.items():
                class_rows.append({"model": name, "op": op, "class": c, **v})
            if len(ev):
                ev["model"], ev["op"] = name, op
                event_rows.append(ev)

    for name, builder in (("lstm", build_lstm), ("gru", build_gru), ("tcn", build_tcn)):
        pred, thresholds = train_deep(name, builder, X, y, meta, folds["folds"], scalers)
        extX, exty, extm = take(X, y, meta, external, scalers["full"])
        extp = pred(extX)
        ext = save_predictions(name, "external", extm, extp)
        thresholds_rows.append({"model": name, "median_fa5_threshold": float(np.median(thresholds)), "thresholds": thresholds})
        for op in ("argmax", "fa5"):
            q = ext if op == "argmax" else apply_threshold(ext, float(np.median(thresholds)))
            s, pc, ev = evaluate_set(q)
            rows.append({"model": name, "op": op, **s})
            for c, v in pc.items():
                class_rows.append({"model": name, "op": op, "class": c, **v})
            if len(ev):
                ev["model"], ev["op"] = name, op
                event_rows.append(ev)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT_DIR / "per_class.csv", index=False)
    if event_rows:
        pd.concat(event_rows, ignore_index=True).to_csv(OUT_DIR / "events.csv", index=False)
    pd.DataFrame(thresholds_rows).to_csv(OUT_DIR / "thresholds.csv", index=False)

    lines = ["=" * 78, "REVERSE DOMAIN EVALUATION: UTM train/calibration -> VirtualBox external test", "=" * 78, ""]
    lines.append(f"UTM training runs: {len(all_utm)}; VirtualBox external runs: {len(external)}")
    lines.append("")
    for op in ("argmax", "fa5"):
        lines.append(("CALIBRATED fa5" if op == "fa5" else "ARGMAX") + " RESULTS")
        lines.append(f"{'model':20s} {'event detect':>14s} {'lead (s)':>10s} {'FA/hour':>10s} {'macro-F1':>10s}")
        for r in [x for x in rows if x["op"] == op]:
            lines.append(f"{r['model']:20s} {r['event_detection_rate']:14.3f} {r['mean_lead_time_sec']:10.2f} {r['false_alarms_per_hour']:10.1f} {r['macro_f1']:10.3f}")
        lines.append("")
    lines.append("Per-class metrics: results/reverse_domain/per_class.csv")
    lines.append("Event metrics: results/reverse_domain/events.csv")
    (OUT_DIR / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
