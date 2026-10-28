"""
Evaluate every model that has predictions in results/predictions/.

Run:  python models/evaluate.py
Out:  results/evaluation/report.txt            the numbers for the thesis
      results/evaluation/summary.csv           model x operating point
      results/evaluation/per_class.csv         model x operating point x class
      results/evaluation/events.csv            one row per onset event
      results/evaluation/thresholds.csv        chosen alarm threshold per fold
      results/evaluation/detection_vs_false_alarms.png
      results/evaluation/confusion_<model>.png
      results/evaluation/event_detection.png
      results/evaluation/lead_time.png

OPERATING POINTS
  argmax      the model warns whenever "some anomaly" is its most likely
              class. Every model then has a DIFFERENT false-alarm rate, so
              detection numbers are not comparable between models.
  fa<N>       (headline) an alarm threshold is chosen per fold on the
              VALIDATION runs so that the model raises at most N false
              alarms per hour of normal traffic; the test runs are then
              scored with that threshold. All models are compared at the
              same false-alarm budget - the standard way to compare
              early-warning systems.
              alarm score = 1 - P(normal); if score >= threshold the
              predicted class is the most likely anomaly class.

METRICS
  event detection  share of onsets with >= 1 CORRECT-class warning in
                   the 10 s before the onset
  lead time        seconds between the earliest correct warning and the
                   onset (max 10), over detected onsets
  false alarms/h   on normal-target windows, consecutive anomaly
                   predictions form one alarm, counted once per started
                   minute it stays on; per hour of normal time
  window F1        per-class and macro F1 over all test windows (secondary)

Every number is computed per fold, then reported as mean +- std.
"""

import _setup_path  # noqa: F401
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common.fold_data import CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
OUT_DIR = PROJECT_ROOT / "results" / "evaluation"
N = len(CLASS_NAMES)
ANOMALIES = CLASS_NAMES[1:]
PROB_COLS = [f"p_{c}" for c in CLASS_NAMES]

FA_TARGETS = [5]                 # false alarms per hour (headline: first)
HEADLINE = f"fa{FA_TARGETS[0]}"
MODEL_ORDER = ["logistic_regression", "random_forest", "lstm", "gru", "tcn"]


# ============================================================
# THRESHOLDING
# ============================================================

def apply_threshold(p, tau):
    """Return a copy of p whose 'pred' uses alarm threshold tau."""
    q = p.copy()
    probs = q[PROB_COLS].to_numpy()
    score = 1.0 - probs[:, 0]
    best_anomaly = probs[:, 1:].argmax(axis=1) + 1
    q["pred"] = np.where(score >= tau, best_anomaly, 0)
    return q


ALARM_COUNT_PERIOD_SEC = 60   # a continuous alarm counts once per minute


def false_alarm_stats(p):
    """False alarms on normal-target windows.

    Consecutive anomaly predictions form one alarm. An alarm that stays on
    counts once per started minute, so a model that alarms constantly is
    charged ~60 alarms/hour instead of looking like a single alarm.
    """
    n = p[p["target"] == 0].sort_values(["run_id", "elapsed_sec"])
    alarms = exc_alarms = 0
    for _, g in n.groupby("run_id"):
        a = (g["pred"] > 0).to_numpy()
        t = g["elapsed_sec"].to_numpy()
        exc = g["benign_excursion"].to_numpy() > 0
        cont = np.r_[False, a[:-1] & (np.diff(t) <= 1.5)]
        seg_start = a & ~cont
        seg_id = np.cumsum(seg_start)
        for sid in np.unique(seg_id[a]):
            m = a & (seg_id == sid)
            dur = t[m][-1] - t[m][0] + 1
            k = int(np.ceil(dur / ALARM_COUNT_PERIOD_SEC))
            alarms += k
            exc_alarms += k if exc[m][0] else 0
    hours = len(n) / 3600
    exc_hours = (n["benign_excursion"] > 0).sum() / 3600
    return {"false_alarms_per_hour": alarms / hours if hours else np.nan,
            "excursion_false_alarms_per_hour":
                exc_alarms / exc_hours if exc_hours else np.nan}


TAU_GRID = np.unique(np.concatenate([
    np.linspace(0.0, 0.9, 46), 1 - np.logspace(-1, -5, 60)]))


def choose_threshold(val, fa_target):
    """Smallest threshold whose validation false-alarm rate <= target."""
    for tau in TAU_GRID:
        if false_alarm_stats(apply_threshold(val, tau))[
                "false_alarms_per_hour"] <= fa_target:
            return float(tau)
    return float(TAU_GRID[-1])


# ============================================================
# METRICS FOR ONE SET OF PREDICTIONS
# ============================================================

def window_metrics(p):
    y, yhat = p["target"].astype(int), p["pred"].astype(int)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, yhat, labels=list(range(N)), zero_division=0)
    present = sup > 0
    _, _, af1, _ = precision_recall_fscore_support(
        y > 0, yhat > 0, average="binary", zero_division=0)
    per_class = {CLASS_NAMES[c]: {"precision": pr[c], "recall": rc[c],
                                  "f1": f1[c], "support": int(sup[c])}
                 for c in range(N)}
    return {"macro_f1": f1[present].mean(), "anomaly_f1": af1}, per_class


def event_table(p):
    ev = p[p["event_id"].fillna("") != ""]
    rows = []
    for eid, g in ev.groupby("event_id"):
        cls = int(g["target"].iloc[0])
        correct = g[g["pred"] == cls]
        rows.append({
            "event_id": eid, "class": CLASS_NAMES[cls], "windows": len(g),
            "detected": len(correct) > 0,
            "detected_any_anomaly": bool((g["pred"] > 0).any()),
            "lead_time_sec": correct["lead_time_sec"].max()
            if len(correct) else 0.0,
        })
    return pd.DataFrame(rows)


def evaluate_set(p):
    w, per_class = window_metrics(p)
    ev = event_table(p)
    fa = false_alarm_stats(p)
    det = ev.groupby("class")["detected"].mean() if len(ev) else pd.Series()
    lead = (ev[ev["detected"]].groupby("class")["lead_time_sec"].mean()
            if len(ev) else pd.Series())
    summary = {**w, **fa,
               "event_detection_rate":
                   ev["detected"].mean() if len(ev) else np.nan,
               "mean_lead_time_sec":
                   ev.loc[ev["detected"], "lead_time_sec"].mean()
                   if len(ev) and ev["detected"].any() else np.nan,
               "events": len(ev)}
    for c in ANOMALIES:
        per_class[c]["event_detection"] = det.get(c, np.nan)
        per_class[c]["lead_time_sec"] = lead.get(c, np.nan)
    return summary, per_class, ev


# ============================================================
# PLOTS
# ============================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(name, preds, op):
    plt = _plt()
    cm = confusion_matrix(preds["target"], preds["pred"], labels=list(range(N)))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    short = ["normal", "congest.", "high lat.", "pkt loss", "bw degr.",
             "queue", "burst"]
    ax.set_xticks(range(N), short, rotation=45, ha="right")
    ax.set_yticks(range(N), short)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center",
                    va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (what starts in the next 10 s)")
    ax.set_title(f"{name} ({op}): test windows, all folds, row-normalised")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def plot_per_class(per_class_df, column, ylabel, fname, ymax=None):
    plt = _plt()
    d = per_class_df[(per_class_df["class"].isin(ANOMALIES))
                     & (per_class_df["op"] == HEADLINE)]
    models = [m for m in MODEL_ORDER if m in d["model"].unique()]
    if not models:
        return
    x = np.arange(len(ANOMALIES))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(models):
        g = d[d["model"] == m].set_index("class").reindex(ANOMALIES)
        ax.bar(x + i * width, g[f"{column}_mean"], width,
               yerr=g[f"{column}_std"], capsize=2, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2, ANOMALIES,
                  rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"at <= {FA_TARGETS[0]} false alarms / hour (thresholds "
                 f"set on validation runs)")
    if ymax:
        ax.set_ylim(0, ymax)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)


def plot_tradeoff(curves):
    """Event detection vs false alarms/h, sweeping the threshold on the
    pooled test predictions of all folds."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in [m for m in MODEL_ORDER if m in curves] + \
             [m for m in curves if m not in MODEL_ORDER]:
        c = pd.DataFrame(curves[m]).sort_values("fa")
        ax.plot(c["fa"], c["det"], marker=".", ms=3, label=m)
    for t in FA_TARGETS:
        ax.axvline(t, color="grey", ls="--", lw=0.8)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false alarms per hour of normal traffic")
    ax.set_ylabel("onsets warned (correct class)")
    ax.set_title("Detection vs false alarms (test runs, all folds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "detection_vs_false_alarms.png", dpi=150)
    plt.close(fig)


def tradeoff_curve(test_parts):
    fa, det = [], []
    for tau in TAU_GRID:
        f_all, d_all = [], []
        for p in test_parts:
            q = apply_threshold(p, tau)
            f_all.append(false_alarm_stats(q)["false_alarms_per_hour"])
            ev = event_table(q)
            d_all.append(ev["detected"].mean())
        fa.append(np.mean(f_all))
        det.append(np.mean(d_all))
    return {"fa": fa, "det": det}


# ============================================================
# MAIN
# ============================================================

def fmt(m, s, nd=3):
    return f"{m:.{nd}f} +- {s:.{nd}f}" if pd.notna(s) else f"{m:.{nd}f}"


def summarise(rows):
    fs = pd.DataFrame(rows)
    out = {}
    for col in fs.columns:
        out[f"{col}_mean"] = fs[col].mean()
        out[f"{col}_std"] = fs[col].std()
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODEL_ORDER if (PRED_DIR / m).exists()]
    models += sorted(p.name for p in PRED_DIR.iterdir()
                     if p.is_dir() and p.name not in models)
    if not models:
        raise SystemExit("No predictions found. Train a model first.")

    summary_rows, class_rows, event_rows, thr_rows = [], [], [], []
    ext_rows, ext_class_rows, ext_event_rows, curves = [], [], [], {}

    for name in models:
        test_files = sorted((PRED_DIR / name).glob("fold*.csv"))
        if not test_files:
            continue
        tests = {f.stem: pd.read_csv(f) for f in test_files}
        vals = {k: PRED_DIR / name / f"val_{k}.csv" for k in tests}
        has_val = all(v.exists() for v in vals.values())
        if not has_val:
            print(f"[WARNING] {name}: no validation predictions - retrain "
                  f"it to get the calibrated ({HEADLINE}) results.")

        ops = {"argmax": {k: None for k in tests}}
        if has_val:
            for t in FA_TARGETS:
                ops[f"fa{t}"] = {}
                for k in tests:
                    tau = choose_threshold(pd.read_csv(vals[k]), t)
                    ops[f"fa{t}"][k] = tau
                    thr_rows.append({"model": name, "op": f"fa{t}",
                                     "fold": k, "threshold": tau})

        for op, taus in ops.items():
            fold_sum, fold_cls, all_preds = [], [], []
            for k, p in tests.items():
                q = p if taus[k] is None else apply_threshold(p, taus[k])
                s, pc, ev = evaluate_set(q)
                fold_sum.append(s)
                for c, v in pc.items():
                    fold_cls.append({"class": c, **v})
                ev["model"], ev["op"], ev["fold"] = name, op, k
                event_rows.append(ev)
                all_preds.append(q)
            summary_rows.append({"model": name, "op": op,
                                 "folds": len(fold_sum),
                                 **summarise(fold_sum)})
            fc = pd.DataFrame(fold_cls)
            agg = fc.groupby("class").agg(["mean", "std"])
            agg.columns = [f"{a}_{b}" for a, b in agg.columns]
            agg = agg.reset_index()
            agg.insert(0, "op", op)
            agg.insert(0, "model", name)
            class_rows.append(agg)
            if op == (HEADLINE if has_val else "argmax"):
                plot_confusion(name, pd.concat(all_preds), op)

        curves[name] = tradeoff_curve(list(tests.values()))

        ext = PRED_DIR / name / "external.csv"
        if ext.exists():
            p = pd.read_csv(ext)
            for op, taus in ops.items():
                tau = None if op == "argmax" else float(
                    np.median(list(taus.values())))
                q = p if tau is None else apply_threshold(p, tau)
                s, ext_pc, ext_ev = evaluate_set(q)
                ext_rows.append({"model": name, "op": op, **s})
                for c, v in ext_pc.items():
                    ext_class_rows.append({"model": name, "op": op,
                                           "class": c, **v})
                ext_ev["model"], ext_ev["op"] = name, op
                ext_event_rows.append(ext_ev)

    summary = pd.DataFrame(summary_rows)
    per_class = pd.concat(class_rows, ignore_index=True)
    events = pd.concat(event_rows, ignore_index=True)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    per_class.to_csv(OUT_DIR / "per_class.csv", index=False)
    events.to_csv(OUT_DIR / "events.csv", index=False)
    if ext_class_rows:
        pd.DataFrame(ext_class_rows).to_csv(
            OUT_DIR / "external_per_class.csv", index=False)
    if ext_event_rows:
        pd.concat(ext_event_rows, ignore_index=True).to_csv(
            OUT_DIR / "external_events.csv", index=False)
    if thr_rows:
        pd.DataFrame(thr_rows).to_csv(OUT_DIR / "thresholds.csv", index=False)
    plot_per_class(per_class, "event_detection",
                   "onsets warned (correct class)", "event_detection.png", 1.05)
    plot_per_class(per_class, "lead_time_sec",
                   "mean lead time of detected onsets (s)", "lead_time.png",
                   10.5)
    plot_tradeoff(curves)

    # ---------------- report ----------------
    L = ["=" * 78, "EVALUATION REPORT (mean +- std over folds)", "=" * 78, ""]

    def overall(op, title):
        s = summary[summary["op"] == op]
        if s.empty:
            return
        L.extend([title, "-" * 78])
        L.append(f"{'model':20s} {'event detect':>15s} {'lead (s)':>13s} "
                 f"{'FA / hour':>13s} {'macro-F1':>15s}")
        for _, r in s.iterrows():
            L.append(
                f"{r['model']:20s} "
                f"{fmt(r['event_detection_rate_mean'], r['event_detection_rate_std']):>15s} "
                f"{fmt(r['mean_lead_time_sec_mean'], r['mean_lead_time_sec_std'], 2):>13s} "
                f"{fmt(r['false_alarms_per_hour_mean'], r['false_alarms_per_hour_std'], 1):>13s} "
                f"{fmt(r['macro_f1_mean'], r['macro_f1_std']):>15s}")
        L.append("")

    for t in FA_TARGETS:
        overall(f"fa{t}", f"A. CALIBRATED: threshold set on validation runs "
                          f"for <= {t} false alarms / hour  (HEADLINE)")
    overall("argmax", "B. DEFAULT: warn when an anomaly class is most likely "
                      "(models not comparable)")
    L.append("event detect = share of onsets with >=1 correct warning in the "
             "10 s before")
    L.append("lead (s)     = mean seconds of warning for detected onsets "
             "(max 10)")
    L.append("FA / hour    = separate false alarms per hour of normal traffic "
             "(test runs)")
    L.append("")

    op_for_classes = HEADLINE if (summary["op"] == HEADLINE).any() else "argmax"
    for name in summary["model"].unique():
        g = per_class[(per_class["model"] == name)
                      & (per_class["op"] == op_for_classes)].set_index("class")
        if g.empty:
            continue
        L += [f"PER CLASS - {name} ({op_for_classes})", "-" * 78]
        L.append(f"{'class':24s} {'event detect':>15s} {'lead (s)':>9s} "
                 f"{'window F1':>15s} {'recall':>7s} {'precision':>9s}")
        for c in CLASS_NAMES:
            r = g.loc[c]
            det = (fmt(r["event_detection_mean"], r["event_detection_std"])
                   if c != "normal" else "-")
            lead = (f"{r['lead_time_sec_mean']:.2f}"
                    if c != "normal" and pd.notna(r["lead_time_sec_mean"])
                    else "-")
            L.append(f"{c:24s} {det:>15s} {lead:>9s} "
                     f"{fmt(r['f1_mean'], r['f1_std']):>15s} "
                     f"{r['recall_mean']:7.3f} {r['precision_mean']:9.3f}")
        L.append("")

    if thr_rows:
        th = pd.DataFrame(thr_rows)
        L += ["CHOSEN ALARM THRESHOLDS (1 - P(normal)), per fold", "-" * 78]
        for (m, op), g in th.groupby(["model", "op"], sort=False):
            L.append(f"{m:20s} {op:6s} " + "  ".join(
                f"{v:.4f}" for v in g["threshold"]))
        L.append("")

    if ext_rows:
        L += ["EXTERNAL (other machine; final model trained on all "
              "main-machine runs)", "-" * 78]
        for r in ext_rows:
            L.append(f"{r['model']:20s} {r['op']:6s} event detect "
                     f"{r['event_detection_rate']:.3f}  lead "
                     f"{r['mean_lead_time_sec']:.2f} s  FA/h "
                     f"{r['false_alarms_per_hour']:.1f}  macro-F1 "
                     f"{r['macro_f1']:.3f}  ({r['events']} events)")
        L.append("Note: external runs contain only the classes collected on "
                 "that machine.")
        if ext_class_rows:
            ext_pc = pd.DataFrame(ext_class_rows)
            ext_op = HEADLINE if (ext_pc["op"] == HEADLINE).any() else "argmax"
            L += ["", "EXTERNAL PER CLASS (headline setting)", "-" * 78]
            L.append(f"{'class':24s} {'event detect':>15s} {'lead (s)':>9s} "
                     f"{'window F1':>15s} {'recall':>7s} {'precision':>9s}")
            for name in ext_pc["model"].unique():
                g = ext_pc[(ext_pc["model"] == name)
                           & (ext_pc["op"] == ext_op)].set_index("class")
                if g.empty:
                    continue
                L.append(f"[{name}]")
                for c in CLASS_NAMES:
                    if c not in g.index:
                        continue
                    r = g.loc[c]
                    det = (f"{r['event_detection']:.3f}"
                           if c != "normal" and pd.notna(r["event_detection"])
                           else "-")
                    lead = (f"{r['lead_time_sec']:.2f}"
                            if c != "normal" and pd.notna(r["lead_time_sec"])
                            else "-")
                    L.append(f"{c:24s} {det:>15s} {lead:>9s} "
                             f"{r['f1']:15.3f} {r['recall']:7.3f} "
                             f"{r['precision']:9.3f}")
            L.append("Interpretation: zero or near-zero bandwidth-degradation "
                     "lead time is consistent with capacity loss becoming "
                     "observable only when throughput is constrained.")
    report = "\n".join(L)
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
