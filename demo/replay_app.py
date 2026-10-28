"""
Replay demo: early warning and detection of network performance anomalies.

Plays back one recorded run second by second, as if it were live:
  * PREDICTION  the model's P(each anomaly starts within 10 s), every second
  * WARNING     raised when the calibrated alarm threshold is crossed
  * DETECTION   rule-based detector (demo/detector.py): what is wrong NOW,
                from the measurements only
  * ground truth (shaded) to check both, and a live score of the run

Data sources (both honest):
  New recording   recorded AFTER training, scored by demo/predict_new_run.py
                  with the final models, which never saw it
  CV test runs    every run is a TEST run in exactly one cross-validation
                  fold; the predictions come from that fold's model

Run from the project root:
    streamlit run demo/replay_app.py
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from detector import detect, MIN_EVIDENCE

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "network_telemetry_clean.csv"
PRED_DIR = ROOT / "results" / "predictions"
THRESH = ROOT / "results" / "evaluation" / "thresholds.csv"
DEMO_RUNS = ROOT / "results" / "demo_runs"

SRC_NEW = "New recording"
SRC_CV = "Cross-validation runs"

CLASSES = ["normal", "congestion", "high_latency", "packet_loss",
           "bandwidth_degradation", "queue_buildup", "traffic_burst"]
NICE = {"normal": "Normal", "congestion": "Congestion",
        "high_latency": "High latency", "packet_loss": "Packet loss",
        "bandwidth_degradation": "Bandwidth degradation",
        "queue_buildup": "Queue build-up", "traffic_burst": "Traffic burst",
        "identifying": "Identifying type"}
COLORS = {"congestion": "#E4572E", "high_latency": "#7B61FF",
          "packet_loss": "#8D6E63", "bandwidth_degradation": "#D6409F",
          "queue_buildup": "#E9A23B", "traffic_burst": "#1597B5"}
MODEL_ORDER = ["lstm", "gru", "tcn", "random_forest", "logistic_regression"]
MODEL_NICE = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN",
              "random_forest": "Random Forest",
              "logistic_regression": "Logistic Regression"}
SCENARIO_ORDER = ["congestion", "high_latency", "packet_loss", "queue_buildup",
                  "traffic_burst", "bandwidth_degradation", "normal"]

WINDOW_SEC = 180            # seconds of history shown
NOMINAL_MBIT = 50.0
BUFFER_MS = 400.0
RECOVER_SEC = 15            # "recovering" status after a detection ends

INK, MUTED, LINE = "#0F172A", "#64748B", "#E6E8EC"
STATE = {   # status kind -> (accent, soft background, label)
    "normal":     ("#13895F", "#EAF6F0", "NORMAL"),
    "warning":    ("#C77C02", "#FDF4E3", "EARLY WARNING"),
    "anomaly":    ("#C8364F", "#FBEBEE", "ANOMALY DETECTED"),
    "recovering": ("#476582", "#EDF1F5", "RECOVERING"),
    "idle":       ("#7A8494", "#F2F3F5", "NO PREDICTION"),
}
VERDICT = {  # code -> (label, colour)
    "correct": ("Correct", "#13895F"),
    "wrong": ("Wrong type", "#D9480F"),
    "early": ("Too early", "#476582"),
    "recovery": ("False · recovery", "#C8364F"),
    "excursion": ("False · excursion", "#C8364F"),
    "false": ("False", "#C8364F"),
    "pending": ("Pending", "#7A8494"),
    "yes": ("Warned", "#13895F"),
    "changed": ("Type changed", "#C77C02"),
    "after": ("After-effect", "#7A8494"),
    "identifying": ("Identifying…", "#7A8494"),
    "no": ("Missed", "#C8364F"),
}


# ============================================================
# STYLE
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input, label {
  font-family: 'Inter', system-ui, sans-serif !important; }
.stApp { background: #F6F7F9; color: #0F172A; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"], #MainMenu, footer, [data-testid="stDecoration"] {
  display: none !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem;
  max-width: 1400px; }
section[data-testid="stSidebar"] { background: #FFFFFF;
  border-right: 1px solid #E6E8EC; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

.app-head { display:flex; align-items:flex-end; justify-content:space-between;
  margin-bottom: .9rem; }
.app-title { font-size: 1.45rem; font-weight: 700; letter-spacing: -.02em;
  color:#0F172A; margin:0; }
.app-sub { color:#64748B; font-size:.86rem; margin-top:.15rem; }
.tag { display:inline-block; padding:.18rem .55rem; border-radius:999px;
  font-size:.72rem; font-weight:600; letter-spacing:.02em;
  background:#EEF1F5; color:#334155; margin-left:.35rem; }

.card { background:#FFFFFF; border:1px solid #E6E8EC; border-radius:14px;
  padding: 1rem 1.15rem; }
.status { display:flex; align-items:center; gap:1.1rem;
  border-left: 6px solid var(--accent); background: var(--soft); }
.status .pill { font-size:.7rem; font-weight:700; letter-spacing:.08em;
  color: var(--accent); }
.status .head { font-size:1.15rem; font-weight:600; color:#0F172A;
  margin-top:.1rem; }
.status .detail { color:#475569; font-size:.84rem; margin-top:.15rem; }
.status .clock { margin-left:auto; text-align:right; }
.status .clock .t { font-family:'JetBrains Mono', monospace; font-size:1.5rem;
  color:#0F172A; }
.status .clock .of { color:#64748B; font-size:.75rem; }
.truth { color:#64748B; font-size:.78rem; margin:.45rem 0 0 .2rem; }

.kpis { display:grid; grid-template-columns: repeat(2, 1fr); gap:.7rem; }
.kpi { background:#FFFFFF; border:1px solid #E6E8EC; border-radius:12px;
  padding:.75rem .9rem; }
.kpi .v { font-size:1.5rem; font-weight:700; color:#0F172A;
  letter-spacing:-.02em; }
.kpi .v small { font-size:.9rem; color:#94A3B8; font-weight:500; }
.kpi .l { color:#64748B; font-size:.74rem; margin-top:.1rem; }
.kpi.bad .v { color:#C8364F; }

.sec { font-size:.72rem; font-weight:700; letter-spacing:.08em;
  color:#64748B; text-transform:uppercase; margin: 1.1rem 0 .45rem .1rem; }

table.res { width:100%; border-collapse:collapse; font-size:.85rem;
  background:#FFFFFF; border:1px solid #E6E8EC; border-radius:12px;
  overflow:hidden; }
table.res th { text-align:left; font-weight:600; color:#64748B;
  font-size:.72rem; letter-spacing:.04em; text-transform:uppercase;
  padding:.6rem .8rem; border-bottom:1px solid #E6E8EC; background:#FAFBFC; }
table.res td { padding:.55rem .8rem; border-bottom:1px solid #F0F1F4;
  color:#0F172A; }
table.res tr:last-child td { border-bottom:none; }
.chip { display:inline-block; padding:.12rem .55rem; border-radius:999px;
  font-size:.74rem; font-weight:600; color:#FFFFFF; }
.dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%;
  margin-right:.45rem; vertical-align:middle; }
.empty { color:#94A3B8; font-size:.85rem; padding:.6rem .2rem; }

.stButton > button { border-radius:10px; border:1px solid #E6E8EC;
  background:#FFFFFF; color:#0F172A; font-weight:500; padding:.35rem .5rem;
  white-space:nowrap; min-height:2.4rem; }
.stButton > button p { font-size:.86rem; white-space:nowrap; }
.stButton > button:hover { border-color:#94A3B8; color:#0F172A; }
.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; }
.stTabs [data-baseweb="tab"] { font-weight:600; color:#64748B; }
div[data-testid="stPlotlyChart"] { background:#FFFFFF; border:1px solid #E6E8EC;
  border-radius:14px; padding:.3rem .4rem; }
</style>
"""

GUIDE_MD = """
**Two parts work together**

- **Prediction** — while the network is normal, the model gives the
  probability that each anomaly type *starts within the next 10 s*. Above
  the calibrated threshold it raises an **early warning**.
- **Detection** — rules on the measurements (RTT, loss, drops, queue,
  delivered load) report that an anomaly *is happening now* and its type.
  Neither uses the ground truth.

**On the charts**

- Light red band: the anomaly is actually happening (ground truth);
  dashed red line: its onset.
- Light grey band: a harmless excursion (small bump in load, delay or loss).
- Coloured strip at the top of the RTT chart: what the detector reports.
- Triangles on the probability chart: warnings, coloured by predicted type.
- Dotted line: the alarm threshold.

**Judging warnings** (each one is judged 10 s after it ends)

- *Correct*: the warned type started within 10 s, and no other type was
  warned on the way.
- *Type changed*: the right type was warned, but another type was shown
  first. *Wrong type*: only other types were warned.
- *Too early*: the anomaly was already building up, but its onset was more
  than 10 s away.
- *False · recovery / excursion / False*: nothing started; raised while the
  network was recovering, during a harmless excursion, or in normal traffic.

**Judging detections** (Detections tab): the detector shows *Identifying
type* until it has 4 seconds of consistent evidence. *Correct* means the
one type it showed was right; *Type changed* means it showed another type
first; *After-effect* is a detection in the recovery right after an
anomaly; *False* is a detection with no anomaly at all.
"""


# ============================================================
# DATA
# ============================================================

def add_display_columns(df):
    df["queue_delay_ms"] = df["qdisc_backlog_bytes"] * 8 / (NOMINAL_MBIT * 1e6) * 1000
    df["offered_mbit"] = df["ingress_rx_bytes_per_sec"] * 8 / 1e6
    df["delivered_mbit"] = df["tx_bytes_per_sec"] * 8 / 1e6
    pps = df["ingress_rx_packets_per_sec"].where(
        df["ingress_rx_packets_per_sec"] > 0)
    df["router_drop_pct"] = (100 * (df["tbf_dropped_per_sec"].fillna(0)
                                    + df["netem_dropped_per_sec"].fillna(0))
                             / pps).fillna(0).clip(0, 100)
    return df


@st.cache_data
def load_signals():
    cols = ["run_id", "scenario", "elapsed_sec", "latency_ms",
            "baseline_latency_ms", "qdisc_backlog_bytes",
            "ingress_rx_bytes_per_sec", "tx_bytes_per_sec", "ping_loss_pct",
            "ingress_rx_packets_per_sec", "tbf_dropped_per_sec",
            "netem_dropped_per_sec", "label", "anomaly_type",
            "phase", "episode_id", "benign_excursion"]
    return add_display_columns(pd.read_csv(CLEAN, usecols=cols))


@st.cache_data
def load_predictions(model):
    parts = []
    for f in sorted((PRED_DIR / model).glob("fold*.csv")):
        p = pd.read_csv(f)
        p["fold"] = f.stem
        parts.append(p)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@st.cache_data
def load_thresholds():
    if not THRESH.exists():
        return pd.DataFrame(columns=["model", "op", "fold", "threshold"])
    return pd.read_csv(THRESH)


@st.cache_data
def load_new_signals(run_dir):
    return add_display_columns(pd.read_csv(Path(run_dir) / "signals.csv"))


@st.cache_data
def load_new_predictions(run_dir, model):
    return pd.read_csv(Path(run_dir) / f"predictions_{model}.csv")


@st.cache_data
def load_new_info(run_dir):
    return json.loads((Path(run_dir) / "info.json").read_text())


@st.cache_data
def detect_run(sig):
    return detect(sig.reset_index(drop=True))


def new_runs():
    if not DEMO_RUNS.exists():
        return []
    return sorted(p for p in DEMO_RUNS.iterdir() if (p / "info.json").exists())


def available_models():
    if not PRED_DIR.exists():
        return []
    found = [m for m in MODEL_ORDER if (PRED_DIR / m).exists()]
    return found + sorted(p.name for p in PRED_DIR.iterdir()
                          if p.is_dir() and p.name not in found)


# ============================================================
# HELPERS
# ============================================================

def spans(t, mask):
    out, start = [], None
    for i in range(len(mask)):
        if mask[i] and start is None:
            start = t[i]
        if not mask[i] and start is not None:
            out.append((start, t[i]))
            start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def onsets_of(sig):
    y = sig["label"].to_numpy()
    idx = np.flatnonzero((y[1:] == 1) & (y[:-1] == 0)) + 1
    return sig["elapsed_sec"].to_numpy()[idx]


def detection_spans(sig):
    t, d, k = (sig["elapsed_sec"].to_numpy(), sig["detected"].to_numpy(),
               sig["det_type"].to_numpy())
    out, start = [], None
    for i in range(len(d)):
        if d[i] and start is None:
            start = i
        if start is not None and (not d[i] or k[i] != k[start]):
            out.append((t[start], t[i], k[start]))
            start = i if d[i] else None
    if start is not None:
        out.append((t[start], t[-1], k[start]))
    return out


def recovering_mask(sig):
    t = sig["elapsed_sec"].to_numpy()
    d = sig["detected"].to_numpy() == 1
    out, last = np.zeros(len(t), bool), -1e9
    for i in range(len(t)):
        if d[i]:
            last = t[i]
        out[i] = t[i] - last <= RECOVER_SEC
    return out


def with_gaps(p, max_step=1.5):
    """Blank rows where seconds are missing, so lines break there."""
    if len(p) < 2:
        return p
    t = p["elapsed_sec"].to_numpy()
    breaks = np.flatnonzero(np.diff(t) > max_step)
    if not len(breaks):
        return p
    return (pd.concat([p, pd.DataFrame({"elapsed_sec": t[breaks] + 0.5})],
                      ignore_index=True)
            .sort_values("elapsed_sec").reset_index(drop=True))


def alarm_column(pred, tau):
    probs = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
    score = 1 - probs[:, 0]
    best = probs[:, 1:].argmax(axis=1) + 1
    if tau is None:
        return np.where(probs.argmax(axis=1) > 0, best, 0), score
    return np.where(score >= tau, best, 0), score


# ============================================================
# STATUS, SCORE
# ============================================================

def status(sig, pred, t_now):
    """(kind, headline, detail) from measurements + model, no ground truth."""
    upto = sig[sig["elapsed_sec"] <= t_now]
    if not len(upto):
        return ("idle", "Run starting",
                "The model needs 30 s of history before its first prediction")
    row = upto.iloc[-1]
    if row["detected"] == 1:
        why = row["reason"] if row["reason"] != "holding" else "settling"
        if row["det_type"] == "identifying":
            return ("anomaly", "Anomaly detected · identifying type",
                    f"Waiting for {MIN_EVIDENCE} s of consistent evidence · "
                    f"signals: {why}")
        return ("anomaly", NICE[row["det_type"]],
                f"Detected from measurements · signals: {why}")
    p = pred[pred["elapsed_sec"] <= t_now]
    fresh = len(p) and abs(p.iloc[-1]["elapsed_sec"] - t_now) < 1.5
    if fresh and p.iloc[-1]["alarm"] > 0:
        last = p.iloc[-1]
        c = CLASSES[int(last["alarm"])]
        return ("warning", f"{NICE[c]} expected within 10 s",
                f"Model probability {last[f'p_{c}']:.2f}")
    if upto["recovering"].iloc[-1]:
        return ("recovering", "Network recovering",
                "An anomaly ended less than 15 s ago")
    if not fresh:
        return ("idle", "No prediction this second",
                "Run start, recording gap, or outside the scored range")
    extra = " · harmless excursion in progress" \
        if row["benign_excursion"] > 0 else ""
    return ("normal", "No anomaly expected in the next 10 s",
            "Model below the alarm threshold" + extra)


def truth_text(sig, t_now):
    upto = sig[sig["elapsed_sec"] <= t_now]
    if not len(upto):
        return "Normal"
    row = upto.iloc[-1]
    if row["label"] == 1:
        return f"{NICE[row['anomaly_type']]} in progress"
    if row["phase"] == "cooldown":
        return "Normal (cool-down after an anomaly)"
    nxt = onsets_of(sig)
    nxt = nxt[nxt > t_now]
    if len(nxt) and nxt[0] - t_now <= 10:
        typ = sig.loc[sig["elapsed_sec"] == nxt[0], "anomaly_type"].iloc[0]
        return f"Normal · {NICE[typ]} starts in {nxt[0] - t_now:.0f} s"
    return "Normal"


def type_sequence(values):
    """Distinct types in the order they were shown (no 'identifying')."""
    out = []
    for v in values:
        if v not in ("identifying", "normal") and (not out or out[-1] != v):
            out.append(v)
    return out


def detection_rows(sig, t_now):
    """Every detection the banner showed so far, and whether it was right."""
    t = sig["elapsed_sec"].to_numpy()
    d = (sig["detected"].to_numpy() == 1) & (t <= t_now)
    y = sig["label"].to_numpy() == 1
    rows, i, n = [], 0, len(t)
    while i < n:
        if not d[i]:
            i += 1
            continue
        j = i
        while j < n and d[j]:
            j += 1
        seq = type_sequence(sig["det_type"].iloc[i:j])
        truth_rows = sig.iloc[i:j][y[i:j]]
        if len(truth_rows):
            truth = truth_rows["anomaly_type"].iloc[0]
            if not seq:
                v = "identifying"
            elif seq == [truth]:
                v = "correct"
            elif truth in seq:
                v = "changed"
            else:
                v = "wrong"
        else:
            truth = None
            prev_end = sig.iloc[max(i - 1, 0)]
            v = "after" if (prev_end["phase"] == "cooldown"
                            or sig["phase"].iloc[i] == "cooldown") else "false"
            if not seq and v == "false":
                v = "false"
        rows.append({"a": t[i], "b": t[j - 1], "seq": seq, "truth": truth,
                     "v": v})
        i = j
    return rows


def onset_rows(sig, pred, t_now, dets):
    rows = []
    for on in onsets_of(sig):
        if on > t_now:
            break
        typ = sig.loc[sig["elapsed_sec"] == on, "anomaly_type"].iloc[0]
        w = pred[(pred["elapsed_sec"] >= on - 10.5) & (pred["elapsed_sec"] < on)]
        hit = w[w["alarm"] == CLASSES.index(typ)]
        other = w[(w["alarm"] > 0) & (w["alarm"] != CLASSES.index(typ))]
        if len(hit):
            warn = "changed" if len(other) else "yes"
        else:
            warn = "wrong" if len(other) else "no"
        seq = type_sequence(CLASSES[int(a)] for a in w["alarm"] if a > 0)
        det = next((r for r in dets if r["a"] <= on + 5 and r["b"] >= on - 1),
                   None)
        rows.append({
            "t": on, "type": typ, "warn": warn, "warn_seq": seq,
            "lead": on - hit["elapsed_sec"].min() if len(hit) else None,
            "det_after": det["a"] - on if det else None,
            "det_seq": det["seq"] if det else None,
            "det_v": det["v"] if det else None})
    return rows


def alarm_rows(sig, pred, t_now):
    p = pred[(pred["elapsed_sec"] <= t_now) & (pred["alarm"] > 0)]
    if not len(p):
        return []
    t = p["elapsed_sec"].to_numpy()
    gid = np.cumsum(np.r_[True, np.diff(t) > 1.5])
    on_t = onsets_of(sig)
    rows = []
    for g in np.unique(gid):
        e = p[gid == g]
        a, b = e["elapsed_sec"].iloc[0], e["elapsed_sec"].iloc[-1]
        seq = type_sequence(CLASSES[int(k)] for k in e["alarm"])
        hits = on_t[(on_t > a) & (on_t <= b + 10.5)]
        if len(hits):
            typ = sig.loc[sig["elapsed_sec"] == hits[0], "anomaly_type"].iloc[0]
            v = ("correct" if seq == [typ] else
                 "changed" if typ in seq else "wrong")
        elif t_now < b + 10:
            v = "pending"
        else:
            r = sig[sig["elapsed_sec"] <= a].iloc[-1]
            if r["episode_id"] > 0:
                v = "early"
            elif r["phase"] == "cooldown" or bool(r["recovering"]):
                v = "recovery"
            elif r["benign_excursion"] > 0:
                v = "excursion"
            else:
                v = "false"
        rows.append({"a": a, "b": b, "seq": seq, "v": v})
    return rows


# ============================================================
# HTML PIECES
# ============================================================

def chip(code):
    label, color = VERDICT[code]
    return f'<span class="chip" style="background:{color}">{label}</span>'


def type_cell(c):
    return (f'<span class="dot" style="background:{COLORS.get(c, "#999")}">'
            f'</span>{NICE[c]}')


def status_html(kind, head, detail, t, t_max, truth):
    accent, soft, label = STATE[kind]
    return f"""
<div class="card status" style="--accent:{accent};--soft:{soft}">
  <div>
    <div class="pill">{label}</div>
    <div class="head">{head}</div>
    <div class="detail">{detail}</div>
  </div>
  <div class="clock"><div class="t">{t} s</div>
    <div class="of">of {t_max} s</div></div>
</div>
<div class="truth">Ground truth&nbsp;·&nbsp;{truth}</div>"""


def seq_cell(seq):
    if not seq:
        return '<span style="color:#94A3B8">—</span>'
    return ' <span style="color:#94A3B8">→</span> '.join(type_cell(k)
                                                         for k in seq)


def kpi_html(ons, als, dets):
    n = len(ons)
    warned = sum(r["warn"] == "yes" for r in ons)
    changed_w = sum(r["warn"] == "changed" for r in ons)
    detected = sum(r["det_v"] == "correct" for r in ons)
    false = sum(r["v"] in ("false", "recovery", "excursion") for r in als)
    false_d = sum(r["v"] == "false" for r in dets)
    issues = (sum(r["v"] in ("wrong", "changed", "early") for r in als)
              + sum(r["v"] in ("wrong", "changed") for r in dets))
    tile = ('<div class="kpi{cls}"><div class="v">{v}</div>'
            '<div class="l">{l}</div></div>')
    return '<div class="kpis">' + "".join([
        tile.format(cls="", v=f"{warned}<small> / {n}</small>",
                    l="Onsets warned in time, right type"
                      + (f" (+{changed_w} after a type change)"
                         if changed_w else "")),
        tile.format(cls="", v=f"{detected}<small> / {n}</small>",
                    l="Onsets detected, right type throughout"),
        tile.format(cls=" bad" if false + false_d else "",
                    v=false + false_d,
                    l=f"False alarms ({false} warnings, {false_d} detections)"),
        tile.format(cls=" bad" if issues else "", v=issues,
                    l="Wrong, changed or too-early types"),
    ]) + "</div>"


def onset_table_html(rows):
    if not rows:
        return '<div class="empty">No onset reached yet.</div>'
    body = ""
    for r in rows:
        warn = chip(r["warn"])
        if r["warn"] in ("wrong", "changed"):
            warn += " " + seq_cell(r["warn_seq"])
        lead = "" if r["lead"] is None else f"{r['lead']:.0f} s before"
        if r["det_seq"] is None:
            det = '<span style="color:#94A3B8">—</span>'
        else:
            det = chip(r["det_v"]) + " " + seq_cell(r["det_seq"]) + (
                f' <span style="color:#94A3B8">'
                f'{r["det_after"]:+.0f} s</span>')
        body += (f"<tr><td>{r['t']:.0f} s</td><td>{type_cell(r['type'])}</td>"
                 f"<td>{warn}</td><td>{lead}</td><td>{det}</td></tr>")
    return ('<table class="res"><tr><th>Onset</th><th>Anomaly</th>'
            '<th>Early warning</th><th>Lead time</th>'
            '<th>Detection (types shown, start vs onset)</th></tr>'
            + body + "</table>")


def when(r):
    a, b = round(r["a"]), round(r["b"])
    return f"{a} s" if a == b else f"{a}–{b} s"


def alarm_table_html(rows):
    if not rows:
        return '<div class="empty">No warning raised yet.</div>'
    body = "".join(
        f"<tr><td>{when(r)}</td><td>{seq_cell(r['seq'])}"
        f"</td><td>{chip(r['v'])}</td></tr>" for r in reversed(rows))
    return ('<table class="res"><tr><th>When</th><th>Warned type(s)</th>'
            '<th>Verdict</th></tr>' + body + "</table>")


def detection_table_html(rows):
    if not rows:
        return '<div class="empty">No detection yet.</div>'
    body = "".join(
        f"<tr><td>{when(r)}</td><td>{seq_cell(r['seq'])}</td>"
        f"<td>{type_cell(r['truth']) if r['truth'] else '—'}</td>"
        f"<td>{chip(r['v'])}</td></tr>" for r in reversed(rows))
    return ('<table class="res"><tr><th>When</th><th>Type(s) shown</th>'
            '<th>Ground truth</th><th>Verdict</th></tr>' + body + "</table>")


# ============================================================
# FIGURES
# ============================================================

FONT = dict(family="Inter, system-ui, sans-serif", size=12, color="#475569")


def build_figure(sig, pred, t_now, t_from, emphasis, tau):
    s = sig[(sig["elapsed_sec"] >= t_from) & (sig["elapsed_sec"] <= t_now)]
    p = pred[(pred["elapsed_sec"] >= t_from) & (pred["elapsed_sec"] <= t_now)]
    pl = with_gaps(p)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.075,
        row_heights=[0.3, 0.3, 0.4],
        specs=[[{}], [{"secondary_y": True}], [{}]],
        subplot_titles=("Round-trip time  ·  ms",
                        "Router queue delay  ·  ms        offered load  ·  Mbit/s",
                        "Probability that an anomaly starts within 10 s"))
    for ann in fig.layout.annotations:
        ann.update(x=0, xanchor="left", font=dict(size=12, color="#334155"))

    t_all = sig["elapsed_sec"].to_numpy()
    for a, b in spans(t_all, sig["label"].to_numpy() == 1):
        if b >= t_from and a <= t_now:
            for r in (1, 2, 3):
                fig.add_vrect(x0=max(a, t_from), x1=min(b, t_now),
                              fillcolor="#C8364F", opacity=0.07, line_width=0,
                              exclude_empty_subplots=False, row=r, col=1)
    for a, b in spans(t_all, sig["benign_excursion"].to_numpy() > 0):
        if b >= t_from and a <= t_now:
            for r in (1, 2, 3):
                fig.add_vrect(x0=max(a, t_from), x1=min(b, t_now),
                              fillcolor="#64748B", opacity=0.07, line_width=0,
                              exclude_empty_subplots=False, row=r, col=1)
    for a, b, k in detection_spans(sig):
        if b >= t_from and a <= t_now:
            fig.add_vrect(x0=max(a, t_from), x1=min(b + 1, t_now),
                          y0=0.93, y1=1.0, fillcolor=COLORS.get(k, "#999"),
                          opacity=0.9, line_width=0,
                          exclude_empty_subplots=False, row=1, col=1)
    for on in onsets_of(sig):
        if t_from <= on <= t_now:
            for r in (1, 2, 3):
                fig.add_vline(x=on, line_color="#C8364F", line_dash="dash",
                              line_width=1, exclude_empty_subplots=False,
                              row=r, col=1)

    fig.add_trace(go.Scatter(x=s["elapsed_sec"], y=s["baseline_latency_ms"],
                             line=dict(color="#94A3B8", dash="dot", width=1),
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=s["elapsed_sec"], y=s["latency_ms"],
                             line=dict(color="#1E293B", width=1.6),
                             name="RTT", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=s["elapsed_sec"], y=s["queue_delay_ms"],
                             line=dict(color="#E9A23B", width=1.4),
                             fill="tozeroy", fillcolor="rgba(233,162,59,.18)",
                             name="queue delay", showlegend=False),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=s["elapsed_sec"], y=s["offered_mbit"],
                             line=dict(color="#3B82F6", width=1.2),
                             name="offered load", showlegend=False),
                  row=2, col=1, secondary_y=True)

    for c in CLASSES[1:]:
        if not len(p):
            break
        if c == emphasis or p[f"p_{c}"].max() > 0.05:
            fig.add_trace(go.Scatter(
                x=pl["elapsed_sec"], y=pl[f"p_{c}"], mode="lines",
                name=NICE[c], line=dict(color=COLORS[c],
                                        width=2.4 if c == emphasis else 1.8)),
                row=3, col=1)
    if tau is not None:
        fig.add_hline(y=tau, line_color="#334155", line_dash="dot",
                      line_width=1, row=3, col=1)
    al = p[p["alarm"] > 0]
    if len(al):
        fig.add_trace(go.Scatter(
            x=al["elapsed_sec"], y=[1.06] * len(al), mode="markers",
            marker=dict(symbol="triangle-down", size=8,
                        color=[COLORS[CLASSES[a]] for a in al["alarm"]]),
            name="warning", showlegend=False, hoverinfo="x"), row=3, col=1)

    fig.update_yaxes(range=[0, 1.12], row=3, col=1, tickformat=".1f")
    fig.update_yaxes(rangemode="tozero", row=2, col=1, secondary_y=False)
    fig.update_yaxes(rangemode="tozero", row=2, col=1, secondary_y=True,
                     showgrid=False, tickfont=dict(color="#3B82F6"))
    fig.update_xaxes(range=[t_from, max(t_now, t_from + 30)],
                     showgrid=False, linecolor=LINE, ticks="outside",
                     tickcolor=LINE)
    fig.update_xaxes(title_text="time in run (s)", row=3, col=1,
                     title_font=dict(size=11, color=MUTED))
    fig.update_yaxes(gridcolor="#EEF0F3", zeroline=False, linecolor=LINE)
    fig.update_layout(height=600, margin=dict(l=48, r=48, t=36, b=36),
                      paper_bgcolor="white", plot_bgcolor="white", font=FONT,
                      legend=dict(orientation="h", y=1.04, x=1, xanchor="right",
                                  yanchor="bottom", font=dict(size=11),
                                  bgcolor="rgba(0,0,0,0)"),
                      hovermode="x unified", uirevision="keep")
    fig.update_layout(legend=dict(y=-0.1, yanchor="top", x=0.5,
                                  xanchor="center"),
                      margin=dict(b=70))
    return fig


NODES = {"h1": (0.0, 1.0), "h2": (0.0, 0.0), "s1": (1.2, 0.5),
         "r1": (2.5, 0.5), "s2": (3.8, 0.5), "h3": (5.0, 1.0),
         "h4": (5.0, 0.0)}
NODE_STYLE = {"h": ("circle", "#334155"), "s": ("square", "#64748B"),
              "r": ("diamond", "#E9A23B")}


def load_color(util, queue_ms):
    if util >= 0.95 or queue_ms >= 100:
        return "#C8364F"
    if util >= 0.7 or queue_ms >= 20:
        return "#E9A23B"
    return "#13895F"


def build_topology(row):
    fig = go.Figure()
    util = row["delivered_mbit"] / NOMINAL_MBIT
    qms = row["queue_delay_ms"]

    def link(a, b, color="#CBD5E1", width=2.5):
        (x0, y0), (x1, y1) = NODES[a], NODES[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color=color, width=width),
                                 hoverinfo="skip", showlegend=False))

    for a, b in (("h1", "s1"), ("h2", "s1"), ("s1", "r1"),
                 ("s2", "h3"), ("s2", "h4")):
        link(a, b)
    link("r1", "s2", color=load_color(util, qms),
         width=3 + 9 * min(row["offered_mbit"] / NOMINAL_MBIT, 1.2))

    for n, (x, y) in NODES.items():
        sym, col = NODE_STYLE[n[0]]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[n],
            textposition=("bottom center" if y < 0.5 else
                          "top center" if y > 0.5 else "bottom center"),
            marker=dict(symbol=sym, size=22, color=col,
                        line=dict(color="white", width=2)),
            textfont=dict(size=12, color="#334155"), hoverinfo="skip",
            showlegend=False))

    fill = min(qms / BUFFER_MS, 1.0)
    rx, ry = NODES["r1"]
    shapes = [dict(type="rect", x0=rx - 0.4, x1=rx + 0.4, y0=ry - 0.44,
                   y1=ry - 0.35, line=dict(color="#CBD5E1", width=1),
                   fillcolor="#F1F5F9")]
    if fill > 0:
        shapes.append(dict(type="rect", x0=rx - 0.4,
                           x1=rx - 0.4 + 0.8 * fill, y0=ry - 0.44,
                           y1=ry - 0.35, line_width=0,
                           fillcolor=load_color(0, qms)))
    small = dict(size=11, color="#475569")
    fig.add_annotation(x=rx, y=ry - 0.55, showarrow=False, font=small,
                       text=f"router queue {qms:.0f} ms")
    fig.add_annotation(x=3.15, y=0.74, showarrow=False, font=small,
                       text=(f"<b>50 Mbit/s bottleneck</b><br>"
                             f"{row['offered_mbit']:.1f} offered · "
                             f"{row['delivered_mbit']:.1f} delivered"))
    fig.add_annotation(x=3.15, y=0.27, showarrow=False, font=small,
                       text=f"drops {row['router_drop_pct']:.1f} %")
    loss = (f" · loss {row['ping_loss_pct']:.0f} %"
            if pd.notna(row["ping_loss_pct"]) else "")
    rtt = row["latency_ms"]
    fig.add_annotation(x=2.5, y=1.18, showarrow=False, font=small,
                       text=(f"ping h2 → h3  ·  RTT "
                             f"{rtt:.0f} ms (baseline "
                             f"{row['baseline_latency_ms']:.0f}){loss}"
                             if pd.notna(rtt) else "ping h2 → h3"))
    fig.update_layout(
        height=250, margin=dict(l=8, r=8, t=8, b=8), shapes=shapes,
        xaxis=dict(visible=False, range=[-0.4, 5.4]),
        yaxis=dict(visible=False, range=[-0.6, 1.32]),
        paper_bgcolor="white", plot_bgcolor="white", font=FONT)
    return fig


# ============================================================
# APP
# ============================================================

_draws = {"n": 0}


def show_fig(placeholder, fig):
    # a unique key per draw: during playback the same placeholder is redrawn
    # many times in one script run, sometimes with an identical figure
    _draws["n"] += 1
    kw = dict(config={"displayModeBar": False}, key=f"fig{_draws['n']}")
    try:
        placeholder.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        placeholder.plotly_chart(fig, use_container_width=True, **kw)


def main():
    st.set_page_config(page_title="Network anomaly early warning",
                       page_icon="📡", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    sources = ([SRC_NEW] if new_runs() else []) + \
        ([SRC_CV] if CLEAN.exists() and available_models() else [])
    if not sources:
        st.error("Nothing to replay. Run demo/predict_new_run.py on a new "
                 "recording, or run the pipeline and train a model first.")
        return

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.markdown('<div class="app-title" style="font-size:1.05rem">'
                    'Replay</div>', unsafe_allow_html=True)
        source = st.radio("Data", sources, label_visibility="collapsed",
                          key="w_source")
        if source == SRC_NEW:
            run_dir = st.selectbox("Recording", new_runs(), key="w_rec",
                                   format_func=lambda p: p.name.split("__")[-1])
            info = load_new_info(str(run_dir))
            models = [m for m in MODEL_ORDER if m in info["models"]] + \
                [m for m in info["models"] if m not in MODEL_ORDER]
        else:
            models = available_models()
        model = st.selectbox("Model", models, key="w_model",
                             format_func=lambda m: MODEL_NICE.get(m, m))
        if source == SRC_CV:
            sig_all = load_signals()
            pred_all = load_predictions(model)
            runs = sorted(pred_all["run_id"].unique())
            present = {r.split("/")[1] for r in runs}
            scen = st.selectbox("Anomaly type", key="w_scen", options=
                                [c for c in SCENARIO_ORDER if c in present]
                                + sorted(present - set(SCENARIO_ORDER)),
                                format_func=lambda s: NICE.get(s, s))
            run = st.selectbox("Run", key="w_run", options=
                               [r for r in runs if r.split("/")[1] == scen],
                               format_func=lambda r: r.split("/")[-1])
        st.divider()
        speed = st.select_slider("Speed (× real time)", [1, 2, 5, 10, 20],
                                 key="w_speed",
                                 value=5)
        calibrated = st.toggle("Calibrated alarm (≤ 5 false alarms/h)",
                               key="w_cal",
                               value=True)
        show_net = st.toggle("Network diagram", value=True, key="w_net")

    # ---------------- data for the chosen run ----------------
    tau = None
    if source == SRC_NEW:
        sig = load_new_signals(str(run_dir)).sort_values("elapsed_sec")
        # always run the CURRENT detector (signals.csv may hold an older one)
        sig = sig.drop(columns=["detected", "det_type", "reason"],
                       errors="ignore").reset_index(drop=True)
        sig = sig.join(detect_run(sig))
        pred = load_new_predictions(str(run_dir), model) \
            .sort_values("elapsed_sec").copy()
        run, emphasis = info["run_id"], None
        if calibrated:
            tau = info["thresholds"].get(model)
        subtitle = (f"{run.split('/')[-1]} · recorded after training · "
                    f"final {MODEL_NICE.get(model, model)} model")
        tag = "UNSEEN DATA"
        about = ("Episode order: " + " → ".join(
            f"{NICE.get(e['type'], e['type'])} ({e['start']:.0f} s)"
            for e in info.get("episodes", [])))
    else:
        sig = sig_all[sig_all["run_id"] == run].sort_values("elapsed_sec")
        sig = sig.reset_index(drop=True).join(detect_run(sig))
        pred = pred_all[pred_all["run_id"] == run] \
            .sort_values("elapsed_sec").copy()
        fold = pred["fold"].iloc[0]
        emphasis = scen
        if calibrated:
            th = load_thresholds()
            m = th[(th["model"] == model) & (th["fold"] == fold)]
            if "op" in m and (m["op"] == "fa5").any():
                m = m[m["op"] == "fa5"]
            tau = float(m["threshold"].iloc[0]) if len(m) else None
        subtitle = (f"{run.split('/')[-1]} · test run of {fold} · "
                    f"{MODEL_NICE.get(model, model)} model of that fold")
        tag = "CROSS-VALIDATION"
        about = ("The model shown was trained on the other folds and never "
                 "saw this run.")
    if calibrated and tau is None:
        st.sidebar.caption("No calibrated threshold found; using argmax.")

    sig = sig.reset_index(drop=True)
    sig["recovering"] = recovering_mask(sig)
    busy = set(sig.loc[sig["detected"] == 1, "elapsed_sec"].round(3))
    pred = pred[~pred["elapsed_sec"].round(3).isin(busy)].copy()
    pred["alarm"], pred["score"] = alarm_column(pred, tau)

    # ---------------- playback state ----------------
    t_min, t_max = 0, int(sig["elapsed_sec"].max())
    first = onsets_of(sig)
    # Replay starts at the beginning of the run, so the score always covers
    # exactly what has been played (use "Next onset" to skip ahead).
    t_start = t_min
    ss = st.session_state
    context = (source, run, model, calibrated)
    if ss.get("context") != context:
        ss.update(context=context, playing=False, t=t_start, t_slider=t_start)
    if ss.pop("sync", False):
        ss["t_slider"] = ss["t"]

    def go_to(t_new):
        ss["t"] = int(min(max(t_new, t_min), t_max))
        ss["t_slider"] = ss["t"]

    def on_play():
        if ss["t"] >= t_max:
            go_to(t_start)
        ss["playing"] = True

    def on_pause():
        ss["playing"] = False
        go_to(ss["t"])

    def on_reset():
        # Clear EVERYTHING: playback, time, sidebar choices and cached data
        # (so files re-written by predict_new_run.py are read again).
        # The app then reopens at 0 s with default settings.
        for k in list(ss.keys()):
            del ss[k]
        st.cache_data.clear()

    def on_next():
        ss["playing"] = False
        nxt = [o for o in onsets_of(sig) if o > ss["t"] + 31]
        if nxt:
            go_to(nxt[0] - 30)

    def on_slider():
        ss["playing"] = False
        ss["t"] = ss["t_slider"]

    # ---------------- layout ----------------
    st.markdown(f"""
<div class="app-head"><div>
  <div class="app-title">Network anomaly early warning
    <span class="tag">{tag}</span></div>
  <div class="app-sub">{subtitle}</div>
</div></div>""", unsafe_allow_html=True)

    widths = [1, 1, 1, 1.35, 4.6]
    try:
        b1, b2, b3, b4, sl = st.columns(widths, vertical_alignment="center")
    except TypeError:                      # Streamlit < 1.36
        b1, b2, b3, b4, sl = st.columns(widths)

    def button(col, label, cb, **kw):
        try:
            col.button(label, on_click=cb, width="stretch", **kw)
        except TypeError:                  # older Streamlit
            col.button(label, on_click=cb, use_container_width=True, **kw)

    button(b1, "▶  Play", on_play)
    button(b2, "⏸  Pause", on_pause)
    button(b3, "↺  Reset", on_reset,
           help="Clear everything and start again at 0 s")
    button(b4, "Next onset  →", on_next,
           help="Jump to 30 s before the next onset")
    sl.slider("Time", t_min, t_max, key="t_slider", on_change=on_slider,
              label_visibility="collapsed")

    status_box = st.empty()
    if show_net:
        left, right = st.columns([1.75, 1], gap="medium")
        net_box = left.empty()
        kpi_box = right.empty()
    else:
        kpi_box = st.empty()
        net_box = None
    chart_box = st.empty()

    tab_on, tab_al, tab_det, tab_help = st.tabs(
        ["Onsets", "Warnings", "Detections", "How to read"])
    with tab_on:
        onset_box = st.empty()
    with tab_al:
        alarm_box = st.empty()
    with tab_det:
        det_box = st.empty()
    with tab_help:
        st.markdown(GUIDE_MD)
        if about:
            st.caption(about)

    def draw(t):
        kind, head, detail = status(sig, pred, t)
        status_box.markdown(status_html(kind, head, detail, t, t_max,
                                        truth_text(sig, t)),
                            unsafe_allow_html=True)
        dets = detection_rows(sig, t)
        ons, als = onset_rows(sig, pred, t, dets), alarm_rows(sig, pred, t)
        kpi_box.markdown(
            f'<div class="sec" style="margin-top:.2rem">Score · '
            f'{t_min}–{t} s of the run</div>'
            + kpi_html(ons, als, dets), unsafe_allow_html=True)
        if net_box is not None:
            upto = sig[sig["elapsed_sec"] <= t]
            show_fig(net_box, build_topology(upto.iloc[-1] if len(upto)
                                             else sig.iloc[0]))
        show_fig(chart_box, build_figure(sig, pred, t,
                                         max(t_min, t - WINDOW_SEC),
                                         emphasis, tau))
        onset_box.markdown(onset_table_html(ons), unsafe_allow_html=True)
        alarm_box.markdown(alarm_table_html(als), unsafe_allow_html=True)
        det_box.markdown(detection_table_html(dets), unsafe_allow_html=True)

    if ss["playing"]:
        t = ss["t"]
        while t <= t_max and ss["playing"]:
            ss["t"] = t
            draw(t)
            time.sleep(1.0 / speed)
            t += 1
        ss["playing"] = False
        ss["sync"] = True
        st.rerun()
    else:
        draw(ss["t"])


if __name__ == "__main__":
    main()
