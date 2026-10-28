"""
Replay demo: early warning of network anomalies.

Plays back one recorded run second by second, as if it were live, and shows
  * the network signals (RTT, router queue delay, offered load)
  * the model's prediction every second: P(each anomaly starts in <= 10 s)
  * the alarm (calibrated threshold, <= 5 false alarms / hour)
  * the true anomaly period and onset, so you can see the warning come first

Honest by construction: every run is a TEST run in exactly one
cross-validation fold, and the predictions shown come from the model of
that fold, which never saw this run during training.

Run (from the project root):
    pip install streamlit plotly
    streamlit run demo/replay_app.py

Needs (produced by the pipeline):
    data/processed/network_telemetry_clean.csv   (Stage 1)
    results/predictions/<model>/fold*.csv         (training scripts)
    results/evaluation/thresholds.csv             (evaluate.py)
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "network_telemetry_clean.csv"
PRED_DIR = ROOT / "results" / "predictions"
THRESH = ROOT / "results" / "evaluation" / "thresholds.csv"

CLASSES = ["normal", "congestion", "high_latency", "packet_loss",
           "bandwidth_degradation", "queue_buildup", "traffic_burst"]
NICE = {"normal": "Normal", "congestion": "Congestion",
        "high_latency": "High latency", "packet_loss": "Packet loss",
        "bandwidth_degradation": "Bandwidth degradation",
        "queue_buildup": "Queue build-up", "traffic_burst": "Traffic burst"}
COLORS = {"congestion": "#d62728", "high_latency": "#9467bd",
          "packet_loss": "#8c564b", "bandwidth_degradation": "#e377c2",
          "queue_buildup": "#ff7f0e", "traffic_burst": "#17becf"}
MODEL_ORDER = ["lstm", "gru", "tcn", "random_forest", "logistic_regression"]
SCENARIO_ORDER = ["congestion", "high_latency", "packet_loss", "queue_buildup",
                  "traffic_burst", "bandwidth_degradation", "normal"]
WINDOW_SEC = 180          # seconds of history shown while playing
NOMINAL_MBIT = 50.0


LEGEND_MD = """
**Markings on the graphs**
- 🟥 **Red shading:** the anomaly is actually happening (ground truth).
- ┆ **Dashed red line:** the exact second the anomaly started (the onset).
- ⬜ **Grey shading:** a harmless excursion, a small normal bump in load,
  delay or loss. The model should *not* warn here; triangles inside grey
  are false alarms.
- ▼ **Triangles (top of the bottom graph):** a warning raised by the model,
  coloured by the anomaly type it expects.
- ┄ **Dotted black line:** the alarm threshold (a warning is raised when the
  model's anomaly probability is above it).
- **Gaps in the probability lines:** seconds the model isn't scored on
  (during the anomaly, during recovery, or early in a long build-up).

**Status banner**
- 🟩 Normal: no anomaly expected in the next 10 s
- 🟨 WARNING: an anomaly is expected within 10 s (the early warning)
- 🟥 ANOMALY IN PROGRESS: the anomaly has started
- 🟦 Recovering, or no prediction for this second

**A good prediction:** the triangles and the yellow banner appear
*before* the dashed red line. The gap between them is the warning time.
"""


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_signals():
    cols = ["run_id", "scenario", "elapsed_sec", "latency_ms",
            "baseline_latency_ms", "qdisc_backlog_bytes",
            "ingress_rx_bytes_per_sec", "tx_bytes_per_sec", "ping_loss_pct",
            "ingress_rx_packets_per_sec", "tbf_dropped_per_sec",
            "netem_dropped_per_sec",
            "label", "phase", "episode_id", "benign_excursion"]
    df = pd.read_csv(CLEAN, usecols=cols)
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


def available_models():
    found = [m for m in MODEL_ORDER if (PRED_DIR / m).exists()]
    return found + sorted(p.name for p in PRED_DIR.iterdir()
                          if p.is_dir() and p.name not in found)


def onsets_of(sig):
    y = sig["label"].to_numpy()
    idx = np.flatnonzero((y[1:] == 1) & (y[:-1] == 0)) + 1
    return sig["elapsed_sec"].to_numpy()[idx]


def anomaly_spans(sig):
    t, y = sig["elapsed_sec"].to_numpy(), sig["label"].to_numpy()
    spans, start = [], None
    for i in range(len(y)):
        if y[i] == 1 and start is None:
            start = t[i]
        if y[i] == 0 and start is not None:
            spans.append((start, t[i]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def with_gaps(p, t_col="elapsed_sec", max_step=1.5):
    """Insert an empty row wherever predictions skip seconds, so plotted
    lines break instead of joining across unscored periods."""
    if len(p) < 2:
        return p
    t = p[t_col].to_numpy()
    breaks = np.flatnonzero(np.diff(t) > max_step)
    if not len(breaks):
        return p
    blanks = pd.DataFrame({t_col: t[breaks] + 0.5})
    return (pd.concat([p, blanks], ignore_index=True)
            .sort_values(t_col).reset_index(drop=True))


def excursion_spans(sig):
    t = sig["elapsed_sec"].to_numpy()
    e = sig["benign_excursion"].to_numpy() > 0
    spans, start = [], None
    for i in range(len(e)):
        if e[i] and start is None:
            start = t[i]
        if not e[i] and start is not None:
            spans.append((start, t[i]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def alarm_column(pred, tau):
    probs = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
    score = 1 - probs[:, 0]
    best = probs[:, 1:].argmax(axis=1) + 1
    if tau is None:                         # argmax mode
        return np.where(probs.argmax(axis=1) > 0, best, 0), score
    return np.where(score >= tau, best, 0), score


# ============================================================
# FIGURE
# ============================================================

def build_figure(sig, pred, t_now, t_from, scenario, tau):
    s = sig[(sig["elapsed_sec"] >= t_from) & (sig["elapsed_sec"] <= t_now)]
    p = pred[(pred["elapsed_sec"] >= t_from) & (pred["elapsed_sec"] <= t_now)]
    pl = with_gaps(p)                      # lines break where unscored
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.32, 0.32, 0.36],
        specs=[[{}], [{"secondary_y": True}], [{}]],
        subplot_titles=("Round-trip time (ms)",
                        "Router queue delay (ms)  |  offered load (Mbit/s)",
                        "Model: probability that this anomaly STARTS within 10 s"))

    # true anomaly periods (ground truth, shaded)
    for a, b in anomaly_spans(sig):
        if b < t_from or a > t_now:
            continue
        for r in (1, 2, 3):
            fig.add_vrect(x0=max(a, t_from), x1=min(b, t_now),
                          fillcolor="red", opacity=0.08, line_width=0, exclude_empty_subplots=False,
                          row=r, col=1)
    for a, b in excursion_spans(sig):
        if b < t_from or a > t_now:
            continue
        for r in (1, 2, 3):
            fig.add_vrect(x0=max(a, t_from), x1=min(b, t_now),
                          fillcolor="grey", opacity=0.12, line_width=0, exclude_empty_subplots=False,
                          row=r, col=1)
    for on in onsets_of(sig):
        if t_from <= on <= t_now:
            for r in (1, 2, 3):
                fig.add_vline(x=on, line_color="red", line_dash="dash",
                              line_width=1.2, exclude_empty_subplots=False,
                              row=r, col=1)

    fig.add_trace(go.Scatter(x=s["elapsed_sec"], y=s["latency_ms"],
                             name="RTT", line=dict(color="#1f77b4")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=s["elapsed_sec"],
                             y=s["baseline_latency_ms"],
                             name="baseline", line=dict(color="grey",
                                                        dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=s["elapsed_sec"], y=s["queue_delay_ms"],
                             name="queue delay", line=dict(color="#ff7f0e")),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=s["elapsed_sec"], y=s["offered_mbit"],
                             name="offered load", line=dict(color="#2ca02c",
                                                            width=1)),
                  row=2, col=1, secondary_y=True)

    for c in CLASSES[1:]:
        show = c == scenario or (len(p) and p[f"p_{c}"].max() > 0.2)
        if not show:
            continue
        fig.add_trace(go.Scatter(
            x=pl["elapsed_sec"], y=pl[f"p_{c}"], mode="lines",
            name=NICE[c], line=dict(color=COLORS[c],
                                    width=3 if c == scenario else 1.5)),
            row=3, col=1)
    if tau is not None:
        fig.add_hline(y=tau, line_color="black", line_dash="dot",
                      annotation_text="alarm threshold",
                      annotation_position="top left", row=3, col=1)

    # alarm markers
    al = p[p["alarm"] > 0]
    if len(al):
        fig.add_trace(go.Scatter(
            x=al["elapsed_sec"], y=[1.04] * len(al), mode="markers",
            marker=dict(symbol="triangle-down", size=9,
                        color=[COLORS[CLASSES[a]] for a in al["alarm"]]),
            name="WARNING raised"), row=3, col=1)

    fig.update_yaxes(range=[0, 1.1], row=3, col=1)
    fig.update_xaxes(range=[t_from, max(t_now, t_from + 30)])
    fig.update_xaxes(title_text="time in run (s)", row=3, col=1)
    fig.update_layout(height=720, margin=dict(l=40, r=20, t=40, b=40),
                      legend=dict(orientation="h", y=-0.12),
                      uirevision="keep")
    return fig


# ============================================================
# NETWORK DIAGRAM
# ============================================================

# Mininet topology used by the collector:
#   h1, h2 --- s1 --- r1 ===(50 Mbit/s bottleneck)=== s2 --- h3, h4
#   background traffic h2 -> h3, cross traffic h1 -> h4, ping h2 -> h3
NODES = {"h1": (0.0, 1.0), "h2": (0.0, 0.0), "s1": (1.2, 0.5),
         "r1": (2.5, 0.5), "s2": (3.8, 0.5), "h3": (5.0, 1.0),
         "h4": (5.0, 0.0)}
NODE_KIND = {"h1": "host", "h2": "host", "h3": "host", "h4": "host",
             "s1": "switch", "s2": "switch", "r1": "router"}
NODE_STYLE = {"host": ("circle", "#4c78a8"), "switch": ("square", "#72b7b2"),
              "router": ("diamond", "#f58518")}
BUFFER_MS = 400.0                 # tbf buffer: 400 ms at 50 Mbit/s
STATUS_COLOR = {"success": "#2ca02c", "warning": "#e6a700",
                "error": "#d62728", "info": "#1f77b4"}


def _load_color(util, queue_ms):
    """green -> yellow -> red from link utilisation and queue delay."""
    if util >= 0.95 or queue_ms >= 100:
        return "#d62728"            # saturated / heavy queueing
    if util >= 0.7 or queue_ms >= 20:
        return "#e6a700"            # filling up
    return "#2ca02c"                # healthy


def build_topology(row, status_kind, status_text):
    fig = go.Figure()
    util = row["delivered_mbit"] / NOMINAL_MBIT
    qms = row["queue_delay_ms"]
    link_col = _load_color(util, qms)

    def link(a, b, color="#9e9e9e", width=3, dash=None):
        (x0, y0), (x1, y1) = NODES[a], NODES[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(color=color, width=width,
                                           dash=dash),
                                 hoverinfo="skip", showlegend=False))

    for a, b in (("h1", "s1"), ("h2", "s1"), ("s1", "r1"),
                 ("s2", "h3"), ("s2", "h4")):
        link(a, b)
    # the bottleneck: width grows with load, colour with congestion
    link("r1", "s2", color=link_col,
         width=4 + 10 * min(row["offered_mbit"] / NOMINAL_MBIT, 1.2))

    for n, (x, y) in NODES.items():
        sym, col = NODE_STYLE[NODE_KIND[n]]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text", text=[n],
            textposition="bottom center" if y < 0.5 else "top center",
            marker=dict(symbol=sym, size=34, color=col,
                        line=dict(color="white", width=2)),
            textfont=dict(size=14), hoverinfo="skip", showlegend=False))

    # router queue: a bar that fills with queue delay (buffer = 400 ms)
    fill = min(qms / BUFFER_MS, 1.0)
    rx, ry = NODES["r1"]
    fig.add_shape(type="rect", x0=rx - 0.35, x1=rx + 0.35, y0=ry - 0.42,
                  y1=ry - 0.30, line=dict(color="#555", width=1))
    if fill > 0:
        fig.add_shape(type="rect", x0=rx - 0.35, x1=rx - 0.35 + 0.7 * fill,
                      y0=ry - 0.42, y1=ry - 0.30, line_width=0,
                      fillcolor=_load_color(0, qms))
    fig.add_annotation(x=rx, y=ry - 0.52, showarrow=False,
                       text=f"router queue {qms:.0f} ms", font=dict(size=12))

    # live numbers on the paths
    fig.add_annotation(x=3.15, y=0.78, showarrow=False, font=dict(size=13),
                       text=(f"<b>bottleneck 50 Mbit/s</b><br>"
                             f"offered {row['offered_mbit']:.1f} &nbsp;|&nbsp; "
                             f"delivered {row['delivered_mbit']:.1f} Mbit/s"))
    fig.add_annotation(x=3.15, y=0.18, showarrow=False, font=dict(size=12),
                       text=f"router drops {row['router_drop_pct']:.1f} %")
    fig.add_annotation(x=0.6, y=-0.42, showarrow=False, font=dict(size=12),
                       text=(f"ping h2 → h3: RTT {row['latency_ms']:.0f} ms "
                             f"(baseline {row['baseline_latency_ms']:.0f})"
                             f" · ping loss {row['ping_loss_pct']:.0f} %"
                             if pd.notna(row["ping_loss_pct"]) else
                             f"ping h2 → h3: RTT {row['latency_ms']:.0f} ms"))
    fig.add_annotation(x=0.6, y=1.32, showarrow=False, font=dict(size=12),
                       text="traffic: h2 → h3 (background), "
                            "h1 → h4 (cross)")

    border = STATUS_COLOR.get(status_kind, "#999")
    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=36, b=10),
        title=dict(text=status_text, font=dict(size=15, color=border),
                   x=0.01),
        xaxis=dict(visible=False, range=[-0.5, 5.5]),
        yaxis=dict(visible=False, range=[-0.65, 1.45]),
        plot_bgcolor="white",
        shapes=list(fig.layout.shapes) + [dict(
            type="rect", xref="paper", yref="paper", x0=0, x1=1, y0=0,
            y1=1, line=dict(color=border, width=4))])
    return fig


# ============================================================
# STATUS BANNER
# ============================================================

def status(sig, pred, t_now):
    row = sig[sig["elapsed_sec"] <= t_now].iloc[-1]
    if row["label"] == 1:
        return ("error", f"ANOMALY IN PROGRESS: {NICE[row['scenario']]}")
    if row["phase"] == "cooldown":
        return ("info", "Network recovering after an anomaly")
    p = pred[pred["elapsed_sec"] <= t_now]
    if not len(p) or abs(p.iloc[-1]["elapsed_sec"] - t_now) >= 1.5:
        return ("info", "No prediction this second (outside the scored "
                        "range: run start, gap, or early in an episode ramp)")
    last = p.iloc[-1]
    if last["alarm"] > 0:
        c = CLASSES[int(last["alarm"])]
        return ("warning", f"WARNING: {NICE[c]} expected within 10 s "
                           f"(P = {last[f'p_{c}']:.2f})")
    extra = " (harmless excursion in progress)" if row["benign_excursion"] > 0 else ""
    return ("success", "Normal: no anomaly expected in the next 10 s" + extra)


def onset_table(sig, pred):
    rows = []
    for on in onsets_of(sig):
        w = pred[(pred["elapsed_sec"] >= on - 10.5) & (pred["elapsed_sec"] < on)]
        cls = CLASSES.index(sig["scenario"].iloc[0])
        hit = w[w["alarm"] == cls]
        rows.append({"onset (s)": round(on),
                     "warned": "yes" if len(hit) else "no",
                     "first warning (s before onset)":
                         f"{on - hit['elapsed_sec'].min():.0f}"
                         if len(hit) else "-"})
    return pd.DataFrame(rows)


# ============================================================
# APP
# ============================================================

def main():
    st.set_page_config(page_title="Network anomaly early warning",
                       layout="wide")
    st.title("Early warning of network performance anomalies")

    if not CLEAN.exists() or not PRED_DIR.exists():
        st.error("Run the pipeline and train a model first "
                 "(data/processed and results/predictions are missing).")
        return

    sig_all = load_signals()
    models = available_models()
    th = load_thresholds()

    with st.sidebar:
        st.header("Replay settings")
        model = st.selectbox("Model", models)
        pred_all = load_predictions(model)
        runs = sorted(pred_all["run_id"].unique())
        present = {r.split("/")[1] for r in runs}
        scen = st.selectbox("Anomaly type",
                            [c for c in SCENARIO_ORDER if c in present]
                            + sorted(present - set(SCENARIO_ORDER)),
                            format_func=lambda s: NICE.get(s, s))
        run = st.selectbox("Run", [r for r in runs if r.split("/")[1] == scen],
                           format_func=lambda r: r.split("/")[-1])
        speed = st.select_slider("Speed (seconds of run per real second)",
                                 [1, 2, 5, 10, 20], value=5)
        calibrated = st.toggle("Calibrated alarm (<= 5 false alarms/h)",
                               value=True)

    sig = sig_all[sig_all["run_id"] == run].sort_values("elapsed_sec")
    pred = pred_all[pred_all["run_id"] == run].sort_values("elapsed_sec").copy()
    fold = pred["fold"].iloc[0]
    tau = None
    if calibrated:
        m = th[(th["model"] == model) & (th["fold"] == fold)]
        tau = float(m["threshold"].iloc[0]) if len(m) else None
        if tau is None:
            st.sidebar.warning("No calibrated threshold found (run "
                               "evaluate.py). Using argmax.")
    pred["alarm"], pred["score"] = alarm_column(pred, tau)

    t_min, t_max = int(sig["elapsed_sec"].min()), int(sig["elapsed_sec"].max())
    key = f"t::{run}"
    if key not in st.session_state:
        first = onsets_of(sig)
        st.session_state[key] = int(max(t_min + 30, first[0] - 30)) \
            if len(first) else t_min + 30
    st.session_state.setdefault("playing", False)

    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 4])
    if c1.button("▶ Play"):
        st.session_state["playing"] = True
    if c2.button("⏸ Pause"):
        st.session_state["playing"] = False
    if c3.button("⏭ 30 s before next onset"):
        nxt = [o for o in onsets_of(sig) if o > st.session_state[key] + 1]
        if nxt:
            st.session_state[key] = int(max(t_min, nxt[0] - 30))
    t_sel = c4.slider("Time (s)", t_min, t_max, int(st.session_state[key]))
    if not st.session_state["playing"]:
        st.session_state[key] = t_sel

    st.caption(f"Run **{run.split('/')[-1]}**: this run was a TEST run in "
               f"{fold}; the {model} model shown never trained on it.")
    with st.expander("ℹ️ How to read this screen", expanded=False):
        st.markdown(LEGEND_MD)
    banner = st.empty()
    show_net = st.toggle("Show network diagram", value=True)
    net = st.empty()
    chart = st.empty()

    def show(placeholder, fig):
        try:
            placeholder.plotly_chart(fig, width="stretch")
        except TypeError:                       # older Streamlit versions
            placeholder.plotly_chart(fig, use_container_width=True)

    def draw(t):
        kind, text = status(sig, pred, t)
        getattr(banner, kind)(text)
        if show_net:
            row = sig[sig["elapsed_sec"] <= t].iloc[-1]
            show(net, build_topology(row, kind, f"t = {t} s  ·  {text}"))
        show(chart, build_figure(sig, pred, t, max(t_min, t - WINDOW_SEC),
                                 scen, tau))

    if st.session_state["playing"]:
        t = st.session_state[key]
        while t <= t_max and st.session_state["playing"]:
            draw(t)
            time.sleep(1.0 / speed)
            t += 1
            st.session_state[key] = t
        st.session_state["playing"] = False
    else:
        draw(st.session_state[key])

    st.subheader("Onsets in this run")
    st.dataframe(onset_table(sig, pred), hide_index=True)


if __name__ == "__main__":
    main()
