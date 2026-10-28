#!/usr/bin/env python3
"""
Mininet telemetry collector for early prediction of network anomalies (v2.9).

v2.9 adds the "mixed" scenario (demo / unseen test run only): one run
containing one episode of EVERY anomaly type in random order, with normal
gaps and benign excursions in between. Each episode is generated, labelled
and onset-detected exactly as in its own single-type scenario. Mixed runs
are written to DEMO_DIR, never to BASE_DIR, so they can never end up in the
training data. The other scenarios are unchanged (same schedules).

Design changes from v1:
  * Randomised schedule (seeded): several anomaly episodes per run at random
    times, with random ramp / hold / recovery lengths and random normal gaps.
  * Gradual ramp-in for every anomaly except traffic bursts, so a genuine
    precursor exists before the anomaly becomes observable.
  * Observation-based onset: `label` turns 1 only when the effect is actually
    measurable (latency rise, loss, drops, queueing delay, delivery
    shortfall), latched until the episode ends.
  * One measurement and qdisc configuration for every scenario, including
    normal: same ping rate, same loss window, same interfaces, real qdisc
    stats always.
  * Fixed qdisc hierarchy on the router installed once and only *changed*
    afterwards (netem root -> tbf child). No qdisc deletion, no counter resets,
    no loss of the baseline link shaping.
  * Traffic is built from fixed-rate flow units started/stopped individually,
    so load changes never restart (and briefly interrupt) other traffic.
  * Background load varies randomly in every run, and benign "excursions"
    (small load bumps, small delay rises with the same ramp shape) occur in
    every scenario as hard negatives.
  * First-row rates are NaN, failed counter reads are NaN, timestamps match
    the sample instant, and a JSON sidecar records seed, schedule, every
    actuation and every observed onset.

Columns starting with `injected_`, plus `phase`, `episode_id`, `benign_excursion`,
`label`, `anomaly_type`, `scenario`, `seed` are METADATA. Never use them as features.

Usage:
  sudo python3 run_experiment.py --scenario congestion --run congestion_01
  sudo python3 run_experiment.py --scenario mixed --run mixed_1
  python3 run_experiment.py --scenario queue_buildup --run qb_01 --dry-run
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import select
import shutil
import signal
import statistics
import subprocess
import time
from datetime import datetime, timedelta, timezone


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/media/sf_ML_PROJECT/data/raw"
# Files are written here during the run and copied to BASE_DIR at the end.
# BASE_DIR is a VirtualBox shared folder; flushing to it every second is a
# likely cause of the multi-second collector stalls seen in test runs.
LOCAL_DIR = "/var/tmp/ml_network_runs"
# Mixed (demo) runs go here, NOT into BASE_DIR, so the pipeline never
# trains on them.
DEMO_DIR = "/media/sf_ML_PROJECT/data/demo_raw"

DEFAULT_DURATION = 900          # seconds
INTERVAL = 1.0                  # sampling interval (s)
MAX_SCHEDULE_LAG = 2.0          # skip slots instead of catching up in a burst

SCENARIOS = [
    "normal", "congestion", "high_latency", "packet_loss",
    "bandwidth_degradation", "queue_buildup", "traffic_burst", "mixed",
]
ANOMALY_TYPES = ["congestion", "high_latency", "packet_loss",
                 "bandwidth_degradation", "queue_buildup", "traffic_burst"]

# --- Router qdisc baseline (identical in every scenario) -----
LINK_DELAY_MS = 5.0             # mirrors the TCLink delay we replace on r1
BOTTLENECK_MBIT = 50.0          # r1 egress towards h3/h4: the only bottleneck
RETURN_RATE_MBIT = 100.0        # r1 egress back towards h1/h2
TBF_BURST = "64kb"              # kilobytes; enough for <=100 Mbit at HZ>=250
TBF_LATENCY = "400ms"           # same buffer for every scenario
NETEM_LIMIT = 10000             # packets

# --- Traffic -------------------------------------------------
FLOW_UNIT_MBIT = 2.5            # every flow is a fixed-rate UDP unit
FLOW_PAYLOAD = 1400             # bytes
BG_LEVELS_MBIT = [10.0 + 2.5 * i for i in range(9)]   # 10 .. 30 Mbit
BG_SEGMENT_SEC = (30.0, 90.0)
MAX_BG_FLOWS = 16               # h2 -> h3
MAX_CROSS_FLOWS = 28            # h1 -> h4
BG_PORT_BASE = 5100
CROSS_PORT_BASE = 5200

# --- Ping ----------------------------------------------------
PING_TARGET = "10.0.1.1"        # h3
PING_INTERVAL = 0.2             # same for every scenario
PING_LOSS_WINDOW = 10.0         # seconds of settled probes used for loss
PING_SETTLE = 2.0               # probes younger than this are not yet judged
PING_MIN_PROBES = 20            # fewer settled probes -> loss = NaN

# --- Schedule ------------------------------------------------
BASELINE_WINDOW = (5.0, 60.0)           # latency baseline measured here
FIRST_EPISODE_START = (150.0, 300.0)
EPISODE_RAMP = (20.0, 60.0)
EPISODE_HOLD = (40.0, 90.0)
EPISODE_RECOVERY = (5.0, 15.0)
EPISODE_GAP = (90.0, 180.0)
END_MARGIN = 30.0
COOLDOWN_SEC = 15.0             # phase="cooldown" after each episode

# Mixed scenario: every type once per round, random order. Gaps use the
# lower part of the normal range so a run stays ~25 min per round.
MIXED_ROUNDS = 1
MIXED_FIRST_START = (150.0, 200.0)
MIXED_GAP = (90.0, 130.0)

BURST_TRAIN = (60.0, 120.0)
BURST_ON = (4.0, 12.0)
BURST_OFF = (6.0, 25.0)
BURST_MBIT = 45.0

EXCURSION_MEAN_SPACING = 180.0
EXCURSION_EARLIEST = 90.0
EXCURSION_CLEARANCE = 30.0              # keep away from anomaly episodes
EXCURSION_RAMP = (10.0, 30.0)
EXCURSION_HOLD = (10.0, 30.0)
EXCURSION_RECOVERY = (5.0, 15.0)
EXCURSION_LOAD_MBIT = (2.5, 10.0)       # max total 40 < 50 bottleneck
EXCURSION_DELAY_MS = (5.0, 25.0)        # below the 50 ms latency threshold
EXCURSION_LOSS_PCT = (0.5, 3.0)         # below the 5% loss threshold, so
                                        # "any loss" is not a free giveaway

# --- Anomaly targets (reached at the end of the ramp) --------
CONGESTION_EXTRA_MBIT = 60.0            # total 70-90 Mbit vs 50 Mbit
# Queue buildup is closed-loop: the load is steered so the router queue
# tracks a target backlog (no buffer overflow), instead of a fixed excess
# that fills the 2.56 MB buffer within seconds (seen in queue_01).
WIRE_OVERHEAD = 1.0549                  # measured ingress / iperf nominal
QUEUE_TARGET_FRACTION = 0.6             # of the tbf buffer (~240 ms delay)
QUEUE_CEILING_FACTOR = 1.15             # open-loop load ceiling during ramp
QUEUE_CTRL_TAU = 6.0                    # s, proportional time constant
QUEUE_CTRL_KI = 0.03                    # 1/s, integral gain
QUEUE_GUARD_FRACTION = 0.85             # above this, back off below capacity
QUEUE_MAX_EXCESS = 0.025                # max load above capacity -> slow growth
QUEUE_CTRL_I_MAX = 2.0                  # Mbit, integral clamp
TRIM_PORT = 5300                        # one fine-rate flow, h1 -> h4
TRIM_MIN_STEP_MBIT = 0.1                # restart trim flow only above this
DEGRADED_RATE_MBIT = 8.0                # below the minimum background
HIGH_DELAY_MS = 150.0
LOSS_TARGET_PCT = 20.0

# --- Observable onset thresholds -----------------------------
LAT_RISE_MS = 50.0              # latency >= baseline + this
LOSS_ONSET_PCT = 5.0            # router-measured random loss (per second)
# Ping loss is averaged over 10 s of probes judged 2 s late, so it lagged the
# real loss by 9-20 s in packet_loss_01; the router counts every packet.
DROP_RATIO_ONSET = 0.02         # qdisc drops / ingress packets in the second
QUEUE_DELAY_ONSET_MS = 50.0     # tbf backlog expressed as queueing delay
SHORTFALL_ONSET = 0.10          # 1 - delivered / offered bytes


# ============================================================
# GLOBAL STATE / SIGNALS
# ============================================================

STOP_REQUESTED = False
INHIBITOR_PROCESS = None


def signal_handler(signum, frame):
    global STOP_REQUESTED
    print("\n[INFO] Stop requested.")
    STOP_REQUESTED = True


# ============================================================
# SMALL HELPERS
# ============================================================

def nan():
    return float("nan")


def isfinite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def fmt(x, nd=3):
    """CSV-friendly number: blank for missing/non-finite."""
    if not isfinite(x):
        return ""
    return round(x, nd) if isinstance(x, float) else x


def seed_from_run(scenario, run_id):
    digest = hashlib.sha256(f"{scenario}/{run_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def uniform(rng, bounds):
    return rng.uniform(bounds[0], bounds[1])


def rate(cur, prev, dt):
    if cur is None or prev is None or not isfinite(dt) or dt <= 0:
        return nan()
    diff = cur - prev
    if diff < 0:                      # counter reset
        return nan()
    return diff / dt


# ============================================================
# SCHEDULE
# ============================================================

def ep_type(ep, scenario):
    """Anomaly type of an episode (mixed runs store it per episode)."""
    return ep.get("type", scenario)


def mixed_episode_types(rng, rounds=MIXED_ROUNDS):
    types = []
    for _ in range(rounds):
        order = ANOMALY_TYPES[:]
        rng.shuffle(order)
        types += order
    return types


def mixed_duration(rng_seed, rounds=MIXED_ROUNDS):
    """Run length that fits every planned episode of a mixed run."""
    import random as _r
    sched = build_schedule("mixed", 10 ** 6, _r.Random(rng_seed), rounds)
    return int(math.ceil(sched["episodes"][-1]["end"] + END_MARGIN + 30))


def build_schedule(scenario, duration, rng, rounds=MIXED_ROUNDS):
    """Random but reproducible plan of background, episodes and excursions."""

    # Background load segments (every scenario, including normal)
    background = []
    t = 0.0
    while t < duration:
        end = min(duration, t + uniform(rng, BG_SEGMENT_SEC))
        background.append({"start": round(t, 3), "end": round(end, 3),
                           "mbit": rng.choice(BG_LEVELS_MBIT)})
        t = end

    # Anomaly episodes
    episodes = []
    mixed = scenario == "mixed"
    types = mixed_episode_types(rng, rounds) if mixed else []
    if scenario != "normal":
        t = uniform(rng, MIXED_FIRST_START if mixed else FIRST_EPISODE_START)
        while True:
            ep = {"id": len(episodes) + 1, "start": round(t, 3)}
            if mixed:
                if len(episodes) == len(types):
                    break
                ep["type"] = types[len(episodes)]
            if ep_type(ep, scenario) == "traffic_burst":
                train_end = t + uniform(rng, BURST_TRAIN)
                bursts, s = [], t
                while s < train_end - 2.0:
                    on_end = min(train_end, s + uniform(rng, BURST_ON))
                    bursts.append([round(s, 3), round(on_end, 3)])
                    s = on_end + uniform(rng, BURST_OFF)
                ep.update(ramp_end=round(t, 3), hold_end=round(train_end, 3),
                          end=round(train_end, 3), bursts=bursts)
            else:
                ramp_end = t + uniform(rng, EPISODE_RAMP)
                hold_end = ramp_end + uniform(rng, EPISODE_HOLD)
                end = hold_end + uniform(rng, EPISODE_RECOVERY)
                ep.update(ramp_end=round(ramp_end, 3),
                          hold_end=round(hold_end, 3), end=round(end, 3))
            if ep["end"] + END_MARGIN > duration:
                if mixed:
                    raise ValueError(
                        f"--duration {duration} is too short for all "
                        f"{len(types)} mixed episodes; omit --duration.")
                break
            episodes.append(ep)
            t = ep["end"] + uniform(rng, MIXED_GAP if mixed else EPISODE_GAP)
        if not episodes:
            raise ValueError(
                f"--duration {duration} is too short for even one episode; "
                f"use at least {int(FIRST_EPISODE_START[1] + 240)} s.")

    # Benign excursions (hard negatives), every scenario
    # Placed in every free gap (not by chance), so anomaly runs get them too.
    excursions = []
    free, a = [], EXCURSION_EARLIEST
    for ep in episodes:
        free.append((a, ep["start"] - EXCURSION_CLEARANCE))
        a = ep["end"] + EXCURSION_CLEARANCE
    free.append((a, duration - END_MARGIN))
    max_len = EXCURSION_RAMP[1] + EXCURSION_HOLD[1] + EXCURSION_RECOVERY[1]
    slots = []
    for a, b in free:
        k = max(1, int(round((b - a) / EXCURSION_MEAN_SPACING)))
        width = (b - a) / k
        slots += [(a + i * width, a + (i + 1) * width) for i in range(k)]
    for a, b in slots:
        length = (uniform(rng, EXCURSION_RAMP), uniform(rng, EXCURSION_HOLD),
                  uniform(rng, EXCURSION_RECOVERY))
        if b - a < sum(length) + 5.0:
            # Shrink to fit rather than skip, so short gaps still get one
            # (a run with 3 long episodes otherwise could get none).
            room = b - a - 5.0
            if room < 20.0:
                continue
            k = room / sum(length)
            length = tuple(x * k for x in length)
        t = rng.uniform(a, b - sum(length))
        ramp_end = t + length[0]
        hold_end = ramp_end + length[1]
        end = hold_end + length[2]
        kind = rng.choice(["load", "delay", "loss"])
        mag = uniform(rng, {"load": EXCURSION_LOAD_MBIT,
                            "delay": EXCURSION_DELAY_MS,
                            "loss": EXCURSION_LOSS_PCT}[kind])
        excursions.append({"id": len(excursions) + 1, "kind": kind,
                           "magnitude": round(mag, 2), "start": round(t, 3),
                           "ramp_end": round(ramp_end, 3),
                           "hold_end": round(hold_end, 3), "end": round(end, 3)})

    return {"background": background, "episodes": episodes,
            "excursions": excursions}


def envelope(t, w):
    """Intensity 0..1 and state of a ramp/hold/recovery window at time t."""
    if t < w["start"] or t >= w["end"]:
        return 0.0, None
    if t < w["ramp_end"]:
        return (t - w["start"]) / (w["ramp_end"] - w["start"]), "ramp"
    if t < w["hold_end"]:
        return 1.0, "hold"
    return (w["end"] - t) / (w["end"] - w["hold_end"]), "recovery"


def desired_state(schedule, scenario, t):
    """What the network should look like at elapsed time t."""
    def bg_at(x):
        return next((s["mbit"] for s in schedule["background"]
                     if s["start"] <= x < s["end"]),
                    schedule["background"][-1]["mbit"])

    # Background is frozen for the whole episode, so a background step can't
    # make an anomaly vanish (or appear) while the label is latched.
    bg = bg_at(t)
    for ep in schedule["episodes"]:
        if ep["start"] <= t < ep["end"]:
            bg = bg_at(ep["start"])
    st = {"bg_mbit": bg, "cross_mbit": 0.0, "delay_ms": LINK_DELAY_MS,
          "loss_pct": 0.0, "rate_mbit": BOTTLENECK_MBIT, "phase": "normal",
          "episode_id": 0, "intensity": 0.0, "burst_on": False,
          "excursion_id": 0}

    for ex in schedule["excursions"]:
        p, state = envelope(t, ex)
        if state:
            st["excursion_id"] = ex["id"]
            if ex["kind"] == "load":
                st["cross_mbit"] += p * ex["magnitude"]
            elif ex["kind"] == "loss":
                st["loss_pct"] += p * ex["magnitude"]
            else:
                st["delay_ms"] += p * ex["magnitude"]

    for ep in schedule["episodes"]:
        if not (ep["start"] <= t < ep["end"]):
            continue
        st["episode_id"] = ep["id"]
        kind = ep_type(ep, scenario)
        st["anomaly_kind"] = kind
        if kind == "traffic_burst":
            on = any(a <= t < b for a, b in ep["bursts"])
            st.update(phase="hold", burst_on=on, intensity=1.0 if on else 0.0)
            if on:
                st["cross_mbit"] += BURST_MBIT
            break
        p, state = envelope(t, ep)
        st.update(phase=state, intensity=p)
        if kind == "congestion":
            st["cross_mbit"] += p * CONGESTION_EXTRA_MBIT
        elif kind == "queue_buildup":
            # Ceiling only; QueueController sets the actual cross load.
            st["cross_mbit"] += p * max(
                0.0, QUEUE_CEILING_FACTOR * BOTTLENECK_MBIT / WIRE_OVERHEAD
                - bg)
            st["queue_ctrl"] = True
        elif kind == "bandwidth_degradation":
            st["rate_mbit"] = BOTTLENECK_MBIT - p * (
                BOTTLENECK_MBIT - DEGRADED_RATE_MBIT)
        elif kind == "high_latency":
            st["delay_ms"] += p * (HIGH_DELAY_MS - LINK_DELAY_MS)
        elif kind == "packet_loss":
            st["loss_pct"] = p * LOSS_TARGET_PCT
        break

    # Metadata only: windowed measurements (e.g. ping loss) still carry the
    # episode for a while after it ends. Lets the pipeline skip these anchors.
    if st["episode_id"] == 0 and any(
            ep["end"] <= t < ep["end"] + COOLDOWN_SEC
            for ep in schedule["episodes"]):
        st["phase"] = "cooldown"

    # Quantise to what the actuators can actually do
    st["bg_flows"] = int(round(st["bg_mbit"] / FLOW_UNIT_MBIT))
    st["cross_flows"] = int(round(st["cross_mbit"] / FLOW_UNIT_MBIT))
    st["delay_ms"] = float(round(st["delay_ms"]))
    st["loss_pct"] = round(st["loss_pct"] * 2) / 2
    st["rate_mbit"] = round(st["rate_mbit"] * 2) / 2
    st["trim_mbit"] = 0.0
    return st


TBF_BUFFER_BYTES = BOTTLENECK_MBIT * 1e6 / 8 * 0.4 + 64 * 1024   # 2.56 MB


class QueueController:
    """PI controller: steer the router backlog to p(t) * fraction * buffer.

    Load is expressed in iperf-nominal Mbit (the unit the flows use).
    The open-loop ceiling from desired_state() still limits how fast load
    can rise during the ramp, so the precursor stays gradual.
    """

    def __init__(self):
        self.integral = 0.0
        self.overhead = WIRE_OVERHEAD      # refined online from measurements

    def reset(self):
        self.integral = 0.0

    def observe(self, ingress_wire_mbit, nominal_mbit):
        """Learn wire/nominal ratio from every sample with real load."""
        if isfinite(ingress_wire_mbit) and nominal_mbit >= 5.0:
            ratio = ingress_wire_mbit / nominal_mbit
            if 0.95 < ratio < 1.2:
                self.overhead += 0.05 * (ratio - self.overhead)

    def step(self, st, backlog_bytes, dt=1.0):
        """Mutates st: sets cross_mbit, cross_flows and trim_mbit."""
        ceiling = st["cross_mbit"]
        bg = st["bg_flows"] * FLOW_UNIT_MBIT
        capacity = BOTTLENECK_MBIT / self.overhead
        target = st["intensity"] * QUEUE_TARGET_FRACTION * TBF_BUFFER_BYTES
        backlog = backlog_bytes if isfinite(backlog_bytes) else 0.0
        err_mbit = (target - backlog) * 8 / 1e6 / self.overhead

        if 0.05 * TBF_BUFFER_BYTES < backlog < 0.95 * TBF_BUFFER_BYTES:
            self.integral += QUEUE_CTRL_KI * err_mbit * dt
            self.integral = max(-QUEUE_CTRL_I_MAX,
                                min(QUEUE_CTRL_I_MAX, self.integral))
        total = capacity + self.integral + err_mbit / QUEUE_CTRL_TAU
        total = min(total, capacity * (1 + QUEUE_MAX_EXCESS))
        if backlog > QUEUE_GUARD_FRACTION * TBF_BUFFER_BYTES:
            total = min(total, 0.95 * capacity)     # never overflow the buffer
        cross = max(0.0, min(ceiling, total - bg))
        set_cross(st, cross)
        st["queue_target_bytes"] = target
        return st


def set_cross(st, cross_mbit):
    """Split a continuous cross load into 2.5 Mbit units + one trim flow."""
    flows = int(cross_mbit // FLOW_UNIT_MBIT)
    trim = cross_mbit - flows * FLOW_UNIT_MBIT
    st["cross_mbit"] = cross_mbit
    st["cross_flows"] = flows
    st["trim_mbit"] = round(trim, 2)


# ============================================================
# PARSERS
# ============================================================

_SIZE_MULT = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}


def parse_tc_stats(text):
    """Parse `tc -s qdisc show` into {handle: {kind, dropped, backlog_*}}."""
    result, current = {}, None
    for line in text.splitlines():
        head = re.match(r"^qdisc\s+(\S+)\s+([0-9a-fA-F]+):", line)
        if head:
            current = {"kind": head.group(1), "dropped": None,
                       "backlog_bytes": None, "backlog_packets": None}
            result[head.group(2).lower()] = current
            continue
        if current is None:
            continue
        m = re.search(r"dropped\s+(\d+)", line)
        if m and current["dropped"] is None:
            current["dropped"] = int(m.group(1))
        m = re.search(r"backlog\s+([0-9.]+)([KMG]?)b\s+(\d+)p", line)
        if m and current["backlog_bytes"] is None:
            current["backlog_bytes"] = int(
                float(m.group(1)) * _SIZE_MULT[m.group(2)])
            current["backlog_packets"] = int(m.group(3))
    return result


STAT_NAMES = ("rx_bytes", "tx_bytes", "rx_packets", "tx_packets",
              "rx_dropped", "tx_dropped", "rx_errors", "tx_errors")
TC_MARKER = "__TC_BEGIN__"


def counters_command(egress, ingress):
    parts = []
    for intf in (egress, ingress):
        for s in STAT_NAMES:
            parts.append(
                f"printf '%s\\n' \"{intf}.{s}=$(cat "
                f"/sys/class/net/{intf}/statistics/{s} 2>/dev/null)\"")
    parts.append(f"echo {TC_MARKER}")
    parts.append(f"tc -s qdisc show dev {egress}")
    return "; ".join(parts)


def parse_counters(output):
    stats_text, _, tc_text = output.partition(TC_MARKER)
    stats = {}
    for line in stats_text.splitlines():
        name, sep, value = line.strip().partition("=")
        if sep:
            try:
                stats[name] = int(value.strip())
            except ValueError:
                stats[name] = None          # failed read -> missing, not 0
    return stats, parse_tc_stats(tc_text)


_PING_REPLY = re.compile(r"icmp_seq=(\d+).*?time[=<]\s*([0-9.]+)\s*ms")
_PING_NOANS = re.compile(r"no answer yet for icmp_seq=(\d+)")


class PingMonitor:
    """Continuous ping; per-sample median RTT and settled windowed loss."""

    def __init__(self):
        self.proc = None
        self.probes = {}       # seq -> {"first": t, "rx": bool}
        self.new_rtts = []

    def start(self, host):
        self.proc = host.popen(
            ["ping", "-n", "-O", "-D", "-i", str(PING_INTERVAL), "-W", "1",
             PING_TARGET],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)

    def feed_line(self, line, now):
        m = _PING_REPLY.search(line)
        if m:
            seq = int(m.group(1))
            self.new_rtts.append(float(m.group(2)))
            probe = self.probes.setdefault(seq, {"first": now, "rx": False})
            probe["rx"] = True           # a late reply overrides "no answer"
            return
        m = _PING_NOANS.search(line)
        if m:
            self.probes.setdefault(int(m.group(1)), {"first": now, "rx": False})

    def poll(self):
        if self.proc is None or self.proc.stdout is None:
            return
        while True:
            ready, _, _ = select.select([self.proc.stdout], [], [], 0)
            if not ready:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            self.feed_line(line, time.monotonic())

    def sample(self, now):
        rtts, self.new_rtts = self.new_rtts, []
        lat = statistics.median(rtts) if rtts else nan()
        lat_max = max(rtts) if rtts else nan()

        lo, hi = now - PING_SETTLE - PING_LOSS_WINDOW, now - PING_SETTLE
        window = [p for p in self.probes.values() if lo <= p["first"] < hi]
        loss = (100.0 * sum(not p["rx"] for p in window) / len(window)
                if len(window) >= PING_MIN_PROBES else nan())

        cutoff = now - PING_SETTLE - PING_LOSS_WINDOW - 5.0
        for seq in [s for s, p in self.probes.items() if p["first"] < cutoff]:
            del self.probes[seq]
        return lat, lat_max, len(rtts), loss

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


# ============================================================
# LABELLING (observation-based onset)
# ============================================================

def onset_criterion(scenario, m, baseline_latency):
    """True when the anomaly is actually observable in this sample."""
    def ge(x, thr):
        return isfinite(x) and x >= thr

    lat_rise = (ge(m["latency_ms"], baseline_latency + LAT_RISE_MS)
                if isfinite(baseline_latency) else False)
    if scenario == "high_latency":
        return lat_rise
    if scenario == "packet_loss":
        # Router count only. Ping loss is a small, lagging sample (it hit
        # 7.8% during a harmless 2.5% loss bump in packet_loss_02); it is
        # used only if the router count is unavailable for this sample.
        if isfinite(m["router_loss_pct"]):
            return m["router_loss_pct"] >= LOSS_ONSET_PCT
        return ge(m["ping_loss_pct"], LOSS_ONSET_PCT)
    if scenario == "congestion":
        return ge(m["drop_ratio"], DROP_RATIO_ONSET) or lat_rise
    if scenario == "queue_buildup":
        return ge(m["queue_delay_ms"], QUEUE_DELAY_ONSET_MS) or lat_rise
    if scenario == "bandwidth_degradation":
        return (ge(m["shortfall"], SHORTFALL_ONSET)
                or ge(m["queue_delay_ms"], QUEUE_DELAY_ONSET_MS) or lat_rise)
    return False


class Labeler:
    """Label 1 from the first observed onset until the episode ends."""

    def __init__(self, scenario, schedule):
        self.scenario = scenario
        self.episodes = {ep["id"]: ep for ep in schedule["episodes"]}
        self.onsets = {}          # episode id -> elapsed of observed onset
        self.burst_onsets = []    # traffic_burst: one per burst

    def label(self, t, st, metrics, baseline_latency):
        if self.scenario == "normal" or st["episode_id"] == 0:
            return 0, "normal"
        ep = self.episodes[st["episode_id"]]
        kind = ep_type(ep, self.scenario)
        if kind == "traffic_burst":
            if st["burst_on"]:
                start = next(a for a, b in ep["bursts"] if a <= t < b)
                if start not in self.burst_onsets:
                    self.burst_onsets.append(start)
                return 1, "traffic_burst"
            return 0, "normal"

        eid = st["episode_id"]
        if (eid not in self.onsets and st["phase"] in ("ramp", "hold")
                and onset_criterion(kind, metrics, baseline_latency)):
            self.onsets[eid] = round(t, 3)
            print(f"[ONSET] episode {eid} ({kind}) observed at t={t:.1f}s "
                  f"(injection started {self.episodes[eid]['start']:.1f}s)")
        if eid in self.onsets:
            return 1, kind
        return 0, "normal"


# ============================================================
# ROUTER QDISC CONTROL
# ============================================================

def run_checked_node_command(node, command, description):
    marker = "__MININET_COMMAND_STATUS__="
    output = node.cmd(f"{command}; command_status=$?; "
                      f"printf '\\n{marker}%s\\n' \"$command_status\"")
    match = re.search(rf"(?:^|\n){re.escape(marker)}(\d+)\s*$", output)
    if not match:
        raise RuntimeError(f"Could not determine status for {description}: "
                           f"{command!r}; output={output.strip()!r}")
    status = int(match.group(1))
    out = output[:match.start()].strip()
    if status != 0:
        raise RuntimeError(f"{description} failed ({status}): "
                           f"{out or '(no output)'}")
    return out


def get_route_interface(r1, destination):
    out = r1.cmd(f"ip -j route get {destination}")
    try:
        for route in json.loads(out):
            if route.get("dev"):
                return route["dev"]
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    tokens = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "",
                    r1.cmd(f"ip route get {destination}")).split()
    if "dev" in tokens and tokens.index("dev") + 1 < len(tokens):
        return tokens[tokens.index("dev") + 1]
    raise RuntimeError(f"Could not resolve route to {destination}: {out!r}")


def netem_args(delay_ms, loss_pct):
    return (f"limit {NETEM_LIMIT} delay {delay_ms:g}ms "
            f"loss {loss_pct:g}%")


def tbf_args(rate_mbit):
    return f"rate {rate_mbit:g}mbit burst {TBF_BURST} latency {TBF_LATENCY}"


class RouterQdisc:
    """netem root (1:) -> tbf child (10:) on both router interfaces.

    Replaces Mininet's TCLink root on r1 with an equivalent delay + rate
    so the baseline shaping is kept, then only ever *changes* parameters.
    """

    def __init__(self, r1, egress, ingress, actions):
        self.r1, self.egress, self.ingress = r1, egress, ingress
        self.actions = actions
        self.delay = LINK_DELAY_MS
        self.loss = 0.0
        self.rate = BOTTLENECK_MBIT

    def _install(self, intf, rate_mbit):
        run_checked_node_command(
            self.r1, f"tc qdisc replace dev {intf} root handle 1: netem "
            f"{netem_args(LINK_DELAY_MS, 0)}", f"install netem on {intf}")
        run_checked_node_command(
            self.r1, f"tc qdisc replace dev {intf} parent 1:1 handle 10: tbf "
            f"{tbf_args(rate_mbit)}", f"install tbf on {intf}")
        shown = run_checked_node_command(
            self.r1, f"tc -s qdisc show dev {intf}", f"verify qdisc on {intf}")
        parsed = parse_tc_stats(shown)
        if parsed.get("1", {}).get("kind") != "netem" or \
                parsed.get("10", {}).get("kind") != "tbf":
            raise RuntimeError(f"Unexpected qdisc layout on {intf}:\n{shown}")

    def install_baseline(self):
        self._install(self.egress, BOTTLENECK_MBIT)
        self._install(self.ingress, RETURN_RATE_MBIT)

    def apply(self, t, delay_ms, loss_pct, rate_mbit):
        if delay_ms != self.delay or loss_pct != self.loss:
            run_checked_node_command(
                self.r1, f"tc qdisc change dev {self.egress} root handle 1: "
                f"netem {netem_args(delay_ms, loss_pct)}", "change netem")
            self.actions.append({"t": round(t, 3), "netem_delay_ms": delay_ms,
                                 "netem_loss_pct": loss_pct})
            self.delay, self.loss = delay_ms, loss_pct
        if rate_mbit != self.rate:
            run_checked_node_command(
                self.r1, f"tc qdisc change dev {self.egress} parent 1:1 "
                f"handle 10: tbf {tbf_args(rate_mbit)}", "change tbf")
            self.actions.append({"t": round(t, 3), "tbf_rate_mbit": rate_mbit})
            self.rate = rate_mbit


# ============================================================
# TRAFFIC (fixed-rate flow units, never restarted)
# ============================================================

class Traffic:
    GROUPS = {
        # name: (client, server, server_ip, port_base, max_flows)
        "bg": ("h2", "h3", "10.0.1.1", BG_PORT_BASE, MAX_BG_FLOWS),
        "cross": ("h1", "h4", "10.0.1.2", CROSS_PORT_BASE, MAX_CROSS_FLOWS),
    }

    def __init__(self, net, actions, warnings):
        self.net = net
        self.actions, self.warnings = actions, warnings
        self.servers = []
        self.flows = {g: [] for g in self.GROUPS}
        self.dying = []
        self.trim, self.trim_mbit = None, 0.0

    def start_servers(self):
        for client, server, ip, base, n in self.GROUPS.values():
            host = self.net.get(server)
            for i in range(n):
                self.servers.append(host.popen(
                    ["iperf", "-s", "-u", "-p", str(base + i)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.servers.append(self.net.get("h4").popen(
            ["iperf", "-s", "-u", "-p", str(TRIM_PORT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(0.5)

    def set_trim(self, t, mbit):
        """One fine-rate flow (0-2.5 Mbit) on top of the units."""
        if abs(mbit - self.trim_mbit) < TRIM_MIN_STEP_MBIT and \
                not (mbit == 0 and self.trim_mbit > 0):
            return
        if self.trim is not None:
            try:
                self.trim.kill()
            except Exception:
                pass
            self.dying.append(self.trim)
            self.trim = None
        if mbit >= TRIM_MIN_STEP_MBIT:
            self.trim = self.net.get("h1").popen(
                ["iperf", "-u", "-c", "10.0.1.2", "-p", str(TRIM_PORT),
                 "-b", f"{int(mbit * 1000)}K", "-l", str(FLOW_PAYLOAD),
                 "-t", "100000"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.trim_mbit = mbit if self.trim is not None else 0.0
        self.actions.append({"t": round(t, 3), "trim_mbit": self.trim_mbit})

    def set_flows(self, t, group, n):
        client, _, ip, base, max_n = self.GROUPS[group]
        if n > max_n:
            self.warnings.append(f"t={t:.1f}: {group} wants {n} flows, "
                                 f"capped at {max_n}")
            n = max_n
        flows = self.flows[group]
        if n == len(flows):
            return
        host = self.net.get(client)
        while len(flows) < n:
            port = base + len(flows)
            flows.append(host.popen(
                ["iperf", "-u", "-c", ip, "-p", str(port),
                 "-b", f"{int(FLOW_UNIT_MBIT * 1000)}K",
                 "-l", str(FLOW_PAYLOAD), "-t", "100000"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        while len(flows) > n:
            # SIGKILL without waiting: SIGTERM makes iperf2 clients linger for
            # seconds sending FINs, which stalled sampling by 2-13 s in v2.0.
            # The server sockets stay bound, so no ICMP unreachable results.
            proc = flows.pop()
            try:
                proc.kill()
            except Exception:
                pass
            self.dying.append(proc)
        self.dying = [p for p in self.dying if p.poll() is None]
        self.actions.append({"t": round(t, 3), f"{group}_flows": n})

    def stop_all(self):
        if self.trim is not None:
            self.dying.append(self.trim)
            self.trim = None
        for proc in self.dying:
            _terminate(proc)
        self.dying = []
        for group in self.flows:
            while self.flows[group]:
                _terminate(self.flows[group].pop())
        while self.servers:
            _terminate(self.servers.pop())


def _terminate(proc):
    try:
        proc.terminate()
        proc.wait(timeout=1)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ============================================================
# SLEEP INHIBITOR
# ============================================================

def start_sleep_inhibitor(duration):
    global INHIBITOR_PROCESS
    try:
        INHIBITOR_PROCESS = subprocess.Popen(
            ["systemd-inhibit", "--what=idle:sleep",
             "--who=ML_NETWORK_EXPERIMENT", "--why=Telemetry collection",
             "--mode=block", "sleep", str(int(duration + 180))],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[OK] Ubuntu sleep inhibitor active.")
    except Exception as exc:
        print(f"[WARNING] Could not start sleep inhibitor: {exc}")


def stop_sleep_inhibitor():
    global INHIBITOR_PROCESS
    if INHIBITOR_PROCESS is not None:
        _terminate(INHIBITOR_PROCESS)
        INHIBITOR_PROCESS = None


# ============================================================
# NETWORK
# ============================================================

def build_network():
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import Node, OVSSwitch
    from mininet.topo import Topo

    class NetworkTopo(Topo):
        def build(self):
            s1 = self.addSwitch("s1", cls=OVSSwitch, failMode="standalone")
            s2 = self.addSwitch("s2", cls=OVSSwitch, failMode="standalone")
            r1 = self.addHost("r1", cls=Node)
            h1 = self.addHost("h1", ip="10.0.0.1/24")
            h2 = self.addHost("h2", ip="10.0.0.2/24")
            h3 = self.addHost("h3", ip="10.0.1.1/24")
            h4 = self.addHost("h4", ip="10.0.1.2/24")
            opts = dict(bw=100, delay=f"{LINK_DELAY_MS:g}ms", use_htb=True)
            for a, b in ((h1, s1), (h2, s1), (s1, r1),
                         (r1, s2), (h3, s2), (h4, s2)):
                self.addLink(a, b, **opts)

    subprocess.run(["mn", "-c"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    try:
        OVSSwitch.setup()
    except Exception as exc:
        print(f"[WARNING] OVS setup returned: {exc}")

    net = Mininet(topo=NetworkTopo(), switch=OVSSwitch, link=TCLink,
                  controller=None, autoSetMacs=True)
    net.start()

    r1 = net.get("r1")
    for cmd in ("ip addr flush dev r1-eth0", "ip addr flush dev r1-eth1",
                "ip addr add 10.0.0.254/24 dev r1-eth0",
                "ip addr add 10.0.1.254/24 dev r1-eth1",
                "ip link set r1-eth0 up", "ip link set r1-eth1 up",
                "sysctl -w net.ipv4.ip_forward=1"):
        r1.cmd(cmd)
    for name, gw in (("h1", "10.0.0.254"), ("h2", "10.0.0.254"),
                     ("h3", "10.0.1.254"), ("h4", "10.0.1.254")):
        net.get(name).cmd(f"ip route replace default via {gw}")
    return net


# ============================================================
# COLLECTION
# ============================================================

FIELDNAMES = [
    "experiment_id", "scenario", "seed", "timestamp", "elapsed_sec",
    "actual_delta_t", "sampling_delay_sec",
    # metadata about what was injected - NEVER model inputs
    "phase", "episode_id", "benign_excursion", "injected_intensity",
    "injected_bg_mbit", "injected_cross_mbit", "injected_delay_ms",
    "injected_loss_pct", "injected_rate_mbit", "injected_queue_target_bytes",
    # measurements
    "latency_ms", "latency_max_ms", "ping_replies", "ping_loss_pct",
    "rx_bytes", "tx_bytes", "rx_packets", "tx_packets",
    "rx_dropped", "tx_dropped", "rx_errors", "tx_errors",
    "rx_bytes_per_sec", "tx_bytes_per_sec",
    "rx_packets_per_sec", "tx_packets_per_sec", "tx_dropped_per_sec",
    "ingress_rx_bytes_per_sec", "ingress_rx_packets_per_sec",
    "qdisc_dropped", "qdisc_dropped_per_sec", "tbf_dropped_per_sec",
    "netem_dropped_per_sec",
    "qdisc_backlog_bytes", "qdisc_backlog_packets", "netem_backlog_packets",
    # ground truth
    "label", "anomaly_type",
]


def collect(net, scenario, run_id, seed, schedule, duration, csv_path,
            sidecar, warnings):
    r1 = net.get("r1")
    egress = get_route_interface(r1, PING_TARGET)
    ingress = get_route_interface(r1, net.get("h2").IP())
    print(f"[INFO] Router egress {egress}, ingress {ingress}.")

    actions = sidecar["actions"]
    qdisc = RouterQdisc(r1, egress, ingress, actions)
    qdisc.install_baseline()

    traffic = Traffic(net, actions, warnings)
    ping = PingMonitor()
    labeler = Labeler(scenario, schedule)
    qctrl = QueueController()
    prev_nominal = None

    try:
        traffic.start_servers()
        st0 = desired_state(schedule, scenario, 0.0)
        traffic.set_flows(0.0, "bg", st0["bg_flows"])
        ping.start(net.get("h2"))
        time.sleep(3.0)                                   # stabilise

        probe, _ = parse_counters(r1.cmd(counters_command(egress, ingress)))
        if not probe.get(f"{egress}.tx_packets"):
            raise RuntimeError("No traffic reaches the router egress.")

        baseline_samples, baseline_latency = [], nan()
        prev_stats, prev_tc, prev_time = None, None, None
        sample_number = 0
        start_mono = time.monotonic()
        start_wall = datetime.now(timezone.utc)

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()

            while not STOP_REQUESTED:
                elapsed = time.monotonic() - start_mono
                if elapsed >= duration:
                    break

                # ---- actuate -------------------------------------
                st = desired_state(schedule, scenario, elapsed)
                traffic.set_flows(elapsed, "bg", st["bg_flows"])
                if not st.get("queue_ctrl"):
                    qctrl.reset()
                    traffic.set_flows(elapsed, "cross", st["cross_flows"])
                    traffic.set_trim(elapsed, st["trim_mbit"])
                qdisc.apply(elapsed, st["delay_ms"], st["loss_pct"],
                            st["rate_mbit"])

                # ---- measure -------------------------------------
                ping.poll()
                sample_time = time.monotonic()
                stats, tc = parse_counters(
                    r1.cmd(counters_command(egress, ingress)))
                lat, lat_max, replies, loss = ping.sample(sample_time)

                dt = (sample_time - prev_time) if prev_time else nan()
                ideal = start_mono + sample_number * INTERVAL
                delay = sample_time - ideal
                if delay > MAX_SCHEDULE_LAG:
                    skipped = int(delay // INTERVAL)
                    warnings.append(f"t={elapsed:.1f}: {delay:.2f}s behind, "
                                    f"skipped {skipped} slot(s)")
                    sample_number += skipped
                    delay = sample_time - (start_mono + sample_number * INTERVAL)

                def s(name, intf=egress):
                    return stats.get(f"{intf}.{name}")

                def ps(name, intf=egress):
                    return prev_stats.get(f"{intf}.{name}") if prev_stats else None

                tbf, net_q = tc.get("10", {}), tc.get("1", {})
                ptbf = prev_tc.get("10", {}) if prev_tc else {}
                pnet = prev_tc.get("1", {}) if prev_tc else {}

                tbf_drop_rate = rate(tbf.get("dropped"), ptbf.get("dropped"), dt)
                netem_drop_rate = rate(net_q.get("dropped"),
                                       pnet.get("dropped"), dt)
                ingress_pps = rate(s("rx_packets", ingress),
                                   ps("rx_packets", ingress), dt)
                ingress_bps = rate(s("rx_bytes", ingress),
                                   ps("rx_bytes", ingress), dt)
                tx_bps = rate(s("tx_bytes"), ps("tx_bytes"), dt)
                backlog = tbf.get("backlog_bytes")
                # netem (root) counts the packets its tbf child rejects as its
                # own drops too, so netem's counter is already the total.
                # v2.0 added the two and double-counted every tbf drop.
                total_dropped = net_q.get("dropped")
                qdrop_rate = netem_drop_rate
                netem_own_rate = (max(0.0, netem_drop_rate - tbf_drop_rate)
                                  if isfinite(tbf_drop_rate)
                                  and isfinite(netem_drop_rate) else nan())

                metrics = {
                    "latency_ms": lat,
                    "ping_loss_pct": loss,
                    "router_loss_pct": (100.0 * netem_own_rate / ingress_pps
                                        if isfinite(netem_own_rate)
                                        and isfinite(ingress_pps)
                                        and ingress_pps > 100 else nan()),
                    "drop_ratio": (tbf_drop_rate / ingress_pps
                                   if isfinite(tbf_drop_rate)
                                   and isfinite(ingress_pps)
                                   and ingress_pps > 0 else nan()),
                    "queue_delay_ms": (backlog * 8 / (qdisc.rate * 1e6) * 1000
                                       if backlog is not None else nan()),
                    "shortfall": (1 - tx_bps / ingress_bps
                                  if isfinite(tx_bps) and isfinite(ingress_bps)
                                  and ingress_bps > 0 else nan()),
                }

                # ---- queue controller: act on the backlog just read ---
                # (acting on the previous tick's reading added a second of
                # loop delay, which made the controller overshoot in tests)
                if st.get("queue_ctrl"):
                    qctrl.step(st, backlog)
                    traffic.set_flows(elapsed, "cross", st["cross_flows"])
                    traffic.set_trim(elapsed, st["trim_mbit"])
                nominal_offered = (len(traffic.flows["bg"])
                                   + len(traffic.flows["cross"])) \
                    * FLOW_UNIT_MBIT + traffic.trim_mbit

                # ---- latency baseline ----------------------------
                if BASELINE_WINDOW[0] <= elapsed < BASELINE_WINDOW[1]:
                    if isfinite(lat):
                        baseline_samples.append(lat)
                elif (elapsed >= BASELINE_WINDOW[1]
                      and not isfinite(baseline_latency)
                      and "baseline_latency_ms" not in sidecar):
                    if baseline_samples:
                        baseline_latency = statistics.median(baseline_samples)
                    else:
                        warnings.append("No latency baseline; latency-based "
                                        "onset criteria disabled.")
                    sidecar["baseline_latency_ms"] = (
                        round(baseline_latency, 3)
                        if isfinite(baseline_latency) else None)
                    print(f"[INFO] Baseline latency: {baseline_latency:.2f} ms")

                label, anomaly_type = labeler.label(
                    elapsed, st, metrics, baseline_latency)

                writer.writerow({
                    "experiment_id": run_id, "scenario": scenario,
                    "seed": seed,
                    "timestamp": (start_wall + timedelta(
                        seconds=sample_time - start_mono)).isoformat(),
                    "elapsed_sec": round(sample_time - start_mono, 3),
                    "actual_delta_t": fmt(dt, 6),
                    "sampling_delay_sec": round(delay, 6),
                    "phase": st["phase"],
                    "episode_id": st["episode_id"],
                    "benign_excursion": st["excursion_id"],
                    "injected_intensity": round(st["intensity"], 4),
                    "injected_bg_mbit": st["bg_flows"] * FLOW_UNIT_MBIT,
                    "injected_cross_mbit": round(min(
                        st["cross_flows"], MAX_CROSS_FLOWS) * FLOW_UNIT_MBIT
                        + traffic.trim_mbit, 2),
                    "injected_queue_target_bytes": round(
                        st.get("queue_target_bytes", 0.0)),
                    "injected_delay_ms": st["delay_ms"],
                    "injected_loss_pct": st["loss_pct"],
                    "injected_rate_mbit": st["rate_mbit"],
                    "latency_ms": fmt(lat),
                    "latency_max_ms": fmt(lat_max),
                    "ping_replies": replies,
                    "ping_loss_pct": fmt(loss, 2),
                    **{n: fmt(s(n)) for n in STAT_NAMES},
                    "rx_bytes_per_sec": fmt(rate(s("rx_bytes"),
                                                 ps("rx_bytes"), dt)),
                    "tx_bytes_per_sec": fmt(tx_bps),
                    "rx_packets_per_sec": fmt(rate(s("rx_packets"),
                                                   ps("rx_packets"), dt)),
                    "tx_packets_per_sec": fmt(rate(s("tx_packets"),
                                                   ps("tx_packets"), dt)),
                    "tx_dropped_per_sec": fmt(rate(s("tx_dropped"),
                                                   ps("tx_dropped"), dt)),
                    "ingress_rx_bytes_per_sec": fmt(ingress_bps),
                    "ingress_rx_packets_per_sec": fmt(ingress_pps),
                    "qdisc_dropped": fmt(total_dropped),
                    "qdisc_dropped_per_sec": fmt(qdrop_rate),
                    "tbf_dropped_per_sec": fmt(tbf_drop_rate),
                    "netem_dropped_per_sec": fmt(netem_own_rate),
                    "qdisc_backlog_bytes": fmt(backlog),
                    "qdisc_backlog_packets": fmt(tbf.get("backlog_packets")),
                    "netem_backlog_packets": fmt(net_q.get("backlog_packets")),
                    "label": label, "anomaly_type": anomaly_type,
                })
                fh.flush()

                prev_stats, prev_tc, prev_time = stats, tc, sample_time
                # ingress over the last interval reflects the load set at
                # the previous tick, so compare with that.
                if isfinite(dt) and dt > MAX_SCHEDULE_LAG and prev_nominal:
                    # If traffic kept flowing during the gap only the
                    # collector was blocked; if it (nearly) stopped, the
                    # whole VM was paused by the host.
                    ratio = (ingress_bps * 8e-6 / (prev_nominal * WIRE_OVERHEAD)
                             if isfinite(ingress_bps) else nan())
                    sidecar.setdefault("stalls", []).append({
                        "t_start": round(sample_time - dt - start_mono, 3),
                        "duration_sec": round(dt, 3),
                        "traffic_ratio": fmt(ratio, 3),
                        "verdict": ("vm_paused" if isfinite(ratio)
                                    and ratio < 0.3 else "collector_blocked"),
                        "episode_id": st["episode_id"],
                        "phase": st["phase"]})
                if isfinite(ingress_bps) and prev_nominal is not None:
                    qctrl.observe(ingress_bps * 8e-6, prev_nominal)
                prev_nominal = nominal_offered
                sample_number += 1

                next_target = start_mono + sample_number * INTERVAL
                sleep = next_target - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    while next_target <= time.monotonic():
                        sample_number += 1
                        next_target = start_mono + sample_number * INTERVAL
    finally:
        ping.stop()
        traffic.stop_all()

    # ---- onset audit ----------------------------------------------
    sidecar["observed_onsets"] = {str(k): v for k, v in labeler.onsets.items()}
    sidecar["burst_onsets"] = labeler.burst_onsets
    if scenario != "normal":
        for ep in schedule["episodes"]:
            if ep_type(ep, scenario) == "traffic_burst":
                continue
            if ep["id"] not in labeler.onsets and ep["end"] <= duration:
                warnings.append(
                    f"Episode {ep['id']} never became observable; it is "
                    f"labelled 0 throughout. Check thresholds/targets.")


# ============================================================
# MAIN
# ============================================================

def write_sidecar(path, sidecar):
    with open(path, "w") as fh:
        json.dump(sidecar, fh, indent=2)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ap = argparse.ArgumentParser(
        description="Mininet anomaly telemetry experiment (v2)")
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--run", required=True, help="Experiment run identifier")
    ap.add_argument("--duration", type=int, default=None,
                    help=f"seconds (default {DEFAULT_DURATION}; mixed: "
                         f"just long enough for all episodes)")
    ap.add_argument("--rounds", type=int, default=MIXED_ROUNDS,
                    help="mixed only: how many times each anomaly type "
                         "occurs (default 1, ~25 min per round)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Schedule seed (default: derived from scenario+run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the schedule only; no Mininet needed")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else \
        seed_from_run(args.scenario, args.run)
    if args.duration is None:
        args.duration = (mixed_duration(seed, args.rounds)
                         if args.scenario == "mixed" else DEFAULT_DURATION)
    rng = random.Random(seed)
    try:
        schedule = build_schedule(args.scenario, args.duration, rng,
                                  args.rounds)
    except ValueError as exc:
        ap.error(str(exc))

    sidecar = {
        "experiment_id": args.run, "scenario": args.scenario, "seed": seed,
        "duration_sec": args.duration, "interval_sec": INTERVAL,
        "config": {k: v for k, v in globals().items()
                   if k.isupper() and isinstance(v, (int, float, str,
                                                     list, tuple))},
        "schedule": schedule, "actions": [], "warnings": [],
    }

    if args.dry_run:
        print(json.dumps({"seed": seed, "schedule": schedule}, indent=2))
        if args.scenario == "mixed":
            print(f"duration {args.duration} s; episode order: "
                  + ", ".join(e["type"] for e in schedule["episodes"]))
        return

    root_dir = DEMO_DIR if args.scenario == "mixed" else BASE_DIR
    out_dir = os.path.join(root_dir, args.scenario)
    os.makedirs(out_dir, exist_ok=True)
    final_csv = os.path.join(out_dir, f"{args.run}.csv")
    final_json = os.path.join(out_dir, f"{args.run}.json")
    if os.path.exists(final_csv):
        ap.error(f"{final_csv} already exists; raw files must not be "
                 f"overwritten. Use a new --run id.")
    # The seed comes from scenario + run id, so a reused or renamed run id
    # silently repeats a whole schedule. Refuse if any saved run has it.
    for root, _, names in [w for d in (BASE_DIR, DEMO_DIR)
                           for w in os.walk(d)]:
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, name)) as fh:
                    other = json.load(fh)
            except Exception:
                continue
            if other.get("seed") == seed:
                ap.error(f"seed {seed} is already used by "
                         f"{os.path.join(root, name)} (run id "
                         f"'{other.get('experiment_id')}'); that schedule "
                         f"would be repeated. Use a different --run id.")
    local_dir = os.path.join(LOCAL_DIR, args.scenario)
    os.makedirs(local_dir, exist_ok=True)
    csv_path = os.path.join(local_dir, f"{args.run}.csv")
    json_path = os.path.join(local_dir, f"{args.run}.json")

    print("=" * 70)
    print(f"Scenario {args.scenario} | run {args.run} | seed {seed} | "
          f"{args.duration}s | episodes {len(schedule['episodes'])} | "
          f"excursions {len(schedule['excursions'])}")
    if args.scenario == "mixed":
        print("Episode order: " + ", ".join(
            f"{e['type']}@{e['start']:.0f}s" for e in schedule["episodes"]))
    print(f"Output: {final_csv} (written locally to {csv_path} first)")
    print("=" * 70)

    write_sidecar(json_path, sidecar)          # schedule saved up front
    net = None
    try:
        start_sleep_inhibitor(args.duration)
        net = build_network()
        loss = net.ping(hosts=[net.get("h2"), net.get("h3")])
        if loss > 0:
            raise RuntimeError("Initial h2 -> h3 connectivity test failed.")
        collect(net, args.scenario, args.run, seed, schedule, args.duration,
                csv_path, sidecar, sidecar["warnings"])
    finally:
        if net is not None:
            try:
                net.stop()
            except Exception:
                pass
        stop_sleep_inhibitor()
        sidecar["finished_at"] = datetime.now(timezone.utc).isoformat()
        sidecar["interrupted"] = STOP_REQUESTED
        write_sidecar(json_path, sidecar)
        try:
            for src, dst in ((csv_path, final_csv), (json_path, final_json)):
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            print(f"[OK] Copied to {out_dir}")
        except Exception as exc:
            print(f"[WARNING] Copy to {out_dir} failed ({exc}); files are "
                  f"still in {local_dir}")
        for w in sidecar["warnings"]:
            print(f"[WARNING] {w}")
        print("[OK] Done.")


if __name__ == "__main__":
    main()
