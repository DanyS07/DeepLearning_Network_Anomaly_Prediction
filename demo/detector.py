"""
Rule-based anomaly DETECTOR (what is wrong right now), used by the demo.

The deep models PREDICT (an anomaly will start within 10 s). Once an
anomaly has started, a prediction is no longer useful; the network operator
needs to know that it is happening and what kind it is. That is done here
with the same physical thresholds the collector uses to decide an onset, so
the red "ANOMALY" banner in the demo comes from measurements only and not
from the ground-truth label.

Rules (per second, measured at the router and by ping):
    RTT          >= baseline + 50 ms
    router loss  >= 5 %     random loss inside the router (netem)
    drop ratio   >= 0.02    queue overflow drops (tbf)
    queue delay  >= 50 ms   router backlog expressed as delay
    shortfall    >= 0.10    fraction of offered bytes not delivered
RTT and shortfall alone must hold for 2 consecutive seconds (single-second
spikes come from sampling jitter). A detection stays on for HOLD_SEC after
the last rule fired, so a single quiet second does not switch it off.

The TYPE of each second is inferred from which signals fire (first match
wins); the type shown for a detection is the most frequent one so far
(a burst-shaped jump at any point makes it a traffic burst):
    packet_loss            router loss is high
    traffic_burst          overload that appeared within 2 s
                           (offered load jumped by >= 25 Mbit/s)
    bandwidth_degradation  the link delivers < 45 Mbit/s and less than
                           is offered, with queueing or drops
                           (the link itself got slower)
    congestion             drops, or offered load >= 1.06 x capacity
    queue_buildup          large queue without drops
    high_latency           RTT up with an empty queue (delay on the path)
"""

import numpy as np
import pandas as pd

NOMINAL_MBIT = 50.0
LAT_RISE_MS = 50.0
LOSS_PCT = 5.0
DROP_RATIO = 0.02
QUEUE_MS = 50.0
SHORTFALL = 0.10
HOLD_SEC = 3
MIN_EVIDENCE = 4          # firing seconds needed before a type is shown
BURST_JUMP_MBIT = 25.0

TYPES = ["identifying", "normal", "congestion", "high_latency", "packet_loss",
         "bandwidth_degradation", "queue_buildup", "traffic_burst"]


def signals(sig):
    """Per-second measurements the rules need (from Stage 1 columns)."""
    s = pd.DataFrame(index=sig.index)
    pps = sig["ingress_rx_packets_per_sec"]
    in_bps = sig["ingress_rx_bytes_per_sec"]
    lat = sig["latency_ms"].ffill()
    s["rtt_rise_ms"] = (lat - sig["baseline_latency_ms"]).fillna(0)
    s["router_loss_pct"] = (100 * sig["netem_dropped_per_sec"] / pps).where(
        pps > 100).fillna(0)
    s["drop_ratio"] = (sig["tbf_dropped_per_sec"] / pps).where(pps > 0).fillna(0)
    s["queue_ms"] = (sig["qdisc_backlog_bytes"].fillna(0) * 8
                     / (NOMINAL_MBIT * 1e6) * 1000)
    s["offered_mbit"] = (in_bps * 8 / 1e6).fillna(0)
    s["delivered_mbit"] = (sig["tx_bytes_per_sec"] * 8 / 1e6).fillna(0)
    s["shortfall"] = (1 - sig["tx_bytes_per_sec"] / in_bps).where(
        in_bps > 0).fillna(0).clip(-1, 1)
    return s


def detect(sig):
    """Returns DataFrame (same index) with columns
        detected  0/1
        det_type  inferred class name ('normal' when not detected)
        reason    which rules fired (text)
    sig: one run, sorted by elapsed_sec, Stage 1 columns.
    """
    s = signals(sig)
    lat = s["rtt_rise_ms"] >= LAT_RISE_MS
    loss = s["router_loss_pct"] >= LOSS_PCT
    drop = s["drop_ratio"] >= DROP_RATIO
    queue = s["queue_ms"] >= QUEUE_MS
    short = s["shortfall"] >= SHORTFALL
    # RTT and shortfall alone are noisy: one late sample splits a second's
    # bytes over two rows (shortfall +0.1 then -0.1) and a single slow ping
    # spikes the RTT. Alone they must hold for 2 consecutive seconds.
    weak = lat | short
    fired = loss | drop | queue | (weak & weak.shift(1, fill_value=False))

    # 1.06 x capacity: the queue_buildup controller keeps the offered load
    # at <= ~1.03 x capacity, congestion and bursts go well above it
    overload = s["offered_mbit"] >= 1.06 * NOMINAL_MBIT
    jump = (s["offered_mbit"] - s["offered_mbit"].shift(2).fillna(0)
            ) >= BURST_JUMP_MBIT
    # the link delivers clearly less than its 50 Mbit/s although there is
    # queueing / loss of delivery -> the link itself got slower
    slow_link = ((s["delivered_mbit"] < 0.9 * NOMINAL_MBIT)
                 & (s["delivered_mbit"] < 0.9 * s["offered_mbit"])
                 & (queue | drop | short)
                 # not while the load is shooting up (start of a burst)
                 & ((s["offered_mbit"] - s["offered_mbit"].shift(1)
                     .fillna(0)) < 5.0))
    # a slower link stays slow: require 2 consecutive seconds, so a single
    # delayed sample (start of a delay rise or of a burst) is not taken
    # for a bandwidth drop
    slow_link = slow_link & slow_link.shift(1, fill_value=False)

    kind = np.select(
        [loss,
         overload & jump,
         slow_link,
         drop | overload,
         queue,
         lat],
        ["packet_loss", "traffic_burst", "bandwidth_degradation",
         "congestion", "queue_buildup", "high_latency"],
        default="")
    # anything that fired but matched no type (e.g. a shortfall spike)
    kind = np.where(fired.to_numpy() & (kind == ""), "congestion", kind)

    names = ["RTT", "loss", "drops", "queue", "shortfall"]
    flags = np.stack([lat, loss, drop, queue, short], axis=1)
    reason = [", ".join(n for n, f in zip(names, row) if f) for row in flags]

    # Hold: stay on for HOLD_SEC after the last firing second.
    # Type of a detection (causal - uses only the seconds seen so far):
    #   a burst-shaped jump at any point -> traffic_burst
    #   otherwise the most frequent per-second type so far
    #   but no type is committed until MIN_EVIDENCE firing seconds agree
    #   (shown as "identifying"), so the first, ambiguous second of an
    #   anomaly never shows a wrong type
    n = len(sig)
    det = np.zeros(n, int)
    typ = np.array(["normal"] * n, dtype=object)
    rsn = np.array([""] * n, dtype=object)
    last_fire, votes = -10 ** 9, {}
    f = fired.to_numpy()
    for i in range(n):
        if f[i]:
            if i - last_fire > HOLD_SEC:        # a new detection starts
                votes = {}
            votes[kind[i]] = votes.get(kind[i], 0) + 1
            last_fire = i
        if i - last_fire <= HOLD_SEC:
            det[i] = 1
            if "traffic_burst" in votes:
                typ[i] = "traffic_burst"
            elif sum(votes.values()) < MIN_EVIDENCE:
                typ[i] = "identifying"
            else:
                top = max(votes.values())
                tied = [k for k, v in votes.items() if v == top]
                typ[i] = kind[last_fire] if kind[last_fire] in tied else tied[0]
            rsn[i] = reason[i] if f[i] else "holding"
    return pd.DataFrame({"detected": det, "det_type": typ, "reason": rsn},
                        index=sig.index)
