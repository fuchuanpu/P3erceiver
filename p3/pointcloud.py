#!/usr/bin/env python3
"""P3receiver point-cloud representation (contribution C1).

SAMPLING-CLOSED per-packet tokens: every channel is a function of that
packet's INTRINSIC attributes only (arrival time, size, direction, protocol),
never of sequence context. Hence for any sampled subset S of a flow F the
token set satisfies  Phi(S) = {phi(p) : p in S}  subset-of  Phi(F)
(exact up to a global time re-anchor when the true first packet is dropped):
surviving packets keep bit-identical tokens under packet sampling.

The init/ v2 token (15-dim) violated this in five channels (log_iat,
norm_pos, cum_bytes_frac, pseudo_direction, size_delta_norm) plus the
model-side sinusoidal INDEX positional encoding. The relational information
those channels carried is recoverable from pairwise differences of the
continuous time coordinate — which is the attention mechanism's job (C2),
not the tokenizer's. `contextual=True` re-adds them ONLY for the
representation-ablation experiment.
"""
from __future__ import annotations

import os

import numpy as np

# Optional RATE-CONSISTENT ESTIMATOR channel (P3_EST_IAT=1): the time gap
# between consecutive SURVIVING packets is a MEASUREMENT of the missing
# mass, not a corrupted IAT. With known sampling rate p the local true IAT
# is estimated as gap * p — exact for systematic 1/k sampling (a survivor
# gap spans exactly k true packets), unbiased in expectation for iid. At
# p=1 the channel IS the true IAT, recovering the relational information
# the closed token set dropped. Strict bit-closure is traded for estimator
# CONSISTENCY (expectation invariant across rates). Opt-in via env var so
# 10-dim checkpoints stay loadable.
EST_IAT = os.environ.get("P3_EST_IAT", "0") == "1"
# ── tokenizer ablation toggles ────────────────────────────────────────
ABL_CONTEXTUAL = os.environ.get("P3_CONTEXTUAL", "0") == "1"  # sampling-fragile
ABL_NO_DIR     = os.environ.get("P3_NO_DIR", "0") == "1"      # drop direction
ABL_NO_BUCKETS = os.environ.get("P3_NO_BUCKETS", "0") == "1"  # drop size buckets

# ── token layout (closed channels) ──────────────────────────────────────
#   idx  channel        meaning
#   0    size_norm      pkt_len / 1500                (size, linear)
#   1    log_length     log1p(pkt_len) / 8            (size, log)   ← Fourier
#   2    log_rel_time   log1p(t - t0)                 (arrival time)← Fourier
#   3    direction      +1 initiator→responder, −1 reverse, 0 unknown
#   4    is_udp         protocol flag
#   5-9  size bucket    one-hot {≤64,64-300,300-700,700-1300,≥1300}
FEATURE_NAMES: list[str] = [
    "size_norm", "log_length", "log_rel_time", "direction", "is_udp",
    "bucket_le64", "bucket_64_300", "bucket_300_700", "bucket_700_1300",
    "bucket_ge1300",
] + (["log_est_iat"] if EST_IAT else [])
D_IN: int = len(FEATURE_NAMES)
CH_EST_IAT = 10                     # valid only when EST_IAT

# ablation-only contextual channels (appended after the closed block)
CONTEXTUAL_NAMES: list[str] = ["log_iat", "norm_pos", "cum_bytes_frac",
                               "size_delta_norm"]
D_CTX: int = D_IN + len(CONTEXTUAL_NAMES)

CH_SIZE_NORM = 0
CH_LOG_LEN   = 1
CH_LOG_RELT  = 2
CH_DIR       = 3
FOURIER_COORDS = (CH_LOG_LEN, CH_LOG_RELT)

MAX_PKTS = 192                      # encoder KV budget (same as init v2)
_SIZE_BUCKETS = (64.0, 300.0, 700.0, 1300.0)

# Time coordinate unit: MILLISECONDS. With seconds, the sub-second handshake
# packets of short web flows all collapse near log1p(0) and the Fourier
# encoding cannot resolve their ORDER (the strongest downstream signal).
# Millisecond units spread the early packets across the log axis:
# log1p(1ms)=0.69 vs log1p(5ms)=1.79, versus 0.001 vs 0.005 in seconds.
TIME_UNIT = 1000.0


def tokens(lens: np.ndarray, ts: np.ndarray, dirs: np.ndarray | None = None,
           is_udp: float = 0.0, *, contextual: bool = False) -> np.ndarray:
    """(lens, ts[, dirs]) of ONE flow -> (n, D) float32 token matrix.

    `ts` may be absolute or relative; it is re-anchored to the first
    OBSERVED packet. `dirs` in {+1,-1} per packet, or None -> 0 (unknown;
    matches datasets without a per-packet direction field)."""
    lens = np.asarray(lens, dtype=np.float64)
    ts   = np.asarray(ts,   dtype=np.float64)
    n = int(min(lens.size, ts.size))
    D = D_CTX if contextual else D_IN
    if n == 0:
        return np.zeros((0, D), dtype=np.float32)
    if n > MAX_PKTS:                # keep the first M (handshake-rich) packets
        lens, ts = lens[:MAX_PKTS], ts[:MAX_PKTS]
        if dirs is not None:
            dirs = np.asarray(dirs)[:MAX_PKTS]
        n = MAX_PKTS

    rel = (ts - ts[0]) * TIME_UNIT
    rel[rel < 0] = 0.0

    b_idx = np.digitize(lens, _SIZE_BUCKETS)          # 0..4
    onehot = np.zeros((n, 5), dtype=np.float64)
    onehot[np.arange(n), b_idx] = 1.0

    contextual = contextual or ABL_CONTEXTUAL
    D = D_CTX if contextual else D_IN
    pc = np.zeros((n, D), dtype=np.float64)
    pc[:, CH_SIZE_NORM] = lens / 1500.0
    pc[:, CH_LOG_LEN]   = np.log1p(lens) / 8.0
    pc[:, CH_LOG_RELT]  = np.log1p(rel)
    if dirs is not None and not ABL_NO_DIR:
        pc[:, CH_DIR] = np.asarray(dirs, dtype=np.float64)[:n]
    pc[:, 4]    = is_udp
    if not ABL_NO_BUCKETS:
        pc[:, 5:10] = onehot
    if EST_IAT:                     # at build time (rate=1) this IS the true IAT
        gap = np.zeros(n)
        if n >= 2:
            gap[1:] = np.maximum(np.diff(rel), 0.0)
        pc[:, CH_EST_IAT] = np.log1p(gap)

    if contextual:                  # ablation block — sampling-FRAGILE on purpose
        c = D_IN
        iat = np.zeros(n); iat[1:] = np.maximum(np.diff(ts), 0.0) * TIME_UNIT
        pc[:, c] = np.log1p(iat)
        pc[:, c + 1] = np.arange(n) / (n - 1) if n >= 2 else 0.0
        total = lens.sum()
        pc[:, c + 2] = np.cumsum(lens) / total if total > 0 else 0.0
        delta = np.zeros(n); delta[1:] = np.diff(lens)
        pc[:, c + 3] = delta / 1500.0
    return pc.astype(np.float32)


def tokens_from_record(rec: dict, *, contextual: bool = False) -> np.ndarray:
    """CSTNet-style preprocessed record (pkt_lens/ts/protocol, no per-packet
    direction) -> token matrix. Direction channel stays 0 (unknown); the
    encoder tolerates this via direction-dropout during pretraining."""
    return tokens(rec.get("pkt_lens", []), rec.get("ts", []), None,
                  is_udp=1.0 if rec.get("protocol") == "udp" else 0.0,
                  contextual=contextual)


# ── sampling operators (the perturbation we are robust to) ──────────────
def sample_mask(n: int, rate: float, rng: np.random.Generator,
                mode: str = "iid", keep_min: int = 2) -> np.ndarray:
    """Boolean keep-mask over n packets simulating router packet sampling.

    mode='iid'        : Bernoulli(rate) per packet (random sampling)
    mode='systematic' : every k-th packet, k = round(1/rate), random phase
                        (deterministic sampled NetFlow/sFlow)
    Guarantees >= keep_min survivors (a flow record with zero packets is
    never emitted by a sampler)."""
    if rate >= 1.0 or n <= keep_min:
        return np.ones(n, dtype=bool)
    if mode == "systematic":
        k = max(1, int(round(1.0 / rate)))
        keep = np.zeros(n, dtype=bool)
        keep[int(rng.integers(k)) :: k] = True
    else:
        keep = rng.random(n) < rate
    short = keep_min - int(keep.sum())
    if short > 0:
        idx = rng.choice(np.flatnonzero(~keep), size=short, replace=False)
        keep[idx] = True
    return keep


def eval_view(pc: np.ndarray, rate: float, rng,
              mode: str = "iid") -> tuple[np.ndarray, float]:
    """Draw the standard eval-side sampled view of a flow.

    Returns (sampled_cloud, effective_rate).  When the drawn mask retains
    EVERY packet (tiny flows clamped by keep_min, or chance full survival)
    the observation is complete, so the flow is conditioned at rate 1 —
    matching the training-time gpu_views fallback, which restores the full
    view with r=1 whenever fewer than 2 packets survive.  Conditioning a
    complete observation on the link rate instead (est_iat x rate, HT n/rate)
    fabricates a population that was never dropped; on single-packet-flow
    attacks (LR scans, reflection DoS) that bias alone flips every
    prediction to benign."""
    keep = sample_mask(pc.shape[0], rate, rng, mode=mode, keep_min=2)
    eff = 1.0 if bool(keep.all()) else rate
    return apply_sampling(pc, keep, eff), eff


def apply_sampling(pc: np.ndarray, keep: np.ndarray,
                   rate: float = 1.0) -> np.ndarray:
    """Subset the cloud and re-anchor time to the first survivor. Closed
    channels stay bit-identical (up to the global time re-anchor of ch 2);
    the optional EST_IAT channel is RECOMPUTED from survivor gaps with the
    known-rate correction  est_true_iat = gap * rate."""
    sub = pc[keep].copy()
    if sub.shape[0]:
        rel = np.expm1(sub[:, CH_LOG_RELT])
        rel = np.maximum(rel - rel[0], 0.0)
        sub[:, CH_LOG_RELT] = np.log1p(rel)
        if EST_IAT:
            gap = np.zeros(sub.shape[0])
            if sub.shape[0] >= 2:
                gap[1:] = np.maximum(np.diff(rel), 0.0) * float(rate)
            sub[:, CH_EST_IAT] = np.log1p(gap)
    return sub


# ── batching ─────────────────────────────────────────────────────────────
def pad_stack(clouds: list[np.ndarray], max_pkts: int = MAX_PKTS):
    """list of (n_i, D) -> RAW (N, M, D) float32 + (N, M) bool mask."""
    D = clouds[0].shape[1] if clouds else D_IN
    X = np.zeros((len(clouds), max_pkts, D), dtype=np.float32)
    M = np.zeros((len(clouds), max_pkts), dtype=bool)
    for i, pc in enumerate(clouds):
        n = min(pc.shape[0], max_pkts)
        if n:
            X[i, :n] = pc[:n]
            M[i, :n] = True
    return X, M


def context_summary(pc: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Observation-context vector (log n, log dur, log bytes) — fed to the
    encoder as ONE extra KV token so the model can calibrate to the sampling
    intensity it is looking at (C2 design element #3).

    RATE CONDITIONING: a router's sampling rate 1/k is a KNOWN configuration
    value, so when `rate` is given the observed count/bytes are inflated to
    POPULATION estimates (n/rate, bytes/rate) — a Horvitz-Thompson style
    correction. Duration needs no correction (the observed span is an
    almost-unbiased estimate of the true span). Keeps the 3-dim contract, so
    pretrained encoders load unchanged and the values stay in-distribution
    (they read as a denser flow's statistics)."""
    n = pc.shape[0]
    if n == 0:
        return np.zeros(3, dtype=np.float32)
    dur   = float(np.expm1(pc[:, CH_LOG_RELT]).max())
    byts  = float(pc[:, CH_SIZE_NORM].sum() * 1500.0)
    r = max(float(rate), 1e-6)
    return np.array([np.log1p(n / r), np.log1p(dur), np.log1p(byts / r) / 16.0],
                    dtype=np.float32)
