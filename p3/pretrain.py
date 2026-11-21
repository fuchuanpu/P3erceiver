#!/usr/bin/env python3
"""P3receiver self-supervised pretraining (contribution C3, objective side).

Two pretext tasks over unlabeled backbone flows, jointly optimised
(L = L_SCC + LAMBDA_MPM * L_MPM):

  SCC  Sampling-Consistency Contrastive — two independent random packet-
       sampled views of the same flow (iid Bernoulli or systematic 1/k,
       mimicking sampled NetFlow/sFlow) are pulled together with NT-Xent
       against in-batch negatives: a sampled sub-cloud must embed where the
       full cloud embeds.
  MPM  Masked Point Modeling — inside view A a further 30% of points are
       hidden from the encoder; the coordinate-query decoder must
       reconstruct each hidden packet's (log-size, direction, size-bucket)
       from its TIME coordinate alone, forcing the latents to internalise
       flow dynamics (handshake rhythm, burst structure).

Direction dropout (p=0.3 per view) makes the encoder tolerate datasets
without a per-packet direction field (e.g. CSTNet preprocessed records).

Usage:
  python pretrain.py --corpus data/mawi_preprocessed/202201011400 [--epochs 20]
      [--batch 256] [--max-steps N] [--tag mawi] [--lr 1e-3]
  --corpus accepts a preprocessed-JSON shard directory (canonical), a single
  .json file, or a legacy .npz corpus.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcloud import (D_IN, CH_DIR, CH_LOG_LEN, CH_LOG_RELT, MAX_PKTS,
                        tokens, sample_mask, apply_sampling, pad_stack,
                        context_summary)
from model import (PerceiverEncoder, ProjectionHead, PointDecoder,
                   config_sig, count_params)
from pretrain_data import load_flows

# ── pretraining hyperparameters ──────────────────────────────────────────
RATE_LO, RATE_HI = 0.25, 1.0      # per-view keep-rate ~ U[RATE_LO, RATE_HI]
# v2 ASYMMETRIC views: one dense, one heavily decimated. Symmetric-rate
# positives are matchable from gross flow statistics alone; a dense/sparse
# pair forces the embedding to align fine structure with its decimated
# shadow instead.
RATE_A_LO, RATE_A_HI = 0.50, 1.00
RATE_B_LO, RATE_B_HI = 0.15, 0.60
# v2 TIME-SCALE augmentation: per-view global speed s ~ logU[1/S, S] on the
# linear timeline. Kills the flow-duration shortcut in SCC and bridges the
# duration domain gap between backbone pretraining flows (median ~7s) and
# short downstream web flows (median ~0.9s).
TIME_SCALE_S     = 4.0
P_SYSTEMATIC     = 0.5            # systematic (1/k) vs iid sampling per view
MPM_RATIO        = 0.30           # fraction of view-A points hidden for MPM
K_MAX            = 48             # decoder query budget per flow
# v3 (anti-missing-data escalation, --v3 implies --v2):
#   - sparse view down to 5% keep-rate (train-time coverage of extreme
#     decimation, matching the 1/32 eval floor)
#   - MPM masking 30% -> 65% (MAE-regime: reconstruction from few points
#     forces a stronger generative prior over flow dynamics)
#   - population-extrapolation aux task: predict the flow's TRUE
#     (log n_total, log duration) from a sampled view's pooled embedding —
#     free supervision from the corpus, teaches sample->population inference
RATE_B_LO_V3     = 0.05
MPM_RATIO_V3     = 0.65
K_MAX_V3         = 96
W_AUX            = 0.3
P_DIR_DROP       = 0.30           # direction-channel dropout per view
NTX_TEMP         = 0.10           # NT-Xent temperature
LAMBDA_MPM       = 1.0
W_LEN, W_DIR, W_BKT = 1.0, 0.5, 0.5

# ── ablation toggles (env vars, default off) ──────────────────────────
ABL_NO_SCC = os.environ.get("P3_NO_SCC", "0") == "1"   # drop contrastive
ABL_NO_MPM = os.environ.get("P3_NO_MPM", "0") == "1"   # drop masked modeling
ABL_NO_AUX = os.environ.get("P3_NO_AUX", "0") == "1"   # drop population aux

LR           = 1e-3
WEIGHT_DECAY = 1e-2
WARMUP_FRAC  = 0.05
GRAD_CLIP    = 1.0

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── batch construction ───────────────────────────────────────────────────
def _make_view(pc: np.ndarray, rng: np.random.Generator,
               rate_lo: float = RATE_LO, rate_hi: float = RATE_HI,
               time_scale: float = 0.0) -> np.ndarray:
    mode = "systematic" if rng.random() < P_SYSTEMATIC else "iid"
    rate = rng.uniform(rate_lo, rate_hi)
    keep = sample_mask(pc.shape[0], rate, rng, mode=mode, keep_min=4)
    sub = apply_sampling(pc, keep, rate)
    if sub.shape[0] > MAX_PKTS:
        sub = sub[:MAX_PKTS]
    sub = sub.copy()
    if time_scale > 1.0:            # exact log-domain global speed change
        s = np.exp(rng.uniform(-np.log(time_scale), np.log(time_scale)))
        sub[:, CH_LOG_RELT] = np.log1p(s * np.expm1(sub[:, CH_LOG_RELT]))
    if rng.random() < P_DIR_DROP:
        sub[:, CH_DIR] = 0.0
    return sub


def collate(clouds: list[np.ndarray], rng: np.random.Generator,
            v2: bool = False):
    """B full clouds -> two sampled views + MPM targets on view A.

    View A keeps ALL its rows in the tensor but the encoder mask hides the
    MPM target rows; the anchor point (index 0) is never hidden so the
    view's time re-anchor stays valid for the decoder queries.

    v2: asymmetric dense/sparse views + time-scale augmentation + ZERO
    context token (the observation-context statistics are an identity
    shortcut for the contrastive task — withheld during pretraining,
    reintroduced at finetune time)."""
    B = len(clouds)
    if v2:
        ts_aug = TIME_SCALE_S
        va = [_make_view(pc, rng, RATE_A_LO, RATE_A_HI, ts_aug) for pc in clouds]
        vb = [_make_view(pc, rng, RATE_B_LO, RATE_B_HI, ts_aug) for pc in clouds]
    else:
        va = [_make_view(pc, rng) for pc in clouds]
        vb = [_make_view(pc, rng) for pc in clouds]

    XA, MA = pad_stack(va)
    XB, MB = pad_stack(vb)
    vis_A = MA.copy()

    t_q   = np.zeros((B, K_MAX), dtype=np.float32)
    y_len = np.zeros((B, K_MAX), dtype=np.float32)
    y_dir = np.zeros((B, K_MAX), dtype=np.int64)
    y_bkt = np.zeros((B, K_MAX), dtype=np.int64)
    q_msk = np.zeros((B, K_MAX), dtype=bool)
    d_msk = np.zeros((B, K_MAX), dtype=bool)

    for i, pc in enumerate(va):
        n = min(pc.shape[0], MAX_PKTS)
        k = min(int(round(n * MPM_RATIO)), K_MAX)
        if n < 6 or k == 0:
            continue
        tgt = rng.choice(np.arange(1, n), size=min(k, n - 1), replace=False)
        vis_A[i, tgt] = False                      # hide from encoder
        kk = tgt.size
        t_q[i, :kk]   = pc[tgt, CH_LOG_RELT]
        y_len[i, :kk] = pc[tgt, CH_LOG_LEN]
        dirs          = pc[tgt, CH_DIR]
        y_dir[i, :kk] = (dirs > 0).astype(np.int64)
        d_msk[i, :kk] = dirs != 0                  # dir may be dropped/unknown
        y_bkt[i, :kk] = pc[tgt, 5:10].argmax(axis=1)
        q_msk[i, :kk] = True

    if v2:
        ctx_A = np.zeros((B, 3), dtype=np.float32)
        ctx_B = np.zeros((B, 3), dtype=np.float32)
    else:
        ctx_A = np.stack([context_summary(x[m]) for x, m in zip(XA, vis_A)])
        ctx_B = np.stack([context_summary(x[m]) for x, m in zip(XB, MB)])
    to = lambda a, dt: torch.as_tensor(a, dtype=dt, device=_DEVICE)
    return (to(XA, torch.float32), to(vis_A, torch.bool), to(ctx_A, torch.float32),
            to(XB, torch.float32), to(MB, torch.bool), to(ctx_B, torch.float32),
            to(t_q, torch.float32), to(y_len, torch.float32),
            to(y_dir, torch.int64), to(y_bkt, torch.int64),
            to(q_msk, torch.bool), to(d_msk, torch.bool))


def nt_xent(za: torch.Tensor, zb: torch.Tensor, temp: float = NTX_TEMP):
    """SimCLR NT-Xent over 2B L2-normalised projections; returns (loss, acc)."""
    B = za.size(0)
    z = torch.cat([za, zb], dim=0)
    sim = z @ z.t() / temp
    sim.fill_diagonal_(float("-inf"))
    target = torch.arange(2 * B, device=z.device)
    target = (target + B) % (2 * B)                # positive = the other view
    loss = F.cross_entropy(sim, target)
    acc = (sim.argmax(dim=1) == target).float().mean()
    return loss, acc


# ── training ─────────────────────────────────────────────────────────────
def pretrain(corpus_path: Path, *, epochs: int, batch: int, lr: float,
             tag: str, max_steps: int | None, seed: int,
             v2: bool = False, v3: bool = False) -> Path:
    global RATE_B_LO, MPM_RATIO, K_MAX
    if v3:
        v2 = True
        RATE_B_LO, MPM_RATIO, K_MAX = RATE_B_LO_V3, MPM_RATIO_V3, K_MAX_V3
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    t0 = time.time()
    flows = load_flows(corpus_path)
    keep = [i for i, f in enumerate(flows) if len(f["lens"]) >= 8]
    clouds = [tokens(flows[i]["lens"], flows[i]["ts"], flows[i]["dirs"],
                     is_udp=flows[i]["is_udp"]) for i in keep]
    # population-extrapolation targets: true packet count + true duration(ms)
    aux_t = np.array([[np.log1p(flows[i]["n_total"]),
                       np.log1p((flows[i]["ts"][-1] - flows[i]["ts"][0]) * 1e3)]
                      for i in keep], dtype=np.float32)
    print(f"[data] {len(clouds):,} flows tokenised "
          f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    allpts = np.concatenate([pc for pc in clouds], axis=0)
    norm_mean, norm_std = allpts.mean(0), allpts.std(0)
    del allpts

    enc = PerceiverEncoder().to(_DEVICE)
    enc.set_norm(norm_mean, norm_std)
    proj = ProjectionHead().to(_DEVICE)
    dec = PointDecoder().to(_DEVICE)
    aux = nn.Sequential(nn.Linear(pool_dim := enc.latents.shape[1], 64),
                        nn.GELU(), nn.Linear(64, 2)).to(_DEVICE) if v3 else None
    params = list(enc.parameters()) + list(proj.parameters()) + list(dec.parameters())
    if aux is not None:
        params += list(aux.parameters())
        aux_all_g = torch.as_tensor(aux_t, device=_DEVICE)
    n_par = count_params(enc) + count_params(proj) + count_params(dec)
    print(f"[model] params enc={count_params(enc):,} total={n_par:,} "
          f"device={_DEVICE.type}", file=sys.stderr, flush=True)

    steps_per_epoch = len(clouds) // batch
    total_steps = min(steps_per_epoch * epochs,
                      max_steps if max_steps else 10**9)
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup
        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup))))
    use_amp = _DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    step = 0
    enc.train(); proj.train(); dec.train()
    for ep in range(epochs):
        order = rng.permutation(len(clouds))
        for bi in range(steps_per_epoch):
            idx = order[bi * batch:(bi + 1) * batch]
            (XA, MA, cA, XB, MB, cB,
             t_q, y_len, y_dir, y_bkt, q_msk, d_msk) = collate(
                [clouds[j] for j in idx], rng, v2=v2)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                lat_A, pool_A = enc(XA, MA, cA)
                _, pool_B = enc(XB, MB, cB)
                loss_nce, nce_acc = nt_xent(proj(pool_A), proj(pool_B))

                p_len, p_dir, p_bkt = dec(lat_A, t_q)
                qm = q_msk.float()
                n_q = qm.sum().clamp(min=1)
                l_len = (F.smooth_l1_loss(p_len, y_len, reduction="none")
                         * qm).sum() / n_q
                l_bkt = (F.cross_entropy(p_bkt.transpose(1, 2), y_bkt,
                                         reduction="none") * qm).sum() / n_q
                dm = (q_msk & d_msk).float()
                l_dir = (F.cross_entropy(p_dir.transpose(1, 2), y_dir,
                                         reduction="none") * dm).sum() / dm.sum().clamp(min=1)
                loss_mpm = W_LEN * l_len + W_DIR * l_dir + W_BKT * l_bkt
                loss = ((0.0 if ABL_NO_SCC else loss_nce)
                        + (0.0 if ABL_NO_MPM else LAMBDA_MPM * loss_mpm))
                l_aux = torch.zeros((), device=_DEVICE)
                if aux is not None and not ABL_NO_AUX:   # population extrapolation
                    tgt = aux_all_g[torch.as_tensor(idx, device=_DEVICE)]
                    l_aux = (F.smooth_l1_loss(aux(pool_A), tgt)
                             + F.smooth_l1_loss(aux(pool_B), tgt))
                    loss = loss + W_AUX * l_aux

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0 or step == total_steps:
                print(f"[pretrain] ep{ep+1} step {step}/{total_steps}  "
                      f"nce={loss_nce.item():.4f} (acc={nce_acc.item():.3f})  "
                      f"mpm={loss_mpm.item():.4f} "
                      f"(len={l_len.item():.4f} dir={l_dir.item():.4f} "
                      f"bkt={l_bkt.item():.4f})  aux={l_aux.item():.4f}  "
                      f"lr={sched.get_last_lr()[0]:.2e}",
                      file=sys.stderr, flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    ckpt_dir = Path(__file__).resolve().parent / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    out = ckpt_dir / f"_ckpt_pretrain_{tag}.pt"
    torch.save({"sig": config_sig(), "encoder": enc.state_dict(),
                "proj": proj.state_dict(), "decoder": dec.state_dict(),
                "corpus": str(corpus_path), "steps": step, "seed": seed,
                "v2": v2, "v3": v3,
                **({"aux": aux.state_dict()} if aux is not None else {})}, out)
    print(f"[done] {step} steps in {time.time()-t0:.0f}s -> {out}",
          file=sys.stderr, flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--tag", default="mawi")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--v2", action="store_true",
                   help="asymmetric views + time-scale aug + no ctx token")
    p.add_argument("--v3", action="store_true",
                   help="v2 + 5%% sparse views + 65%% MPM + population-"
                        "extrapolation aux head")
    a = p.parse_args()
    pretrain(a.corpus, epochs=a.epochs, batch=a.batch, lr=a.lr, tag=a.tag,
             max_steps=a.max_steps, seed=a.seed, v2=a.v2, v3=a.v3)


if __name__ == "__main__":
    main()
