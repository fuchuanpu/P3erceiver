#!/usr/bin/env python3
"""Downstream transfer + sampling-robustness evaluation for P3receiver.

Loads the CSTNet-TLS-1.3 split (init/'s 00_split.json), initialises the
encoder from a pretraining checkpoint (or from scratch for the ablation),
finetunes the 120-class head, and evaluates accuracy / macro-F1 under
packet sampling rates {1, 1/2, 1/4, 1/8, 1/16, 1/32} x {iid, systematic}.

Usage:
  python finetune.py --split ../init/20260624-195147_CSTNet_opus_perceiver_v2/00_split.json \
      [--pretrained _ckpt_pretrain_mawi.pt] [--epochs 120] [--sample-aug]
      [--tag ft] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from pointcloud import (MAX_PKTS, EST_IAT, CH_EST_IAT, CH_LOG_RELT,
                        CH_SIZE_NORM, tokens_from_record, sample_mask,
                        apply_sampling, eval_view, pad_stack, context_summary)
from model import P3Classifier, config_sig, count_params

LR           = 8e-4
WEIGHT_DECAY = 4e-2
LABEL_SMOOTH = 0.05
WARMUP_FRAC  = 0.10
GRAD_CLIP    = 1.0
BATCH        = 512
AUG_RATE_MIN = 0.03125       # --sample-aug: keep-rate ~ logU[1/32, 1] per flow
LAMBDA_CONS  = 1.0           # --consistency: KL(sampled || full.detach()) weight

EVAL_RATES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
EVAL_MODES = ("iid", "systematic")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── data ─────────────────────────────────────────────────────────────────
def load_side(split_path: Path, side: str):
    meta = json.loads(split_path.read_text())
    cache: dict[str, list] = {}
    clouds, ys = [], []
    for s in meta["samples"][side]:
        jp = s["json_path"]
        if jp not in cache:
            cache[jp] = json.loads(Path(jp).read_text())
        rec = cache[jp][int(s["rec_idx"])]
        clouds.append(tokens_from_record(rec))
        ys.append(int(s["class_id"]))
    return clouds, np.asarray(ys, dtype=np.int64)


def batch_tensors(clouds, rng=None, rates=None):
    """Pad/stack + rate-conditioned context to device.

    rng given  -> log-uniform sampling augmentation, keep-rate per flow
                  ~ logU[AUG_RATE_MIN, 1] (covers the whole eval range).
    rates      -> KNOWN per-flow sampling rates for the population-corrected
                  context token (defaults to 1.0 = unsampled)."""
    if rng is not None:
        out, rates = [], []
        for pc in clouds:
            rate = float(np.exp(rng.uniform(np.log(AUG_RATE_MIN), 0.0)))
            keep = sample_mask(pc.shape[0], rate, rng, mode="iid", keep_min=2)
            eff = 1.0 if bool(keep.all()) else rate
            out.append(apply_sampling(pc, keep, eff))
            rates.append(eff)
        clouds = out
    if rates is None:
        rates = [1.0] * len(clouds)
    X, M = pad_stack(clouds)
    ctx = np.stack([context_summary(pc[:MAX_PKTS], r)
                    for pc, r in zip(clouds, rates)])
    return (torch.as_tensor(X, device=_DEVICE),
            torch.as_tensor(M, device=_DEVICE),
            torch.as_tensor(ctx, device=_DEVICE))


# ── GPU-resident view construction (training path) ──────────────────────
# The model is tiny (288K params, ~10ms/step) while the per-flow Python
# collate costs ~450ms/step, so the GPU idles >95% of each step. Training
# views are therefore built ON DEVICE from a padded tensor kept resident:
# Bernoulli keep-masks, log-domain time re-anchor, rate-corrected est_iat
# via a shifted cummax (packets are time-ordered), and HT-corrected context
# — all vectorised. EVALUATION keeps the CPU reference path (batch_tensors)
# so every eval number stays protocol-identical to earlier results.
def gpu_views(Xb: torch.Tensor, Mb: torch.Tensor, sample: bool):
    """(B,M,D) raw padded batch + valid mask -> (X', keep_mask, ctx).

    sample=False: full view (rate 1). sample=True: per-flow keep-rate
    ~ logU[AUG_RATE_MIN, 1]; flows left with <2 survivors fall back to the
    full view (the CPU path's keep_min analogue)."""
    B, M, _ = Xb.shape
    dev = Xb.device
    if sample:
        r = torch.exp(torch.rand(B, device=dev)
                      * float(np.log(AUG_RATE_MIN)))
        keep = (torch.rand(B, M, device=dev) < r.unsqueeze(1)) & Mb
        few = keep.sum(1) < 2
        if few.any():
            keep[few] = Mb[few]
            r = torch.where(few, torch.ones_like(r), r)
    else:
        r = torch.ones(B, device=dev)
        keep = Mb
    Xv = Xb.clone()
    rel = torch.expm1(Xb[..., CH_LOG_RELT])
    big = torch.finfo(rel.dtype).max
    t0 = torch.where(keep, rel, torch.full_like(rel, big)).min(1, keepdim=True).values
    t0 = torch.where(t0 >= big, torch.zeros_like(t0), t0)
    rel_a = torch.clamp(rel - t0, min=0.0)
    Xv[..., CH_LOG_RELT] = torch.log1p(rel_a)
    if EST_IAT:
        src = torch.where(keep, rel_a, torch.full_like(rel_a, -1.0))
        prev = torch.cummax(
            torch.cat([torch.full_like(src[:, :1], -1.0), src[:, :-1]], 1),
            dim=1).values
        gap = torch.where(keep & (prev >= 0),
                          torch.clamp(rel_a - prev, min=0.0) * r.unsqueeze(1),
                          torch.zeros_like(rel_a))
        Xv[..., CH_EST_IAT] = torch.log1p(gap)
    n = keep.sum(1).float()
    dur = (rel_a * keep).max(1).values
    byts = (Xb[..., CH_SIZE_NORM] * keep).sum(1) * 1500.0
    ctx = torch.stack([torch.log1p(n / r), torch.log1p(dur),
                       torch.log1p(byts / r) / 16.0], dim=1)
    return Xv, keep, ctx


# ── train ────────────────────────────────────────────────────────────────
def finetune(split: Path, *, pretrained: Path | None, epochs: int,
             sample_aug: bool, tag: str, seed: int,
             linear_probe: bool = False, encoder_lr_scale: float = 1.0,
             label_frac: float = 1.0, label_flip: float = 0.0,
             consistency: bool = False,
             cls_weight: str = "inv") -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    t0 = time.time()

    tr_clouds, y_tr = load_side(split, "train")
    va_clouds, y_va = load_side(split, "val")

    if label_frac < 1.0:            # label-efficiency regime: per-class subsample
        keep_idx = []
        for c in np.unique(y_tr):
            idx = np.flatnonzero(y_tr == c)
            k = max(2, int(round(idx.size * label_frac)))
            keep_idx.append(rng.choice(idx, size=min(k, idx.size), replace=False))
        keep_idx = np.concatenate(keep_idx)
        tr_clouds = [tr_clouds[i] for i in keep_idx]
        y_tr = y_tr[keep_idx]
        print(f"[data] label_frac={label_frac} -> {len(tr_clouds)} train flows",
              file=sys.stderr, flush=True)
    if label_flip > 0.0:            # contamination regime: flip TRAIN labels only
        n_fl = int(round(len(y_tr) * label_flip))
        if n_fl:
            fl = rng.choice(len(y_tr), size=n_fl, replace=False)
            n_c = int(y_tr.max()) + 1
            y_tr = y_tr.copy()
            # binary -> invert; multi-class -> reassign to a different class
            y_tr[fl] = (1 - y_tr[fl]) if n_c == 2 else \
                (y_tr[fl] + rng.integers(1, n_c, size=n_fl)) % n_c
        print(f"[data] label_flip={label_flip} -> {n_fl} corrupted train labels",
              file=sys.stderr, flush=True)
    n_classes = int(max(y_tr.max(), y_va.max())) + 1
    print(f"[data] train={len(tr_clouds)} val={len(va_clouds)} "
          f"classes={n_classes} ({time.time()-t0:.0f}s)",
          file=sys.stderr, flush=True)

    model = P3Classifier(n_classes).to(_DEVICE)
    if pretrained is not None:
        blob = torch.load(pretrained, map_location=_DEVICE)
        assert blob["sig"] == config_sig(), \
            f"pretrain sig mismatch: {blob['sig']} vs {config_sig()}"
        model.encoder.load_state_dict(blob["encoder"])
        print(f"[init] encoder <- {pretrained.name} (steps={blob['steps']})",
              file=sys.stderr, flush=True)
    else:
        allpts = np.concatenate(tr_clouds, axis=0)
        model.encoder.set_norm(allpts.mean(0), allpts.std(0))
        print("[init] from scratch (norm fit on train)",
              file=sys.stderr, flush=True)
    print(f"[model] params={count_params(model):,}", file=sys.stderr, flush=True)

    if linear_probe:                 # frozen-encoder evaluation of the SSL features
        for p_ in model.encoder.parameters():
            p_.requires_grad_(False)
        model.encoder.eval()
        print("[init] linear probe: encoder frozen", file=sys.stderr, flush=True)

    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    task_type = json.loads(split.read_text()).get("task_type", "classification")
    if task_type == "detection" or cls_weight == "sqrt":
        # DOUBLE COMPENSATION guard: the sqrt-inverse sampler already
        # oversamples the rare class by sqrt(ratio); full inverse-frequency
        # CE weights on top give ratio^1.5 total compensation, which under
        # extreme imbalance (a 60:1 detection task) collapses the model to
        # all-attack predictions. sqrt-inverse weights make
        # sampler x weights ~ exactly inverse frequency.
        cw = np.sqrt(counts.sum() / (n_classes * counts))
    elif cls_weight == "none":
        cw = np.ones_like(counts)
    else:
        cw = counts.sum() / (n_classes * counts)
    cw = cw / cw.mean()
    class_w = torch.tensor(cw, dtype=torch.float32, device=_DEVICE)
    if task_type != "detection" and cls_weight == "none":
        sample_w = np.ones(len(y_tr), dtype=np.float64)
    else:
        sample_w = (1.0 / np.sqrt(counts))[y_tr]
    sample_w = sample_w / sample_w.sum()

    N = len(tr_clouds)
    steps_per_epoch = (N + BATCH - 1) // BATCH
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    param_groups = [
        {"params": [p_ for p_ in model.encoder.parameters() if p_.requires_grad],
         "lr": LR * encoder_lr_scale},
        {"params": model.head.parameters(), "lr": LR},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup
        else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total_steps - warmup))))
    crit = nn.CrossEntropyLoss(weight=class_w, label_smoothing=LABEL_SMOOTH)
    use_amp = _DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    yt = torch.as_tensor(y_tr, device=_DEVICE)

    # GPU-resident padded training tensors (see gpu_views)
    Xall, Mall = pad_stack(tr_clouds)
    Xg = torch.as_tensor(Xall, device=_DEVICE)
    Mg = torch.as_tensor(Mall, device=_DEVICE)
    del Xall, Mall

    model.train()
    if linear_probe:
        model.encoder.eval()         # keep dropout off in the frozen encoder
    step = 0
    for ep in range(epochs):
        epoch_idx = rng.choice(N, size=N, replace=True, p=sample_w)
        ep_loss = 0.0
        for bi in range(0, N, BATCH):
            b = epoch_idx[bi:bi + BATCH]
            bt = torch.as_tensor(b, device=_DEVICE, dtype=torch.long)
            Xb, Mb = Xg[bt], Mg[bt]
            opt.zero_grad(set_to_none=True)
            if consistency:
                # multi-rate consistency distillation: every flow is seen
                # FULL and SAMPLED in the same step; the sampled prediction
                # is pulled onto the full one (task-level analogue of the
                # pretraining SCC objective, now with labels).
                Xf, Mf, Cf = gpu_views(Xb, Mb, sample=False)
                Xs, Ms, Cs = gpu_views(Xb, Mb, sample=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    lf = model(Xf, Mf, Cf)
                    ls = model(Xs, Ms, Cs)
                    loss = (0.5 * crit(lf, yt[b]) + 0.5 * crit(ls, yt[b])
                            + LAMBDA_CONS * F.kl_div(
                                F.log_softmax(ls.float(), dim=1),
                                F.softmax(lf.float().detach(), dim=1),
                                reduction="batchmean"))
            else:
                X, M, ctx = gpu_views(Xb, Mb, sample=sample_aug)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = crit(model(X, M, ctx), yt[b])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            ep_loss += loss.item() * len(b)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[ft] epoch {ep+1:>3}/{epochs}  loss={ep_loss/N:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}", file=sys.stderr, flush=True)

    # ── sampling-robustness evaluation ──────────────────────────────────
    model.eval()
    results = {"tag": tag, "pretrained": str(pretrained) if pretrained else None,
               "sample_aug": sample_aug, "consistency": consistency,
               "epochs": epochs, "seed": seed, "linear_probe": linear_probe,
               "encoder_lr_scale": encoder_lr_scale, "label_frac": label_frac,
               "label_flip": label_flip,
               "n_train": len(tr_clouds), "eval": {}}

    @torch.no_grad()
    def eval_at(rate: float, mode: str) -> dict:
        e_rng = np.random.default_rng(12345)      # fixed sampling for fairness
        probs = []
        preds = []
        for bi in range(0, len(va_clouds), BATCH):
            sub, effs = [], []
            for pc in va_clouds[bi:bi + BATCH]:
                view, eff = eval_view(pc, rate, e_rng, mode=mode)
                sub.append(view)
                effs.append(eff)
            # the deployed router's 1/k is known -> rate-conditioned context
            # (fully-retained flows are complete observations -> rate 1)
            X, M, ctx = batch_tensors(sub, rates=effs)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X, M, ctx)
            lg = logits.float()
            preds.append(lg.argmax(1).cpu().numpy())
            if n_classes == 2:      # attack-class probability for the ROC
                probs.append(torch.softmax(lg, 1)[:, 1].cpu().numpy())
        yp = np.concatenate(preds)
        acc = float(accuracy_score(y_va, yp))
        p, r, f1, _ = precision_recall_fscore_support(
            y_va, yp, average="macro", zero_division=0)
        out = {"acc": round(acc, 4), "macro_f1": round(float(f1), 4),
               "macro_p": round(float(p), 4), "macro_r": round(float(r), 4)}
        if n_classes == 2 and len(set(y_va.tolist())) == 2:
            out["auc"] = round(float(roc_auc_score(y_va, np.concatenate(probs))), 4)
            # attack-class F1: the metric the paper reports for detection
            # (eval_att_f1_rates.py / Table 2's supervised row)
            out["att_f1"] = round(float(f1_score(y_va, yp, pos_label=1)), 4)
        return out

    for mode in EVAL_MODES:
        for rate in EVAL_RATES:
            if rate == 1.0 and mode == "systematic":
                continue                          # identical to iid @ 1.0
            m = eval_at(rate, mode)
            results["eval"][f"{mode}@{rate:g}"] = m
            print(f"[eval] {mode:>10s} rate={rate:<7g} acc={m['acc']:.4f} "
                  f"macroF1={m['macro_f1']:.4f}", file=sys.stderr, flush=True)

    here = Path(__file__).resolve().parent
    (here / "ckpt").mkdir(exist_ok=True)
    (here / "results").mkdir(exist_ok=True)
    torch.save({"sig": config_sig(), "model": model.state_dict(),
                "n_classes": n_classes, "results": results},
               here / "ckpt" / f"_ckpt_finetune_{tag}.pt")
    out_json = here / "results" / f"_results_finetune_{tag}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"[done] {time.time()-t0:.0f}s -> {out_json}",
          file=sys.stderr, flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--sample-aug", action="store_true")
    p.add_argument("--consistency", action="store_true",
                   help="multi-rate consistency distillation (implies aug)")
    p.add_argument("--linear-probe", action="store_true")
    p.add_argument("--encoder-lr-scale", type=float, default=1.0)
    p.add_argument("--label-frac", type=float, default=1.0)
    p.add_argument("--label-flip", type=float, default=0.0,
                   help="fraction of TRAIN labels to corrupt (contamination study)")
    p.add_argument("--tag", default="ft")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cls-weight", choices=["inv", "sqrt", "none"],
                   default="inv",
                   help="CE class-weight scheme for classification tasks "
                        "(detection always uses the sqrt guard)")
    a = p.parse_args()
    finetune(a.split, pretrained=a.pretrained, epochs=a.epochs,
             sample_aug=a.sample_aug, tag=a.tag, seed=a.seed,
             linear_probe=a.linear_probe,
             encoder_lr_scale=a.encoder_lr_scale, label_frac=a.label_frac,
             label_flip=a.label_flip,
             consistency=a.consistency, cls_weight=a.cls_weight)


if __name__ == "__main__":
    main()
