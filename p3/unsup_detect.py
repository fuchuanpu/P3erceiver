#!/usr/bin/env python3
"""Unsupervised (one-class) detection with the frozen pretrained encoder.

No attack labels are used anywhere: the FROZEN pretrained encoder embeds
the train-side BENIGN flows, a k-NN distance detector is fit on those
embeddings, and val-side flows (benign + attack) are scored by their mean
distance to the k nearest benign neighbours. Reports AUROC and F1 at the
95th-percentile-of-benign threshold, across sampling rates (iid, known-rate
conditioned context, same protocol as finetune.py's eval_at).

This is the "pretrain once, reuse everywhere" claim in its purest form:
zero task-specific training, the encoder has never seen an attack.

Usage (P3_EST_IAT must match the checkpoint's token width):
  P3_EST_IAT=1 python unsup_detect.py --split splits/det_X_split.json \
      --pretrained ckpt/_ckpt_pretrain_mawi_v4.pt --tag det_X [--k 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from pointcloud import sample_mask, apply_sampling, eval_view
from model import PerceiverEncoder, config_sig
from finetune import load_side, batch_tensors, EVAL_RATES

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def embed(enc, clouds, rate: float, mode: str, rng) -> np.ndarray:
    out = []
    for bi in range(0, len(clouds), 512):
        sub, effs = [], []
        for pc in clouds[bi:bi + 512]:
            view, eff = eval_view(pc, rate, rng, mode=mode)
            sub.append(view)
            effs.append(eff)
        X, M, ctx = batch_tensors(sub, rates=effs)
        _, pooled = enc(X, M, ctx)
        out.append(torch.nn.functional.normalize(pooled, dim=-1)
                   .float().cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()
    t0 = time.time()

    tr_clouds, y_tr = load_side(a.split, "train")
    va_clouds, y_va = load_side(a.split, "val")
    benign = [pc for pc, y in zip(tr_clouds, y_tr) if y == 0]

    enc = PerceiverEncoder().to(_DEVICE)
    blob = torch.load(a.pretrained, map_location=_DEVICE)
    assert blob["sig"] == config_sig(), "sig mismatch (check P3_EST_IAT)"
    enc.load_state_dict(blob["encoder"])
    enc.eval()

    rng0 = np.random.default_rng(0)
    z_ben = embed(enc, benign, 1.0, "iid", rng0)
    nn_ = NearestNeighbors(n_neighbors=a.k).fit(z_ben)
    # benign self-scores (k+1 neighbours, drop self) -> threshold @ p95
    d_self, _ = NearestNeighbors(n_neighbors=a.k + 1).fit(z_ben).kneighbors(z_ben)
    thr = float(np.percentile(d_self[:, 1:].mean(1), 95))

    results = {"tag": a.tag, "pretrained": str(a.pretrained), "k": a.k,
               "n_benign_fit": len(benign), "n_val": len(va_clouds),
               "thr_p95": thr, "eval": {}}
    for rate in EVAL_RATES:
        e_rng = np.random.default_rng(12345)
        z = embed(enc, va_clouds, rate, "iid", e_rng)
        d, _ = nn_.kneighbors(z)
        score = d.mean(1)
        auroc = float(roc_auc_score(y_va, score))
        f1 = float(f1_score(y_va, (score > thr).astype(int)))
        # rate-calibrated threshold: the operator replays its benign corpus
        # through the sampler at the deployed rate and re-takes p95 there;
        # this removes the score-distribution shift that a full-rate
        # threshold suffers under sampling.
        if rate >= 1.0:
            thr_r = thr
        else:
            zb_r = embed(enc, benign, rate, "iid", np.random.default_rng(7))
            # k+1 neighbours, drop the closest: the sampled view of a flow
            # quasi-matches its own full-rate index entry (same convention
            # as the rate-1 self-score threshold).
            db_r, _ = nn_.kneighbors(zb_r, n_neighbors=a.k + 1)
            thr_r = float(np.percentile(db_r[:, 1:].mean(1), 95))
        f1r = float(f1_score(y_va, (score > thr_r).astype(int)))
        results["eval"][f"iid@{rate:g}"] = {"auroc": round(auroc, 4),
                                            "f1_p95": round(f1, 4),
                                            "f1_p95r": round(f1r, 4)}
        print(f"[unsup] rate={rate:<7g} auroc={auroc:.4f} f1@p95={f1:.4f} "
              f"f1@p95r={f1r:.4f}", file=sys.stderr, flush=True)

    out = Path(__file__).resolve().parent / "results" / \
        f"_results_unsup_{a.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] {time.time()-t0:.0f}s -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
