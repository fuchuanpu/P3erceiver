#!/usr/bin/env python3
"""Evaluate a fine-tuned P3erceiver checkpoint under packet sampling.

Loads a split, tokenizes every held-out flow into a packet point cloud,
replays the flow at the requested keep rates with independent (Bernoulli)
sampling, and reports accuracy / macro-F1 per rate.

  python -m p3.evaluate --split splits/cipherspec_split.json \
      --ckpt ckpt/cipherspec_finetuned.pt --rates 1,0.5,0.25
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The released checkpoints carry the rate-consistent estimator channel
# (Section 4.1), i.e. an 11-channel token. This must be set before
# `pointcloud` is imported, because the channel layout is fixed at import.
os.environ.setdefault('P3_EST_IAT', '1')

import numpy as np
import torch

from pointcloud import (tokens_from_record, eval_view,
                        pad_stack, context_summary)
from model import P3Classifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_side(split_path: Path, side: str, root: Path):
    """Read one side of the split; json_path is relative to `root`."""
    meta = json.loads(split_path.read_text())
    cache: dict[str, list] = {}
    clouds, ys = [], []
    for s in meta['samples'][side]:
        jp = s['json_path']
        if jp not in cache:
            p = Path(jp)
            cache[jp] = json.loads((p if p.is_absolute() else root / p).read_text())
        clouds.append(tokens_from_record(cache[jp][int(s['rec_idx'])]))
        ys.append(int(s['class_id']))
    return clouds, np.asarray(ys, dtype=np.int64), meta


@torch.no_grad()
def evaluate(model, clouds, y, rate: float, seed: int = 12345, batch: int = 512):
    """Replay every flow at `rate` and return (accuracy, macro-F1).

    Uses the same `eval_view` helper as the training code, so a flow whose
    mask happens to retain every packet is conditioned at rate 1 (it is a
    complete observation) rather than at the link rate."""
    rng = np.random.default_rng(seed)
    preds = np.empty(len(clouds), dtype=np.int64)
    for i in range(0, len(clouds), batch):
        chunk = clouds[i:i + batch]
        views, rates = [], []
        for pc in chunk:
            view, eff = eval_view(pc, rate, rng, mode='iid')
            views.append(view); rates.append(eff)
        X, M = pad_stack(views)
        ctx = np.stack([context_summary(v[:192], r) for v, r in zip(views, rates)])
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE),
                       torch.as_tensor(M, device=DEVICE),
                       torch.as_tensor(ctx, dtype=torch.float32, device=DEVICE))
        preds[i:i + len(chunk)] = logits.argmax(-1).cpu().numpy()
    acc = float((preds == y).mean())
    f1s = []
    for c in np.unique(y):
        tp = int(((preds == c) & (y == c)).sum())
        fp = int(((preds == c) & (y != c)).sum())
        fn = int(((preds != c) & (y == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return acc, float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--side', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--rates', default='1,0.5,0.25',
                    help='comma-separated keep rates')
    ap.add_argument('--seed', type=int, default=12345,
                    help='sampling seed; 12345 reproduces the paper numbers')
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent      # artifact/
    clouds, y, meta = load_side(a.split, a.side, root)
    n_classes = int(meta['n_classes'])
    print(f'[data] {a.side}: {len(clouds)} flows, {n_classes} classes')

    model = P3Classifier(n_classes).to(DEVICE)
    sd = torch.load(a.ckpt, map_location=DEVICE)
    model.load_state_dict(sd['model'] if 'model' in sd else sd)
    model.eval()
    print(f'[model] loaded {a.ckpt}  '
          f'({sum(p.numel() for p in model.parameters()):,} parameters)')

    print(f"\n{'rate':>8s} {'accuracy':>10s} {'macro-F1':>10s}")
    for rk in a.rates.split(','):
        rate = float(rk)
        acc, f1 = evaluate(model, clouds, y, rate, seed=a.seed)
        label = 'x1' if rate >= 1.0 else f'x{rate:g}'
        print(f'{label:>8s} {acc:>10.4f} {f1:>10.4f}')


if __name__ == '__main__':
    main()
