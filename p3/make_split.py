#!/usr/bin/env python3
"""Generate a 00_split-style stratified split for any preprocessed
classification dataset (one JSON per class under by_class/ by_service/
by_browser/ by_dest/ by_task/ ...).

Mirrors the schema of init/'s 00_split.json (task_type, class_names,
samples.{train,val,test} with {class_id, class_name, json_path, rec_idx})
so finetune.py's load_side() consumes it unchanged.

Usage:
  python make_split.py --dataset data/traffic_classification_preprocessed/ustc_tfc2016 \
      [--ratios 0.3 0.3 0.4] [--min-per-class 5] [--seed 42]
      [--out splits/<name>_split.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BY_DIRS = ("by_class", "by_service", "by_browser", "by_dest", "by_task")


def make_split(dataset: Path, ratios: tuple[float, float, float],
               min_per_class: int, seed: int, out: Path) -> dict:
    by_dir = next((dataset / b for b in BY_DIRS if (dataset / b).is_dir()), None)
    if by_dir is None:
        raise FileNotFoundError(f"no {BY_DIRS} subdir under {dataset}")
    files = sorted(by_dir.glob("*.json"))
    rng = np.random.default_rng(seed)

    class_names, class_dirs, n_per_class = [], {}, {}
    samples = {"train": [], "val": [], "test": []}
    skipped = []
    for f in files:
        n = len(json.loads(f.read_text()))
        name = f.stem
        if n < min_per_class:
            skipped.append((name, n))
            continue
        cid = len(class_names)
        class_names.append(name)
        class_dirs[name] = str(f.resolve())
        n_per_class[name] = n
        idx = rng.permutation(n)
        n_tr = max(1, int(round(n * ratios[0])))
        n_va = max(1, int(round(n * ratios[1])))
        bounds = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va],
                  "test": idx[n_tr + n_va:]}
        for side, ids in bounds.items():
            samples[side].extend(
                {"class_id": cid, "class_name": name,
                 "json_path": str(f.resolve()), "rec_idx": int(i)}
                for i in ids)
    if skipped:
        print(f"[skip] {len(skipped)} classes < {min_per_class} flows: "
              f"{skipped}", file=sys.stderr)

    split = {"task_type": "classification", "input_format": "json",
             "n_classes": len(class_names), "class_names": class_names,
             "is_binary": len(class_names) == 2,
             "train_ratio": ratios[0], "val_ratio": ratios[1],
             "test_ratio": ratios[2], "seed": seed, "granularity": "flow",
             "class_dirs": class_dirs, "n_per_class": n_per_class,
             "samples": samples}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split))
    print(f"[done] {dataset.name}: {len(class_names)} classes  "
          + "  ".join(f"{s}={len(samples[s])}" for s in samples)
          + f"  -> {out}", file=sys.stderr)
    return split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--ratios", type=float, nargs=3, default=(0.3, 0.3, 0.4))
    p.add_argument("--min-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = a.out or Path(__file__).resolve().parent / "splits" / \
        f"{a.dataset.name}_split.json"
    make_split(a.dataset.resolve(), tuple(a.ratios), a.min_per_class,
               a.seed, out)


if __name__ == "__main__":
    main()
