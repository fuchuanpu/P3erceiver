#!/usr/bin/env bash
# Minimal working example: reproduce the Cipher-Spec. classification cell
# from the shipped split + checkpoint. CPU-only, ~1 minute, no API access.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export P3_EST_IAT=1
PY_BIN="${PYTHON:-python}"
command -v "$PY_BIN" >/dev/null || PY_BIN=python3
command -v "$PY_BIN" >/dev/null || { echo "no python on PATH -- run: conda activate p3" >&2; exit 1; }

"$PY_BIN" p3/evaluate.py \
    --split splits/cipherspec_split.json \
    --ckpt  ckpt/cipherspec_finetuned.pt \
    --rates 1,0.5,0.25

echo
echo "Expected (paper): x1 0.9584 | x1/2 0.8391 | x1/4 0.6662 macro-F1"
