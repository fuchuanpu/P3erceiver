#!/usr/bin/env bash
# Fine-tune the shipped pre-trained encoder on the CipherSpectrum split,
# reproducing ckpt/cipherspec_finetuned.pt. ~10 min on one GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export P3_EST_IAT=1
PY_BIN="${PYTHON:-python}"
command -v "$PY_BIN" >/dev/null || PY_BIN=python3
command -v "$PY_BIN" >/dev/null || { echo "no python on PATH -- run: conda activate p3" >&2; exit 1; }

"$PY_BIN" p3/finetune.py \
    --split       splits/cipherspec_split.json \
    --pretrained  ckpt/pretrained.pt \
    --tag         cipherspec_repro \
    --epochs      120 \
    --seed        42 \
    --consistency

echo
echo "Evaluate the result with:"
echo "  python p3/evaluate.py --split splits/cipherspec_split.json \\"
echo "      --ckpt ckpt/_ckpt_finetune_cipherspec_repro.pt --rates 1,0.5,0.25"
