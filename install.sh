#!/usr/bin/env bash
# One-shot environment setup for the P3erceiver artifact.
#   bash install.sh                 # conda env `p3`, CUDA build of PyTorch
#   bash install.sh --env my-env    # different env name
#   bash install.sh --cpu           # CPU-only PyTorch wheel
set -euo pipefail

ENV_NAME=p3
CPU_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --cpu) CPU_ONLY=1; shift ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

command -v conda >/dev/null || { echo "conda not found on PATH" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[install] updating existing env '$ENV_NAME'"
else
  echo "[install] creating env '$ENV_NAME' (python 3.12)"
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

if [[ "$CPU_ONLY" == "1" ]]; then
  echo "[install] CPU-only PyTorch"
  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
  grep -v '^torch==' requirements.txt | pip install -r /dev/stdin
else
  pip install -r requirements.txt
fi

echo "[install] smoke test"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$HERE/p3" P3_EST_IAT=1 python - <<'PY'
import torch, numpy, sklearn
from pointcloud import D_IN, MAX_PKTS
from model import P3Classifier, count_params
m = P3Classifier(41)
print(f"  torch {torch.__version__}, cuda={torch.cuda.is_available()}")
print(f"  token channels D_IN={D_IN}, max packets={MAX_PKTS}")
print(f"  encoder parameters: {count_params(m.encoder):,}")
PY
echo "[install] done -- run: conda activate $ENV_NAME && bash examples/run_minimal_cipherspec.sh"
