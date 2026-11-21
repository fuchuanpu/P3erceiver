# P3erceiver

P3erceiver is a sampling-resilient traffic analysis framework. It views
the packets of a flow as a **point cloud** rather than a sequence: every
packet becomes a point whose coordinates (arrival time, size, direction)
depend on that packet alone. Packet sampling then only *thins* the cloud
instead of reshaping it, so the surviving tokens stay bitwise identical
and one pre-trained encoder serves both attack detection and encrypted
traffic classification under sampling.

The pipeline has four parts, mirroring Section 4 of the paper:

1. **Sampling-resilient tokenization** (`p3/pointcloud.py`) — turns each
   packet into an 11-channel token whose every channel but one is a
   function of that packet alone, plus a rate-consistent estimator
   channel that rescales the gap between surviving packets by the known
   sampling rate.
2. **Perceiver fusion** (`p3/model.py`) — a fixed-size latent array
   (96 x 128) iteratively cross-attends over the variable-cardinality
   point set, so the flow embedding is 128-d regardless of how many
   packets survive. The whole encoder is 288,020 parameters.
3. **Sampling-invariant pre-training** (`p3/pretrain.py`) — three
   objectives driven by the sampling process itself: contrastive
   alignment of two independently sampled views, masked point modeling
   from time coordinates alone, and population extrapolation.
4. **Downstream adaptation** (`p3/finetune.py`, `p3/unsup_detect.py`) —
   supervised fine-tuning with multi-rate consistency distillation, or a
   benign-only k-NN detector over frozen embeddings.

> **Anonymity notice (double-blind submission).** This artifact has been
> scrubbed of author-identifying content — names, affiliations,
> repository URLs, and site-specific paths — before release. It is safe
> to inspect and run under double-blind review.

> **Safety and scope.**
>
> - **No network access is required or performed.** Every script reads
>   local preprocessed JSON; the pipeline never opens a socket and never
>   replays traffic onto a live network.
> - **No credentials, no API keys.** Unlike LLM-driven pipelines, all
>   computation is local. Nothing is uploaded.
> - **Payload is never read.** The tokenizer consumes only packet
>   length, timestamp, direction, and the transport protocol; the shipped
>   JSON records carry no payload bytes.
> - **Shipped data is public.** The bundled CipherSpectrum split comes
>   from a public dataset (see `data/README.md`). It contains RFC-1918
>   client addresses from the original capture; no end-user identity is
>   recoverable.

```
artifact/
├── p3/                        # library
│   ├── pointcloud.py          # sampling-resilient tokenization + samplers
│   ├── model.py               # Perceiver encoder + heads (288,020 params)
│   ├── pretrain.py            # SCC / masked point modeling / population aux
│   ├── pretrain_data.py       # backbone-trace flow extraction
│   ├── finetune.py            # supervised fine-tuning + consistency distillation
│   ├── unsup_detect.py        # benign-kNN unsupervised detection
│   ├── make_split.py          # train/val/test split builder
│   └── evaluate.py            # sampled replay of a fine-tuned checkpoint
├── ckpt/
│   ├── pretrained.pt          # encoder pre-trained on one 15-min backbone trace
│   └── cipherspec_finetuned.pt# fine-tuned on the shipped CipherSpectrum split
├── data/
│   ├── README.md              # JSON schema, dataset URLs, preprocessing recipe
│   └── cipherspec/by_dest/    # 41-class encrypted-TLS destination split (35 MB)
├── splits/
│   └── cipherspec_split.json  # frozen 3:3:4 flow-level split
├── examples/
│   ├── run_minimal_cipherspec.sh   # reproduce a paper cell (~1 min, CPU-only)
│   ├── run_finetune_cipherspec.sh  # fine-tune from the pre-trained encoder
│   └── run_pretrain.sh             # pre-train from a raw backbone trace
├── results/
│   └── cipherspec_metrics.json# frozen paper-side metrics for the same cell
├── install.sh
├── requirements.txt
└── README.md                  # this file
```

## Installation

### Hardware requirements

Minimum (evaluation and fine-tuning of the shipped cell, CPU-only):

- x86_64 Linux host with >= 8 GB RAM and >= 1 GB free disk.

Recommended (pre-training from a raw backbone trace):

- NVIDIA GPU with >= 8 GB VRAM and a CUDA 12.x-compatible driver.
- >= 32 GB system RAM for the backbone flow extraction.

The model itself is tiny (288,020 parameters, 2 MB per checkpoint), so
the GPU is a throughput convenience rather than a requirement.

### Test environment

The artifact is developed and verified end-to-end on:

| Component | Version |
|---|---|
| OS           | Ubuntu (kernel 5.15) |
| CPU          | AMD EPYC 9755 (512 logical cores) |
| GPU          | NVIDIA H200 141 GB (optional) |
| Python       | 3.12 (conda env `p3`) |
| PyTorch      | 2.11.0 + CUDA 12.8 |
| scikit-learn | 1.8.0 |
| dpkt         | 1.9.8 |

### Quick start

```bash
# Creates conda env `p3`, installs the pinned deps, imports the
# modules as a smoke test.
bash install.sh

# Different env name:
bash install.sh --env my-env

# CPU-only wheel (skips the CUDA build of PyTorch):
bash install.sh --cpu
```

The installer is re-runnable — it updates an existing env rather than
clobbering it.

## Minimal working example — reproduce a paper cell in ~1 minute

To make the artifact self-contained we ship one fully preprocessed
dataset (the 41-class encrypted-TLS destination-domain split of
**CipherSpectrum**, ~35 MB at `data/cipherspec/by_dest/`) together with
the checkpoint fine-tuned on it (`ckpt/cipherspec_finetuned.pt`, 2 MB).
Together they reproduce the Cipher-Spec. column of the paper's
classification table:

```bash
conda activate p3
bash examples/run_minimal_cipherspec.sh
```

which runs

```bash
python p3/evaluate.py \
    --split splits/cipherspec_split.json \
    --ckpt  ckpt/cipherspec_finetuned.pt \
    --rates 1,0.5,0.25
```

and prints

```
[data] val: 12382 flows, 41 classes
[model] loaded ckpt/cipherspec_finetuned.pt  (293,439 parameters)

    rate   accuracy   macro-F1
      x1     0.9585     0.9585
    x0.5     0.8392     0.8393
   x0.25     0.6644     0.6663
```

Against the numbers reported in the paper:

| Keep rate | Reported macro-F1 | This artifact | Delta |
|---|---|---|---|
| x1   | 0.9584 | 0.9585 | 0.0001 |
| x1/2 | 0.8391 | 0.8393 | 0.0002 |
| x1/4 | 0.6662 | 0.6663 | 0.0001 |

The residual delta is mixed-precision noise: the paper-side runs
evaluate under `torch.amp.autocast`, whereas `evaluate.py` runs in fp32
so that the CPU-only path gives identical output. Sampling itself is
deterministic given `--seed` (default `12345`, the seed used for every
number in the paper). Wall time is ~1 minute on a single CPU core;
the 293,439 reported parameters are the 288,020-parameter encoder plus
the 41-class linear head.

To fine-tune the same cell from the pre-trained encoder instead of
using the shipped checkpoint (~10 min on one GPU):

```bash
bash examples/run_finetune_cipherspec.sh
```

## Data layout

`p3/` reads **preprocessed JSON** (one flow per list entry, no raw pcap
at model time):

```json
{
  "pcap": "<source capture>", "flow_idx": 0,
  "src_ip": "10.0.2.15", "dst_ip": "...", "src_port": 43868,
  "dst_port": 443, "protocol": "tcp",
  "pkt_lens": [60, 60, 44, 40, ...],
  "ts":       [0.0, 1.005, 1.407, ...]
}
```

Only `pkt_lens`, `ts`, the direction derived from the address pair, and
`protocol` reach the tokenizer — addresses and ports are dropped, since
learning them invites shortcut learning.

A split file freezes the flow-level partition:

```json
{"task_type": "classification", "n_classes": 41,
 "class_names": [...], "train_ratio": 0.3, "val_ratio": 0.3,
 "samples": {"train": [{"class_id": 0, "json_path": "data/...json",
                        "rec_idx": 656}, ...],
             "val": [...], "test": [...]}}
```

`json_path` is resolved relative to the artifact root, so the bundle is
relocatable. See `data/README.md` for the URLs of every public dataset
used in the paper and the pcap -> JSON preprocessing recipe.

## Running the pipeline

```bash
conda activate p3

# 1. Build a split from a directory of per-class JSON files
python p3/make_split.py --dataset data/cipherspec/by_dest \
    --out splits/my_split.json --train-ratio 0.3 --val-ratio 0.3

# 2. Pre-train the encoder on an unlabeled backbone trace
python p3/pretrain.py --pcap /path/to/backbone.pcap --tag my_backbone

# 3. Fine-tune for a downstream task
python p3/finetune.py --split splits/my_split.json \
    --pretrained ckpt/pretrained.pt --tag my_task \
    --sample-aug --consistency

# 4. Evaluate under sampling
python p3/evaluate.py --split splits/my_split.json \
    --ckpt ckpt/_ckpt_finetune_my_task.pt --rates 1,0.5,0.25,0.125
```

For unsupervised detection (no attack labels), fit the benign-only k-NN
detector over the frozen encoder instead of step 3:

```bash
python p3/unsup_detect.py --split splits/my_detection_split.json \
    --ckpt ckpt/pretrained.pt
```

## Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--rates` | `1,0.5,0.25` | Keep rates replayed at evaluation time |
| `--seed` | `12345` | Sampling seed; this value reproduces the paper |
| `--side` | `val` | Split side to evaluate |
| `--sample-aug` | off | Per-flow keep rate ~ logU[1/32, 1] during fine-tuning |
| `--consistency` | off | Multi-rate consistency distillation loss |
| `--pretrained` | none | Encoder checkpoint to initialise from |

The environment variable `P3_EST_IAT=1` enables the rate-consistent
estimator channel and is required by the released checkpoints (they are
11-channel). `evaluate.py` sets it automatically; the training scripts
expect it in the environment, and `examples/*.sh` export it.

## Reproducing other paper cells

The shipped bundle covers one classification cell end-to-end. The
remaining datasets are public but too large to ship (the detection
benchmarks alone exceed 100 GB of pcap); `data/README.md` lists the
download URL and the preprocessing command for each of them, so any cell
can be rebuilt with the same three commands as above. The pre-trained
encoder in `ckpt/pretrained.pt` is the single checkpoint reused by every
downstream task in the paper — no per-dataset pre-training is needed.
