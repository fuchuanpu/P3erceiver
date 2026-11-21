# Datasets

`p3/` never touches raw pcap at model time. Every dataset is first
converted into **preprocessed JSON**, one file per class (classification)
or one benign/attack pair per group (detection), with one flow per list
entry.

## Record schema

```json
{
  "pcap": "<source capture file>",
  "flow_idx": 0,
  "src_ip": "10.0.2.15",
  "dst_ip": "148.251.232.132",
  "src_port": 43868,
  "dst_port": 443,
  "protocol": "tcp",
  "pkt_lens": [60, 60, 44, 40],
  "ts":       [0.0, 1.005, 1.407, 1.407]
}
```

`pkt_lens` and `ts` are parallel arrays over the packets of one
bidirectional flow, ordered by arrival; `ts` is in seconds relative to
the first packet of that flow.

**What the tokenizer actually consumes.** Only `pkt_lens`, `ts`, the
direction implied by the address pair, and `protocol` reach
`tokens_from_record`. Addresses and ports are *not* features — learning
them lets a model shortcut on the capture environment rather than on
traffic behaviour. They are retained in the JSON only so that flows can
be traced back to the source capture for auditing.

No payload bytes are stored at any point.

## Shipped dataset

`cipherspec/by_dest/` — the 41-class encrypted-TLS destination-domain
split of **CipherSpectrum**, ~35 MB, 41,205 flows. This is the only
dataset bundled with the artifact; it backs the minimal working example
in the top-level README.

## Public datasets used in the paper

| Dataset | Task | Source |
|---|---|---|
| CipherSpectrum | classification (41 destination domains) | published with the Enigma-NTC paper (IEEE S&P 2025) |
| CSTNet-TLS 1.3 | classification (120 TLS services) | published with the ET-BERT paper (WWW 2022) |
| ISCX-VPN 2016 | classification (VPN service types) | Canadian Institute for Cybersecurity |
| ISCX-Tor 2016 | classification (Tor service types) | Canadian Institute for Cybersecurity |
| USTC-TFC 2016 | classification (application / malware) | published with the USTC-TFC dataset |
| Browser traffic | classification (4 browsers) | published with the FlowPrint paper (NDSS 2020) |
| CIC-IDS 2017 / 2018 | detection (intrusions) | Canadian Institute for Cybersecurity |
| CIC-DDoS 2019 | detection (volumetric floods) | Canadian Institute for Cybersecurity |
| CIC-AndMal | detection (Android malware) | Canadian Institute for Cybersecurity |
| CIC-IoT | detection (IoT botnet) | Canadian Institute for Cybersecurity |
| CIC-DoH | detection (DoH covert channels) | Canadian Institute for Cybersecurity |
| CTU-13 | detection (campus botnet) | CTU University |
| MAWI backbone traces | pre-training (unlabeled) | MAWI Working Group traffic archive |

Each is distributed by its original authors under its own terms; we
redistribute none of them except the CipherSpectrum split above, which
is public. Consult the respective sites for download links and licence
conditions.

## pcap to JSON

The converter walks a pcap, groups packets into bidirectional flows by
the 5-tuple, and emits the schema above:

```bash
python p3/make_split.py --help     # split builder over an existing JSON tree
```

Flow assembly rules used throughout the paper:

- A flow is the set of packets sharing the 5-tuple
  (protocol, src IP, dst IP, src port, dst port), with the two
  directions merged and direction recorded per packet.
- Flows with fewer than 4 packets are dropped at preprocessing time.
- At most 192 packets per flow reach the model (`MAX_PKTS`); longer
  flows keep their handshake-rich prefix.
- Timestamps are re-anchored to the first packet of the flow.

## Pre-training corpus

`ckpt/pretrained.pt` was pre-trained on a single 15-minute MAWI backbone
trace (35,609,745 packets, 255,399 unlabeled bidirectional flows). No
labels are used at any point in pre-training, and the same checkpoint is
reused by every downstream task in the paper — there is no per-dataset
pre-training.

The trace is not shipped (it exceeds the artifact size budget); it can
be downloaded from the MAWI archive and converted with
`p3/pretrain_data.py`, which is what `examples/run_pretrain.sh` calls.
