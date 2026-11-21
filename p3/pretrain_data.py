#!/usr/bin/env python3
"""Build the P3receiver pretraining corpus (contribution C3, data side).

Streams a backbone pcap (MAWI-style, payload-truncated) with dpkt in TWO
passes and emits a random sample of bidirectional flows as PREPROCESSED
JSON shards in the SAME record format as the downstream
traffic_classification_preprocessed datasets (cstnet_tls_1.3 etc.):

  {"src_ip": ..., "src_port": ..., "dst_ip": ..., "dst_port": ...,
   "protocol": "tcp"|"udp", "pcap": "<trace>.pcap", "flow_idx": int,
   "pkt_lens": [...], "ts": [rel seconds, 0-based]}

plus two MAWI-only OPTIONAL fields (absent in CSTNet, tolerated by all
loaders): "dirs" (+1 packet from flow initiator / -1 reverse) and
"n_total" (true packet count before the --max-store cap). src is the flow
INITIATOR (first endpoint seen).

  pass 1  count packets per canonical 5-tuple (bidirectional key)
  pass 2  for a RANDOM subset of keys with >= --min-pkts packets, collect
          per-packet (rel_ts, ip_total_len, direction) + flow protocol

Only IP length, timestamp, direction and L4 protocol feed the model — no
payload bytes are read. Packet sizes use the IP TOTAL length (MAWI
truncates captured payload); MAWI IPs are pre-anonymised by the publisher.

Usage:
  python pretrain_data.py --pcap ~/dataset/202201011400.pcap \
      --out-dir data/mawi_preprocessed/202201011400 [--min-pkts 8]
      [--max-flows 300000] [--max-store 256] [--shard-size 20000]
      [--limit-pkts N] [--seed 42]
"""
from __future__ import annotations

import argparse
import gzip
import json
import socket
import sys
import time
from collections import Counter
from pathlib import Path

import dpkt
import numpy as np


def _flow_key(buf: bytes):
    """Ethernet frame -> (canonical_key, endpoint_a, is_udp) or None.

    canonical_key is direction-agnostic (sorted endpoints); endpoint_a is
    THIS packet's source endpoint, used to assign per-packet direction."""
    eth = dpkt.ethernet.Ethernet(buf)
    ip = eth.data
    if isinstance(ip, dpkt.ip.IP):
        size = ip.len
    elif isinstance(ip, dpkt.ip6.IP6):
        size = ip.plen + 40
    else:
        return None
    l4 = ip.data
    if isinstance(l4, dpkt.tcp.TCP):
        is_udp = 0
    elif isinstance(l4, dpkt.udp.UDP):
        is_udp = 1
    else:
        return None
    a = ip.src + l4.sport.to_bytes(2, "big")
    b = ip.dst + l4.dport.to_bytes(2, "big")
    key = bytes([is_udp]) + (a + b if a <= b else b + a)
    return key, a, b, size, is_udp


def _ep_str(ep: bytes) -> tuple[str, int]:
    """(ip_bytes + port_bytes) endpoint -> (ip string, port int)."""
    fam = socket.AF_INET if len(ep) == 6 else socket.AF_INET6
    return socket.inet_ntop(fam, ep[:-2]), int.from_bytes(ep[-2:], "big")


def _iter_pcap(path: Path, limit: int | None):
    _open = gzip.open if str(path).endswith(".gz") else open
    with _open(path, "rb") as f:
        for i, (ts, buf) in enumerate(dpkt.pcap.Reader(f)):
            if limit is not None and i >= limit:
                return
            yield i, ts, buf


def build_corpus(pcap: Path, out_dir: Path, *, min_pkts: int, max_flows: int,
                 max_store: int, shard_size: int, limit_pkts: int | None,
                 seed: int) -> None:
    rng = np.random.default_rng(seed)

    # ── pass 1: packet count per canonical key ──────────────────────────
    t0 = time.time()
    counts: Counter = Counter()
    n_pkts = 0
    for i, _ts, buf in _iter_pcap(pcap, limit_pkts):
        n_pkts = i + 1
        try:
            parsed = _flow_key(buf)
        except Exception:
            continue
        if parsed is not None:
            counts[parsed[0]] += 1
        if n_pkts % 5_000_000 == 0:
            print(f"[pass1] {n_pkts/1e6:.0f}M pkts  {len(counts)/1e6:.2f}M keys  "
                  f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    qualifying = [k for k, c in counts.items() if c >= min_pkts]
    print(f"[pass1] done: {n_pkts:,} pkts, {len(counts):,} keys, "
          f"{len(qualifying):,} with >={min_pkts} pkts  "
          f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    del counts

    # ── random flow selection (the "randomly sampled flows" of C3) ──────
    if len(qualifying) > max_flows:
        sel = rng.choice(len(qualifying), size=max_flows, replace=False)
        qualifying = [qualifying[i] for i in sel]
    selected = set(qualifying)
    del qualifying
    print(f"[select] {len(selected):,} flows", file=sys.stderr, flush=True)

    # ── pass 2: collect per-packet attributes for selected keys ─────────
    t1 = time.time()
    # key -> [initiator_ep, responder_ep, [ts], [len], [dir], is_udp, n_total]
    flows: dict[bytes, list] = {}
    for i, ts, buf in _iter_pcap(pcap, limit_pkts):
        try:
            parsed = _flow_key(buf)
        except Exception:
            continue
        if parsed is None or parsed[0] not in selected:
            continue
        key, src_ep, dst_ep, size, is_udp = parsed
        st = flows.get(key)
        if st is None:
            st = flows[key] = [src_ep, dst_ep, [], [], [], is_udp, 0]
        st[6] += 1
        if len(st[2]) < max_store:
            st[2].append(ts)
            st[3].append(size)
            st[4].append(1 if src_ep == st[0] else -1)
        if (i + 1) % 5_000_000 == 0:
            print(f"[pass2] {(i+1)/1e6:.0f}M pkts  {time.time()-t1:.0f}s",
                  file=sys.stderr, flush=True)

    # ── CSTNet-style preprocessed JSON shards ────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(flows)           # deterministic layout
    n_stored, shard, shard_id = 0, [], 0

    def _flush(shard_recs: list, sid: int) -> None:
        p = out_dir / f"shard_{sid:03d}.json"
        p.write_text(json.dumps(shard_recs))
        print(f"[write] {p.name}  ({len(shard_recs)} flows)",
              file=sys.stderr, flush=True)

    for j, k in enumerate(order):
        init_ep, resp_ep, tss, lens, dirs, is_udp, n_total = flows[k]
        t_arr = np.asarray(tss, dtype=np.float64)
        rel = np.round(t_arr - t_arr[0], 6)
        sip, sport = _ep_str(init_ep)
        dip, dport = _ep_str(resp_ep)
        shard.append({
            "src_ip": sip, "src_port": sport,
            "dst_ip": dip, "dst_port": dport,
            "protocol": "udp" if is_udp else "tcp",
            "pcap": pcap.name, "flow_idx": j,
            "pkt_lens": [int(v) for v in lens],
            "ts": rel.tolist(),
            "dirs": dirs,                    # MAWI extra: +1 initiator, -1 back
            "n_total": n_total,              # MAWI extra: count before cap
        })
        n_stored += len(tss)
        if len(shard) >= shard_size:
            _flush(shard, shard_id)
            shard, shard_id = [], shard_id + 1
    if shard:
        _flush(shard, shard_id)
    meta = {"pcap": str(pcap), "n_pkts_scanned": n_pkts,
            "min_pkts": min_pkts, "max_flows": max_flows,
            "max_store": max_store, "seed": seed,
            "n_flows": len(order), "n_pkts_stored": n_stored,
            "n_shards": shard_id + 1}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(order):,} flows, {n_stored:,} stored pkts -> "
          f"{out_dir}/ ({shard_id + 1} shards)  "
          f"(pass2 {time.time()-t1:.0f}s, total {time.time()-t0:.0f}s)",
          file=sys.stderr, flush=True)


def records_to_flows(recs: list[dict]) -> list[dict]:
    """Preprocessed JSON records (CSTNet/MAWI style) -> flow dicts for the
    tokenizer. `dirs` and `n_total` are optional (absent in CSTNet)."""
    out = []
    for r in recs:
        n = len(r["pkt_lens"])
        dirs = r.get("dirs")
        out.append({"ts": np.asarray(r["ts"], dtype=np.float64),
                    "lens": np.asarray(r["pkt_lens"], dtype=np.float64),
                    "dirs": (np.asarray(dirs, dtype=np.float64)
                             if dirs is not None else None),
                    "is_udp": 1.0 if r.get("protocol") == "udp" else 0.0,
                    "n_total": int(r.get("n_total", n))})
    return out


def load_flows(path: Path) -> list[dict]:
    """Load a pretraining corpus from any supported form:
    - directory of preprocessed shard_*.json (canonical)
    - a single preprocessed .json file
    - legacy ragged .npz (backward compatibility)
    """
    path = Path(path)
    if path.is_dir():
        flows = []
        for p in sorted(path.glob("shard_*.json")):
            flows.extend(records_to_flows(json.loads(p.read_text())))
        if not flows:
            raise FileNotFoundError(f"no shard_*.json under {path}")
        return flows
    if path.suffix == ".json":
        return records_to_flows(json.loads(path.read_text()))
    return load_corpus(path)


def load_corpus(path: Path):
    """LEGACY npz -> list of dicts (ts rel, lens, dirs, is_udp, n_total)."""
    z = np.load(path, allow_pickle=False)
    # bind each member ONCE — NpzFile.__getitem__ decompresses the whole
    # array on every access, so indexing it inside the loop is O(F * P)
    offsets = z["offsets"]
    ts, lens, dirs = z["ts"], z["lens"], z["dirs"]
    is_udp, n_total = z["is_udp"], z["n_total"]
    out = []
    for j in range(len(offsets) - 1):
        a, b = offsets[j], offsets[j + 1]
        out.append({"ts": ts[a:b].astype(np.float64),
                    "lens": lens[a:b].astype(np.float64),
                    "dirs": dirs[a:b].astype(np.float64),
                    "is_udp": float(is_udp[j]),
                    "n_total": int(n_total[j])})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pcap", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="directory for preprocessed shard_*.json + meta.json")
    p.add_argument("--min-pkts", type=int, default=8)
    p.add_argument("--max-flows", type=int, default=300_000)
    p.add_argument("--max-store", type=int, default=256)
    p.add_argument("--shard-size", type=int, default=20_000)
    p.add_argument("--limit-pkts", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build_corpus(a.pcap, a.out_dir, min_pkts=a.min_pkts, max_flows=a.max_flows,
                 max_store=a.max_store, shard_size=a.shard_size,
                 limit_pkts=a.limit_pkts, seed=a.seed)


if __name__ == "__main__":
    main()
