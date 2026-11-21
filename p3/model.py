#!/usr/bin/env python3
"""P3receiver Perceiver architecture (contribution C2).

Set-semantic fusion of the packet point cloud:
  1. CONTINUOUS-COORDINATE Fourier encoding replaces the index PE — position
     information comes from the physical (log-size, log-arrival-time)
     coordinates, so the encoder is permutation-invariant over points and
     indifferent to index reshuffling under sampling (pairs with C1).
  2. A fixed latent array (N_LAT x D_LAT) iteratively CROSS-ATTENDS over the
     masked point set (weight-tied across N_ITER): variable-cardinality,
     unordered, possibly-decimated inputs are the native case, cost O(N*L).
  3. An OBSERVATION-CONTEXT token (log n_obs / dur_obs / bytes_obs) is
     appended to the KV set so the model can calibrate to how much of the
     flow it is seeing.
  4. A Perceiver-IO style COORDINATE-QUERY decoder reconstructs masked
     packets from their time coordinate alone (pretraining head, detachable).
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn

# ── encoder ablation toggles (env vars, default off) ──────────────────
ABL_NO_FOURIER = os.environ.get("P3_NO_FOURIER", "0") == "1"
ABL_NO_CTX     = os.environ.get("P3_NO_CTX", "0") == "1"
ABL_NO_SA      = os.environ.get("P3_NO_SA", "0") == "1"
import torch.nn.functional as F

from pointcloud import D_IN, FOURIER_COORDS, CH_LOG_RELT, MAX_PKTS

# ── hyperparameters ──────────────────────────────────────────────────────
NUM_FOURIER_BANDS = 16
MAX_FREQ          = 16.0
N_LAT             = int(os.environ.get("P3_N_LAT", "96"))  # latent-size sweep
D_LAT             = 128
N_HEADS           = 8
N_ITER            = 6          # weight-tied: depth is parameter-free
N_SA              = 1
FF_MULT           = 2
DROPOUT           = 0.10
D_PROJ            = 128        # contrastive projection width

QUERY_BANDS       = 32         # Fourier bands for decoder time queries


def config_sig() -> dict:
    return {"D_IN": D_IN, "MAX_PKTS": MAX_PKTS, "N_LAT": N_LAT,
            "D_LAT": D_LAT, "N_HEADS": N_HEADS, "N_ITER": N_ITER,
            "N_SA": N_SA, "NUM_FOURIER_BANDS": NUM_FOURIER_BANDS,
            "MAX_FREQ": MAX_FREQ, "FF_MULT": FF_MULT,
            "QUERY_BANDS": QUERY_BANDS}


# ── shared blocks (SDPA attention, pre-norm, as in init v2) ─────────────
class FourierEncode(nn.Module):
    """Concatenate sin/cos Fourier features of selected coordinate channels."""
    def __init__(self, coord_idx, num_bands, max_freq):
        super().__init__()
        self.coord_idx = list(coord_idx)
        self.register_buffer("freqs", torch.linspace(1.0, max_freq, num_bands))
        self.out_extra = len(self.coord_idx) * num_bands * 2

    def forward(self, x):                      # (B, M, D_IN)
        coords = x[..., self.coord_idx]
        scaled = coords.unsqueeze(-1) * self.freqs * torch.pi
        enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return torch.cat([x, enc.flatten(start_dim=-2)], dim=-1)


class MHA(nn.Module):
    """Multi-head attention via F.scaled_dot_product_attention."""
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, xq, xkv, key_padding=None):
        B, Lq, _ = xq.shape
        Lk = xkv.shape[1]
        q = self.q(xq).view(B, Lq, self.h, self.dh).transpose(1, 2)
        k = self.k(xkv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.v(xkv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        attn_mask = None
        if key_padding is not None:            # (B, Lk) True = pad
            attn_mask = torch.zeros(B, 1, 1, Lk, device=xq.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(
                key_padding[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(out.transpose(1, 2).reshape(B, Lq, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MHA(dim, heads)
        self.ff = FeedForward(dim, FF_MULT, dropout)

    def forward(self, z):
        z = z + self.attn(self.norm(z), self.norm(z))
        return z + self.ff(z)


# ── encoder ──────────────────────────────────────────────────────────────
class PerceiverEncoder(nn.Module):
    """Point cloud (B,M,D_IN) + mask (B,M) + context (B,3) -> latents
    (B,N_LAT,D_LAT) and pooled embedding (B,D_LAT).

    Per-channel normalisation lives in buffers so checkpoints are
    self-contained (set_norm() once from corpus statistics)."""
    def __init__(self, d_in: int = D_IN):
        super().__init__()
        self.fourier = FourierEncode(FOURIER_COORDS, NUM_FOURIER_BANDS, MAX_FREQ)
        in_dim = d_in if ABL_NO_FOURIER else d_in + self.fourier.out_extra
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, D_LAT)
        self.ctx_proj = nn.Linear(3, D_LAT)

        self.latents = nn.Parameter(torch.randn(N_LAT, D_LAT) * 0.02)
        self.cross_q_norm  = nn.LayerNorm(D_LAT)
        self.cross_kv_norm = nn.LayerNorm(D_LAT)
        self.cross_attn = MHA(D_LAT, N_HEADS)
        self.cross_ff = FeedForward(D_LAT, FF_MULT, DROPOUT)
        self.sa_blocks = nn.ModuleList(
            [SelfAttnBlock(D_LAT, N_HEADS, DROPOUT) for _ in range(N_SA)])
        self.final_norm = nn.LayerNorm(D_LAT)

        self.register_buffer("norm_mean", torch.zeros(d_in))
        self.register_buffer("norm_std", torch.ones(d_in))

    @torch.no_grad()
    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        std = std.copy()
        std[std < 1e-6] = 1.0
        self.norm_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.norm_std.copy_(torch.from_numpy(std.astype(np.float32)))

    def forward(self, x, mask, ctx):
        # x is RAW (pre-norm); normalise on-device, zero the pads
        x = (x - self.norm_mean) / self.norm_std
        x = x * mask.unsqueeze(-1)
        feat = x if ABL_NO_FOURIER else self.fourier(x)
        kv = self.input_proj(self.input_norm(feat))
        if not ABL_NO_CTX:                     # observation-context token
            kv = torch.cat([kv, self.ctx_proj(ctx).unsqueeze(1)], dim=1)
            key_padding = torch.cat(
                [~mask, torch.zeros(mask.size(0), 1, dtype=torch.bool,
                                    device=mask.device)], dim=1)
        else:
            key_padding = ~mask
        kv_n = self.cross_kv_norm(kv)
        z = self.latents.unsqueeze(0).expand(x.size(0), -1, -1)
        for _ in range(N_ITER):
            q = self.cross_q_norm(z)
            z = z + self.cross_attn(q, kv_n, key_padding=key_padding)
            z = z + self.cross_ff(z)
            if not ABL_NO_SA:
                for sa in self.sa_blocks:
                    z = sa(z)
        z = self.final_norm(z)
        return z, z.mean(dim=1)


# ── pretraining heads ────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """Pooled embedding -> L2-normalised contrastive space (SCC)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D_LAT, D_LAT), nn.GELU(),
                                 nn.Linear(D_LAT, D_PROJ))

    def forward(self, pooled):
        return F.normalize(self.net(pooled), dim=-1)


class PointDecoder(nn.Module):
    """Perceiver-IO coordinate-query decoder (MPM).

    A masked packet is queried by its TIME coordinate alone: Fourier-lift
    log_rel_time -> query vector -> one cross-attention over the latents ->
    predict (log_length, direction class, size-bucket class)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.linspace(1.0, MAX_FREQ, QUERY_BANDS))
        self.q_proj = nn.Linear(QUERY_BANDS * 2 + 1, D_LAT)
        self.q_norm = nn.LayerNorm(D_LAT)
        self.kv_norm = nn.LayerNorm(D_LAT)
        self.attn = MHA(D_LAT, N_HEADS)
        self.ff = FeedForward(D_LAT, FF_MULT, DROPOUT)
        self.out_norm = nn.LayerNorm(D_LAT)
        self.head_len = nn.Linear(D_LAT, 1)      # log1p(len)/8 regression
        self.head_dir = nn.Linear(D_LAT, 2)      # {reverse, forward} CE
        self.head_bkt = nn.Linear(D_LAT, 5)      # size bucket CE

    def forward(self, latents, t_coords):        # latents (B,L,D), t (B,K)
        scaled = t_coords.unsqueeze(-1) * self.freqs * torch.pi
        q_in = torch.cat([t_coords.unsqueeze(-1),
                          torch.sin(scaled), torch.cos(scaled)], dim=-1)
        q = self.q_norm(self.q_proj(q_in))
        h = q + self.attn(q, self.kv_norm(latents))
        h = self.out_norm(h + self.ff(h))
        return self.head_len(h).squeeze(-1), self.head_dir(h), self.head_bkt(h)


# ── downstream head ──────────────────────────────────────────────────────
class P3Classifier(nn.Module):
    """Encoder + linear classification head (finetune / linear probe)."""
    def __init__(self, n_classes: int, d_in: int = D_IN):
        super().__init__()
        self.encoder = PerceiverEncoder(d_in)
        self.head = nn.Linear(D_LAT, n_classes)

    def forward(self, x, mask, ctx):
        _, pooled = self.encoder(x, mask, ctx)
        return self.head(pooled)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
