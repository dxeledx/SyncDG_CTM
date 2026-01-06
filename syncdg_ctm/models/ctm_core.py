from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CTMCoreConfig:
    d_model: int = 128
    d_token: int = 64
    n_ticks: int = 8
    memory_len: int = 8
    attn_heads: int = 4
    synapse_hidden: int = 256
    nlm_hidden: int = 64
    neuron_embed_dim: int = 16
    dropout: float = 0.1


class CTMCore(nn.Module):
    """
    EEG-adapted CTM core:
    - Cross-attn read on token sequence
    - Synapse update produces pre-activations
    - Shared NLM with neuron embedding (FiLM-like via concat)
    """

    def __init__(self, cfg: CTMCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.d_model <= 0 or cfg.d_token <= 0:
            raise ValueError("d_model/d_token must be > 0")
        if cfg.n_ticks <= 0:
            raise ValueError("n_ticks must be > 0")
        if cfg.memory_len <= 0:
            raise ValueError("memory_len must be > 0")
        if cfg.attn_heads <= 0:
            raise ValueError("attn_heads must be > 0")

        self.z0 = nn.Parameter(torch.zeros(cfg.d_model))
        self.a0 = nn.Parameter(torch.zeros(cfg.d_model, cfg.memory_len))

        self.query = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.d_token))
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_token, num_heads=cfg.attn_heads, dropout=cfg.dropout, batch_first=True
        )

        self.synapse = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.d_token, cfg.synapse_hidden * 2),
            nn.GLU(dim=-1),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.synapse_hidden, cfg.d_model),
        )

        self.neuron_embed = nn.Embedding(cfg.d_model, cfg.neuron_embed_dim)
        self.nlm = nn.Sequential(
            nn.Linear(cfg.memory_len + cfg.neuron_embed_dim, cfg.nlm_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.nlm_hidden, 1),
        )
        self.post_act = nn.Tanh()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, L, d_token)
        Returns:
            z_ticks: (B, T, d_model)
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be (B,L,d_token), got {tuple(tokens.shape)}")
        b, _, d_token = tokens.shape
        if d_token != self.cfg.d_token:
            raise ValueError(f"tokens last dim {d_token} != cfg.d_token {self.cfg.d_token}")

        z = self.z0.unsqueeze(0).expand(b, -1)  # (B,D)
        a_hist = self.a0.unsqueeze(0).expand(b, -1, -1)  # (B,D,M)

        neuron_idx = torch.arange(self.cfg.d_model, device=tokens.device)
        neuron_emb = self.neuron_embed(neuron_idx)  # (D,E)
        neuron_emb = neuron_emb.unsqueeze(0).expand(b, -1, -1)  # (B,D,E)

        z_ticks = []
        for _ in range(self.cfg.n_ticks):
            q = self.query(z).unsqueeze(1)  # (B,1,d_token)
            o, _ = self.attn(q, tokens, tokens, need_weights=False)
            o = o.squeeze(1)  # (B,d_token)

            a = self.synapse(torch.cat([z, o], dim=-1))  # (B,D)
            if self.cfg.memory_len > 1:
                a_hist = torch.cat([a_hist[:, :, 1:], a.unsqueeze(-1)], dim=-1)
            else:
                a_hist = a.unsqueeze(-1)

            inp = torch.cat([a_hist, neuron_emb], dim=-1)  # (B,D,M+E)
            z_next = self.nlm(inp.view(b * self.cfg.d_model, -1)).view(b, self.cfg.d_model)
            z = self.post_act(z_next)
            z_ticks.append(z)

        return torch.stack(z_ticks, dim=1)

