"""Bidirectional Gated DeltaNet blocks for ProtSyntax."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroCenteredRMSNorm(nn.Module):
    """RMS normalization initialized around an identity scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / (rms + self.eps) * (1.0 + self.weight)


class GatedDeltaNetCore(nn.Module):
    """Unidirectional gated DeltaNet recurrence.

    The state update follows a gated delta rule:
    ``S_t = S_{t-1} * alpha_t * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T``.
    """

    def __init__(self, d_model: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_inner = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)

        self.alpha_proj = nn.Linear(d_model, self.num_heads, bias=False)
        self.beta_proj = nn.Linear(d_model, self.num_heads, bias=False)
        self.struct_bias = nn.Parameter(torch.zeros(self.num_heads))

        self.q_norm = ZeroCenteredRMSNorm(self.head_dim)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        if reverse:
            x = torch.flip(x, dims=[1])

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        q = F.silu(self.q_norm(q))
        k = F.silu(self.k_norm(k))
        v = F.silu(v)

        alpha_logits = self.alpha_proj(x) + self.struct_bias
        alpha = torch.sigmoid(alpha_logits).view(batch, seq_len, self.num_heads, 1)
        beta = torch.sigmoid(self.beta_proj(x)).view(batch, seq_len, self.num_heads, 1)

        states = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        outputs = []
        identity = torch.eye(self.head_dim, device=x.device, dtype=x.dtype).view(
            1, 1, self.head_dim, self.head_dim
        )

        for t in range(seq_len):
            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]
            alpha_t = alpha[:, t]
            beta_t = beta[:, t]

            k_col = k_t.unsqueeze(-1)
            k_row = k_t.unsqueeze(-2)
            v_col = v_t.unsqueeze(-1)

            update_matrix = beta_t.unsqueeze(-1) * torch.matmul(v_col, k_row)
            decay_matrix = alpha_t.unsqueeze(-1) * (
                identity - beta_t.unsqueeze(-1) * torch.matmul(k_col, k_row)
            )

            states = torch.matmul(states, decay_matrix) + update_matrix
            h_t = torch.matmul(states, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(h_t)

        h_sequence = torch.stack(outputs, dim=1)

        if reverse:
            h_sequence = torch.flip(h_sequence, dims=[1])

        return h_sequence.reshape(batch, seq_len, self.d_inner)


class BiGatedDeltaNet(nn.Module):
    """Bidirectional DeltaNet with zero-parameter cross-gating fusion."""

    def __init__(self, d_model: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.forward_core = GatedDeltaNetCore(d_model, num_heads, head_dim)
        self.backward_core = GatedDeltaNetCore(d_model, num_heads, head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim * 2, d_model, bias=False)
        self.out_gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_forward = self.forward_core(x, reverse=False)
        h_backward = self.backward_core(x, reverse=True)

        cross_forward = h_forward * torch.sigmoid(h_backward)
        cross_backward = h_backward * torch.sigmoid(h_forward)
        h_fused = torch.cat([cross_forward, cross_backward], dim=-1)

        output = self.out_proj(h_fused)
        return output * F.silu(self.out_gate(x))
