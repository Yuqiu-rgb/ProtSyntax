"""Geometric Gated Attention used by ProtSyntax.

The module augments semantic multi-head attention with a learnable geometric
penalty computed from residue-frame probe points. It is designed for PTM
reasoning where sequence-compatible residues must also be plausible in 3D
microenvironments.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricGatedAttention(nn.Module):
    """Structure-aware gated attention for residue-level protein modeling.

    Args:
        dim: Hidden dimension of the input representation.
        num_heads: Number of semantic attention heads.
        num_points: Number of learnable 3D probe points per attention head.
        dropout: Dropout applied to attention probabilities.
    """

    def __init__(
        self,
        dim: int = 1280,
        num_heads: int = 16,
        num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_points = num_points
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        point_dim = num_heads * num_points * 3
        self.q_point_proj = nn.Linear(dim, point_dim, bias=False)
        self.k_point_proj = nn.Linear(dim, point_dim, bias=False)

        self.gamma = nn.Parameter(torch.zeros(num_heads))
        self.gate_proj = nn.Linear(dim, num_heads, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize geometry projections as an identity-like warm start."""

        nn.init.zeros_(self.q_point_proj.weight)
        nn.init.zeros_(self.k_point_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _apply_rigid_transform(
        self,
        points: torch.Tensor,
        rots: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """Map local residue-frame probe points into global coordinates.

        Args:
            points: Local probe points with shape ``[B, L, H, P, 3]``.
            rots: Residue rotation matrices with shape ``[B, L, 3, 3]``.
            trans: Residue translations with shape ``[B, L, 3]``.

        Returns:
            Global probe points with shape ``[B, L, H, P, 3]``.
        """

        batch, length, heads, points_per_head, _ = points.shape
        flat_points = points.reshape(batch, length, heads * points_per_head, 3)
        rotated = torch.matmul(rots.unsqueeze(2), flat_points.unsqueeze(-1)).squeeze(-1)
        global_points = rotated + trans.unsqueeze(2)
        return global_points.reshape(batch, length, heads, points_per_head, 3)

    @staticmethod
    def _prepare_attention_mask(
        attention_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Convert common mask layouts to additive attention logits."""

        if attention_mask is None:
            return None

        if attention_mask.dim() == 2:
            additive_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
            return additive_mask
        if attention_mask.dim() == 3:
            additive_mask = (1.0 - attention_mask[:, None, :, :].to(dtype)) * torch.finfo(dtype).min
            return additive_mask
        return attention_mask.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rots: Optional[torch.Tensor] = None,
        trans: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run geometric gated attention.

        Args:
            hidden_states: Input tensor with shape ``[B, L, D]``.
            rots: Optional residue-frame rotations with shape ``[B, L, 3, 3]``.
            trans: Optional residue-frame translations with shape ``[B, L, 3]``.
            attention_mask: Optional additive or binary attention mask.

        Returns:
            A tuple of ``(output, attention_weights)``.
        """

        batch, length, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if rots is not None and trans is not None:
            q_points_local = self.q_point_proj(hidden_states).view(
                batch, length, self.num_heads, self.num_points, 3
            )
            k_points_local = self.k_point_proj(hidden_states).view(
                batch, length, self.num_heads, self.num_points, 3
            )

            q_points_global = self._apply_rigid_transform(q_points_local, rots, trans).transpose(1, 2)
            k_points_global = self._apply_rigid_transform(k_points_local, rots, trans).transpose(1, 2)

            q_pts_exp = q_points_global.unsqueeze(3)
            k_pts_exp = k_points_global.unsqueeze(2)
            sq_dist = torch.sum((q_pts_exp - k_pts_exp) ** 2, dim=(-2, -1))

            gamma = F.softplus(self.gamma).view(1, self.num_heads, 1, 1)
            distance_scale = math.sqrt(2.0 / (9.0 * self.num_points))
            attn_logits = attn_logits - gamma * sq_dist * distance_scale

        additive_mask = self._prepare_attention_mask(attention_mask, attn_logits.dtype)
        if additive_mask is not None:
            attn_logits = attn_logits + additive_mask

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v).transpose(1, 2)
        gates = torch.sigmoid(self.gate_proj(hidden_states)).unsqueeze(-1)
        gated_attn_output = attn_output * gates

        output = self.out_proj(gated_attn_output.contiguous().view(batch, length, self.dim))
        return output, attn_weights
