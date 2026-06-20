"""Bio-RoPE positional encoding for protein language modeling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BioRoPE(nn.Module):
    """Biophysically modulated rotary positional encoding.

    Bio-RoPE partitions hidden channels into structural periodic, amino-acid
    physicochemical, and standard long-range RoPE components.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("dim must be even for rotary position encoding.")

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_properties = 3

        self.dim_per = int(dim * 0.2)
        if self.dim_per % 2 != 0:
            self.dim_per += 1

        self.dim_physio = int(dim * 0.3)
        if self.dim_physio % 2 != 0:
            self.dim_physio += 1

        self.dim_std = dim - self.dim_per - self.dim_physio
        if self.dim_std <= 0 or self.dim_std % 2 != 0:
            raise ValueError("Channel partitioning produced an invalid standard RoPE dimension.")

        periods = [2.0, 3.0, 3.6, 5.1]
        per_freqs = [2.0 * math.pi / periods[i % len(periods)] for i in range(self.dim_per // 2)]
        self.register_buffer("freqs_per", torch.tensor(per_freqs, dtype=torch.float32, device=device))

        physio_freqs = 1.0 / (
            base ** (torch.arange(0, self.dim_physio, 2, dtype=torch.float32, device=device) / self.dim_physio)
        )
        self.register_buffer("freqs_physio", physio_freqs)

        aa_properties = torch.tensor(
            [
                [1.8, 2.1, 0.5],      # A
                [-3.1, 0.2, 1.8],     # R
                [-2.8, -0.5, 1.0],    # N
                [-2.9, -0.8, 0.9],    # D
                [2.5, -1.0, 1.1],     # C
                [-2.5, -0.5, 1.3],    # Q
                [-2.7, 1.2, 1.2],     # E
                [-0.5, -2.5, -3.1],   # G
                [-1.5, -0.8, 1.6],    # H
                [3.0, 1.5, 1.6],      # I
                [2.8, 2.1, 1.5],      # L
                [-3.1, 0.1, 1.7],     # K
                [1.9, 1.5, 1.5],      # M
                [2.8, 0.3, 2.5],      # F
                [-1.0, -2.0, 0.8],    # P
                [-0.8, -0.2, -1.0],   # S
                [-0.6, 0.1, -0.5],    # T
                [2.0, -0.5, 3.1],     # W
                [1.2, -1.0, 2.2],     # Y
                [2.6, 1.1, 1.3],      # V
                [0.0, 0.0, 0.0],      # X/PAD
            ],
            dtype=torch.float32,
            device=device,
        )
        self.aa_embedding = nn.Embedding.from_pretrained(aa_properties, freeze=True)

        std_freqs = 1.0 / (
            base ** (torch.arange(0, self.dim_std, 2, dtype=torch.float32, device=device) / self.dim_std)
        )
        self.register_buffer("freqs_std", std_freqs)
        self.register_buffer("positions", torch.arange(max_seq_len, dtype=torch.float32, device=device))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_tokens: torch.Tensor) -> torch.Tensor:
        """Apply Bio-RoPE to query or key states.

        Args:
            x: Tensor with shape ``[batch, seq_len, dim]``.
            seq_tokens: Amino-acid token IDs with shape ``[batch, seq_len]``.

        Returns:
            Position-encoded tensor with the same shape as ``x``.
        """

        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}.")

        dtype = x.dtype
        device = x.device
        positions = self.positions[:seq_len].to(device=device, dtype=dtype)

        freqs_per = self.freqs_per.to(device=device, dtype=dtype)
        freqs_t_per = torch.einsum("i,j->ij", positions, freqs_per)
        emb_per = torch.cat((freqs_t_per, freqs_t_per), dim=-1)

        token_ids = seq_tokens.to(device=device, dtype=torch.long).clamp(min=0, max=20)
        props = self.aa_embedding(token_ids).to(dtype=dtype)
        repeats = (self.dim_physio // 2) // self.num_properties + 1
        props_expanded = props.repeat(1, 1, repeats)[..., : self.dim_physio // 2]

        freqs_physio = self.freqs_physio.to(device=device, dtype=dtype)
        freqs_t_physio = torch.einsum("i,j->ij", positions, freqs_physio)
        freqs_t_physio_shifted = freqs_t_physio.unsqueeze(0) + props_expanded
        emb_physio = torch.cat((freqs_t_physio_shifted, freqs_t_physio_shifted), dim=-1)

        freqs_std = self.freqs_std.to(device=device, dtype=dtype)
        freqs_t_std = torch.einsum("i,j->ij", positions, freqs_std)
        emb_std = torch.cat((freqs_t_std, freqs_t_std), dim=-1)

        emb_per_batch = emb_per.unsqueeze(0).expand(batch_size, -1, -1)
        emb_std_batch = emb_std.unsqueeze(0).expand(batch_size, -1, -1)
        emb_total = torch.cat([emb_per_batch, emb_physio, emb_std_batch], dim=-1)

        cos_emb = emb_total.cos()
        sin_emb = emb_total.sin()
        return (x * cos_emb) + (self._rotate_half(x) * sin_emb)
