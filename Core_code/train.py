# -*- coding: utf-8 -*-
"""Train ProtSyntax with a shared sequence-structure backbone and PACE-Nash.


Input files:
    metadata.pt       Vocabulary, frozen amino acid properties, descriptors for
                      40 PTM classes, and class ordering.
    partitions.json   Global partition assignments shared by all data sources.
    train_batches.pt  Dynamically padded batches, each containing a single task.
    val_batches.pt    Validation batches disjoint from training proteins and
                      their homology clusters.

Data preparation:
    1. Pool full-length proteins across all tasks and cluster with CD-HIT at 50%
       sequence identity. Assign whole clusters to train/val/test with seed=42,
       targeting an 8:1:1 ratio. Generate windows after assigning partitions.
    2. Kinetic records must also satisfy MMseqs2 identity=0.50 and coverage=0.85,
       and preserve connected components of the enzyme-cluster/ligand-reaction
       graph. Km and Ki share a molecular identity namespace; kcat records are
       grouped by the complete substrate set.
    3. PTM and kinase inputs use 55-residue windows centered on candidate sites.
       Negative sites must be chemically compatible and lack annotations for
       the target modification across all sources. Exclude unknown labels with
       label_mask rather than treating them as negatives.
    4. Extract frozen ESM-C and SaProt features offline. AlphaFold2 structures
       provide C-alpha translations and local rotations derived from N/C-alpha/C
       backbone atoms. Mask missing or unreliable geometry with geometry_mask.
    5. Freeze ibm/MoLFormer-XL-both-10pct and enable deterministic attention.
       Mean-pool non-padding, non-special tokens and map features to 1280
       dimensions. Exclude molecular inputs exceeding 202 tokens. Canonicalize
       with RDKit, preserving stereochemistry and charges and removing atom maps.
    6. Standardize kcat to s^-1 and Km/Ki to mM. Exclude non-positive or censored
       measurements and records with ambiguous required ligands. Deduplicate
       measurements, then aggregate within each enzyme/target/ligand context:
       maximum for kcat and geometric mean for Km/Ki. Apply log10 afterward.
       Retain temperature, pH, and assay conditions as metadata, not model inputs.

Batch tensors (B: records; W: windows; L: padded sequence length):
    task: ptm / kinase / crosstalk / kcat / km / ki
    record_ids: length B; window_owner: [W], record index for each window
    aa_ids, positions, token_mask, geometry_mask: [W,L]
    esm_c: [W,L,D_esm]; saprot: [W,L,D_saprot]
    rotations: [W,L,3,3]; translations: [W,L,3], coordinates in angstroms
    pool_weight: [W,L], inverse residue coverage for overlapping kinetic windows;
                 zero for padding
    physchem: [B,D_phys], standardized using parameters fitted on training data
    PTM/kinase: center_index [B]; labels/label_mask [B,C], with C=40/4
    crosstalk: source_index/target_index/source_type/target_type [B];
               labels/label_mask [B,1]; both sites must lie in the same context
    Classification: contrast_labels/contrast_mask [B,40] for contrastive learning
    Kinetics: substrate_sum/target_substrate/inhibitor [B,1280]; target_log10 [B]
    Fill invalid features with finite zeros; use masks to determine validity.

metadata.pt fields: aa_vocab, padding_id, aa_properties [V,3], ptm_names,
ptm_descriptors [40,D_label], esm_dim, saprot_dim, and feature_version.
Each partitions.json record contains record_id/task/split/protein_id/sequence_hash/
cdhit_cluster; kinetic records also require enzyme_group/ligand_group/joint_component.
Proteins marked reserved_downstream=True and their homology clusters are restricted
to the test or external partition.

Usage:
    python ProtSyntax_Train.py --data-dir DATA --config CONFIG.json --out MODEL_DIR

Required CONFIG.json settings correspond to TrainConfig fields without defaults.
Experimental settings not specified in the paper must be supplied explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog, minimize
from sklearn.metrics import average_precision_score, matthews_corrcoef
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


CLASS_TASKS = ("ptm", "kinase", "crosstalk")
KINETIC_TASKS = ("kcat", "km", "ki")
TASKS = CLASS_TASKS + KINETIC_TASKS


@dataclass
class TrainConfig:
    # Supply these settings explicitly; the paper omits their values or task mappings.
    expert_hidden: int
    gga_heads: int
    max_steps: int
    early_stop_patience: int       # Validation checks allowed without improvement.
    focal_gamma_positive: float
    focal_gamma_negative: float
    physicochemical_margin: float
    batch_sizes: dict[str, int]    # 256 or 512 records per task batch.
    selection_task: str           # Select checkpoints by validation AP or negative MAE.

    seed: int = 42
    hidden: int = 1280
    layers: int = 33
    delta_heads: int = 16
    delta_head_dim: int = 80
    ptm_window: int = 55
    max_length: int = 1024
    window_stride: int = 512
    experts: int = 16
    top_k: int = 2
    residual_dropout: float = 0.1
    attention_dropout: float = 0.1
    norm_eps: float = 1e-6
    structural_fraction: float = 0.2
    physicochemical_fraction: float = 0.3
    rope_base: float = 10000.0
    geometric_probes: int = 4
    geometric_scale_init: float = 0.0
    contrastive_temperature: float = 0.7
    jaccard_threshold: float = 0.1
    probability_clip: float = 0.05
    variance_regularization: float = 0.01
    load_balance_coefficient: float = 0.01
    nash_update_frequency: int = 10
    nash_coefficient_lr: float = 0.01
    peak_lr: float = 4e-4
    min_lr: float = 4e-5
    warmup_steps: int = 5000
    weight_decay: float = 0.1
    validation_frequency: int = 500
    gradient_checkpointing: bool = True
    device: str = "cuda"

    def validate(self):
        assert self.layers == 33 and self.hidden == 1280
        assert self.delta_heads * self.delta_head_dim == self.hidden
        assert self.hidden % self.gga_heads == 0
        assert (self.hidden // self.gga_heads) % 2 == 0
        assert self.expert_hidden > 0 and self.max_steps > self.warmup_steps
        assert self.early_stop_patience > 0 and self.selection_task in TASKS
        assert set(self.batch_sizes) == set(TASKS)
        assert set(self.batch_sizes.values()) <= {256, 512}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def audit_partitions(records):
    """Verify global partition isolation across tasks using the existing split map."""
    scopes = {k: {} for k in (
        "protein_id", "sequence_hash", "cdhit_cluster", "enzyme_group",
        "ligand_group", "joint_component",
    )}
    reserved = {r["cdhit_cluster"] for r in records if r.get("reserved_downstream")}
    ids = set()
    for r in records:
        assert r["record_id"] not in ids, "Duplicate record ID"
        ids.add(r["record_id"])
        assert r["split"] in {"train", "val", "test", "external"}
        for key in ("protein_id", "sequence_hash", "cdhit_cluster"):
            assert r.get(key) is not None
        if r["task"] in KINETIC_TASKS:
            assert all(r.get(k) is not None for k in
                       ("enzyme_group", "ligand_group", "joint_component"))
        if r["cdhit_cluster"] in reserved:
            assert r["split"] in {"test", "external"}, "Reserved downstream protein in train or val"
        for key, mapping in scopes.items():
            if r.get(key) is None:
                continue
            previous = mapping.setdefault(str(r[key]), r["split"])
            assert previous == r["split"], f"Cross-partition leakage: {key}={r[key]}"


def load_batches(data_dir, split, records, cfg):
    batches = torch.load(Path(data_dir) / f"{split}_batches.pt",
                         map_location="cpu", weights_only=True)
    allowed = {r["record_id"]: r["task"] for r in records if r["split"] == split}
    seen = set()
    for b in batches:
        task = b["task"]
        assert task in TASKS and b["record_ids"]
        assert len(b["record_ids"]) <= cfg.batch_sizes[task]
        assert b["aa_ids"].shape[1] <= cfg.max_length
        assert b["token_mask"].any(dim=1).all()
        for rid in b["record_ids"]:
            assert rid in allowed and allowed[rid] == task
            assert rid not in seen, "Record appears in multiple batches within a partition"
            seen.add(rid)
        if task in CLASS_TASKS:
            assert torch.equal(b["window_owner"], torch.arange(len(b["record_ids"])))
            assert b["label_mask"].any(), "Classification batch contains no observed labels"
        if split == "train" and task in ("ptm", "kinase"):
            assert b["aa_ids"].shape[1] == cfg.ptm_window
    assert seen == set(allowed), "Batch records do not match the partition manifest"
    return batches


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # Supplementary Eqs. (5)-(6): divide by RMS + epsilon and scale by 1 + gamma.
        rms = x.float().square().mean(-1, keepdim=True).sqrt()
        return (x.float() / (rms + self.eps) * (1 + self.gamma)).to(x.dtype)


class BioRoPE(nn.Module):
    """Bio-RoPE, Supplementary Eqs. (1)-(3).

    Allocate paired channels to periodic (20%), physicochemical (30%), and
    standard rotary (50%) components, rounding to whole channel pairs.
    """

    def __init__(self, head_dim, aa_properties, cfg):
        super().__init__()
        pairs = head_dim // 2
        self.n_periodic = round(pairs * cfg.structural_fraction)
        self.n_physical = round(pairs * cfg.physicochemical_fraction)
        self.n_standard = pairs - self.n_periodic - self.n_physical
        periods = torch.tensor([2.0, 3.0, 3.6, 4.4])
        self.register_buffer("periods", periods.repeat(math.ceil(self.n_periodic / 4))
                             [:self.n_periodic])
        self.register_buffer("base_freq", cfg.rope_base **
                             (-torch.arange(pairs, dtype=torch.float32) / pairs))
        self.properties = nn.Embedding.from_pretrained(aa_properties.float(), freeze=True)
        self.phase_projection = nn.Linear(3, self.n_physical, bias=False)

    def forward(self, x, aa_ids, positions):
        # x: [W,H,L,D]. Both scan directions retain the original residue coordinates.
        p = positions.float().unsqueeze(-1)
        a = self.n_periodic
        b = a + self.n_physical
        phase = torch.cat((p * (2 * math.pi / self.periods),
                           p * self.base_freq[a:b] +
                           self.phase_projection(self.properties(aa_ids)).float(),
                           p * self.base_freq[b:]), dim=-1).unsqueeze(1)
        pair = x.float().reshape(*x.shape[:-1], -1, 2)
        even, odd = pair[..., 0], pair[..., 1]
        cos, sin = phase.cos(), phase.sin()
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
        return rotated.flatten(-2).to(x.dtype)


class GatedDeltaCore(nn.Module):
    """Supplementary Eqs. (4)-(9), with state axes ordered as value then key."""

    def __init__(self, cfg, aa_properties):
        super().__init__()
        self.heads, self.dim = cfg.delta_heads, cfg.delta_head_dim
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(self.dim, cfg.norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.dim, cfg.norm_eps)
        self.decay = nn.Linear(cfg.hidden, self.heads, bias=False)
        self.write = nn.Linear(cfg.hidden, self.heads, bias=False)
        self.structural_bias = nn.Parameter(torch.zeros(self.heads))
        self.rope = BioRoPE(self.dim, aa_properties, cfg)

    def forward(self, x, batch, reverse=False):
        w, length, _ = x.shape
        q, k, v = self.qkv(x).reshape(w, length, 3, self.heads, self.dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q = self.rope(F.silu(self.q_norm(q)), batch["aa_ids"], batch["positions"])
        k = self.rope(F.silu(self.k_norm(k)), batch["aa_ids"], batch["positions"])
        v = F.silu(v)
        alpha = torch.sigmoid(self.decay(x) + self.structural_bias).float()
        beta = torch.sigmoid(self.write(x)).float()
        state = torch.zeros(w, self.heads, self.dim, self.dim,
                            device=x.device, dtype=torch.float32)
        outputs = [None] * length
        order = range(length - 1, -1, -1) if reverse else range(length)
        for t in order:
            qt, kt, vt = q[:, :, t].float(), k[:, :, t].float(), v[:, :, t].float()
            a, b = alpha[:, t, :, None, None], beta[:, t, :, None, None]
            old_prediction = torch.einsum("bhvk,bhk->bhv", state, kt)
            # S_t = alpha*S - alpha*beta*(S*k)*k^T + beta*v*k^T.
            proposal = a * state + b * ((vt - a.squeeze(-1) * old_prediction)
                                        .unsqueeze(-1) * kt.unsqueeze(-2))
            valid = batch["token_mask"][:, t, None, None, None]
            state = torch.where(valid, proposal, state)
            out = torch.einsum("bhvk,bhk->bhv", state, qt)
            outputs[t] = out * batch["token_mask"][:, t, None, None]
        return torch.stack(outputs, dim=1).flatten(-2).to(x.dtype)


class BiGatedDeltaNet(nn.Module):
    def __init__(self, cfg, aa_properties):
        super().__init__()
        self.forward_core = GatedDeltaCore(cfg, aa_properties)
        self.backward_core = GatedDeltaCore(cfg, aa_properties)
        self.output = nn.Linear(2 * cfg.hidden, cfg.hidden, bias=False)
        self.gate = nn.Linear(cfg.hidden, cfg.hidden, bias=False)

    def forward(self, x, batch):
        fwd = self.forward_core(x, batch)
        bwd = self.backward_core(x, batch, reverse=True)
        # Supplementary Eqs. (11)-(12): parameter-free cross-gating, followed by
        # concatenation, projection, and an input-conditioned SiLU output gate.
        fused = torch.cat((fwd * bwd.sigmoid(), bwd * fwd.sigmoid()), dim=-1)
        return self.output(fused) * F.silu(self.gate(x))


class GeometricGatedAttention(nn.Module):
    def __init__(self, cfg, aa_properties):
        super().__init__()
        self.heads, self.dim = cfg.gga_heads, cfg.hidden // cfg.gga_heads
        self.probes = cfg.geometric_probes
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden, bias=False)
        self.points = nn.Linear(cfg.hidden, 2 * self.heads * self.probes * 3)
        nn.init.zeros_(self.points.weight)
        nn.init.zeros_(self.points.bias)
        self.gamma = nn.Parameter(torch.full((self.heads,), cfg.geometric_scale_init))
        self.output_gate = nn.Linear(cfg.hidden, self.heads)
        self.output = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.dropout = nn.Dropout(cfg.attention_dropout)
        self.rope = BioRoPE(self.dim, aa_properties, cfg)

    def forward(self, x, batch):
        w, length, _ = x.shape
        q, k, v = self.qkv(x).reshape(w, length, 3, self.heads, self.dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q = self.rope(q, batch["aa_ids"], batch["positions"])
        k = self.rope(k, batch["aa_ids"], batch["positions"])
        logits = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.dim)
        local = self.points(x).reshape(w, length, 2, self.heads, self.probes, 3)
        global_points = torch.einsum("blij,blshmj->blshmi",
                                     batch["rotations"].float(), local.float())
        global_points = global_points + batch["translations"][:, :, None, None, None, :]
        qp, kp = global_points.unbind(2)
        qp, kp = (p.permute(0, 2, 1, 3, 4).flatten(-2) for p in (qp, kp))
        distance2 = (qp.square().sum(-1, keepdim=True) +
                     kp.square().sum(-1).unsqueeze(-2) -
                     2 * (qp @ kp.transpose(-2, -1))).clamp_min(0)
        valid_geometry = batch["geometry_mask"] & batch["token_mask"]
        pair_mask = valid_geometry[:, None, :, None] & valid_geometry[:, None, None, :]
        penalty = (F.softplus(self.gamma)[None, :, None, None] * distance2 *
                   math.sqrt(2 / (9 * self.probes)))
        # Supplementary Eq. (15): softplus(gamma) equals log(2) when gamma is zero.
        # Zero local probes retain residue translations after transformation,
        # so zero initialization does not imply a zero geometric penalty.
        logits = logits - penalty.masked_fill(~pair_mask, 0)
        logits = logits.masked_fill(~batch["token_mask"][:, None, None, :], -torch.inf)
        attention = self.dropout(logits.softmax(-1)).to(v.dtype)
        out = attention @ v
        gate = self.output_gate(x).sigmoid().transpose(1, 2).unsqueeze(-1)
        out = (out * gate).transpose(1, 2).reshape(w, length, -1)
        return self.output(out) * batch["token_mask"].unsqueeze(-1)


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden, inner):
        super().__init__()
        self.input = nn.Linear(hidden, 2 * inner, bias=False)
        self.output = nn.Linear(inner, hidden, bias=False)

    def forward(self, x):
        gate, value = self.input(x).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class SparseMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.hidden, cfg.experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUExpert(cfg.hidden, cfg.expert_hidden)
                                     for _ in range(cfg.experts))

    def forward(self, x, mask):
        tokens = x[mask]
        probabilities = self.router(tokens).float().softmax(-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        routed = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_id, slot = torch.where(indices == expert_id)
            values = expert(tokens[token_id]) * weights[token_id, slot, None].to(x.dtype)
            routed.index_add_(0, token_id, values)
        out = torch.zeros_like(x)
        out[mask] = routed
        counts = F.one_hot(indices, len(self.experts)).float().sum(1).mean(0) / self.top_k
        # Standard routing balance term. Table S12 gives its coefficient (0.01)
        # without specifying the full formula.
        balance = len(self.experts) * (counts * probabilities.mean(0)).sum()
        return out, balance


class HybridBlock(nn.Module):
    def __init__(self, cfg, aa_properties, use_geometry):
        super().__init__()
        self.norm1 = ZeroCenteredRMSNorm(cfg.hidden, cfg.norm_eps)
        self.norm2 = ZeroCenteredRMSNorm(cfg.hidden, cfg.norm_eps)
        module = GeometricGatedAttention if use_geometry else BiGatedDeltaNet
        self.mixer = module(cfg, aa_properties)
        self.moe = SparseMoE(cfg)
        self.dropout = nn.Dropout(cfg.residual_dropout)

    def forward(self, x, batch):
        x = x + self.dropout(self.mixer(self.norm1(x), batch))
        routed, balance = self.moe(self.norm2(x), batch["token_mask"])
        x = x + self.dropout(routed)
        return x * batch["token_mask"].unsqueeze(-1), balance


class ProtSyntax(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden
        self.token = nn.Embedding(len(metadata["aa_vocab"]), hidden,
                                  padding_idx=metadata["padding_id"])
        self.esm_projection = nn.Linear(metadata["esm_dim"], hidden)
        self.saprot_projection = nn.Linear(metadata["saprot_dim"], hidden)
        self.task_embedding = nn.Embedding(len(TASKS), hidden)
        self.substrate_projection = nn.Linear(1280, hidden, bias=False)
        self.target_projection = nn.Linear(1280, hidden, bias=False)
        self.inhibitor_projection = nn.Linear(1280, hidden, bias=False)
        # [Delta, Delta, Delta, GGA] * 8 + Delta gives 25 Delta and 8 GGA blocks.
        self.blocks = nn.ModuleList(HybridBlock(cfg, metadata["aa_properties"], i % 4 == 3)
                                    for i in range(cfg.layers))
        self.final_norm = ZeroCenteredRMSNorm(hidden, cfg.norm_eps)
        self.register_buffer("ptm_descriptors", metadata["ptm_descriptors"].float())
        label_dim = self.ptm_descriptors.shape[1]
        # Task-head MLP widths are unspecified; use the shared hidden dimension.
        self.label_encoder = nn.Sequential(nn.Linear(label_dim, hidden), nn.SiLU(),
                                           nn.Linear(hidden, hidden))
        self.site_projection = nn.Linear(hidden, hidden)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.ptm_bias = nn.Parameter(torch.zeros(()))
        self.kinase_head = nn.Linear(hidden, 4)
        self.crosstalk_head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(),
                                            nn.Linear(hidden, 1))
        self.kinetic_heads = nn.ModuleDict({t: nn.Linear(hidden, 2) for t in KINETIC_TASKS})

    def shared_parameters(self):
        # Compute Nash gradient inner products over the shared backbone parameters.
        for module in (self.token, self.esm_projection, self.saprot_projection,
                       self.task_embedding, self.blocks, self.final_norm):
            yield from (p for p in module.parameters() if p.requires_grad)

    def encode(self, batch):
        task = batch["task"]
        x = (self.token(batch["aa_ids"]) + self.esm_projection(batch["esm_c"].detach()) +
             self.saprot_projection(batch["saprot"].detach()))
        x = x + self.task_embedding.weight[TASKS.index(task)]
        if task in KINETIC_TASKS:
            ligand = self.substrate_projection(batch["substrate_sum"].detach())
            if task == "km":
                ligand = ligand + self.target_projection(batch["target_substrate"].detach())
            if task == "ki":
                ligand = ligand + self.inhibitor_projection(batch["inhibitor"].detach())
            x = x + ligand[batch["window_owner"], None, :]
        x = x * batch["token_mask"].unsqueeze(-1)
        balance = x.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            if self.training and self.cfg.gradient_checkpointing:
                x, auxiliary = checkpoint(block, x, batch, use_reentrant=False)
            else:
                x, auxiliary = block(x, batch)
            balance = balance + auxiliary
        return self.final_norm(x), balance / len(self.blocks)

    def score_ptm(self, site_hidden, descriptors=None):
        descriptors = self.ptm_descriptors if descriptors is None else descriptors
        z = F.normalize(self.site_projection(site_hidden), dim=-1)
        query = F.normalize(self.label_encoder(descriptors), dim=-1)
        return self.log_scale.exp() * (z @ query.T) + self.ptm_bias

    def pool_records(self, hidden, batch):
        # Give each residue a total weight of one across overlapping windows,
        # then normalize the accumulated representation for each protein.
        weights = batch["pool_weight"] * batch["token_mask"]
        n = len(batch["record_ids"])
        sums = hidden.new_zeros(n, hidden.shape[-1])
        denominators = hidden.new_zeros(n, 1)
        sums.index_add_(0, batch["window_owner"], (hidden * weights[..., None]).sum(1))
        denominators.index_add_(0, batch["window_owner"], weights.sum(1, keepdim=True))
        return sums / denominators.clamp_min(1e-8)

    def forward(self, batch):
        hidden, balance = self.encode(batch)
        task = batch["task"]
        rows = torch.arange(len(batch["record_ids"]), device=hidden.device)
        out = {"balance": balance}
        if task in ("ptm", "kinase"):
            z = hidden[rows, batch["center_index"]]
            out["logits"] = self.score_ptm(z) if task == "ptm" else self.kinase_head(z)
        elif task == "crosstalk":
            source = hidden[rows, batch["source_index"]]
            target = hidden[rows, batch["target_index"]]
            labels = self.label_encoder(self.ptm_descriptors)
            pair = torch.cat((source, target, labels[batch["source_type"]],
                              labels[batch["target_type"]]), dim=-1)
            out["logits"] = self.crosstalk_head(pair)
            z = (source + target) / 2
        else:
            z = self.pool_records(hidden, batch)
            out["mu"], out["log_variance"] = self.kinetic_heads[task](z).float().unbind(-1)
            out["log_variance"] = out["log_variance"].clamp(-12, 12)
        out["representation"] = z
        return out


def asymmetric_focal(logits, targets, mask, cfg):
    p = logits.float().sigmoid()
    negative_p = (1 - p + cfg.probability_clip).clamp(max=1)
    positive = -targets * (1 - p).pow(cfg.focal_gamma_positive) * p.clamp_min(1e-8).log()
    negative = -(1 - targets) * (1 - negative_p).pow(cfg.focal_gamma_negative)
    negative = negative * negative_p.clamp_min(1e-8).log()
    return ((positive + negative) * mask).sum() / mask.sum().clamp_min(1)


def correlation_contrastive(z, labels, observed, cfg):
    """Supplementary Eqs. (17)-(19), using jointly observed labels for Jaccard overlap."""
    n = len(z)
    if n < 2:
        return z.sum() * 0
    shared = observed[:, None, :].bool() & observed[None, :, :].bool()
    positive = labels.bool()
    intersection = (positive[:, None, :] & positive[None, :, :] & shared).sum(-1)
    union = ((positive[:, None, :] | positive[None, :, :]) & shared).sum(-1)
    weights = intersection.float() / union.clamp_min(1)
    diagonal = torch.eye(n, dtype=torch.bool, device=z.device)
    weights = weights.masked_fill(diagonal | (weights < cfg.jaccard_threshold), 0)
    z = F.normalize(z.float(), dim=-1)
    similarities = z @ z.T / cfg.contrastive_temperature
    denominator = similarities.masked_fill(diagonal, -torch.inf).logsumexp(-1, keepdim=True)
    log_probability = (similarities - denominator).masked_fill(diagonal, 0)
    per_anchor = -(weights * log_probability).sum(-1) / weights.sum(-1).clamp_min(1e-8)
    return per_anchor.mean()


def physicochemical_penalty(z, descriptors, margin):
    n = len(z)
    if n < 2:
        return z.sum() * 0
    latent_distance = torch.cdist(z.float(), z.float())
    physical_distance = torch.cdist(descriptors.float(), descriptors.float())
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return F.relu(margin * physical_distance - latent_distance)[off_diagonal].mean()


def gaussian_kinetic_loss(mu, log_variance, target, cfg):
    # Supplementary Eq. (20): targets, means, and variances refer to log10 space.
    variance = log_variance.exp()
    nll = 0.5 * (math.log(2 * math.pi) + log_variance + (target - mu).square() / variance)
    return nll.mean() + cfg.variance_regularization * variance.square().mean()


class PACENash(nn.Module):
    """Balance L_PTM, L_kin, and L_phys using shared-parameter gradient products.

    The Supplementary Materials do not specify the inner Nash solver. This
    implementation maximizes sum(log(G @ a)) over simplex-constrained weights
    with positive gradient utilities, using SLSQP and a coefficient update
    rate of 0.01.

    Table S12 lists both 'Nash Inner LR=10' and 'Nash Coefficient Learning
    Rate=0.01'. The former is ambiguous and is not interpreted as an iteration
    count or used to override the latter.

    This strategy provides no guarantee of global Pareto optimality after
    non-convex training.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("coefficients", torch.full((3,), 1 / 3))

    def task_losses(self, outputs_and_batches):
        ptm, kinetic, physical, auxiliary = [], [], [], []
        for out, b in outputs_and_batches:
            z = out["representation"]
            physical.append(physicochemical_penalty(z, b["physchem"],
                                                     self.cfg.physicochemical_margin))
            auxiliary.append(out["balance"])
            if b["task"] in CLASS_TASKS:
                focal = asymmetric_focal(out["logits"], b["labels"].float(),
                                         b["label_mask"], self.cfg)
                contrast = correlation_contrastive(z, b["contrast_labels"],
                                                   b["contrast_mask"], self.cfg)
                ptm.append(focal + contrast)
            else:
                kinetic.append(gaussian_kinetic_loss(out["mu"], out["log_variance"],
                                                      b["target_log10"], self.cfg))
        # Aggregate batches from individual tasks into three objectives per step.
        # Require each objective to be present before taking its mean.
        assert ptm and kinetic and physical
        losses = torch.stack((torch.stack(ptm).mean(), torch.stack(kinetic).mean(),
                              torch.stack(physical).mean()))
        return losses, torch.stack(auxiliary).mean()

    def update_coefficients(self, losses, parameters):
        parameters = tuple(parameters)
        gradients = [torch.autograd.grad(loss, parameters, retain_graph=True,
                                         allow_unused=True) for loss in losses]
        gram = torch.zeros(3, 3, device=losses.device, dtype=torch.float64)
        # Accumulate inner products per parameter to avoid concatenating full gradients.
        for i in range(3):
            for j in range(i, 3):
                terms = [(a.detach().double() * b.detach().double()).sum()
                         for a, b in zip(gradients[i], gradients[j])
                         if a is not None and b is not None]
                if terms:
                    gram[i, j] = gram[j, i] = torch.stack(terms).sum()
        del gradients
        g = gram.cpu().numpy()
        scale = np.max(np.abs(g))
        if not np.isfinite(scale) or scale <= 1e-12:
            return
        g = g / scale
        eps = 1e-8
        current = self.coefficients.detach().cpu().double().numpy()
        # Find weights with positive utility for every objective. Retain the previous
        # coefficients if no feasible solution exists.
        feasible = linprog(np.zeros(3), A_ub=-g, b_ub=-np.ones(3) * eps,
                           A_eq=np.ones((1, 3)), b_eq=[1.0], bounds=[(0, 1)] * 3,
                           method="highs")
        if not feasible.success:
            return
        initial = current if np.all(g @ current >= eps) else feasible.x
        solution = minimize(
            lambda a: -np.log(np.maximum(g @ a, eps)).sum(), initial,
            jac=lambda a: -(g.T @ (1 / np.maximum(g @ a, eps))),
            method="SLSQP", bounds=[(0, 1)] * 3,
            constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1},
                         {"type": "ineq", "fun": lambda a: g @ a - eps}],
        )
        if solution.success and np.all(g @ solution.x >= eps * 0.9):
            new = (1 - self.cfg.nash_coefficient_lr) * initial
            new += self.cfg.nash_coefficient_lr * solution.x
            new = np.maximum(new, 0)
            self.coefficients.copy_(torch.as_tensor(new / new.sum(),
                                    dtype=self.coefficients.dtype, device=losses.device))


def learning_rate(step, cfg):
    if step <= cfg.warmup_steps:
        return cfg.peak_lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def infinite_batches(batches, seed):
    rng = random.Random(seed)
    while True:
        order = list(range(len(batches)))
        rng.shuffle(order)
        for i in order:
            yield batches[i]


@torch.no_grad()
def validation_predictions(model, batches, device):
    model.eval()
    grouped = {}
    for raw in batches:
        b = to_device(raw, device)
        out = model(b)
        task = b["task"]
        store = grouped.setdefault(task, {"prediction": [], "target": [], "mask": []})
        if task in CLASS_TASKS:
            store["prediction"].append(out["logits"].sigmoid().float().cpu().numpy())
            store["target"].append(b["labels"].cpu().numpy())
            store["mask"].append(b["label_mask"].cpu().numpy())
        else:
            store["prediction"].append(out["mu"].cpu().numpy())
            store["target"].append(b["target_log10"].cpu().numpy())
    return {task: {key: np.concatenate(value) for key, value in store.items() if value}
            for task, store in grouped.items()}


def fit_validation_thresholds(grouped):
    """Maximize validation MCC per class when both label outcomes are observed."""
    thresholds = {}
    for task, values in grouped.items():
        if task not in CLASS_TASKS:
            continue
        selected = []
        for col in range(values["prediction"].shape[1]):
            mask = values["mask"][:, col].astype(bool)
            y, p = values["target"][mask, col], values["prediction"][mask, col]
            if len(np.unique(y)) < 2:
                selected.append(None)
                continue
            # Threshold search is an implementation choice. Derive candidates solely
            # from validation predictions; test labels never enter this procedure.
            candidates = np.unique(np.r_[0.0, np.quantile(p, np.linspace(0, 1, 201)), 1.0])
            scores = [matthews_corrcoef(y, p >= threshold) for threshold in candidates]
            selected.append(float(candidates[int(np.argmax(scores))]))
        thresholds[task] = selected
    return thresholds


def checkpoint_selection_score(grouped, task):
    data = grouped[task]
    if task in KINETIC_TASKS:
        return -float(np.abs(data["target"] - data["prediction"]).mean())
    scores = []
    for col in range(data["prediction"].shape[1]):
        valid = data["mask"][:, col].astype(bool)
        y, p = data["target"][valid, col], data["prediction"][valid, col]
        if len(np.unique(y)) == 2:
            scores.append(average_precision_score(y, p))
    if not scores:
        raise ValueError("No validation class supports AP for the checkpoint-selection task")
    return float(np.mean(scores))


def train(cfg, metadata, train_batches, val_batches, output_dir):
    cfg.validate()
    seed_everything(cfg.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ProtSyntax(cfg, metadata).to(cfg.device)
    criterion = PACENash(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    # Draw one batch per task at each step, then combine the three Nash objectives.
    # Keep site windows and full-length enzyme records in separate task batches.
    by_task = {task: [b for b in train_batches if b["task"] == task] for task in TASKS}
    assert all(by_task.values()), "Joint training requires supervised batches for all six tasks"
    streams = {task: infinite_batches(batches, cfg.seed + i)
               for i, (task, batches) in enumerate(by_task.items())}
    best_score, no_improvement = -math.inf, 0
    for step in range(1, cfg.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, cfg)
        forward_results = []
        for task in TASKS:
            batch = to_device(next(streams[task]), cfg.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=cfg.device.startswith("cuda")):
                forward_results.append((model(batch), batch))
        losses, load_balance = criterion.task_losses(forward_results)
        if step % cfg.nash_update_frequency == 0:
            criterion.update_coefficients(losses, model.shared_parameters())
        total = (criterion.coefficients.detach() * losses).sum()
        total = total + cfg.load_balance_coefficient * load_balance
        if not torch.isfinite(total):
            raise FloatingPointError(f"step={step}: non-finite loss")
        total.backward()
        optimizer.step()
        if step % cfg.validation_frequency == 0 or step == cfg.max_steps:
            predictions = validation_predictions(model, val_batches, cfg.device)
            score = checkpoint_selection_score(predictions, cfg.selection_task)
            print(f"step={step} loss={total.item():.6f} validation_score={score:.6f}")
            if score > best_score:
                best_score, no_improvement = score, 0
                torch.save({
                    "model": model.state_dict(), "config": asdict(cfg),
                    "metadata": metadata, "step": step, "validation_score": score,
                    "thresholds": fit_validation_thresholds(predictions),
                    "threshold_source": "validation", "target_transform": "log10",
                    "nash_coefficients": criterion.coefficients.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                }, output_dir / "best.pt")
            else:
                no_improvement += 1
            if no_improvement >= cfg.early_stop_patience:
                break
    return output_dir / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Train ProtSyntax")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = TrainConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    metadata = torch.load(args.data_dir / "metadata.pt", map_location="cpu", weights_only=True)
    records = json.loads((args.data_dir / "partitions.json").read_text(encoding="utf-8"))
    assert len(metadata["ptm_names"]) == metadata["ptm_descriptors"].shape[0] == 40
    audit_partitions(records)
    train_batches = load_batches(args.data_dir, "train", records, cfg)
    val_batches = load_batches(args.data_dir, "val", records, cfg)
    print(train(cfg, metadata, train_batches, val_batches, args.out))


if __name__ == "__main__":
    main()
