"""PACE-Nash multi-task learning losses for ProtSyntax."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PACENashLoss(nn.Module):
    """Physicochemical-aware contrastive and evidential Nash loss."""

    def __init__(
        self,
        num_tasks: int = 3,
        temperature: float = 0.07,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.gamma = focal_gamma
        self.alpha = focal_alpha

    def _correlation_aware_contrastive_loss(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        tau: float = 0.1,
    ) -> torch.Tensor:
        z = F.normalize(z, p=2, dim=1)
        similarity_matrix = torch.matmul(z, z.T) / self.temperature

        intersection = torch.matmul(labels, labels.T)
        sum_labels = labels.sum(dim=1, keepdim=True)
        union = sum_labels + sum_labels.T - intersection + 1e-8
        jaccard_weights = intersection / union

        mask = torch.eye(labels.size(0), dtype=torch.bool, device=z.device)
        jaccard_weights = jaccard_weights.masked_fill(mask, 0.0)
        jaccard_weights = torch.where(
            jaccard_weights > tau,
            jaccard_weights,
            torch.zeros_like(jaccard_weights),
        )

        exp_sim = torch.exp(similarity_matrix)
        exp_sim_sum = exp_sim.masked_fill(mask, 0.0).sum(dim=1, keepdim=True) + 1e-8
        log_prob = similarity_matrix - torch.log(exp_sim_sum)
        loss = -(jaccard_weights * log_prob).sum(dim=1) / (jaccard_weights.sum(dim=1) + 1e-8)

        valid_mask = jaccard_weights.sum(dim=1) > 0
        if not valid_mask.any():
            return z.sum() * 0.0
        return loss[valid_mask].mean()

    def _asymmetric_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = focal_weight * torch.pow(1.0 - p_t, self.gamma) * bce_loss
        return loss.mean()

    @staticmethod
    def _evidential_lognormal_loss(
        mu: torch.Tensor,
        log_var: torch.Tensor,
        targets: torch.Tensor,
        lambda_var: float = 0.01,
    ) -> torch.Tensor:
        var = torch.exp(log_var) + 1e-6
        nll_loss = 0.5 * (math.log(2.0 * math.pi) + log_var + ((targets - mu) ** 2) / var)
        var_reg = lambda_var * torch.norm(var, p=2)
        return nll_loss.mean() + var_reg

    @staticmethod
    def _physicochemical_manifold_penalty(
        z: torch.Tensor,
        physico_props: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        z_dist = torch.cdist(z, z, p=2)
        p_dist = torch.cdist(physico_props, physico_props, p=2)

        penalty = F.relu(margin * p_dist - z_dist)
        mask = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        penalty = penalty.masked_fill(mask, 0.0)
        return penalty.mean()

    def compute_task_losses(
        self,
        embeddings: torch.Tensor,
        class_logits: torch.Tensor,
        class_targets: torch.Tensor,
        reg_mu: torch.Tensor,
        reg_logvar: torch.Tensor,
        reg_targets: torch.Tensor,
        physico_props: torch.Tensor,
    ) -> torch.Tensor:
        l_contrast = self._correlation_aware_contrastive_loss(embeddings, class_targets)
        l_focal = self._asymmetric_focal_loss(class_logits, class_targets)
        l_class = l_contrast + l_focal

        l_reg = self._evidential_lognormal_loss(reg_mu, reg_logvar, reg_targets)
        l_physico = self._physicochemical_manifold_penalty(embeddings, physico_props)

        return torch.stack([l_class, l_reg, l_physico])


class NashMTLOptimizer:
    """Nash-style gradient aggregation for shared multi-task parameters."""

    def __init__(
        self,
        model: nn.Module,
        shared_layer_names: Sequence[str],
        num_tasks: int = 3,
        update_freq: int = 10,
        lr: float = 0.1,
    ) -> None:
        self.model = model
        self.shared_layer_names = tuple(shared_layer_names)
        self.num_tasks = num_tasks
        self.update_freq = update_freq
        self.lr = lr
        self.step = 0
        self.alpha = torch.ones(num_tasks) / num_tasks

    def _shared_parameters(self) -> list[torch.nn.Parameter]:
        return [
            param
            for name, param in self.model.named_parameters()
            if param.requires_grad and any(layer_name in name for layer_name in self.shared_layer_names)
        ]

    @staticmethod
    def _flatten_grads(
        grads: Iterable[torch.Tensor | None],
        params: Sequence[torch.nn.Parameter],
    ) -> torch.Tensor:
        flat = []
        for grad, param in zip(grads, params):
            if grad is None:
                flat.append(torch.zeros_like(param).reshape(-1))
            else:
                flat.append(grad.reshape(-1))
        return torch.cat(flat)

    def backward_nash(self, losses: torch.Tensor) -> torch.Tensor:
        if losses.numel() != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} task losses, got {losses.numel()}.")

        device = losses.device
        self.alpha = self.alpha.to(device)

        if self.step % self.update_freq == 0:
            shared_params = self._shared_parameters()
            if not shared_params:
                raise ValueError("No shared parameters matched shared_layer_names.")

            grad_vectors = []
            for task_idx in range(self.num_tasks):
                grads = torch.autograd.grad(
                    losses[task_idx],
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_vectors.append(self._flatten_grads(grads, shared_params))

            grad_matrix = torch.stack(grad_vectors)
            gram = torch.matmul(grad_matrix, grad_matrix.T)

            alpha = torch.ones(self.num_tasks, device=device) / self.num_tasks
            for _ in range(50):
                projected = torch.clamp(torch.matmul(gram, alpha), min=1e-8)
                alpha = alpha + self.lr * (1.0 / projected - alpha * (1.0 / projected).sum())
                alpha = torch.clamp(alpha, min=1e-4)
                alpha = alpha / alpha.sum()

            self.alpha = alpha.detach()

        self.model.zero_grad(set_to_none=True)
        total_loss = torch.sum(self.alpha * losses)
        total_loss.backward()
        self.step += 1
        return total_loss
