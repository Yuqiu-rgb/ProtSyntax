# -*- coding: utf-8 -*-
"""Evaluate ProtSyntax, profile protein PTM sites, and predict enzyme kinetics.

References: main text, Sections 4.1-4.4; Supplementary Materials, Sections 2.2
and 4.7.2, and Table S12. Place this file beside ProtSyntax_Train.py to reuse
the model architecture and batch tensor conventions.

The test directory contains metadata.pt, partitions.json, and test_batches.pt.
Test records inherit the global partition assignments shared by all supervision
sources. Do not resample records or randomly repartition individual windows.
Load validation-selected thresholds from best.pt; use test labels only for metrics.

Outputs:
    metrics.json       Per-class and macro MCC/AUC/AP for PTM, kinase, and
                       crosstalk tasks; R2 and MAE in log10 space for kcat/Km/Ki.
    predictions.csv    Per-record probabilities and predicted labels, or kinetic
                       means and uncertainty estimates.

Protein inference helpers:
    profile_ptm()          Use 55-residue centered windows and report native
                           residue positions with one-based indexing.
    make_kinetic_batch()   Build windows of up to 1024 residues at stride 512,
                           with inverse-coverage pooling weights.
    predict_zero_shot()    Generate PTM queries from chemical descriptors while
                           keeping all model parameters fixed.
    crosstalk_direction()  Compare target-site probabilities before and after
                           a substitution that removes source-site modifiability.

Usage:
    python ProtSyntax_Test.py --data-dir DATA --checkpoint MODEL_DIR/best.pt --output RESULTS
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from ProtSyntax_Train import (
    CLASS_TASKS,
    KINETIC_TASKS,
    ProtSyntax,
    TrainConfig,
    audit_partitions,
    load_batches,
    to_device,
)


def restore_model(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = TrainConfig(**state["config"])
    cfg.device = device
    model = ProtSyntax(cfg, state["metadata"])
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    assert state["threshold_source"] == "validation"
    assert state["target_transform"] == "log10"
    return model, state


def classification_metrics(targets, probabilities, mask, thresholds, class_names):
    per_class = {}
    for col, name in enumerate(class_names):
        valid = mask[:, col].astype(bool)
        y = targets[valid, col].astype(int)
        p = probabilities[valid, col]
        threshold = thresholds[col]
        result = {"n": len(y), "positives": int(y.sum()), "threshold": threshold,
                  "MCC": None, "AUC": None, "AP": None,
                  "accuracy": None, "F1": None}
        # Leave metrics unset when either binary label outcome is absent.
        if len(np.unique(y)) == 2:
            result["AUC"] = float(roc_auc_score(y, p))
            result["AP"] = float(average_precision_score(y, p))
            if threshold is not None:
                predicted = p >= threshold
                result["MCC"] = float(matthews_corrcoef(y, predicted))
                result["accuracy"] = float(accuracy_score(y, predicted))
                result["F1"] = float(f1_score(y, predicted, zero_division=0))
        per_class[name] = result
    macro, counts = {}, {}
    for metric in ("MCC", "AUC", "AP", "accuracy", "F1"):
        values = [item[metric] for item in per_class.values() if item[metric] is not None]
        macro[metric] = float(np.mean(values)) if values else None
        counts[metric] = len(values)
    return {"per_class": per_class, "macro": macro, "macro_valid_class_counts": counts}


def kinetic_metrics(target_log10, mu):
    return {
        "n": len(mu),
        "target_space": "log10",
        "R2": float(r2_score(target_log10, mu))
              if len(mu) > 1 and np.var(target_log10) > 0 else None,
        "MAE": float(mean_absolute_error(target_log10, mu)),
    }


def original_scale_prediction(mu, log_variance):
    sigma = math.sqrt(math.exp(log_variance))
    # 10**mu is the median in the original units, not the log-normal expectation.
    return {"median_original_units": 10.0 ** mu,
            "lower_95_original_units": 10.0 ** (mu - 1.96 * sigma),
            "upper_95_original_units": 10.0 ** (mu + 1.96 * sigma)}


@torch.inference_mode()
def evaluate(model, batches, state):
    grouped, rows = {}, []
    for raw in batches:
        b = to_device(raw, model.cfg.device)
        out = model(b)
        task = b["task"]
        store = grouped.setdefault(task, {"target": [], "prediction": [], "mask": []})
        if task in CLASS_TASKS:
            p = out["logits"].float().sigmoid().cpu().numpy()
            y = b["labels"].cpu().numpy()
            mask = b["label_mask"].cpu().numpy().astype(bool)
            thresholds = state["thresholds"][task]
            names = (state["metadata"]["ptm_names"] if task == "ptm" else
                     state["metadata"].get("kinase_names", [f"kinase_{i}" for i in range(4)])
                     if task == "kinase" else ["crosstalk"])
            assert len(names) == p.shape[1] == len(thresholds)
            for i, record_id in enumerate(b["record_ids"]):
                for col, name in enumerate(names):
                    threshold = thresholds[col]
                    rows.append({
                        "record_id": record_id, "task": task, "label": name,
                        "target": int(y[i, col]) if mask[i, col] else None,
                        "evaluated": bool(mask[i, col]), "probability": float(p[i, col]),
                        "threshold": threshold,
                        "predicted_label": int(p[i, col] >= threshold)
                                           if threshold is not None else None,
                    })
            store["mask"].append(mask)
            store["class_names"] = names
        else:
            p = out["mu"].cpu().numpy()
            log_variance = out["log_variance"].cpu().numpy()
            y = b["target_log10"].cpu().numpy()
            for i, record_id in enumerate(b["record_ids"]):
                mu, lv = float(p[i]), float(log_variance[i])
                rows.append({"record_id": record_id, "task": task,
                             "target_log10": float(y[i]), "mu_log10": mu,
                             "variance_log10": math.exp(lv),
                             "unit": "s^-1" if task == "kcat" else "mM",
                             **original_scale_prediction(mu, lv)})
        store["target"].append(y)
        store["prediction"].append(p)
    metrics = {}
    for task, data in grouped.items():
        y, p = np.concatenate(data["target"]), np.concatenate(data["prediction"])
        if task in CLASS_TASKS:
            metrics[task] = classification_metrics(
                y, p, np.concatenate(data["mask"]), state["thresholds"][task], data["class_names"])
        else:
            metrics[task] = kinetic_metrics(y, p)
    return metrics, rows


def export_results(metrics, rows, state, checkpoint_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(checkpoint_path), "checkpoint_step": state["step"],
              "feature_version": state["metadata"]["feature_version"],
              "inference_date_utc": datetime.now(timezone.utc).isoformat(),
              "threshold_source": "validation", "metrics": metrics}
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# Cached protein features align with native residue coordinates across the full
# sequence. Exclude additional rows for CLS/EOS tokens.
RESIDUE_FIELDS = ("aa_ids", "esm_c", "saprot", "rotations", "translations", "geometry_mask")


def pad_window(protein, start, length, padding_id):
    """Extract a window at any start index and mask positions outside the protein.

    Valid residues retain their native zero-based positions. Padded positions
    carry no biological coordinates and are excluded by token_mask.
    """
    n = len(protein["aa_ids"])
    positions = torch.arange(start, start + length)
    valid = (positions >= 0) & (positions < n)
    index = positions[valid]
    window = {"positions": positions.clamp_min(0), "token_mask": valid}
    for key in RESIDUE_FIELDS:
        source = protein[key]
        shape = (length,) + tuple(source.shape[1:])
        fill = padding_id if key == "aa_ids" else 0
        value = torch.full(shape, fill, dtype=source.dtype)
        value[valid] = source[index]
        window[key] = value
    window["geometry_mask"] = window["geometry_mask"].bool() & valid
    return window


def pack_windows(windows):
    return {key: torch.stack([window[key] for window in windows]) for key in windows[0]}


@torch.inference_mode()
def profile_ptm(model, protein, metadata, compatibility, thresholds, batch_size=256):
    """Profile candidate sites using a [residue_count,40] compatibility mask.

    Use the same 55-residue centered windows as training and convert native
    zero-based indices to one-based positions in the output. For sequence-only
    inputs, supply zero coordinate and rotation tensors with the expected
    shapes and set every geometry_mask entry to False.
    """
    length = len(protein["aa_ids"])
    assert compatibility.shape == (length, len(metadata["ptm_names"]))
    assert length > 0
    half = model.cfg.ptm_window // 2
    candidates = torch.where(compatibility.any(dim=-1))[0].tolist()
    results = []
    for offset in range(0, len(candidates), batch_size):
        sites = candidates[offset:offset + batch_size]
        windows = [pad_window(protein, site - half, model.cfg.ptm_window,
                              metadata["padding_id"]) for site in sites]
        batch = pack_windows(windows)
        batch.update(task="ptm", record_ids=[f"{protein['protein_id']}:{s + 1}" for s in sites],
                     window_owner=torch.arange(len(sites)),
                     center_index=torch.full((len(sites),), half, dtype=torch.long))
        probabilities = model(to_device(batch, model.cfg.device))["logits"].sigmoid().cpu()
        for row, site in enumerate(sites):
            for col in torch.where(compatibility[site])[0].tolist():
                p, threshold = float(probabilities[row, col]), thresholds[col]
                results.append({"protein_id": protein["protein_id"], "position": site + 1,
                                "ptm": metadata["ptm_names"][col], "probability": p,
                                "threshold": threshold,
                                "predicted_label": int(p >= threshold)
                                if threshold is not None else None})
    return results


def overlapping_starts(length, window=1024, stride=512):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def make_kinetic_batch(protein, ligand, task, metadata, cfg):
    """Build all windows for one enzyme while preserving its ligand context."""
    assert task in KINETIC_TASKS
    n = len(protein["aa_ids"])
    assert n > 0
    width = min(n, cfg.max_length)
    starts = overlapping_starts(n, cfg.max_length, cfg.window_stride)
    coverage = torch.zeros(n)
    for start in starts:
        coverage[start:start + width] += 1
    windows = []
    for start in starts:
        window = pad_window(protein, start, width, metadata["padding_id"])
        window["pool_weight"] = 1 / coverage[start:start + width]
        windows.append(window)
    batch = pack_windows(windows)
    batch.update(task=task, record_ids=[protein["protein_id"]],
                 window_owner=torch.zeros(len(windows), dtype=torch.long))
    # Sum embeddings over the complete reaction substrate set; do not average them.
    batch["substrate_sum"] = ligand["substrate_sum"].reshape(1, 1280)
    if task == "km":
        batch["target_substrate"] = ligand["target_substrate"].reshape(1, 1280)
    if task == "ki":
        batch["inhibitor"] = ligand["inhibitor"].reshape(1, 1280)
    return batch


@torch.inference_mode()
def predict_zero_shot(model, batch, unseen_descriptors):
    """Generate descriptor-based queries using Supplementary Eqs. (23)-(24).

    Keep the label encoder frozen and do not fit target-class thresholds. The
    checkpoint must come from leave-one-type/family-out training, with target
    positives, support/query proteins, and their homology clusters excluded
    from the relevant supervision sources according to the experiment protocol.
    A standard checkpoint trained on all 40 classes cannot establish zero-shot
    generalization through this helper alone.
    """
    assert batch["task"] == "ptm"
    batch = to_device(batch, model.cfg.device)
    hidden, _ = model.encode(batch)
    rows = torch.arange(len(batch["record_ids"]), device=hidden.device)
    sites = hidden[rows, batch["center_index"]]
    return model.score_ptm(sites, unseen_descriptors.to(hidden.device)).sigmoid().cpu()


def crosstalk_direction(native_probability, perturbed_probability,
                        same_residue=False, validation_delta_threshold=0.05):
    """Classify crosstalk direction as described in Supplementary Section 4.7.2.

    Define delta as P(target | source disabled) - P(target | native). Before
    predicting the perturbed sequence, recompute its ESM-C/SaProt features and
    residue-dependent physicochemical phases while retaining the original
    backbone geometry. Updating aa_ids alone would leave stale contextual
    features. The paper uses alanine substitution at the source site and checks
    direction consistency with residue-specific nonmodifiable substitutions
    and masked-site perturbations.

    The default threshold, 0.05, was selected on the paper's validation data.
    For a new experiment, pass the threshold fixed on its validation partition.
    """
    if same_residue:
        return {"direction": "mutually_exclusive_same_site", "delta": None}
    delta = perturbed_probability - native_probability
    direction = ("unresolved" if abs(delta) < validation_delta_threshold else
                 "cooperative" if delta < 0 else "antagonistic")
    return {"direction": direction, "delta": delta}


def main():
    parser = argparse.ArgumentParser(description="Test ProtSyntax")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model, state = restore_model(args.checkpoint, args.device)
    metadata = torch.load(args.data_dir / "metadata.pt", map_location="cpu", weights_only=True)
    assert metadata["ptm_names"] == state["metadata"]["ptm_names"], "Test class order differs from model"
    assert metadata["feature_version"] == state["metadata"]["feature_version"]
    records = json.loads((args.data_dir / "partitions.json").read_text(encoding="utf-8"))
    audit_partitions(records)
    batches = load_batches(args.data_dir, "test", records, model.cfg)
    metrics, predictions = evaluate(model, batches, state)
    export_results(metrics, predictions, state, args.checkpoint, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
