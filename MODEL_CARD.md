# ProtSyntax Model Card

## Model Summary

ProtSyntax is a PTM-aware protein language model for residue-level post-translational modification syntax learning and protein-level functional consequence modeling.

The released code focuses on the core research modules:

- Bio-RoPE positional encoding;
- Bi-Gated DeltaNet bidirectional sequence propagation;
- Geometric Gated Attention for structure-constrained residue interactions;
- PACE-Nash multi-task learning loss.

## Intended Use

ProtSyntax is intended for research use in:

- PTM site prediction;
- PTM type recovery and cloze-style syntax probing;
- PTM crosstalk modeling;
- kinase-specific phosphorylation analysis;
- enzyme kinetic representation learning;
- hypothesis generation for experimental follow-up.

## Out-of-Scope Use

ProtSyntax should not be used as a standalone clinical, diagnostic, therapeutic, or safety-critical decision system. Predictions should be interpreted as computational evidence and validated with appropriate biological experiments.

## Inputs

Typical inputs include:

- amino-acid sequences;
- residue-centered PTM candidate sites;
- optional protein structure features, such as C-alpha coordinates or residue-frame transforms;
- task-specific labels for PTM classes, kinase classes, crosstalk labels, or enzyme kinetic targets.

## Outputs

Depending on the task head, ProtSyntax may produce:

- PTM-site probabilities;
- PTM-type probabilities;
- kinase-specific phosphorylation predictions;
- crosstalk probabilities;
- enzyme kinetic regression estimates and uncertainty terms;
- intermediate residue or protein embeddings.

## Training Data

The ProtSyntax benchmark dataset is maintained separately from this code repository. See [Data/README.md](Data/README.md) for dataset access and recommended versioning practice.

## Limitations

- PTM databases are incomplete and biased toward well-studied organisms, proteins, residues, and modification types.
- Static or predicted structures do not fully represent conformational ensembles, transient complexes, membrane environments, or condensate dynamics.
- Rare PTM classes may remain data-limited even with transfer learning.
- PTM predictions may be affected by sequence isoform mismatch, uncertain residue localization, and noisy negative sampling.

## Recommended Reporting

When reporting ProtSyntax results, include:

- model checkpoint version;
- dataset revision and split;
- PTM classes evaluated;
- structural source and confidence filtering;
- thresholding strategy;
- metrics such as MCC, AUROC, AUPRC, AP, MAE, and R2 where appropriate.
