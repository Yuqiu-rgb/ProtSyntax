# Data

The ProtSyntax benchmark dataset is hosted on Hugging Face:

https://huggingface.co/datasets/Ethan-Lin/PTMBenchmark-ProtSyntax

This repository does not store raw training, validation, or test data files. Keeping the dataset in a dedicated data repository makes versioning clearer and avoids committing large parquet artifacts to git.

## Dataset Scope

| Block | Description |
|---|---|
| General PTM site prediction | Site-centered samples across 40 PTM classes |
| Kinase-specific phosphorylation | Kinase-substrate recognition tasks |
| PTM crosstalk | Conditional dependencies between PTM events |
| Enzyme kinetic regression | Full-length protein records for kinetic parameters |

## Loading Example

```python
from datasets import load_dataset

dataset = load_dataset("Ethan-Lin/PTMBenchmark-ProtSyntax")
print(dataset)
```

## Recommended Practice

- Pin the dataset revision used in experiments.
- Keep downloaded artifacts outside the git repository or under a locally ignored cache directory.
- Report dataset split names and filtering rules in downstream papers or benchmarks.
