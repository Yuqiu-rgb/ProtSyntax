# Contributing

Thank you for your interest in ProtSyntax.

This repository is maintained as a research code release. Contributions that improve documentation, examples, packaging, reproducibility, or model integration are welcome.

## Development Workflow

1. Create a focused branch for your change.
2. Keep pull requests small and easy to review.
3. Run syntax checks before submitting:

```bash
python -m py_compile Core_code/*.py demo.py
```

4. Do not commit datasets, checkpoints, manuscript drafts, or local cache files.

## Reporting Issues

When reporting a bug, include:

- the Python and PyTorch versions;
- the operating system and GPU/CUDA environment, if relevant;
- the exact command that failed;
- a minimal traceback or reproduction snippet.
