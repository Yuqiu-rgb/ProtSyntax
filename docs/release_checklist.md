# Release Checklist

Use this checklist before making a public release for manuscript review.

## Repository

- [ ] Confirm the repository is public.
- [ ] Confirm the default branch is `main`.
- [ ] Confirm `README.md` renders figures correctly on GitHub.
- [ ] Confirm Apache-2.0 license metadata appears in GitHub.
- [ ] Confirm manuscript source files and local model checkpoints are not tracked.

## Data And Weights

- [ ] Pin the Hugging Face dataset revision used in the manuscript.
- [ ] Confirm `Data/README.md` points to the public dataset.
- [ ] Confirm model checkpoints are hosted externally.
- [ ] Confirm `Model_weight/README.md` contains no checkpoint URL if the weight link should remain unpublished during review.

## Validation

- [ ] Run `python -m py_compile Core_code/*.py demo.py`.
- [ ] Test the demo in an environment with PyTorch, Transformers, and Biopython installed.
- [ ] Verify the released checkpoint can be loaded with `trust_remote_code=True`.
- [ ] Verify the example PDB preprocessing path matches the model's expected structural input format.

## Manuscript Metadata

- [ ] Update `CITATION.cff` when final author list and venue information are available.
- [ ] Replace the placeholder BibTeX entry in `README.md` after acceptance or preprint release.
- [ ] Add DOI, model checkpoint revision, and dataset revision to the final release notes.
