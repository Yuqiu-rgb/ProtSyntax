# Model Weights

ProtSyntax model checkpoints are hosted externally and are intentionally not tracked in this git repository.

Do not commit model files such as `.pt`, `.pth`, `.ckpt`, `.bin`, or `.safetensors` to this directory. The repository keeps only the source code, demo, documentation, and release metadata.

## Expected Local Layout

After obtaining the released checkpoint files, place them locally in a structure compatible with the demo:

```text
Model_weight/
`-- ProtSyntax/
    |-- config.json
    |-- tokenizer.json
    |-- tokenizer_config.json
    |-- special_tokens_map.json
    `-- model.safetensors
```

Then pass `--model /path/to/checkpoint` to `demo.py` if your checkpoint is stored elsewhere.
