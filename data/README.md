# Data Notes

- `data/raw/`: optional raw corpora and references.
- `data/processed/`: instruction-format JSONL splits used for training/eval.

Each JSONL row must include:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "Classification: ...\n\nReasoning:\n1. ...\nRisk factors: ...\nRecommended action: ..."
}
```

Class targets:
- LEGITIMATE ~40%
- SUSPICIOUS ~30%
- FRAUDULENT ~30%
