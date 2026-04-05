# Fraud LLM Fine-tuning (QLoRA)

Fine-tuning pipeline for 3-class fraud narrative classification:

- `LEGITIMATE`
- `SUSPICIOUS`
- `FRAUDULENT`

## Live Demo

| Interface | URL |
|---|---|
| Interactive Classifier | https://fraud-llm-demo-5tphgb6fsa-as.a.run.app |

Includes:
- Data preparation and JSONL formatting
- QLoRA training script (real trainer + local smoke fallback)
- Evaluation pipeline (accuracy/F1/latency)
- Adapter merge/export flow
- Gradio + FastAPI inference deployment

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
```

Flexible install without the lockfile:

```bash
pip install -r requirements.txt
```

Notes:
- `trl` is pinned to `<0.9` because `SFTTrainer` constructor args changed across releases.
- Default mixed precision config targets Kaggle T4/P100 (`fp16=true`, `bf16=false`).

## Generate Processed Dataset

```bash
python3 src/dataset.py --generate --total_examples 6000 --output_dir data/processed
```

## Smoke Training (CPU-safe)

```bash
python3 src/train.py --config configs/qlora_config.yaml --training_config configs/training_config.yaml --use_mock_trainer --max_steps 10 --dataset_size 50
```

## Evaluate

```bash
python3 src/evaluate.py --model_path results/lora_adapter --test_data data/processed/test.jsonl --output results/metrics/eval_metrics.json
```

## Merge

```bash
python3 src/merge.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --adapter_path results/lora_adapter --output_dir results/merged_model --use_mock_merge
```

## Inference API

```bash
pip install -r inference/requirements.lock.txt
uvicorn inference.predict:app --reload --port 8002
```

## Gradio Demo

```bash
python3 inference/app.py
```

## Reproducible Dependency Audit

```bash
pip-audit -r requirements.lock.txt --no-deps --disable-pip
pip-audit -r inference/requirements.lock.txt --no-deps --disable-pip
```
