# Fraud LLM Fine-tune

QLoRA fine-tuning pipeline for fraud narrative classification (`LEGITIMATE`, `SUSPICIOUS`, `FRAUDULENT`).

## Live Demo (Deployment Placeholder)

- HuggingFace Space: `https://huggingface.co/spaces/alazkiyai09/fraud-llm`
- Optional API endpoint: `https://fraud-llm-api-xxxxx-as.a.run.app`
- Status: `pending deployment`

## Notes

- `trl` pinned to `<0.9` for SFTTrainer API stability.
- Default mixed precision for T4/P100: `fp16=true`, `bf16=false`.

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate Dataset

```bash
python3 src/dataset.py --generate --total_examples 6000 --output_dir data/processed
```

## Smoke Train / Merge / Evaluate

```bash
python3 src/train.py --config configs/qlora_config.yaml --training_config configs/training_config.yaml --use_mock_trainer --max_steps 10 --dataset_size 50
python3 src/merge.py --base_model mistralai/Mistral-7B-Instruct-v0.3 --adapter_path results/lora_adapter --output_dir results/merged_model --use_mock_merge
python3 src/evaluate.py --model_path results/lora_adapter --test_data data/processed/test.jsonl --output results/metrics/eval_metrics.json
```

## Serve

```bash
uvicorn inference.predict:app --reload --port 8002
python3 inference/app.py
```

Update live links above once deployed.
