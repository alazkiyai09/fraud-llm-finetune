import argparse
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ProjectConfig
from src.dataset import FraudDatasetIO
from src.inference import FraudLLMInference


class MockKeywordTrainer:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_rows: list[dict], val_rows: list[dict], epochs: int) -> dict:
        label_keywords: dict[str, dict[str, int]] = {"LEGITIMATE": {}, "SUSPICIOUS": {}, "FRAUDULENT": {}}

        for row in train_rows:
            parsed = FraudLLMInference.parse_output(row["output"])
            label = parsed["classification"]
            tokens = [token.strip(".,:;()[]").lower() for token in row["input"].split()]
            for token in tokens:
                if len(token) < 4:
                    continue
                label_keywords[label][token] = label_keywords[label].get(token, 0) + 1

        keyword_rules = {
            label: [token for token, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:25]]
            for label, freq in label_keywords.items()
        }

        losses = []
        current = 1.25
        for epoch in range(epochs):
            current = max(0.12, current * 0.72)
            losses.append({"epoch": epoch + 1, "train_loss": round(current, 4), "eval_loss": round(current + 0.05, 4)})

        (self.output_dir / "keyword_rules.json").write_text(json.dumps(keyword_rules, indent=2), encoding="utf-8")
        (self.output_dir / "mock_adapter.json").write_text(
            json.dumps(
                {
                    "type": "mock_lora_adapter",
                    "epochs": epochs,
                    "train_size": len(train_rows),
                    "val_size": len(val_rows),
                    "losses": losses,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return {"losses": losses, "train_size": len(train_rows), "val_size": len(val_rows)}


def _build_sft_trainer(
    sft_trainer_cls: type,
    model,
    training_args,
    train_ds,
    val_ds,
    tokenizer,
    max_seq_length: int,
):
    # Handle minor TRL API differences without forcing immediate refactors.
    signature = inspect.signature(sft_trainer_cls.__init__)
    valid_params = set(signature.parameters.keys())

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
    }

    if "tokenizer" in valid_params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in valid_params:
        kwargs["processing_class"] = tokenizer

    if "max_seq_length" in valid_params:
        kwargs["max_seq_length"] = max_seq_length

    if "dataset_text_field" in valid_params:
        kwargs["dataset_text_field"] = "text"

    return sft_trainer_cls(**kwargs)



def run_real_qlora_training(config: ProjectConfig, train_rows: list[dict], val_rows: list[dict], output_dir: str) -> dict:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except Exception as exc:
        raise RuntimeError("Real QLoRA dependencies are not available. Use --use_mock_trainer.") from exc

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    model = get_peft_model(model, lora_cfg)

    def to_text(row: dict) -> dict:
        text = f"Instruction: {row['instruction']}\nInput: {row['input']}\nOutput: {row['output']}"
        return {"text": text}

    train_ds = Dataset.from_list([to_text(row) for row in train_rows])
    val_ds = Dataset.from_list([to_text(row) for row in val_rows])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        evaluation_strategy=config.training.evaluation_strategy,
        report_to=[config.training.report_to],
    )

    trainer = _build_sft_trainer(
        sft_trainer_cls=SFTTrainer,
        model=model,
        training_args=training_args,
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        max_seq_length=config.training.max_seq_length,
    )

    train_result = trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_loss": train_result.metrics.get("train_loss", None),
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA training entrypoint")
    parser.add_argument("--config", default="configs/qlora_config.yaml")
    parser.add_argument("--training_config", default="configs/training_config.yaml")
    parser.add_argument("--train_data", default="data/processed/train.jsonl")
    parser.add_argument("--val_data", default="data/processed/val.jsonl")
    parser.add_argument("--output_dir", default="results/lora_adapter")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--dataset_size", type=int, default=0)
    parser.add_argument("--use_mock_trainer", action="store_true")
    args = parser.parse_args()

    config = ProjectConfig.from_yaml_files(args.config, args.training_config)
    train_rows = FraudDatasetIO.load_jsonl(args.train_data)
    val_rows = FraudDatasetIO.load_jsonl(args.val_data)

    if args.dataset_size > 0:
        train_rows = train_rows[: args.dataset_size]
        val_rows = val_rows[: max(1, min(len(val_rows), args.dataset_size // 5))]

    epochs = config.training.num_epochs
    if args.max_steps > 0:
        epochs = max(1, min(epochs, args.max_steps))

    if args.use_mock_trainer:
        metrics = MockKeywordTrainer(args.output_dir).train(train_rows, val_rows, epochs)
    else:
        metrics = run_real_qlora_training(config, train_rows, val_rows, args.output_dir)

    metrics_path = Path("results/metrics/training_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
