import json
from pathlib import Path

from src.dataset import FraudDatasetBuilder, FraudDatasetIO, split_rows
from src.evaluate import evaluate_model
from src.merge import run_mock_merge
from src.train import MockKeywordTrainer


def test_mock_pipeline_end_to_end(tmp_path: Path) -> None:
    rows = [sample.to_dict() for sample in FraudDatasetBuilder(seed=123).generate(total_examples=300)]
    train_rows, val_rows, test_rows = split_rows(rows, seed=123)

    processed_dir = tmp_path / "processed"
    FraudDatasetIO.write_jsonl(str(processed_dir / "train.jsonl"), train_rows)
    FraudDatasetIO.write_jsonl(str(processed_dir / "val.jsonl"), val_rows)
    FraudDatasetIO.write_jsonl(str(processed_dir / "test.jsonl"), test_rows)

    adapter_dir = tmp_path / "lora_adapter"
    trainer_metrics = MockKeywordTrainer(str(adapter_dir)).train(train_rows, val_rows, epochs=1)

    assert (adapter_dir / "mock_adapter.json").exists()
    assert (adapter_dir / "keyword_rules.json").exists()
    assert trainer_metrics["train_size"] == len(train_rows)

    merged_dir = tmp_path / "merged_model"
    run_mock_merge(str(adapter_dir), str(merged_dir))

    assert (merged_dir / "mock_merged.json").exists()

    metrics = evaluate_model(str(merged_dir), str(processed_dir / "test.jsonl"))

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert metrics["format_compliance"] >= 0.95
    assert "confusion_matrix" in metrics

    output = tmp_path / "eval_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    assert output.exists()
