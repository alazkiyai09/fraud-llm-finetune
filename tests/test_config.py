from pathlib import Path

import pytest

from src.config import ProjectConfig


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_project_config_from_yaml(tmp_path: Path) -> None:
    qlora = tmp_path / "qlora.yaml"
    training = tmp_path / "training.yaml"

    _write(
        qlora,
        """
base_model: "Qwen/Qwen2-7B-Instruct"
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj"]
  bias: "none"
  task_type: "CAUSAL_LM"
""".strip(),
    )
    _write(
        training,
        """
training:
  num_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 2
  learning_rate: 0.0001
  report_to: "none"
""".strip(),
    )

    cfg = ProjectConfig.from_yaml_files(str(qlora), str(training))

    assert cfg.base_model == "Qwen/Qwen2-7B-Instruct"
    assert cfg.lora.r == 8
    assert cfg.lora.lora_alpha == 16
    assert cfg.training.num_epochs == 1
    assert cfg.training.per_device_train_batch_size == 2
    assert cfg.training.report_to == "none"


def test_load_defaults_when_yaml_is_empty(tmp_path: Path) -> None:
    qlora = tmp_path / "qlora_empty.yaml"
    training = tmp_path / "training_empty.yaml"
    _write(qlora, "{}")
    _write(training, "{}")

    cfg = ProjectConfig.from_yaml_files(str(qlora), str(training))

    assert cfg.base_model == "mistralai/Mistral-7B-Instruct-v0.3"
    assert cfg.lora.r == 16
    assert cfg.training.num_epochs == 3


def test_missing_config_file_raises(tmp_path: Path) -> None:
    qlora = tmp_path / "missing.yaml"
    training = tmp_path / "training.yaml"
    _write(training, "{}")

    with pytest.raises(FileNotFoundError):
        ProjectConfig.from_yaml_files(str(qlora), str(training))


def test_invalid_yaml_root_type_raises(tmp_path: Path) -> None:
    qlora = tmp_path / "qlora_invalid.yaml"
    training = tmp_path / "training_valid.yaml"
    _write(qlora, "- not-an-object")
    _write(training, "{}")

    with pytest.raises(ValueError):
        ProjectConfig.from_yaml_files(str(qlora), str(training))
