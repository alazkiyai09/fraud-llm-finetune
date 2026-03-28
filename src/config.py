from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass(slots=True)
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(slots=True)
class TrainingConfig:
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_seq_length: int = 1024
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    report_to: str = "wandb"


@dataclass(slots=True)
class ProjectConfig:
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml_files(cls, qlora_path: str, training_path: str) -> "ProjectConfig":
        qlora_payload = _read_yaml(qlora_path)
        training_payload = _read_yaml(training_path)

        quantization = QuantizationConfig(**qlora_payload.get("quantization", {}))
        lora = LoRAConfig(**qlora_payload.get("lora", {}))
        training = TrainingConfig(**training_payload.get("training", {}))

        return cls(
            base_model=qlora_payload.get("base_model", "mistralai/Mistral-7B-Instruct-v0.3"),
            quantization=quantization,
            lora=lora,
            training=training,
        )



def _read_yaml(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {file_path} must be a YAML object")
    return payload
