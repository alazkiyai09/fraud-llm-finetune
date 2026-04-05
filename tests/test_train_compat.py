from src.config import ProjectConfig
from src.train import _build_sft_trainer, _build_training_args_kwargs


class _TrainerWithTokenizer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        tokenizer,
        max_seq_length,
        dataset_text_field,
    ) -> None:
        self.payload = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
            "dataset_text_field": dataset_text_field,
        }


class _TrainerWithProcessingClass:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        processing_class,
        max_seq_length,
    ) -> None:
        self.payload = {
            "model": model,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": processing_class,
            "max_seq_length": max_seq_length,
        }


def test_build_sft_trainer_uses_tokenizer_signature() -> None:
    trainer = _build_sft_trainer(
        sft_trainer_cls=_TrainerWithTokenizer,
        model="m",
        training_args="a",
        train_ds="train",
        val_ds="val",
        tokenizer="tok",
        max_seq_length=1024,
    )
    assert trainer.payload["tokenizer"] == "tok"
    assert trainer.payload["dataset_text_field"] == "text"


def test_build_sft_trainer_uses_processing_class_signature() -> None:
    trainer = _build_sft_trainer(
        sft_trainer_cls=_TrainerWithProcessingClass,
        model="m",
        training_args="a",
        train_ds="train",
        val_ds="val",
        tokenizer="tok",
        max_seq_length=1024,
    )
    assert trainer.payload["processing_class"] == "tok"
    assert trainer.payload["max_seq_length"] == 1024


def test_build_training_args_kwargs_includes_max_steps_when_positive() -> None:
    kwargs = _build_training_args_kwargs(ProjectConfig(), "results/out", max_steps=25)
    assert kwargs["max_steps"] == 25


def test_build_training_args_kwargs_omits_max_steps_when_zero() -> None:
    kwargs = _build_training_args_kwargs(ProjectConfig(), "results/out", max_steps=0)
    assert "max_steps" not in kwargs
