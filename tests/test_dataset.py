from src.dataset import FraudDatasetBuilder, FraudDatasetIO, FraudDatasetValidator, split_rows


def test_generate_default_distribution_and_validity() -> None:
    rows = [sample.to_dict() for sample in FraudDatasetBuilder(seed=7).generate(total_examples=1000)]

    assert len(rows) == 1000
    assert all(FraudDatasetValidator.is_valid_row(row) for row in rows)

    dist = FraudDatasetValidator.class_distribution(rows)
    assert 0.35 <= dist["LEGITIMATE"] <= 0.45
    assert 0.25 <= dist["SUSPICIOUS"] <= 0.35
    assert 0.25 <= dist["FRAUDULENT"] <= 0.35


def test_generate_has_minimum_size_guardrail() -> None:
    rows = FraudDatasetBuilder(seed=1).generate(total_examples=10)
    assert len(rows) == 300


def test_split_rows_ratio() -> None:
    rows = [sample.to_dict() for sample in FraudDatasetBuilder(seed=9).generate(total_examples=1000)]
    train, val, test = split_rows(rows, seed=9)

    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_jsonl_roundtrip(tmp_path) -> None:
    rows = [sample.to_dict() for sample in FraudDatasetBuilder(seed=42).generate(total_examples=300)]
    target = tmp_path / "sample.jsonl"

    FraudDatasetIO.write_jsonl(str(target), rows)
    loaded = FraudDatasetIO.load_jsonl(str(target))

    assert loaded == rows
