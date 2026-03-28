import argparse
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FraudDatasetIO, FraudDatasetValidator, LABELS
from src.inference import FraudLLMInference

TARGETS = {
    "accuracy": 0.85,
    "f1_macro": 0.82,
    "f1_fraudulent": 0.80,
    "precision_fraudulent": 0.85,
    "recall_suspicious": 0.80,
    "inference_time_per_sample_ms": 500,
}



def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator



def _compute_metrics(y_true: list[str], y_pred: list[str], inference_times: list[float]) -> dict:
    labels = LABELS
    confusion = {actual: {pred: 0 for pred in labels} for actual in labels}

    for actual, pred in zip(y_true, y_pred, strict=True):
        confusion[actual][pred] += 1

    correct = sum(confusion[label][label] for label in labels)
    accuracy = _safe_div(correct, len(y_true))

    per_class = {}
    f1_values = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }
        f1_values.append(f1)

    avg_latency = _safe_div(sum(inference_times), len(inference_times))

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(sum(f1_values) / len(f1_values), 4),
        "f1_fraudulent": per_class["FRAUDULENT"]["f1"],
        "precision_fraudulent": per_class["FRAUDULENT"]["precision"],
        "recall_suspicious": per_class["SUSPICIOUS"]["recall"],
        "inference_time_per_sample_ms": round(avg_latency, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }



def evaluate_model(model_path: str, test_data: str) -> dict:
    inference = FraudLLMInference(model_path=model_path)
    rows = FraudDatasetIO.load_jsonl(test_data)

    y_true = []
    y_pred = []
    latencies = []
    format_ok = 0

    for row in rows:
        gold = FraudDatasetValidator.parse_output(row["output"])
        if not gold:
            continue

        started = time.perf_counter()
        prediction = inference.classify(row["input"])
        latency_ms = (time.perf_counter() - started) * 1000

        if all(key in prediction for key in ("classification", "reasoning", "risk_factors", "recommended_action")):
            format_ok += 1

        y_true.append(gold["classification"])
        y_pred.append(prediction["classification"])
        latencies.append(latency_ms)

    metrics = _compute_metrics(y_true=y_true, y_pred=y_pred, inference_times=latencies)
    metrics["format_compliance"] = round(_safe_div(format_ok, len(y_true)), 4)
    metrics["targets"] = TARGETS
    metrics["model_path"] = model_path

    return metrics



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned fraud model")
    parser.add_argument("--model_path", default="results/lora_adapter")
    parser.add_argument("--test_data", default="data/processed/test.jsonl")
    parser.add_argument("--output", default="results/metrics/eval_metrics.json")
    args = parser.parse_args()

    metrics = evaluate_model(model_path=args.model_path, test_data=args.test_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
