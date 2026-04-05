import os
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FraudLLMInference


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "results" / "merged_model"))
ALLOW_RULE_BASED_FALLBACK = _env_flag("ALLOW_RULE_BASED_FALLBACK", False)
INFER = FraudLLMInference(
    model_path=MODEL_PATH,
    strict_loading=not ALLOW_RULE_BASED_FALLBACK,
    require_artifacts=not ALLOW_RULE_BASED_FALLBACK,
)

EXAMPLES = [
    [
        "Wire transfer of $49,900 to a newly opened account in the Cayman Islands. "
        "Account holder registered 3 days ago with minimal KYC documentation. "
        "Transaction initiated at 2:47 AM local time."
    ],
    [
        "Monthly payroll direct deposit of $3,500 from ABC Corp to employee checking account. "
        "Regular recurring transaction on the 15th of each month for the past 2 years."
    ],
    [
        "Series of 5 cash deposits of $9,800 each made at different branch locations within a 3-hour window. "
        "All deposits made to the same business account by the same individual."
    ],
    [
        "Online purchase of $89.99 from Amazon using a credit card on file. "
        "Shipping address matches billing address. "
        "Customer has 5-year account history with consistent spending patterns."
    ],
    [
        "Cryptocurrency purchase of $48,000 funded by multiple incoming wire transfers from 3 different countries "
        "received within the past 48 hours. Account opened 2 weeks ago."
    ],
]


def classify_transaction(description: str) -> str:
    if not description or not description.strip():
        return "Please provide a transaction narrative before running classification."

    result = INFER.classify(description=description, max_tokens=256, temperature=0.1)
    reasoning_lines = "\n".join(f"{idx}. {line}" for idx, line in enumerate(result["reasoning"], start=1))

    return (
        f"Classification: {result['classification']}\n"
        f"Recommended Action: {result['recommended_action']}\n\n"
        f"Reasoning:\n{reasoning_lines}\n\n"
        f"Risk Factors: {', '.join(result['risk_factors'])}\n"
        f"Inference Mode: {result['mode']}\n"
        f"Latency: {result['inference_time_ms']} ms"
    )



demo = gr.Interface(
    fn=classify_transaction,
    inputs=gr.Textbox(
        label="Transaction Description",
        lines=5,
        placeholder="Describe a banking transaction to classify...",
    ),
    outputs=[gr.Textbox(label="Classification & Analysis", lines=10)],
    title="🏦 FraudLLM — Transaction Narrative Classifier",
    description=(
        "Fine-tuned Mistral-7B (QLoRA) for banking fraud detection. "
        "Classifies transactions as LEGITIMATE, SUSPICIOUS, or FRAUDULENT "
        "with detailed reasoning."
    ),
    examples=EXAMPLES,
    theme=gr.themes.Base(primary_hue="blue", secondary_hue="slate"),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
