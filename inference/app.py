import os
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FraudLLMInference

MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "results" / "merged_model"))
INFER = FraudLLMInference(model_path=MODEL_PATH)



def classify_transaction(description: str) -> str:
    result = INFER.classify(description=description, max_tokens=256, temperature=0.1)
    reasoning_lines = "\n".join(f"{idx}. {line}" for idx, line in enumerate(result["reasoning"], start=1))

    return (
        f"Classification: {result['classification']}\n\n"
        f"Reasoning:\n{reasoning_lines}\n\n"
        f"Risk factors: {', '.join(result['risk_factors'])}\n"
        f"Recommended action: {result['recommended_action']}\n"
        f"Inference mode: {result['mode']}\n"
        f"Latency: {result['inference_time_ms']} ms"
    )



demo = gr.Interface(
    fn=classify_transaction,
    inputs=gr.Textbox(
        label="Transaction Description",
        lines=6,
        placeholder="Enter a transaction description to classify...",
    ),
    outputs=gr.Textbox(label="Classification & Analysis", lines=14),
    title="FraudLLM - Transaction Narrative Classifier",
    description="Fine-tuned-style classifier for LEGITIMATE/SUSPICIOUS/FRAUDULENT narratives",
    examples=[
        [
            "Wire transfer of $49,900 to a newly opened account in the Cayman Islands. "
            "Account holder registered 3 days ago with minimal KYC documentation."
        ],
        ["Monthly salary deposit of $5,200 from Acme Corp to checking account. Regular pattern for 18 months."],
        ["Three cash deposits of $9,500, $9,800, and $9,700 within 2 hours at different branches."],
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
