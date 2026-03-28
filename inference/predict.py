import os
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FraudLLMInference


class ClassifyRequest(BaseModel):
    description: str = Field(..., min_length=10)
    max_tokens: int = Field(default=256, ge=32, le=512)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    classification: str
    reasoning: str
    risk_factors: list[str]
    recommended_action: str
    inference_time_ms: float
    mode: str


MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "results" / "merged_model"))
INFER = FraudLLMInference(model_path=MODEL_PATH)

app = FastAPI(title="FraudLLM API", version="0.1.0")


@app.post("/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest) -> ClassifyResponse:
    result = INFER.classify(
        description=payload.description,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
    )

    return ClassifyResponse(
        classification=result["classification"],
        reasoning="\n".join(result["reasoning"]),
        risk_factors=result["risk_factors"],
        recommended_action=result["recommended_action"],
        inference_time_ms=result["inference_time_ms"],
        mode=result["mode"],
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "mode": INFER.mode}
