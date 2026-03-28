from src.inference import FraudLLMInference


def test_build_prompt_contains_description() -> None:
    infer = FraudLLMInference(model_path="/tmp/non-existent-model")
    prompt = infer.build_prompt("Monthly salary deposit from Acme Corp")

    assert "Classify the following transaction description" in prompt
    assert "Transaction: Monthly salary deposit from Acme Corp" in prompt


def test_parse_output_structured() -> None:
    output = (
        "Classification: FRAUDULENT\n\n"
        "Reasoning:\n"
        "1. Amount near threshold\n"
        "2. New account with weak KYC\n"
        "3. Offshore routing pattern\n\n"
        "Risk factors: structuring, weak kyc, offshore transfer\n"
        "Recommended action: BLOCK and file SAR"
    )

    parsed = FraudLLMInference.parse_output(output)

    assert parsed["classification"] == "FRAUDULENT"
    assert len(parsed["reasoning"]) >= 3
    assert "structuring" in parsed["risk_factors"]
    assert parsed["recommended_action"] == "BLOCK and file SAR"


def test_parse_output_fallback_uses_detected_label() -> None:
    parsed = FraudLLMInference.parse_output("This looks FRAUDULENT based on multiple red flags")
    assert parsed["classification"] == "FRAUDULENT"
    assert parsed["recommended_action"] == "REVIEW"


def test_rule_based_classification_fraudulent() -> None:
    infer = FraudLLMInference(model_path="/tmp/non-existent-model")
    description = (
        "Wire transfer just below threshold to newly opened offshore account with minimal KYC. "
        "Pattern suggests structuring and immediate onward transfer."
    )

    result = infer.classify(description)

    assert result["classification"] in {"FRAUDULENT", "SUSPICIOUS"}
    assert isinstance(result["risk_factors"], list)
    assert "recommended_action" in result


def test_format_output_roundtrip_parse() -> None:
    infer = FraudLLMInference(model_path="/tmp/non-existent-model")
    parsed = {
        "classification": "LEGITIMATE",
        "reasoning": [
            "Pattern matches recurring payroll behavior",
            "Counterparty and KYC profile are stable",
            "No high-risk indicators detected",
        ],
        "risk_factors": ["low behavioral variance", "verified counterparty"],
        "recommended_action": "APPROVE",
    }

    text = infer.format_output(parsed)
    reparsed = infer.parse_output(text)

    assert reparsed["classification"] == "LEGITIMATE"
    assert reparsed["recommended_action"] == "APPROVE"
    assert "verified counterparty" in reparsed["risk_factors"]
