import html
import os
import re
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

EXAMPLE_LIBRARY = {
    "Offshore Wire Escalation": {
        "Baseline": (
            "Wire transfer of $49,900 to a newly opened account in the Cayman Islands. "
            "Account holder registered 3 days ago with minimal KYC documentation. "
            "Transaction initiated at 2:47 AM local time."
        ),
        "Variation A": (
            "Cross-border transfer of $48,750 to an offshore beneficiary created this week. "
            "Customer profile has documentation gaps and transaction executed outside normal operating hours."
        ),
        "Variation B": (
            "High-value transfer of $49,950 to a new international recipient. "
            "Sender account shows sudden activity spike and immediate onward transfer intent."
        ),
    },
    "Routine Payroll": {
        "Baseline": (
            "Monthly payroll direct deposit of $3,500 from ABC Corp to employee checking account. "
            "Regular recurring transaction on the 15th of each month for the past 2 years."
        ),
        "Variation A": (
            "Recurring salary payment from verified employer to employee account. "
            "Amount and date pattern are stable over 18 months."
        ),
        "Variation B": (
            "Biweekly payroll credit from known corporate account to long-standing beneficiary. "
            "No unusual timing or counterparty changes."
        ),
    },
    "Structuring Pattern": {
        "Baseline": (
            "Series of 5 cash deposits of $9,800 each made at different branch locations within a 3-hour window. "
            "All deposits made to the same business account by the same individual."
        ),
        "Variation A": (
            "Multiple sub-threshold deposits were made to one account across several branches in a short period. "
            "Behavior suggests possible structuring activity."
        ),
        "Variation B": (
            "Cash deposits just below reporting threshold repeated four times in one afternoon for a newly active account."
        ),
    },
    "Consumer Purchase": {
        "Baseline": (
            "Online purchase of $89.99 from Amazon using a credit card on file. "
            "Shipping address matches billing address. "
            "Customer has 5-year account history with consistent spending patterns."
        ),
        "Variation A": (
            "Small e-commerce purchase from a familiar merchant using known device and matching address details."
        ),
        "Variation B": (
            "Routine retail payment with low amount and stable customer behavior profile."
        ),
    },
}


def _mode_status(mode: str) -> tuple[str, str]:
    if mode == "transformers":
        return (
            "LIVE MODEL MODE",
            "Using transformer model artifacts for generation-based classification.",
        )
    if mode == "mock_adapter":
        return (
            "DEMO ADAPTER MODE",
            "Using bundled demo adapter artifacts (deterministic local inference behavior).",
        )
    return (
        "RULE-BASED FALLBACK MODE",
        "Using keyword-rule fallback logic. Results are demo-safe and not full model inference.",
    )


def _class_badge(classification: str) -> str:
    colors = {
        "LEGITIMATE": ("#0F766E", "#CCFBF1"),
        "SUSPICIOUS": ("#9A3412", "#FFEDD5"),
        "FRAUDULENT": ("#991B1B", "#FEE2E2"),
    }
    bg, fg = colors.get(classification, ("#1D4ED8", "#DBEAFE"))
    return (
        f"<div style='display:inline-block;background:{bg};color:{fg};"
        "padding:0.35rem 0.75rem;border-radius:999px;font-weight:700;'>"
        f"{classification}</div>"
    )


def _result_cards(result: dict) -> tuple[str, str, str, str, str, str]:
    classification = str(result.get("classification", "UNKNOWN"))
    action = str(result.get("recommended_action", "REVIEW"))
    reasoning = result.get("reasoning") or []
    risk_factors = result.get("risk_factors") or []
    mode_title, mode_detail = _mode_status(str(result.get("mode", "rule_based")))

    classification_card = "### Classification\n" f"{_class_badge(classification)}"
    action_card = "### Recommended Action\n" f"**{action}**"

    if reasoning:
        reasons = "\n".join(f"{idx}. {line}" for idx, line in enumerate(reasoning[:5], start=1))
    else:
        reasons = "1. No reasoning returned by inference runtime."
    reasons_card = f"### Top Reasons\n{reasons}"

    if risk_factors:
        factor_lines = "\n".join(f"- `{factor}`" for factor in risk_factors)
    else:
        factor_lines = "- `none`"
    factors_card = f"### Risk Factors\n{factor_lines}"

    metadata_card = (
        "### Runtime\n"
        f"- Mode: **{mode_title}**\n"
        f"- Detail: {mode_detail}\n"
        f"- Inference Key: `{result.get('mode', 'rule_based')}`\n"
        f"- Latency: **{result.get('inference_time_ms', '-')} ms**"
    )

    raw_output = str(result.get("raw_output", ""))
    return classification_card, action_card, reasons_card, factors_card, metadata_card, raw_output


def _empty_result(message: str) -> tuple[str, str, str, str, str, str]:
    return (
        "### Classification\nNo result",
        "### Recommended Action\nNo result",
        f"### Top Reasons\n1. {message}",
        "### Risk Factors\n- `n/a`",
        "### Runtime\n- Mode: waiting\n- Latency: n/a",
        "",
    )


def _highlight_terms(risk_factors: list[str]) -> list[str]:
    terms: list[str] = []
    for factor in risk_factors:
        raw = str(factor).strip().lower()
        if not raw:
            continue
        if raw not in terms:
            terms.append(raw)
        for token in re.findall(r"[a-z0-9]+", raw):
            if len(token) >= 4 and token not in terms:
                terms.append(token)
    terms = sorted(terms, key=len, reverse=True)
    return terms[:18]


def _highlight_narrative(text: str, terms: list[str], classification: str) -> str:
    if not text.strip():
        return "<em>No narrative provided.</em>"

    palette = {
        "FRAUDULENT": ("rgba(220, 38, 38, 0.45)", "#FEE2E2"),
        "SUSPICIOUS": ("rgba(217, 119, 6, 0.45)", "#FFEDD5"),
        "LEGITIMATE": ("rgba(13, 148, 136, 0.35)", "#CCFBF1"),
    }
    bg_color, font_color = palette.get(classification, ("rgba(59, 130, 246, 0.35)", "#DBEAFE"))
    if not terms:
        return f"<span style='color:#E2E8F0;'>{html.escape(text)}</span>"

    pattern = re.compile("|".join(re.escape(term) for term in terms), flags=re.IGNORECASE)
    chunks = []
    cursor = 0
    for match in pattern.finditer(text):
        chunks.append(html.escape(text[cursor : match.start()]))
        matched = html.escape(text[match.start() : match.end()])
        chunks.append(
            "<mark style='background:"
            f"{bg_color};color:{font_color};padding:0 0.18rem;border-radius:4px;'>{matched}</mark>"
        )
        cursor = match.end()
    chunks.append(html.escape(text[cursor:]))
    return "".join(chunks)


def _attention_panel(description: str, result: dict) -> str:
    classification = str(result.get("classification", "UNKNOWN"))
    risk_factors = [str(item) for item in (result.get("risk_factors") or [])]
    terms = _highlight_terms(risk_factors)
    highlighted = _highlight_narrative(description, terms, classification)

    if terms:
        triggers = ", ".join(f"<code>{html.escape(term)}</code>" for term in terms[:8])
    else:
        triggers = "<code>no explicit trigger terms detected</code>"

    return (
        "<div class='attention-map'>"
        "<h4 style='margin:0 0 0.55rem 0;'>Explainability Map</h4>"
        "<p style='margin:0 0 0.55rem 0;color:#CBD5E1;'>Highlighted phrases indicate tokens linked to model reasoning.</p>"
        f"<div style='line-height:1.7;background:#0F172A;border:1px solid #334155;border-radius:10px;padding:0.85rem;'>{highlighted}</div>"
        f"<p style='margin:0.7rem 0 0 0;color:#94A3B8;'>Trigger terms: {triggers}</p>"
        "</div>"
    )


def classify_transaction(description: str):
    if not description or not description.strip():
        empty = _empty_result("Provide a transaction narrative before running classification.")
        return (*empty, "<div class='attention-map'><em>No narrative to explain yet.</em></div>")

    result = INFER.classify(description=description, max_tokens=256, temperature=0.1)
    cards = _result_cards(result)
    attention_html = _attention_panel(description, result)
    return (*cards, attention_html)


def _compact_result(title: str, result: dict) -> str:
    classification = str(result.get("classification", "UNKNOWN"))
    action = str(result.get("recommended_action", "REVIEW"))
    latency = result.get("inference_time_ms", "-")
    factors = result.get("risk_factors") or []
    top_factor = str(factors[0]) if factors else "n/a"
    return (
        f"### {title}\n"
        f"{_class_badge(classification)}\n\n"
        f"- Action: **{action}**\n"
        f"- Top signal: `{top_factor}`\n"
        f"- Latency: **{latency} ms**"
    )


def compare_transactions(base_description: str, variant_description: str):
    if not base_description.strip() or not variant_description.strip():
        message = "Provide both original and variant narratives to compare."
        base_md = "### Original Narrative\nNo result"
        variant_md = "### Variant Narrative\nNo result"
        compare_md = f"### Delta Summary\n- {message}"
        empty_html = "<div class='attention-map'><em>No explainability map available.</em></div>"
        return base_md, variant_md, compare_md, empty_html, empty_html

    base_result = INFER.classify(description=base_description, max_tokens=256, temperature=0.1)
    variant_result = INFER.classify(description=variant_description, max_tokens=256, temperature=0.1)

    base_md = _compact_result("Original Narrative", base_result)
    variant_md = _compact_result("Variant Narrative", variant_result)

    base_cls = str(base_result.get("classification", "UNKNOWN"))
    variant_cls = str(variant_result.get("classification", "UNKNOWN"))
    base_action = str(base_result.get("recommended_action", "REVIEW"))
    variant_action = str(variant_result.get("recommended_action", "REVIEW"))
    same_class = base_cls == variant_cls
    same_action = base_action == variant_action
    delta_sentence = "Classification changed." if not same_class else "Classification stayed the same."
    action_sentence = "Action changed." if not same_action else "Action stayed the same."

    compare_md = (
        "### Delta Summary\n"
        f"- Original: **{base_cls}** → {base_action}\n"
        f"- Variant: **{variant_cls}** → {variant_action}\n"
        f"- {delta_sentence} {action_sentence}\n"
        f"- Original latency: **{base_result.get('inference_time_ms', '-')} ms**\n"
        f"- Variant latency: **{variant_result.get('inference_time_ms', '-')} ms**"
    )

    base_attention = _attention_panel(base_description, base_result)
    variant_attention = _attention_panel(variant_description, variant_result)
    return base_md, variant_md, compare_md, base_attention, variant_attention


def _template_names() -> list[str]:
    return list(EXAMPLE_LIBRARY.keys())


def _variation_names(template_name: str) -> list[str]:
    template = EXAMPLE_LIBRARY.get(template_name) or {}
    return list(template.keys())


def _template_text(template_name: str, variation_name: str) -> str:
    template = EXAMPLE_LIBRARY.get(template_name) or {}
    if variation_name in template:
        return template[variation_name]
    if template:
        first_key = next(iter(template))
        return template[first_key]
    return ""


def on_template_change(template_name: str):
    variations = _variation_names(template_name)
    if not variations:
        return gr.update(choices=[], value=None), ""
    selected = variations[0]
    preview = _template_text(template_name, selected)
    return gr.update(choices=variations, value=selected), preview


def on_variation_change(template_name: str, variation_name: str):
    return _template_text(template_name, variation_name)


def apply_selected_template(template_name: str, variation_name: str):
    return _template_text(template_name, variation_name)


def seed_compare_from_template(template_name: str):
    baseline = _template_text(template_name, "Baseline")
    variation = _template_text(template_name, "Variation A") or _template_text(template_name, "Variation B")
    if not variation:
        variation = baseline
    return baseline, variation


initial_template = _template_names()[0]
initial_variation = _variation_names(initial_template)[0]
initial_preview = _template_text(initial_template, initial_variation)
runtime_title, runtime_detail = _mode_status(INFER.mode)

UI_THEME = gr.themes.Base(primary_hue="blue", secondary_hue="slate")
UI_CSS = """
    .gradio-container { max-width: 1380px !important; }
    .sidebar-panel {
      background:#0F172A;
      border:1px solid #334155;
      border-radius:12px;
      padding:0.9rem;
    }
    .attention-map code {
      background:#1E293B;
      border:1px solid #334155;
      border-radius:4px;
      padding:0.1rem 0.3rem;
      color:#E2E8F0;
    }
    """

with gr.Blocks(title="FraudLLM Sandbox") as demo:
    gr.Markdown(
        "# FraudLLM - Prompt & Evaluation Sandbox\n"
        "Template-driven narrative testing with explainability mapping and A/B comparison."
    )
    gr.Markdown(
        f"### Runtime Mode: **{runtime_title}**\n"
        f"{runtime_detail}\n\n"
        f"- Mode key: `{INFER.mode}`\n"
        f"- Rule fallback allowed: `{ALLOW_RULE_BASED_FALLBACK}`\n"
        f"- Model path: `{MODEL_PATH}`"
    )

    with gr.Row():
        with gr.Column(scale=3, elem_classes=["sidebar-panel"]):
            gr.Markdown("### Template Sidebar")
            template_name = gr.Dropdown(
                label="Template",
                choices=_template_names(),
                value=initial_template,
            )
            template_variation = gr.Dropdown(
                label="Variation",
                choices=_variation_names(initial_template),
                value=initial_variation,
            )
            template_preview = gr.Textbox(
                label="Template Preview",
                value=initial_preview,
                lines=10,
                interactive=False,
            )
            use_template_btn = gr.Button("Use In Sandbox", variant="secondary")
            seed_compare_btn = gr.Button("Seed A/B Compare", variant="secondary")

        with gr.Column(scale=9):
            with gr.Tabs():
                with gr.TabItem("Sandbox"):
                    gr.Markdown(
                        "#### Single Narrative Classification\n"
                        "1. Pick template/variation from the sidebar.\n"
                        "2. Edit the narrative.\n"
                        "3. Run classification and inspect explainability highlights."
                    )
                    input_box = gr.Textbox(
                        label="Transaction Narrative",
                        lines=9,
                        placeholder="Describe a banking transaction to classify...",
                    )
                    run_btn = gr.Button("Classify Transaction", variant="primary")

                    with gr.Row():
                        classification_md = gr.Markdown("### Classification\n_No result yet_")
                        action_md = gr.Markdown("### Recommended Action\n_No result yet_")

                    with gr.Row():
                        reasons_md = gr.Markdown("### Top Reasons\n1. Run classification to see reasoning.")
                        factors_md = gr.Markdown("### Risk Factors\n- `n/a`")

                    runtime_md = gr.Markdown("### Runtime\n- Mode: waiting\n- Latency: n/a")
                    attention_html = gr.HTML("<div class='attention-map'><em>Explainability map will appear after classification.</em></div>")
                    raw_output_box = gr.Textbox(label="Raw Model Output", lines=6)

                with gr.TabItem("A/B Compare"):
                    gr.Markdown(
                        "#### Compare Original vs Variant\n"
                        "Use this mode to evaluate whether a narrative tweak changes model decisions."
                    )
                    with gr.Row():
                        compare_base_box = gr.Textbox(label="Original Narrative", lines=9)
                        compare_variant_box = gr.Textbox(label="Variant Narrative", lines=9)
                    compare_btn = gr.Button("Run A/B Comparison", variant="primary")

                    with gr.Row():
                        compare_base_md = gr.Markdown("### Original Narrative\n_No result yet_")
                        compare_variant_md = gr.Markdown("### Variant Narrative\n_No result yet_")
                    compare_delta_md = gr.Markdown("### Delta Summary\n- Run comparison to see changes.")

                    with gr.Row():
                        compare_base_attention = gr.HTML(
                            "<div class='attention-map'><em>Original explainability map will appear here.</em></div>"
                        )
                        compare_variant_attention = gr.HTML(
                            "<div class='attention-map'><em>Variant explainability map will appear here.</em></div>"
                        )

                with gr.TabItem("Explanation"):
                    gr.Markdown(
                        "## What This Project Does\n"
                        "FraudLLM reads transaction narratives and classifies them into:\n"
                        "- `LEGITIMATE`\n"
                        "- `SUSPICIOUS`\n"
                        "- `FRAUDULENT`\n\n"
                        "## Process\n"
                        "1. Parse narrative features (timing, counterparties, behavioral clues).\n"
                        "2. Run inference based on runtime mode.\n"
                        "3. Return class, recommended action, reasons, and risk factors.\n"
                        "4. Highlight trigger terms in the narrative for explainability.\n"
                        "5. Compare original vs variant narratives in A/B mode.\n\n"
                        "## Runtime Modes\n"
                        "- **LIVE MODEL MODE**: full local transformer artifacts loaded.\n"
                        "- **DEMO ADAPTER MODE**: deterministic adapter behavior for demos.\n"
                        "- **RULE-BASED FALLBACK MODE**: keyword-rule fallback when full artifacts are unavailable."
                    )

    run_btn.click(
        fn=classify_transaction,
        inputs=input_box,
        outputs=[classification_md, action_md, reasons_md, factors_md, runtime_md, raw_output_box, attention_html],
    )
    compare_btn.click(
        fn=compare_transactions,
        inputs=[compare_base_box, compare_variant_box],
        outputs=[
            compare_base_md,
            compare_variant_md,
            compare_delta_md,
            compare_base_attention,
            compare_variant_attention,
        ],
    )

    template_name.change(
        fn=on_template_change,
        inputs=template_name,
        outputs=[template_variation, template_preview],
    )
    template_variation.change(
        fn=on_variation_change,
        inputs=[template_name, template_variation],
        outputs=template_preview,
    )
    use_template_btn.click(
        fn=apply_selected_template,
        inputs=[template_name, template_variation],
        outputs=input_box,
    )
    seed_compare_btn.click(
        fn=seed_compare_from_template,
        inputs=template_name,
        outputs=[compare_base_box, compare_variant_box],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        theme=UI_THEME,
        css=UI_CSS,
    )
