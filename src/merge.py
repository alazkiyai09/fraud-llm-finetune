import argparse
import json
import shutil
from pathlib import Path



def run_real_merge(base_model: str, adapter_path: str, output_dir: str) -> None:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Transformers/PEFT dependencies unavailable for real merge.") from exc

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged = peft_model.merge_and_unload()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



def run_mock_merge(adapter_path: str, output_dir: str) -> None:
    src = Path(adapter_path)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for file_name in ("mock_adapter.json", "keyword_rules.json"):
        source_file = src / file_name
        if source_file.exists():
            shutil.copy2(source_file, dst / file_name)

    (dst / "mock_merged.json").write_text(
        json.dumps(
            {
                "type": "mock_merged_model",
                "source_adapter": str(src),
                "status": "ready",
            },
            indent=2,
        ),
        encoding="utf-8",
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter_path", default="results/lora_adapter")
    parser.add_argument("--output_dir", default="results/merged_model")
    parser.add_argument("--use_mock_merge", action="store_true")
    args = parser.parse_args()

    if args.use_mock_merge:
        run_mock_merge(adapter_path=args.adapter_path, output_dir=args.output_dir)
    else:
        run_real_merge(base_model=args.base_model, adapter_path=args.adapter_path, output_dir=args.output_dir)

    print(json.dumps({"output_dir": args.output_dir, "mode": "mock" if args.use_mock_merge else "real"}, indent=2))


if __name__ == "__main__":
    main()
