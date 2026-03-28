import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

LABELS = ["LEGITIMATE", "SUSPICIOUS", "FRAUDULENT"]
INSTRUCTION = (
    "Classify the following transaction description as LEGITIMATE, "
    "SUSPICIOUS, or FRAUDULENT. Explain your reasoning."
)

OUTPUT_PATTERN = re.compile(
    r"Classification:\s*(LEGITIMATE|SUSPICIOUS|FRAUDULENT).*?"
    r"Reasoning:\s*(.*?)\s*"
    r"Risk factors:\s*(.*?)\s*"
    r"Recommended action:\s*(.*)",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class FraudSample:
    instruction: str
    input: str
    output: str

    def to_dict(self) -> dict:
        return {"instruction": self.instruction, "input": self.input, "output": self.output}


class FraudDatasetBuilder:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def generate(self, total_examples: int = 6000) -> list[FraudSample]:
        total_examples = max(300, total_examples)

        legit_n = int(total_examples * 0.4)
        suspicious_n = int(total_examples * 0.3)
        fraudulent_n = total_examples - legit_n - suspicious_n

        samples: list[FraudSample] = []
        samples.extend(self._build_legitimate_samples(legit_n))
        samples.extend(self._build_suspicious_samples(suspicious_n))
        samples.extend(self._build_fraudulent_samples(fraudulent_n))

        self.rng.shuffle(samples)
        return samples

    def _build_legitimate_samples(self, n: int) -> list[FraudSample]:
        employers = ["Acme Corp", "Delta Logistics", "Nusa Retail", "Helios Tech"]
        categories = ["salary deposit", "monthly invoice payment", "utilities payment", "merchant settlement"]
        channels = ["mobile banking", "payroll gateway", "online banking", "ACH"]

        rows = []
        for _ in range(n):
            amount = self.rng.randint(350, 8900)
            employer = self.rng.choice(employers)
            category = self.rng.choice(categories)
            channel = self.rng.choice(channels)
            month_count = self.rng.randint(6, 36)

            narrative = (
                f"Transaction from {employer} via {channel} for {category}. "
                f"Amount ${amount:,} follows recurring monthly pattern for {month_count} months. "
                "Customer profile and destination account are stable with complete KYC and no prior alerts."
            )

            reasoning = [
                "Payment pattern is consistent with established historical behavior",
                "Counterparty and destination account are verified and longstanding",
                "No threshold gaming, unusual timing, or suspicious jurisdiction signals",
            ]
            risk_factors = ["low behavioral variance", "verified counterparty", "stable KYC profile"]
            rows.append(self._build_sample("LEGITIMATE", narrative, reasoning, risk_factors, "APPROVE"))

        return rows

    def _build_suspicious_samples(self, n: int) -> list[FraudSample]:
        regions = ["Dubai", "Singapore", "Hong Kong", "Cayman Islands"]
        channels = ["online banking", "wire", "mobile app"]

        rows = []
        for _ in range(n):
            amount = self.rng.randint(9000, 32000)
            velocity = self.rng.randint(2, 5)
            region = self.rng.choice(regions)
            channel = self.rng.choice(channels)
            account_age = self.rng.randint(20, 75)

            narrative = (
                f"{velocity} transfers initiated via {channel} totaling ${amount:,} to beneficiary in {region}. "
                f"Receiving account age is {account_age} days with recent profile updates and moderate documentation gaps. "
                "Activity differs from baseline business hours but still has partially plausible invoice context."
            )

            reasoning = [
                "Transfer velocity and destination profile deviate from normal account behavior",
                "Counterparty documentation is incomplete and account history is relatively short",
                "Signals are concerning but not conclusive for confirmed fraud",
            ]
            risk_factors = ["velocity anomaly", "counterparty age", "cross-border exposure"]
            rows.append(self._build_sample("SUSPICIOUS", narrative, reasoning, risk_factors, "REVIEW"))

        return rows

    def _build_fraudulent_samples(self, n: int) -> list[FraudSample]:
        offshore = ["Cayman Islands", "British Virgin Islands", "Panama", "Belize"]

        rows = []
        for _ in range(n):
            near_threshold = self.rng.randint(49500, 49999)
            account_age = self.rng.randint(1, 10)
            hour = self.rng.randint(0, 4)
            location = self.rng.choice(offshore)
            split_count = self.rng.randint(2, 4)

            narrative = (
                f"{split_count} wire transfers between $9,600 and ${near_threshold:,} sent to newly opened account "
                f"({account_age} days old) in {location}. "
                f"Transactions occurred around {hour}:30 local time with minimal KYC and immediate onward transfers. "
                "Pattern indicates structuring and likely mule-account laundering behavior."
            )

            reasoning = [
                "Amounts are intentionally near reporting thresholds indicating structuring",
                "Newly opened recipient account with minimal KYC suggests mule-account usage",
                "Offshore destination and off-hours execution increase laundering risk materially",
            ]
            risk_factors = ["structuring", "new account", "offshore transfer", "unusual timing"]
            rows.append(self._build_sample("FRAUDULENT", narrative, reasoning, risk_factors, "BLOCK and file SAR"))

        return rows

    @staticmethod
    def _build_sample(
        classification: str,
        narrative: str,
        reasoning: list[str],
        risk_factors: list[str],
        action: str,
    ) -> FraudSample:
        reasoning_text = "\n".join(f"{idx}. {line}" for idx, line in enumerate(reasoning, start=1))
        output = (
            f"Classification: {classification}\n\n"
            f"Reasoning:\n{reasoning_text}\n\n"
            f"Risk factors: {', '.join(risk_factors)}\n"
            f"Recommended action: {action}"
        )

        return FraudSample(instruction=INSTRUCTION, input=narrative, output=output)


class FraudDatasetIO:
    @staticmethod
    def write_jsonl(path: str, rows: list[dict]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    @staticmethod
    def load_jsonl(path: str) -> list[dict]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        rows = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows


class FraudDatasetValidator:
    @staticmethod
    def parse_output(output: str) -> dict | None:
        matched = OUTPUT_PATTERN.search(output)
        if not matched:
            return None

        classification, reasoning, risk_factors, action = matched.groups()
        parsed = {
            "classification": classification.upper().strip(),
            "reasoning": reasoning.strip(),
            "risk_factors": [item.strip() for item in risk_factors.split(",") if item.strip()],
            "recommended_action": action.strip(),
        }
        return parsed

    @classmethod
    def is_valid_row(cls, row: dict) -> bool:
        if not isinstance(row, dict):
            return False
        for key in ("instruction", "input", "output"):
            if key not in row or not isinstance(row[key], str) or not row[key].strip():
                return False

        narrative_len = len(row["input"].strip())
        if narrative_len < 50 or narrative_len > 500:
            return False

        parsed = cls.parse_output(row["output"])
        if not parsed:
            return False

        if parsed["classification"] not in LABELS:
            return False

        return True

    @classmethod
    def class_distribution(cls, rows: list[dict]) -> dict[str, float]:
        counts = {label: 0 for label in LABELS}
        for row in rows:
            parsed = cls.parse_output(row.get("output", ""))
            if parsed and parsed["classification"] in counts:
                counts[parsed["classification"]] += 1

        total = max(1, len(rows))
        return {label: counts[label] / total for label in LABELS}



def split_rows(rows: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    train_end = int(len(shuffled) * 0.8)
    val_end = int(len(shuffled) * 0.9)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    return train, val, test



def _generate_and_save(total_examples: int, output_dir: str, seed: int) -> None:
    rows = [sample.to_dict() for sample in FraudDatasetBuilder(seed=seed).generate(total_examples=total_examples)]

    invalid_rows = [row for row in rows if not FraudDatasetValidator.is_valid_row(row)]
    if invalid_rows:
        raise RuntimeError(f"Generated invalid dataset rows: {len(invalid_rows)}")

    train, val, test = split_rows(rows, seed=seed)

    out = Path(output_dir)
    FraudDatasetIO.write_jsonl(str(out / "train.jsonl"), train)
    FraudDatasetIO.write_jsonl(str(out / "val.jsonl"), val)
    FraudDatasetIO.write_jsonl(str(out / "test.jsonl"), test)

    dist = FraudDatasetValidator.class_distribution(rows)
    print(
        json.dumps(
            {
                "total_examples": len(rows),
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "class_distribution": dist,
            },
            indent=2,
        )
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud dataset preparation")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic processed dataset")
    parser.add_argument("--total_examples", type=int, default=6000)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.generate:
        _generate_and_save(total_examples=args.total_examples, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
