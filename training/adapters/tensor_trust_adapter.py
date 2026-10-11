from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TENSOR_TRUST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tensor_trust"
    / "source"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "tensor_trust"
)


HIJACKING_PATH = Path(
    "benchmarks/hijacking-robustness/v1/"
    "hijacking_robustness_dataset.jsonl"
)

EXTRACTION_PATH = Path(
    "benchmarks/extraction-robustness/v1/"
    "extraction_robustness_dataset.jsonl"
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc


def build_common_record(
    *,
    sample_id: str,
    text: str,
    attack_types: list[str],
    original_label: str,
    metadata: Dict[str, Any],
    target_data_types: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Convert one Tensor Trust benchmark sample into our canonical schema.
    """

    return {
        "sample_id": sample_id,

        "text": text,

        "normalized_text": None,

        "is_attack": True,

        "attack_types": attack_types,

        "source_type": "user_message",

        "trust_level": "untrusted",

        "directness": "direct",

        "encoding_type": "unknown",

        "language": "en",

        "malicious_spans": [text] if text else [],

        "requested_actions": [],

        "target_data_types": target_data_types or [],

        "severity": "high",

        "dataset_source": "tensor_trust",

        "original_label": original_label,

        "split": "unassigned",

        "metadata": metadata,
    }


def convert_hijacking_sample(
    row: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert Tensor Trust hijacking robustness sample.

    Tensor Trust defines these attacks as attempts to make the model
    produce "Access Granted" despite the attack not being the true
    access code.

    We conservatively map this benchmark to:
        instruction_override
    """

    original_id = row["sample_id"]
    attack = row.get("attack") or ""

    metadata = {
        "tensor_trust_sample_id": original_id,
        "tensor_trust_task": "hijacking_robustness",
        "benchmark_version": "v1",

        # We keep these fields for research/debugging.
        # They should NOT become model input automatically.
        "pre_prompt": row.get("pre_prompt"),
        "post_prompt": row.get("post_prompt"),
        "access_code": row.get("access_code"),
    }

    return build_common_record(
        sample_id=f"tensor_trust_hijacking_{original_id}",
        text=attack,
        attack_types=[
            "instruction_override"
        ],
        original_label="hijacking_robustness",
        metadata=metadata,
    )


def convert_extraction_sample(
    row: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert Tensor Trust extraction robustness sample.

    Tensor Trust defines these attacks as attempts to cause the model
    to reveal a secret access code.

    We conservatively map this benchmark to:
        secret_extraction
    """

    original_id = row["sample_id"]
    attack = row.get("attack") or ""

    metadata = {
        "tensor_trust_sample_id": original_id,
        "tensor_trust_task": "extraction_robustness",
        "benchmark_version": "v1",

        # Preserved for provenance/research only.
        "pre_prompt": row.get("pre_prompt"),
        "post_prompt": row.get("post_prompt"),
        "access_code": row.get("access_code"),
    }

    return build_common_record(
        sample_id=f"tensor_trust_extraction_{original_id}",
        text=attack,
        attack_types=[
            "secret_extraction"
        ],
        original_label="extraction_robustness",
        metadata=metadata,
        target_data_types=[
            "secret_access_code"
        ],
    )


def write_jsonl(
    records: Iterable[Dict[str, Any]],
    path: Path,
) -> int:
    """Write records to JSONL and return record count."""

    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(
                record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")
            count += 1

    return count


def validate_record(record: Dict[str, Any]) -> None:
    """
    Basic canonical-schema validation.

    Full JSON Schema validation will be added later.
    """

    required_fields = {
        "sample_id",
        "text",
        "is_attack",
        "attack_types",
        "source_type",
        "dataset_source",
    }

    missing = required_fields - record.keys()

    if missing:
        raise ValueError(
            f"Record {record.get('sample_id')} missing fields: "
            f"{sorted(missing)}"
        )

    if not isinstance(record["text"], str):
        raise TypeError(
            f"{record['sample_id']}: text must be string"
        )

    if not isinstance(record["is_attack"], bool):
        raise TypeError(
            f"{record['sample_id']}: is_attack must be bool"
        )

    if not isinstance(record["attack_types"], list):
        raise TypeError(
            f"{record['sample_id']}: attack_types must be list"
        )


def convert_dataset(
    tensor_trust_root: Path,
    output_dir: Path,
) -> None:

    hijacking_file = tensor_trust_root / HIJACKING_PATH
    extraction_file = tensor_trust_root / EXTRACTION_PATH

    if not hijacking_file.exists():
        raise FileNotFoundError(
            f"Hijacking dataset not found: {hijacking_file}"
        )

    if not extraction_file.exists():
        raise FileNotFoundError(
            f"Extraction dataset not found: {extraction_file}"
        )

    # -----------------------------
    # Hijacking benchmark
    # -----------------------------

    hijacking_records = []

    for row in read_jsonl(hijacking_file):
        record = convert_hijacking_sample(row)
        validate_record(record)
        hijacking_records.append(record)

    hijacking_output = (
        output_dir
        / "hijacking_canonical.jsonl"
    )

    hijacking_count = write_jsonl(
        hijacking_records,
        hijacking_output,
    )

    # -----------------------------
    # Extraction benchmark
    # -----------------------------

    extraction_records = []

    for row in read_jsonl(extraction_file):
        record = convert_extraction_sample(row)
        validate_record(record)
        extraction_records.append(record)

    extraction_output = (
        output_dir
        / "extraction_canonical.jsonl"
    )

    extraction_count = write_jsonl(
        extraction_records,
        extraction_output,
    )

    # -----------------------------
    # Combined malicious dataset
    # -----------------------------

    combined_records = (
        hijacking_records
        + extraction_records
    )

    combined_output = (
        output_dir
        / "tensor_trust_curated_attacks.jsonl"
    )

    combined_count = write_jsonl(
        combined_records,
        combined_output,
    )

    print("=" * 70)
    print("Tensor Trust adapter complete")
    print("=" * 70)

    print(
        f"Hijacking records : {hijacking_count}"
    )

    print(
        f"Extraction records : {extraction_count}"
    )

    print(
        f"Combined records   : {combined_count}"
    )

    print()

    print(
        f"Output directory   : {output_dir}"
    )

    print()

    print("Files:")
    print(f"  {hijacking_output}")
    print(f"  {extraction_output}")
    print(f"  {combined_output}")


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Convert Tensor Trust curated robustness benchmarks "
            "into the Prompt Firewall canonical dataset format."
        )
    )

    parser.add_argument(
        "--tensor-trust-root",
        type=Path,
        default=DEFAULT_TENSOR_TRUST_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    convert_dataset(
        tensor_trust_root=args.tensor_trust_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
