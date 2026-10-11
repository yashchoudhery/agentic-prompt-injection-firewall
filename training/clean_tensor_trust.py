from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "tensor_trust"
    / "tensor_trust_curated_attacks.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "tensor_trust"
    / "tensor_trust_cleaned.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "tensor_trust_cleaning_report.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

    return records


def normalize_for_dedup(text: str) -> str:
    """
    Conservative duplicate normalization.

    Important:
    We do NOT remove punctuation, Unicode or encoding artifacts,
    because these can be meaningful adversarial features.

    We only:
      - lowercase
      - collapse whitespace
      - strip outer whitespace
    """
    return " ".join(text.lower().split())


def merge_group(
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    base = dict(records[0])

    # Union all attack labels in case future datasets overlap.
    attack_types = sorted(
        {
            attack_type
            for record in records
            for attack_type in record.get(
                "attack_types",
                [],
            )
        }
    )

    base["attack_types"] = attack_types

    metadata = dict(
        base.get("metadata") or {}
    )

    metadata["duplicate_count"] = len(records)

    metadata["source_sample_ids"] = [
        record["sample_id"]
        for record in records
    ]

    metadata["original_labels_seen"] = sorted(
        {
            str(record.get("original_label"))
            for record in records
        }
    )

    base["metadata"] = metadata

    return base


def write_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:
            json.dump(
                record,
                f,
                ensure_ascii=False,
            )

            f.write("\n")


def main() -> None:

    records = read_jsonl(INPUT_FILE)

    groups: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    empty_records = []

    for record in records:

        text = record.get(
            "text",
            "",
        )

        if not text.strip():
            empty_records.append(record)
            continue

        key = normalize_for_dedup(
            text
        )

        groups[key].append(
            record
        )

    cleaned_records = []

    duplicate_groups = 0
    duplicated_rows_removed = 0

    for grouped_records in groups.values():

        if len(grouped_records) > 1:
            duplicate_groups += 1

            duplicated_rows_removed += (
                len(grouped_records) - 1
            )

        cleaned_records.append(
            merge_group(
                grouped_records
            )
        )

    cleaned_records.sort(
        key=lambda record: (
            record["attack_types"],
            record["sample_id"],
        )
    )

    write_jsonl(
        cleaned_records,
        OUTPUT_FILE,
    )

    label_counts: dict[str, int] = defaultdict(int)

    for record in cleaned_records:
        for attack_type in record[
            "attack_types"
        ]:
            label_counts[
                attack_type
            ] += 1

    max_length = max(
        (
            len(record["text"])
            for record in cleaned_records
        ),
        default=0,
    )

    long_2000 = sum(
        len(record["text"]) > 2000
        for record in cleaned_records
    )

    long_4000 = sum(
        len(record["text"]) > 4000
        for record in cleaned_records
    )

    report = {
        "input_records": len(records),

        "empty_records_removed": len(
            empty_records
        ),

        "duplicate_groups": duplicate_groups,

        "duplicate_rows_removed": (
            duplicated_rows_removed
        ),

        "output_unique_records": len(
            cleaned_records
        ),

        "attack_type_counts": dict(
            sorted(label_counts.items())
        ),

        "length_flags": {
            "greater_than_2000_chars": (
                long_2000
            ),
            "greater_than_4000_chars": (
                long_4000
            ),
            "maximum_chars": max_length,
        },

        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
    }

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("Tensor Trust cleaning complete")
    print("=" * 70)

    print(
        f"Input records           : {len(records)}"
    )

    print(
        f"Empty records removed   : {len(empty_records)}"
    )

    print(
        f"Duplicate groups        : {duplicate_groups}"
    )

    print(
        "Duplicate rows removed : "
        f"{duplicated_rows_removed}"
    )

    print(
        "Unique output records  : "
        f"{len(cleaned_records)}"
    )

    print()

    print("Attack labels:")

    for label, count in sorted(
        label_counts.items()
    ):
        print(
            f"  {label:25} {count}"
        )

    print()

    print(
        f">2000 chars             : {long_2000}"
    )

    print(
        f">4000 chars             : {long_4000}"
    )

    print(
        f"Maximum length          : {max_length}"
    )

    print()

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()
