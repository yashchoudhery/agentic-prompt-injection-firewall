from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "tensor_trust"
)

HIJACKING_FILE = DATA_DIR / "hijacking_canonical.jsonl"
EXTRACTION_FILE = DATA_DIR / "extraction_canonical.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
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


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * p
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return float(values[lower])

    weight = k - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def normalize_for_duplicate_check(text: str) -> str:
    return " ".join(text.lower().split())


def describe_dataset(
    name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    texts = [
        record.get("text", "")
        for record in records
    ]

    normalized_texts = [
        normalize_for_duplicate_check(text)
        for text in texts
    ]

    char_lengths = [
        len(text)
        for text in texts
    ]

    word_lengths = [
        len(text.split())
        for text in texts
    ]

    duplicate_counter = Counter(normalized_texts)

    duplicate_groups = {
        text: count
        for text, count in duplicate_counter.items()
        if text and count > 1
    }

    empty_count = sum(
        1
        for text in texts
        if not text.strip()
    )

    unique_count = len(
        {
            text
            for text in normalized_texts
            if text
        }
    )

    report = {
        "dataset": name,
        "records": len(records),
        "empty_texts": empty_count,
        "unique_normalized_texts": unique_count,
        "duplicate_rows_estimate": (
            len(records)
            - unique_count
            - empty_count
        ),

        "character_length": {
            "min": min(char_lengths) if char_lengths else 0,
            "median": statistics.median(char_lengths)
            if char_lengths else 0,
            "mean": statistics.mean(char_lengths)
            if char_lengths else 0,
            "p95": percentile(char_lengths, 0.95),
            "p99": percentile(char_lengths, 0.99),
            "max": max(char_lengths) if char_lengths else 0,
        },

        "word_length": {
            "min": min(word_lengths) if word_lengths else 0,
            "median": statistics.median(word_lengths)
            if word_lengths else 0,
            "mean": statistics.mean(word_lengths)
            if word_lengths else 0,
            "p95": percentile(word_lengths, 0.95),
            "p99": percentile(word_lengths, 0.99),
            "max": max(word_lengths) if word_lengths else 0,
        },

        "duplicate_groups": len(duplicate_groups),
    }

    return report


def print_report(report: dict[str, Any]) -> None:

    print()
    print("=" * 72)
    print(report["dataset"])
    print("=" * 72)

    print(
        f"Records                  : {report['records']}"
    )

    print(
        f"Empty texts              : {report['empty_texts']}"
    )

    print(
        "Unique normalized texts  : "
        f"{report['unique_normalized_texts']}"
    )

    print(
        "Duplicate rows estimate  : "
        f"{report['duplicate_rows_estimate']}"
    )

    print(
        f"Duplicate groups         : {report['duplicate_groups']}"
    )

    print()

    print("Character length")
    for key, value in report["character_length"].items():
        print(
            f"  {key:8}: {value:.2f}"
            if isinstance(value, float)
            else f"  {key:8}: {value}"
        )

    print()

    print("Word length")
    for key, value in report["word_length"].items():
        print(
            f"  {key:8}: {value:.2f}"
            if isinstance(value, float)
            else f"  {key:8}: {value}"
        )


def find_cross_dataset_overlap(
    hijacking_records: list[dict[str, Any]],
    extraction_records: list[dict[str, Any]],
) -> set[str]:

    hijacking_texts = {
        normalize_for_duplicate_check(
            record.get("text", "")
        )
        for record in hijacking_records
        if record.get("text", "").strip()
    }

    extraction_texts = {
        normalize_for_duplicate_check(
            record.get("text", "")
        )
        for record in extraction_records
        if record.get("text", "").strip()
    }

    return hijacking_texts & extraction_texts


def longest_examples(
    records: list[dict[str, Any]],
    count: int = 5,
) -> list[dict[str, Any]]:

    return sorted(
        records,
        key=lambda record: len(
            record.get("text", "")
        ),
        reverse=True,
    )[:count]


def print_longest_examples(
    name: str,
    records: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 72)
    print(f"Longest examples: {name}")
    print("=" * 72)

    for index, record in enumerate(
        longest_examples(records),
        start=1,
    ):

        text = record.get("text", "")

        preview = text[:300].replace(
            "\n",
            "\\n",
        )

        print(
            f"{index}. "
            f"{record['sample_id']} "
            f"chars={len(text)}"
        )

        print(
            f"   {preview}"
        )


def main() -> None:

    hijacking = load_jsonl(HIJACKING_FILE)
    extraction = load_jsonl(EXTRACTION_FILE)

    hijacking_report = describe_dataset(
        "Tensor Trust — Hijacking",
        hijacking,
    )

    extraction_report = describe_dataset(
        "Tensor Trust — Extraction",
        extraction,
    )

    print_report(hijacking_report)
    print_report(extraction_report)

    overlap = find_cross_dataset_overlap(
        hijacking,
        extraction,
    )

    print()
    print("=" * 72)
    print("Cross-dataset overlap")
    print("=" * 72)

    print(
        "Identical normalized attack texts appearing "
        f"in BOTH classes: {len(overlap)}"
    )

    if overlap:
        print()
        print("First few overlaps:")

        for text in list(sorted(overlap))[:10]:
            print(
                f"- {text[:250]}"
            )

    print_longest_examples(
        "Hijacking",
        hijacking,
    )

    print_longest_examples(
        "Extraction",
        extraction,
    )


if __name__ == "__main__":
    main()
