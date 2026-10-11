from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "hard_negatives"
    / "bipia_clean_contexts.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "hard_negatives"
    / "bipia_clean_contexts_dedup_balanced.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "bipia_benign_cleaning_report.json"
)

RANDOM_SEED = 2026

MAX_PER_SOURCE = {
    "email": 200,
    "source_code": 200,
    "table": 200,
}


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from exc

    return records


def normalize(
    text: str,
) -> str:

    return " ".join(
        str(text)
        .lower()
        .split()
    )


def main() -> None:

    records = read_jsonl(
        INPUT_FILE
    )

    # --------------------------------
    # Deduplicate globally
    # --------------------------------

    seen = set()
    unique_records = []

    for record in records:

        key = normalize(
            record.get(
                "text",
                "",
            )
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        unique_records.append(
            record
        )

    # --------------------------------
    # Group by source type
    # --------------------------------

    grouped = defaultdict(list)

    for record in unique_records:

        grouped[
            record["source_type"]
        ].append(
            record
        )

    before_balance = {
        source: len(items)
        for source, items
        in grouped.items()
    }

    # --------------------------------
    # Balance
    # --------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    balanced = []

    for source, items in sorted(
        grouped.items()
    ):

        limit = MAX_PER_SOURCE.get(
            source,
            200,
        )

        if len(items) > limit:

            selected = rng.sample(
                items,
                limit,
            )

        else:
            selected = items

        balanced.extend(
            selected
        )

    balanced.sort(
        key=lambda r: (
            r["source_type"],
            r["sample_id"],
        )
    )

    # --------------------------------
    # Write output
    # --------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in balanced:

            json.dump(
                record,
                f,
                ensure_ascii=False,
            )

            f.write("\n")

    after_balance = Counter(
        record["source_type"]
        for record in balanced
    )

    report = {
        "input_records": len(
            records
        ),

        "unique_after_dedup": len(
            unique_records
        ),

        "duplicates_removed": (
            len(records)
            - len(unique_records)
        ),

        "source_counts_before_balance": (
            dict(
                sorted(
                    before_balance.items()
                )
            )
        ),

        "source_counts_after_balance": (
            dict(
                sorted(
                    after_balance.items()
                )
            )
        ),

        "final_records": len(
            balanced
        ),

        "random_seed": (
            RANDOM_SEED
        ),
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

        f.write("\n")

    print("=" * 72)
    print(
        "BIPIA benign cleaning complete"
    )
    print("=" * 72)

    print(
        "Input records          :",
        len(records),
    )

    print(
        "Unique after dedup     :",
        len(unique_records),
    )

    print(
        "Duplicates removed     :",
        len(records)
        - len(unique_records),
    )

    print()

    print(
        "Before balancing:"
    )

    for source, count in sorted(
        before_balance.items()
    ):

        print(
            f"  {source:20} {count}"
        )

    print()

    print(
        "After balancing:"
    )

    for source, count in sorted(
        after_balance.items()
    ):

        print(
            f"  {source:20} {count}"
        )

    print()

    print(
        "Final records          :",
        len(balanced),
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Report:",
        REPORT_FILE,
    )


if __name__ == "__main__":
    main()
