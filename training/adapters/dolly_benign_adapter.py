from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dolly"
    / "source"
    / "databricks-dolly-15k.jsonl"
)

ORDINARY_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dolly"
    / "dolly_benign.jsonl"
)

HARD_NEGATIVE_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "hard_negatives"
    / "dolly_security_hard_negatives.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "dolly_conversion_report.json"
)


SECURITY_TERMS = [
    "ignore previous",
    "ignore all previous",
    "previous instructions",
    "system prompt",
    "prompt injection",
    "jailbreak",
    "password",
    "credential",
    "api key",
    "access token",
    "secret",
    "execute",
    "tool call",
    "developer message",
]


def normalize(text: str) -> str:
    return " ".join(
        str(text)
        .lower()
        .split()
    )


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


def build_input_text(
    instruction: str,
    context: str,
) -> str:

    instruction = instruction.strip()
    context = context.strip()

    if not context:
        return instruction

    return (
        instruction
        + "\n\nContext:\n"
        + context
    )


def security_matches(
    text: str,
) -> list[str]:

    lower = text.lower()

    return [
        term
        for term in SECURITY_TERMS
        if term in lower
    ]


def make_sample_id(
    index: int,
    text: str,
) -> str:

    raw = (
        f"dolly|{index}|{text}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"dolly_benign_{digest}"
    )


def canonical_record(
    *,
    index: int,
    row: dict[str, Any],
    text: str,
    matches: list[str],
) -> dict[str, Any]:

    instruction = str(
        row.get(
            "instruction",
            "",
        )
        or ""
    ).strip()

    context = str(
        row.get(
            "context",
            "",
        )
        or ""
    ).strip()

    category = str(
        row.get(
            "category",
            "unknown",
        )
        or "unknown"
    )

    is_hard_negative = bool(
        matches
    )

    return {
        "sample_id": make_sample_id(
            index,
            text,
        ),

        "text": text,

        "normalized_text": None,

        "is_attack": False,

        "attack_types": [],

        # The complete Dolly example represents
        # a legitimate user request.
        "source_type": "user_message",

        # User-controlled text is not inherently
        # trusted just because it is benign.
        "trust_level": "untrusted",

        "directness": "unknown",

        "encoding_type": "plain",

        "language": "en",

        "malicious_spans": [],

        "requested_actions": [],

        "target_data_types": [],

        "severity": "none",

        "dataset_source": (
            "databricks_dolly_15k"
        ),

        "original_label": (
            category
        ),

        # Final splitting happens after
        # global corpus construction.
        "split": "unassigned",

        "metadata": {
            "original_row_index": index,

            "dolly_category": category,

            "has_context": bool(
                context
            ),

            "hard_negative": (
                is_hard_negative
            ),

            "hard_negative_reason": (
                "security_like_benign_language"
                if is_hard_negative
                else None
            ),

            "security_term_matches": (
                matches
            ),
        },

        # Schema-v2 fields
        "user_instruction": (
            instruction
        ),

        "external_content": (
            context
            if context
            else None
        ),

        "attack_payload": None,

        "poisoned_content": None,

        "attack_position": None,

        "provenance": {
            "user_instruction_origin": (
                "user"
            ),

            # It is authoritative as the user's
            # requested task, even though the raw
            # input remains security-untrusted.
            "user_instruction_trust": (
                "trusted"
            ),

            "external_content_origin": (
                "user_supplied_context"
                if context
                else None
            ),

            "external_content_trust": (
                "untrusted"
                if context
                else None
            ),

            "attack_origin": None,
        },
    }


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

    raw_rows = read_jsonl(
        INPUT_FILE
    )

    ordinary = []
    hard_negatives = []

    seen = set()

    duplicates_removed = 0
    empty_removed = 0

    category_counts_before = Counter()
    category_counts_after = Counter()

    security_term_counts = Counter()

    rows_with_context = 0

    for index, row in enumerate(
        raw_rows
    ):

        instruction = str(
            row.get(
                "instruction",
                "",
            )
            or ""
        ).strip()

        context = str(
            row.get(
                "context",
                "",
            )
            or ""
        ).strip()

        category = str(
            row.get(
                "category",
                "unknown",
            )
            or "unknown"
        )

        category_counts_before[
            category
        ] += 1

        if not instruction:
            empty_removed += 1
            continue

        text = build_input_text(
            instruction,
            context,
        )

        normalized = normalize(
            text
        )

        if not normalized:
            empty_removed += 1
            continue

        if normalized in seen:
            duplicates_removed += 1
            continue

        seen.add(
            normalized
        )

        if context:
            rows_with_context += 1

        matches = security_matches(
            text
        )

        record = canonical_record(
            index=index,
            row=row,
            text=text,
            matches=matches,
        )

        category_counts_after[
            category
        ] += 1

        if matches:

            hard_negatives.append(
                record
            )

            for term in matches:
                security_term_counts[
                    term
                ] += 1

        else:

            ordinary.append(
                record
            )

    write_jsonl(
        ordinary,
        ORDINARY_OUTPUT,
    )

    write_jsonl(
        hard_negatives,
        HARD_NEGATIVE_OUTPUT,
    )

    final_total = (
        len(ordinary)
        + len(hard_negatives)
    )

    report = {
        "input_rows": len(
            raw_rows
        ),

        "empty_removed": (
            empty_removed
        ),

        "duplicates_removed": (
            duplicates_removed
        ),

        "unique_canonical_records": (
            final_total
        ),

        "ordinary_benign_records": (
            len(ordinary)
        ),

        "security_hard_negative_records": (
            len(hard_negatives)
        ),

        "rows_with_context": (
            rows_with_context
        ),

        "rows_without_context": (
            final_total
            - rows_with_context
        ),

        "category_counts_before": dict(
            sorted(
                category_counts_before.items()
            )
        ),

        "category_counts_after": dict(
            sorted(
                category_counts_after.items()
            )
        ),

        "security_term_record_counts": dict(
            sorted(
                security_term_counts.items()
            )
        ),

        "ordinary_output": str(
            ORDINARY_OUTPUT
        ),

        "hard_negative_output": str(
            HARD_NEGATIVE_OUTPUT
        ),

        "split": "unassigned",
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
        "Dolly benign conversion complete"
    )
    print("=" * 72)

    print(
        "Input rows                 :",
        len(raw_rows),
    )

    print(
        "Empty removed              :",
        empty_removed,
    )

    print(
        "Exact normalized duplicates:",
        duplicates_removed,
    )

    print(
        "Unique canonical records   :",
        final_total,
    )

    print()

    print(
        "Ordinary benign            :",
        len(ordinary),
    )

    print(
        "Security hard negatives    :",
        len(hard_negatives),
    )

    print()

    print(
        "With supplied context      :",
        rows_with_context,
    )

    print(
        "Without supplied context   :",
        final_total
        - rows_with_context,
    )

    print()

    print(
        "Hard-negative term counts:"
    )

    for term, count in sorted(
        security_term_counts.items()
    ):

        print(
            f"  {term:25} {count}"
        )

    print()

    print(
        "Ordinary output:",
        ORDINARY_OUTPUT,
    )

    print(
        "Hard negatives:",
        HARD_NEGATIVE_OUTPUT,
    )

    print(
        "Report:",
        REPORT_FILE,
    )


if __name__ == "__main__":
    main()
