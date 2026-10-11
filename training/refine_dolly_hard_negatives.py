from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORDINARY_INPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dolly"
    / "dolly_benign.jsonl"
)

CANDIDATE_INPUT = (
    PROJECT_ROOT
    / "data"
    / "hard_negatives"
    / "dolly_security_hard_negatives.jsonl"
)

ORDINARY_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dolly"
    / "dolly_benign_refined.jsonl"
)

HARD_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "hard_negatives"
    / "dolly_true_hard_negatives.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "dolly_hard_negative_refinement_report.json"
)


STRONG_SECURITY_TERMS = [
    "password",
    "passwords",
    "credential",
    "credentials",
    "api key",
    "access token",
    "authentication",
    "multi-factor authentication",
    "security breach",
    "secret key",
]


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    records = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


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


def is_true_hard_negative(
    record: dict[str, Any],
) -> tuple[bool, list[str]]:

    # Important:
    # classify primarily using the USER INSTRUCTION,
    # not arbitrary context words.

    instruction = str(
        record.get(
            "user_instruction",
            "",
        )
        or ""
    ).lower()

    matches = [
        term
        for term in STRONG_SECURITY_TERMS
        if term in instruction
    ]

    return (
        bool(matches),
        matches,
    )


def main() -> None:

    ordinary = read_jsonl(
        ORDINARY_INPUT
    )

    candidates = read_jsonl(
        CANDIDATE_INPUT
    )

    true_hard = []
    returned_to_ordinary = []

    for record in candidates:

        is_hard, matches = (
            is_true_hard_negative(
                record
            )
        )

        record["metadata"][
            "security_term_matches"
        ] = matches

        if is_hard:

            record["metadata"][
                "hard_negative"
            ] = True

            record["metadata"][
                "hard_negative_reason"
            ] = (
                "benign_security_instruction"
            )

            true_hard.append(
                record
            )

        else:

            record["metadata"][
                "hard_negative"
            ] = False

            record["metadata"][
                "hard_negative_reason"
            ] = None

            returned_to_ordinary.append(
                record
            )

    refined_ordinary = (
        ordinary
        + returned_to_ordinary
    )

    write_jsonl(
        refined_ordinary,
        ORDINARY_OUTPUT,
    )

    write_jsonl(
        true_hard,
        HARD_OUTPUT,
    )

    report = {
        "original_ordinary": len(
            ordinary
        ),

        "original_hard_candidates": len(
            candidates
        ),

        "true_hard_negatives": len(
            true_hard
        ),

        "returned_to_ordinary": len(
            returned_to_ordinary
        ),

        "final_ordinary": len(
            refined_ordinary
        ),

        "final_total": (
            len(refined_ordinary)
            + len(true_hard)
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
            indent=2,
        )

        f.write("\n")

    print("=" * 72)
    print(
        "Dolly hard-negative refinement complete"
    )
    print("=" * 72)

    for key, value in report.items():
        print(
            f"{key:30} : {value}"
        )

    print()
    print(
        "Ordinary:",
        ORDINARY_OUTPUT,
    )

    print(
        "Hard negatives:",
        HARD_OUTPUT,
    )

    print(
        "Report:",
        REPORT_FILE,
    )


if __name__ == "__main__":
    main()
