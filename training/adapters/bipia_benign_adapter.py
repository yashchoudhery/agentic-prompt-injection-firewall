from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BIPIA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bipia"
    / "source"
    / "benchmark"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "hard_negatives"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "bipia_clean_contexts.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "bipia_benign_report.json"
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
                    f"Invalid JSON: "
                    f"{path}:{line_number}"
                ) from exc

    return records


def stringify_context(
    record: dict[str, Any],
) -> str:

    context = record.get(
        "context",
        "",
    )

    if isinstance(context, list):
        return "\n".join(
            str(x)
            for x in context
        )

    return str(context)


def make_id(
    source_name: str,
    index: int,
    text: str,
) -> str:

    raw = (
        f"{source_name}|"
        f"{index}|"
        f"{text}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"bipia_benign_{digest}"
    )


def source_type_for(
    source_name: str,
) -> str:

    mapping = {
        "email": "email",
        "table": "table",
        "code": "source_code",
    }

    return mapping[
        source_name
    ]


def get_user_instruction(
    source_name: str,
    record: dict[str, Any],
) -> str | None:

    if source_name in {
        "email",
        "table",
    }:

        question = record.get(
            "question"
        )

        if question is not None:
            return str(question)

    return None


def build_record(
    source_name: str,
    index: int,
    row: dict[str, Any],
) -> dict[str, Any]:

    text = stringify_context(
        row
    ).strip()

    user_instruction = (
        get_user_instruction(
            source_name,
            row,
        )
    )

    metadata = {
        "bipia_source_task": (
            source_name
        ),

        "bipia_context_index": (
            index
        ),

        "official_bipia_split": (
            "train"
        ),

        "ideal_answer": (
            row.get("ideal")
        ),
    }

    if source_name == "code":

        metadata[
            "original_error"
        ] = row.get(
            "error"
        )

        metadata[
            "original_code"
        ] = row.get(
            "code"
        )

        metadata[
            "context_url"
        ] = row.get(
            "context_url"
        )

    return {
        "sample_id": make_id(
            source_name,
            index,
            text,
        ),

        "text": text,

        "normalized_text": None,

        "is_attack": False,

        "attack_types": [],

        "source_type": (
            source_type_for(
                source_name
            )
        ),

        # External content is untrusted,
        # but untrusted does NOT mean malicious.
        "trust_level": "untrusted",

        "directness": "unknown",

        "encoding_type": "plain",

        "language": "en",

        "malicious_spans": [],

        "requested_actions": [],

        "target_data_types": [],

        "severity": "none",

        "dataset_source": (
            "bipia_clean"
        ),

        "original_label": (
            "benign_context"
        ),

        "split": "train",

        "metadata": metadata,

        # Schema-v2 fields
        "user_instruction": (
            user_instruction
        ),

        "external_content": text,

        "attack_payload": None,

        "poisoned_content": None,

        "attack_position": None,

        "provenance": {
            "user_instruction_origin": (
                "user"
            ),

            "user_instruction_trust": (
                "trusted"
                if user_instruction
                is not None
                else "unknown"
            ),

            "external_content_origin": (
                source_name
            ),

            "external_content_trust": (
                "untrusted"
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

    sources = {
        "email": (
            BIPIA_ROOT
            / "email"
            / "train.jsonl"
        ),

        "table": (
            BIPIA_ROOT
            / "table"
            / "train.jsonl"
        ),

        "code": (
            BIPIA_ROOT
            / "code"
            / "train.jsonl"
        ),
    }

    output = []

    skipped_empty = 0

    for source_name, path in (
        sources.items()
    ):

        rows = read_jsonl(
            path
        )

        for index, row in enumerate(
            rows
        ):

            text = stringify_context(
                row
            ).strip()

            if not text:
                skipped_empty += 1
                continue

            output.append(
                build_record(
                    source_name,
                    index,
                    row,
                )
            )

    write_jsonl(
        output,
        OUTPUT_FILE,
    )

    source_counts = Counter(
        record["source_type"]
        for record in output
    )

    unique_texts = {
        " ".join(
            record["text"]
            .lower()
            .split()
        )
        for record in output
    }

    duplicate_rows = (
        len(output)
        - len(unique_texts)
    )

    report = {
        "total_benign_records": (
            len(output)
        ),

        "source_type_counts": dict(
            sorted(
                source_counts.items()
            )
        ),

        "skipped_empty": (
            skipped_empty
        ),

        "unique_normalized_texts": (
            len(unique_texts)
        ),

        "duplicate_rows_estimate": (
            duplicate_rows
        ),

        "is_attack": False,

        "split": "train",
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
        "BIPIA benign conversion complete"
    )
    print("=" * 72)

    print(
        "Benign records          :",
        len(output),
    )

    print(
        "Unique normalized texts :",
        len(unique_texts),
    )

    print(
        "Duplicate rows          :",
        duplicate_rows,
    )

    print(
        "Skipped empty           :",
        skipped_empty,
    )

    print()

    print(
        "Source counts:"
    )

    for source, count in sorted(
        source_counts.items()
    ):

        print(
            f"  {source:20} {count}"
        )

    print()

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
