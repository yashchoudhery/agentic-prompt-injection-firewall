from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TENSOR_TRUST_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "tensor_trust"
    / "tensor_trust_cleaned.jsonl"
)

HACKAPROMPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hackaprompt"
    / "hackaprompt_cleaned.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "unified"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "unified_direct_attacks.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "direct_attack_unification_report.json"
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


def normalize_for_dedup(
    text: str,
) -> str:

    return " ".join(
        str(text)
        .lower()
        .split()
    )


def make_unified_id(
    normalized_text: str,
) -> str:

    digest = hashlib.sha256(
        normalized_text.encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    return f"direct_attack_{digest}"


def merge_records(
    records: list[dict[str, Any]],
    normalized_text: str,
) -> dict[str, Any]:

    representative = records[0]

    attack_types = sorted(
        {
            label
            for record in records
            for label in record.get(
                "attack_types",
                [],
            )
        }
    )

    dataset_sources = sorted(
        {
            str(
                record.get(
                    "dataset_source",
                    "unknown",
                )
            )
            for record in records
        }
    )

    original_sample_ids = [
        record.get("sample_id")
        for record in records
    ]

    original_labels = sorted(
        {
            str(
                record.get(
                    "original_label",
                    "",
                )
            )
            for record in records
            if record.get(
                "original_label"
            )
        }
    )

    source_types = sorted(
        {
            str(
                record.get(
                    "source_type",
                    "unknown",
                )
            )
            for record in records
        }
    )

    target_data_types = sorted(
        {
            item
            for record in records
            for item in record.get(
                "target_data_types",
                [],
            )
        }
    )

    requested_actions = sorted(
        {
            item
            for record in records
            for item in record.get(
                "requested_actions",
                [],
            )
        }
    )

    metadata = {
        "dataset_sources_seen": (
            dataset_sources
        ),

        "source_sample_ids": (
            original_sample_ids
        ),

        "original_labels_seen": (
            original_labels
        ),

        "merged_record_count": len(
            records
        ),

        "source_types_seen": (
            source_types
        ),

        # Preserve complete original
        # source metadata for research.
        "source_metadata": [
            {
                "dataset_source": (
                    record.get(
                        "dataset_source"
                    )
                ),
                "sample_id": (
                    record.get(
                        "sample_id"
                    )
                ),
                "metadata": (
                    record.get(
                        "metadata",
                        {},
                    )
                ),
            }
            for record in records
        ],
    }

    return {
        "sample_id": make_unified_id(
            normalized_text
        ),

        "text": representative.get(
            "text",
            "",
        ),

        "normalized_text": None,

        "is_attack": True,

        "attack_types": attack_types,

        "source_type": (
            representative.get(
                "source_type",
                "user_message",
            )
        ),

        "trust_level": (
            representative.get(
                "trust_level",
                "untrusted",
            )
        ),

        "directness": "direct",

        "encoding_type": (
            representative.get(
                "encoding_type",
                "unknown",
            )
        ),

        "language": (
            representative.get(
                "language",
                "en",
            )
        ),

        "malicious_spans": [
            representative.get(
                "text",
                "",
            )
        ],

        "requested_actions": (
            requested_actions
        ),

        "target_data_types": (
            target_data_types
        ),

        "severity": "high",

        "dataset_source": (
            "unified_direct"
        ),

        "original_label": (
            "|".join(
                original_labels
            )
            if original_labels
            else None
        ),

        "split": "unassigned",

        "metadata": metadata,
    }


def main() -> None:

    tensor_records = read_jsonl(
        TENSOR_TRUST_FILE
    )

    hackaprompt_records = read_jsonl(
        HACKAPROMPT_FILE
    )

    all_records = (
        tensor_records
        + hackaprompt_records
    )

    groups: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    for record in all_records:

        text = record.get(
            "text",
            "",
        )

        if not text.strip():
            continue

        key = normalize_for_dedup(
            text
        )

        groups[key].append(
            record
        )

    unified = []

    cross_dataset_overlap = 0

    overlap_examples = []

    for normalized_text, records in groups.items():

        sources = {
            record.get(
                "dataset_source"
            )
            for record in records
        }

        if len(sources) > 1:

            cross_dataset_overlap += 1

            if len(
                overlap_examples
            ) < 10:

                overlap_examples.append(
                    {
                        "text": (
                            records[0][
                                "text"
                            ][:500]
                        ),
                        "sources": sorted(
                            str(x)
                            for x in sources
                        ),
                        "labels": sorted(
                            {
                                label
                                for record
                                in records
                                for label
                                in record.get(
                                    "attack_types",
                                    [],
                                )
                            }
                        ),
                    }
                )

        unified.append(
            merge_records(
                records,
                normalized_text,
            )
        )

    unified.sort(
        key=lambda record: (
            record["sample_id"]
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in unified:

            json.dump(
                record,
                f,
                ensure_ascii=False,
            )

            f.write("\n")

    label_counts = defaultdict(
        int
    )

    multilabel_count = 0

    for record in unified:

        for label in record[
            "attack_types"
        ]:

            label_counts[
                label
            ] += 1

        if len(
            record["attack_types"]
        ) > 1:

            multilabel_count += 1

    report = {
        "tensor_trust_records": len(
            tensor_records
        ),

        "hackaprompt_records": len(
            hackaprompt_records
        ),

        "input_records_total": len(
            all_records
        ),

        "cross_dataset_overlap_groups": (
            cross_dataset_overlap
        ),

        "unified_unique_records": len(
            unified
        ),

        "rows_removed_by_global_dedup": (
            len(all_records)
            - len(unified)
        ),

        "attack_type_counts": dict(
            sorted(
                label_counts.items()
            )
        ),

        "multi_label_records": (
            multilabel_count
        ),

        "overlap_examples": (
            overlap_examples
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

    print("=" * 72)
    print(
        "Unified direct attack corpus"
    )
    print("=" * 72)

    print(
        "Tensor Trust records      : "
        f"{len(tensor_records)}"
    )

    print(
        "HackAPrompt records       : "
        f"{len(hackaprompt_records)}"
    )

    print(
        "Input records total       : "
        f"{len(all_records)}"
    )

    print(
        "Cross-dataset overlaps    : "
        f"{cross_dataset_overlap}"
    )

    print(
        "Unified unique records    : "
        f"{len(unified)}"
    )

    print(
        "Rows removed globally     : "
        f"{len(all_records) - len(unified)}"
    )

    print()

    print(
        "Attack labels:"
    )

    for label, count in sorted(
        label_counts.items()
    ):

        print(
            f"  {label:25} {count}"
        )

    print()

    print(
        "Multi-label records       : "
        f"{multilabel_count}"
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
