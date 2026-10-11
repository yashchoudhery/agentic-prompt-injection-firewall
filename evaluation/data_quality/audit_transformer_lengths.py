from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "microsoft/deberta-v3-small"

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.jsonl"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "transformer_length_audit.json"
)


def safe(value):

    if value is None:
        return ""

    return str(value).strip()


def build_input(row):

    source_type = safe(
        row.get("source_type")
    )

    user_instruction = safe(
        row.get("user_instruction")
    )

    external_content = safe(
        row.get("external_content")
    )

    raw_text = safe(
        row.get("text")
    )

    return "\n".join(
        [
            "[SOURCE_TYPE]",
            source_type
            if source_type
            else "<UNKNOWN>",

            "[USER_INSTRUCTION]",
            user_instruction
            if user_instruction
            else "<NONE>",

            "[EXTERNAL_CONTENT]",
            external_content
            if external_content
            else "<NONE>",

            "[CONTENT_TO_INSPECT]",
            raw_text,
        ]
    )


def read_jsonl(path):

    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


def percentile(values, q):

    if len(values) == 0:
        return 0.0

    return float(
        np.percentile(
            values,
            q,
        )
    )


def stats(values):

    if len(values) == 0:

        return {
            "count": 0,
        }

    arr = np.asarray(
        values,
        dtype=np.int64,
    )

    return {
        "count": int(
            len(arr)
        ),

        "min": int(
            arr.min()
        ),

        "mean": float(
            arr.mean()
        ),

        "median": float(
            np.median(arr)
        ),

        "p90": percentile(
            arr,
            90,
        ),

        "p95": percentile(
            arr,
            95,
        ),

        "p99": percentile(
            arr,
            99,
        ),

        "max": int(
            arr.max()
        ),

        "over_256": int(
            (arr > 256).sum()
        ),

        "over_384": int(
            (arr > 384).sum()
        ),

        "over_512": int(
            (arr > 512).sum()
        ),

        "over_768": int(
            (arr > 768).sum()
        ),

        "over_1024": int(
            (arr > 1024).sum()
        ),
    }


def analyze_split(
    name,
    rows,
    tokenizer,
):

    all_lengths = []

    by_label = defaultdict(
        list
    )

    by_source = defaultdict(
        list
    )

    by_attack_type = defaultdict(
        list
    )

    bipia_position_lengths = (
        defaultdict(
            list
        )
    )

    longest = []

    for index, row in enumerate(rows):

        text = build_input(
            row
        )

        token_ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        length = len(
            token_ids
        )

        all_lengths.append(
            length
        )

        label = (
            "attack"
            if row["is_attack"]
            else "benign"
        )

        by_label[
            label
        ].append(
            length
        )

        source = row[
            "metadata"
        ].get(
            "training_source_group",
            "unknown",
        )

        by_source[
            source
        ].append(
            length
        )

        for attack_type in (
            row.get(
                "attack_types",
                []
            )
            or []
        ):

            by_attack_type[
                attack_type
            ].append(
                length
            )

        if source == "bipia_indirect":

            position = row.get(
                "attack_position"
            )

            if position is None:

                position = row[
                    "metadata"
                ].get(
                    "attack_position",
                    "unknown",
                )

            bipia_position_lengths[
                str(position)
            ].append(
                length
            )

        longest.append(
            {
                "tokens": length,

                "sample_id": row.get(
                    "sample_id"
                ),

                "source": source,

                "label": label,

                "attack_types": row.get(
                    "attack_types",
                    [],
                ),

                "attack_position": row.get(
                    "attack_position"
                ),

                "text_preview": text[:500],
            }
        )

    longest.sort(
        key=lambda x: x[
            "tokens"
        ],
        reverse=True,
    )

    report = {
        "split": name,

        "overall": stats(
            all_lengths
        ),

        "by_label": {
            key: stats(value)
            for key, value
            in sorted(
                by_label.items()
            )
        },

        "by_source": {
            key: stats(value)
            for key, value
            in sorted(
                by_source.items()
            )
        },

        "by_attack_type": {
            key: stats(value)
            for key, value
            in sorted(
                by_attack_type.items()
            )
        },

        "bipia_attack_position": {
            key: stats(value)
            for key, value
            in sorted(
                bipia_position_lengths.items()
            )
        },

        "top_20_longest": (
            longest[:20]
        ),
    }

    return report


def print_stats(
    title,
    result,
):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    overall = result[
        "overall"
    ]

    for key, value in (
        overall.items()
    ):

        print(
            f"{key:15}: {value}"
        )

    print()
    print("BY SOURCE")
    print("-" * 90)

    for source, values in (
        result[
            "by_source"
        ].items()
    ):

        print()
        print(source)

        print(
            f"  count    : {values['count']}"
        )

        print(
            f"  median   : {values.get('median', 0):.1f}"
        )

        print(
            f"  p95      : {values.get('p95', 0):.1f}"
        )

        print(
            f"  p99      : {values.get('p99', 0):.1f}"
        )

        print(
            f"  max      : {values.get('max', 0)}"
        )

        print(
            f"  >512     : {values.get('over_512', 0)}"
        )

    print()
    print("BIPIA ATTACK POSITION")
    print("-" * 90)

    for position, values in (
        result[
            "bipia_attack_position"
        ].items()
    ):

        print()
        print(position)

        print(
            f"  count    : {values['count']}"
        )

        print(
            f"  median   : {values.get('median', 0):.1f}"
        )

        print(
            f"  p95      : {values.get('p95', 0):.1f}"
        )

        print(
            f"  max      : {values.get('max', 0)}"
        )

        print(
            f"  >512     : {values.get('over_512', 0)}"
        )


def main():

    print(
        "Loading tokenizer:",
        MODEL_NAME,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    train_rows = read_jsonl(
        TRAIN_FILE
    )

    validation_rows = read_jsonl(
        VALIDATION_FILE
    )

    print(
        "Train rows:",
        len(train_rows)
    )

    print(
        "Validation rows:",
        len(validation_rows)
    )

    train_report = analyze_split(
        "train",
        train_rows,
        tokenizer,
    )

    validation_report = (
        analyze_split(
            "validation",
            validation_rows,
            tokenizer,
        )
    )

    report = {
        "model_name": MODEL_NAME,

        "note": (
            "Tokenizer model_max_length is not used "
            "as a runtime limit; explicit sequence "
            "length will be selected after this audit."
        ),

        "train": train_report,

        "validation": (
            validation_report
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

    print_stats(
        "TRAIN TOKEN LENGTHS",
        train_report,
    )

    print_stats(
        "VALIDATION TOKEN LENGTHS",
        validation_report,
    )

    print()
    print("=" * 90)
    print("LONGEST VALIDATION EXAMPLES")
    print("=" * 90)

    for row in (
        validation_report[
            "top_20_longest"
        ][:10]
    ):

        print()
        print(
            "tokens:",
            row["tokens"]
        )

        print(
            "source:",
            row["source"]
        )

        print(
            "label:",
            row["label"]
        )

        print(
            "attack_position:",
            row[
                "attack_position"
            ]
        )

        print(
            "sample_id:",
            row["sample_id"]
        )

    print()
    print(
        "Report:",
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
