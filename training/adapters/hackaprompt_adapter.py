from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hackaprompt"
    / "source"
    / "hackaprompt.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hackaprompt"
)


LEVEL_LABEL_MAP = {
    0: ["instruction_override"],
    1: ["instruction_override"],
    2: ["secret_extraction"],
    3: ["instruction_override"],
    4: ["instruction_override"],
    5: ["instruction_override"],

    # Level 6 is a two-stage attack:
    # user input -> model 1 output -> model 2.
    6: [
        "instruction_override",
        "multi_step_jailbreak",
    ],

    7: ["instruction_override"],
    8: ["instruction_override"],
    9: ["instruction_override"],
}


def normalize_for_dedup(text: str) -> str:
    """
    Conservative deduplication normalization.

    Preserve punctuation and adversarial characters.
    Only lowercase and collapse whitespace.
    """

    return " ".join(
        str(text)
        .lower()
        .split()
    )


def make_sample_id(
    normalized_text: str,
) -> str:

    digest = hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()[:16]

    return f"hackaprompt_{digest}"


def valid_successful_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    filtered = df[
        df["correct"] == True
    ].copy()

    filtered = filtered[
        filtered["user_input"].notna()
    ].copy()

    filtered[
        "user_input_clean"
    ] = (
        filtered["user_input"]
        .astype(str)
        .str.strip()
    )

    filtered = filtered[
        filtered["user_input_clean"] != ""
    ].copy()

    # Level 10 has no successful records in the
    # downloaded dataset, but explicitly restrict
    # conversion to levels whose semantics we mapped.
    filtered = filtered[
        filtered["level"].isin(
            LEVEL_LABEL_MAP.keys()
        )
    ].copy()

    filtered[
        "_normalized"
    ] = filtered[
        "user_input_clean"
    ].map(
        normalize_for_dedup
    )

    return filtered


def build_record(
    normalized_text: str,
    group: pd.DataFrame,
) -> dict[str, Any]:

    representative = group.iloc[0]

    levels = sorted(
        int(level)
        for level in group[
            "level"
        ].unique()
    )

    models = sorted(
        str(model)
        for model in group[
            "model"
        ].dropna().unique()
    )

    source_datasets = sorted(
        str(source)
        for source in group[
            "dataset"
        ].dropna().unique()
    )

    attack_types = sorted(
        {
            attack_type
            for level in levels
            for attack_type in LEVEL_LABEL_MAP[
                level
            ]
        }
    )

    expected_completions = sorted(
        {
            str(value)
            for value in group[
                "expected_completion"
            ].dropna().unique()
        }
    )

    target_data_types = []

    if "secret_extraction" in attack_types:
        target_data_types.append(
            "secret_token"
        )

    original_labels = [
        f"level_{level}"
        for level in levels
    ]

    return {
        "sample_id": make_sample_id(
            normalized_text
        ),

        # Preserve the original first-seen attack.
        "text": str(
            representative[
                "user_input_clean"
            ]
        ),

        "normalized_text": None,

        "is_attack": True,

        "attack_types": attack_types,

        "source_type": "user_message",

        "trust_level": "untrusted",

        "directness": "direct",

        # We do not infer encoding simply from level.
        "encoding_type": "unknown",

        "language": "en",

        "malicious_spans": [
            str(
                representative[
                    "user_input_clean"
                ]
            )
        ],

        "requested_actions": [],

        "target_data_types": (
            target_data_types
        ),

        "severity": "high",

        "dataset_source": "hackaprompt",

        "original_label": "|".join(
            original_labels
        ),

        "split": "unassigned",

        "metadata": {
            "hackaprompt_levels": levels,

            "successful_models": models,

            "source_datasets": (
                source_datasets
            ),

            "success_count": int(
                len(group)
            ),

            "expected_completions": (
                expected_completions
            ),

            "minimum_token_count": int(
                group[
                    "token_count"
                ].min()
            ),

            "maximum_token_count": int(
                group[
                    "token_count"
                ].max()
            ),

            "original_row_count": int(
                len(group)
            ),
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


def validate_record(
    record: dict[str, Any],
) -> None:

    required = {
        "sample_id",
        "text",
        "is_attack",
        "attack_types",
        "source_type",
        "dataset_source",
    }

    missing = (
        required
        - record.keys()
    )

    if missing:
        raise ValueError(
            f"Missing fields: {missing}"
        )

    if not record["text"].strip():
        raise ValueError(
            "Empty attack text"
        )

    if not record["attack_types"]:
        raise ValueError(
            "Attack record has no labels"
        )


def convert(
    input_path: Path,
    output_dir: Path,
) -> None:

    print(
        f"Loading {input_path} ..."
    )

    df = pd.read_parquet(
        input_path
    )

    total_rows = len(df)

    successful = valid_successful_rows(
        df
    )

    print(
        f"Raw rows                  : {total_rows}"
    )

    print(
        "Valid successful rows      : "
        f"{len(successful)}"
    )

    grouped = successful.groupby(
        "_normalized",
        sort=False,
    )

    records = []

    for normalized_text, group in grouped:

        record = build_record(
            normalized_text,
            group,
        )

        validate_record(
            record
        )

        records.append(
            record
        )

    records.sort(
        key=lambda r: r[
            "sample_id"
        ]
    )

    output_file = (
        output_dir
        / "hackaprompt_cleaned.jsonl"
    )

    write_jsonl(
        records,
        output_file,
    )

    # ---------------------------------
    # Statistics
    # ---------------------------------

    label_counts: dict[
        str,
        int,
    ] = defaultdict(int)

    multilabel_count = 0

    for record in records:

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
        "raw_rows": int(
            total_rows
        ),

        "successful_rows_before_dedup": int(
            len(successful)
        ),

        "unique_attack_records": int(
            len(records)
        ),

        "duplicates_collapsed": int(
            len(successful)
            - len(records)
        ),

        "attack_type_counts": dict(
            sorted(
                label_counts.items()
            )
        ),

        "multi_label_records": int(
            multilabel_count
        ),

        "level_mapping": {
            str(level): labels
            for level, labels
            in LEVEL_LABEL_MAP.items()
        },
    }

    report_file = (
        PROJECT_ROOT
        / "evaluation"
        / "data_quality"
        / "hackaprompt_conversion_report.json"
    )

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print(
        "HackAPrompt conversion complete"
    )
    print("=" * 72)

    print(
        f"Successful rows         : {len(successful)}"
    )

    print(
        f"Unique attack records   : {len(records)}"
    )

    print(
        "Duplicates collapsed    : "
        f"{len(successful) - len(records)}"
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
        f"Multi-label records      : {multilabel_count}"
    )

    print()

    print(
        f"Output: {output_file}"
    )

    print(
        f"Report: {report_file}"
    )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    convert(
        args.input,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
