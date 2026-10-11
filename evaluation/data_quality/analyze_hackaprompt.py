from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hackaprompt"
    / "source"
    / "hackaprompt.parquet"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "hackaprompt_quality_report.json"
)


def normalize_text(text: str) -> str:
    return " ".join(
        str(text)
        .lower()
        .split()
    )


def percentile(
    values: list[int],
    p: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * p

    lower = int(k)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    if lower == upper:
        return float(
            values[lower]
        )

    weight = k - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def main() -> None:

    df = pd.read_parquet(
        DATA_FILE
    )

    print("=" * 72)
    print("HackAPrompt quality analysis")
    print("=" * 72)

    print(
        f"Total rows                  : {len(df)}"
    )

    # ---------------------------------------
    # Successful attacks
    # ---------------------------------------

    successful = df[
        df["correct"] == True
    ].copy()

    print(
        f"correct=True rows           : {len(successful)}"
    )

    # ---------------------------------------
    # Remove missing attack text
    # ---------------------------------------

    successful_non_null = successful[
        successful["user_input"].notna()
    ].copy()

    print(
        "Successful + user_input      : "
        f"{len(successful_non_null)}"
    )

    successful_non_null[
        "user_input_clean"
    ] = successful_non_null[
        "user_input"
    ].astype(str).str.strip()

    successful_non_empty = (
        successful_non_null[
            successful_non_null[
                "user_input_clean"
            ] != ""
        ]
        .copy()
    )

    print(
        "Successful + non-empty input : "
        f"{len(successful_non_empty)}"
    )

    # ---------------------------------------
    # Exact/normalized duplicates
    # ---------------------------------------

    normalized = successful_non_empty[
        "user_input_clean"
    ].map(
        normalize_text
    )

    unique_normalized = normalized.nunique()

    duplicates_removed = (
        len(successful_non_empty)
        - unique_normalized
    )

    print(
        f"Unique normalized inputs    : {unique_normalized}"
    )

    print(
        f"Duplicate successful rows   : {duplicates_removed}"
    )

    # ---------------------------------------
    # Length statistics
    # ---------------------------------------

    char_lengths = (
        successful_non_empty[
            "user_input_clean"
        ]
        .str.len()
        .tolist()
    )

    token_counts = (
        successful_non_empty[
            "token_count"
        ]
        .astype(int)
        .tolist()
    )

    print()
    print("Character lengths")

    print(
        f"  min     : {min(char_lengths)}"
    )

    print(
        f"  median  : {statistics.median(char_lengths):.2f}"
    )

    print(
        f"  mean    : {statistics.mean(char_lengths):.2f}"
    )

    print(
        f"  p95     : {percentile(char_lengths, 0.95):.2f}"
    )

    print(
        f"  p99     : {percentile(char_lengths, 0.99):.2f}"
    )

    print(
        f"  max     : {max(char_lengths)}"
    )

    print()
    print("Reported token_count")

    print(
        f"  min     : {min(token_counts)}"
    )

    print(
        f"  median  : {statistics.median(token_counts):.2f}"
    )

    print(
        f"  mean    : {statistics.mean(token_counts):.2f}"
    )

    print(
        f"  p95     : {percentile(token_counts, 0.95):.2f}"
    )

    print(
        f"  p99     : {percentile(token_counts, 0.99):.2f}"
    )

    print(
        f"  max     : {max(token_counts)}"
    )

    # ---------------------------------------
    # Distributions
    # ---------------------------------------

    print()
    print("=" * 72)
    print("Successful rows by level")
    print("=" * 72)

    print(
        successful_non_empty[
            "level"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 72)
    print("Successful rows by model")
    print("=" * 72)

    print(
        successful_non_empty[
            "model"
        ]
        .value_counts()
    )

    print()
    print("=" * 72)
    print("Successful rows by source dataset")
    print("=" * 72)

    print(
        successful_non_empty[
            "dataset"
        ]
        .value_counts()
    )

    # ---------------------------------------
    # Success rates
    # ---------------------------------------

    print()
    print("=" * 72)
    print("Success rate by level")
    print("=" * 72)

    success_by_level = (
        df.groupby("level")["correct"]
        .agg(
            total="count",
            successful="sum",
            success_rate="mean",
        )
    )

    success_by_level[
        "success_rate"
    ] *= 100

    print(
        success_by_level
    )

    print()
    print("=" * 72)
    print("Success rate by model")
    print("=" * 72)

    success_by_model = (
        df.groupby("model")["correct"]
        .agg(
            total="count",
            successful="sum",
            success_rate="mean",
        )
    )

    success_by_model[
        "success_rate"
    ] *= 100

    print(
        success_by_model
    )

    # ---------------------------------------
    # Preview unique successful attacks
    # ---------------------------------------

    temp = successful_non_empty.copy()

    temp[
        "_normalized"
    ] = normalized

    unique_success = (
        temp.drop_duplicates(
            subset=["_normalized"],
            keep="first",
        )
    )

    print()
    print("=" * 72)
    print("First 10 unique successful attacks")
    print("=" * 72)

    for i, row in (
        unique_success
        .head(10)
        .iterrows()
    ):

        text = row[
            "user_input_clean"
        ]

        print()
        print(
            f"level={row['level']} "
            f"model={row['model']} "
            f"tokens={row['token_count']}"
        )

        print(
            text[:500]
            .replace("\n", "\\n")
        )

    # ---------------------------------------
    # Build JSON report
    # ---------------------------------------

    report = {
        "total_rows": int(
            len(df)
        ),

        "correct_true_rows": int(
            len(successful)
        ),

        "successful_non_null_input": int(
            len(successful_non_null)
        ),

        "successful_non_empty_input": int(
            len(successful_non_empty)
        ),

        "unique_successful_normalized_inputs": int(
            unique_normalized
        ),

        "duplicate_successful_rows": int(
            duplicates_removed
        ),

        "character_length": {
            "min": int(
                min(char_lengths)
            ),
            "median": float(
                statistics.median(
                    char_lengths
                )
            ),
            "mean": float(
                statistics.mean(
                    char_lengths
                )
            ),
            "p95": float(
                percentile(
                    char_lengths,
                    0.95,
                )
            ),
            "p99": float(
                percentile(
                    char_lengths,
                    0.99,
                )
            ),
            "max": int(
                max(char_lengths)
            ),
        },

        "token_count": {
            "min": int(
                min(token_counts)
            ),
            "median": float(
                statistics.median(
                    token_counts
                )
            ),
            "mean": float(
                statistics.mean(
                    token_counts
                )
            ),
            "p95": float(
                percentile(
                    token_counts,
                    0.95,
                )
            ),
            "p99": float(
                percentile(
                    token_counts,
                    0.99,
                )
            ),
            "max": int(
                max(token_counts)
            ),
        },

        "successful_by_level": {
            str(k): int(v)
            for k, v in (
                successful_non_empty[
                    "level"
                ]
                .value_counts()
                .sort_index()
                .items()
            )
        },

        "successful_by_model": {
            str(k): int(v)
            for k, v in (
                successful_non_empty[
                    "model"
                ]
                .value_counts()
                .items()
            )
        },

        "successful_by_dataset": {
            str(k): int(v)
            for k, v in (
                successful_non_empty[
                    "dataset"
                ]
                .value_counts()
                .items()
            )
        },
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

    print()
    print(
        "Report written to:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
