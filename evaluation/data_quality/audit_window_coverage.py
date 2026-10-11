from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "microsoft/deberta-v3-small"

FILES = {
    "train": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "train.jsonl"
    ),
    "validation": (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "validation.jsonl"
    ),
}

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "window_coverage_report.json"
)


MAX_LENGTH = 512

# Reserve a small budget for:
# source type + authoritative user task.
HEADER_TOKEN_BUDGET = 96

# Overlap between consecutive content windows.
STRIDE = 128


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


def safe(value):
    if value is None:
        return ""

    return str(value).strip()


def build_header(
    row,
    tokenizer,
):
    source_type = safe(
        row.get("source_type")
    )

    user_instruction = safe(
        row.get("user_instruction")
    )

    header = (
        "[SOURCE_TYPE]\n"
        f"{source_type or '<UNKNOWN>'}\n\n"
        "[USER_INSTRUCTION]\n"
        f"{user_instruction or '<NONE>'}"
    )

    # Prevent an abnormally long user instruction
    # from consuming the entire 512-token budget.
    ids = tokenizer(
        header,
        add_special_tokens=False,
        truncation=True,
        max_length=HEADER_TOKEN_BUDGET,
    )["input_ids"]

    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
    )


def make_windows(
    row,
    tokenizer,
):
    header = build_header(
        row,
        tokenizer,
    )

    content = safe(
        row.get("text")
    )

    encoded = tokenizer(
        header,
        content,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_attention_mask=False,
    )

    return encoded["input_ids"]


def select_three_windows(
    windows,
):
    n = len(windows)

    if n <= 3:
        return list(
            range(n)
        )

    indices = [
        0,
        n // 2,
        n - 1,
    ]

    # Defensive dedup.
    return sorted(
        set(indices)
    )


def contains_subsequence(
    sequence,
    subsequence,
):
    if not subsequence:
        return False

    if len(subsequence) > len(sequence):
        return False

    limit = (
        len(sequence)
        - len(subsequence)
        + 1
    )

    for i in range(limit):
        if (
            sequence[
                i : i + len(subsequence)
            ]
            == subsequence
        ):
            return True

    return False


def percentile(values, q):
    if len(values) == 0:
        return 0.0

    return float(
        np.percentile(
            values,
            q,
        )
    )


def analyze(
    split,
    rows,
    tokenizer,
):
    window_counts = []

    by_source = defaultdict(
        list
    )

    long_documents = 0
    three_window_documents = 0

    bipia_total = 0

    full_coverage = 0
    selected_coverage = 0

    position_stats = defaultdict(
        Counter
    )

    misses = []

    for row in rows:
        windows = make_windows(
            row,
            tokenizer,
        )

        n_windows = len(
            windows
        )

        window_counts.append(
            n_windows
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
            n_windows
        )

        if n_windows > 1:
            long_documents += 1

        selected_indices = (
            select_three_windows(
                windows
            )
        )

        if len(
            selected_indices
        ) == 3:
            three_window_documents += 1

        # --------------------------------------------------
        # BIPIA malicious examples:
        # use gold payload ONLY to audit window coverage.
        # It is NOT used as an inference feature.
        # --------------------------------------------------

        if (
            source
            != "bipia_indirect"
        ):
            continue

        payload = safe(
            row.get(
                "attack_payload"
            )
        )

        if not payload:
            continue

        bipia_total += 1

        payload_ids = tokenizer(
            payload,
            add_special_tokens=False,
        )["input_ids"]

        payload_in_all = any(
            contains_subsequence(
                window,
                payload_ids,
            )
            for window
            in windows
        )

        payload_in_selected = any(
            contains_subsequence(
                windows[index],
                payload_ids,
            )
            for index
            in selected_indices
        )

        if payload_in_all:
            full_coverage += 1

        if payload_in_selected:
            selected_coverage += 1

        position = safe(
            row.get(
                "attack_position"
            )
        ) or "unknown"

        position_stats[
            position
        ][
            "total"
        ] += 1

        if payload_in_all:
            position_stats[
                position
            ][
                "all_windows_hit"
            ] += 1

        if payload_in_selected:
            position_stats[
                position
            ][
                "three_windows_hit"
            ] += 1

        if not payload_in_selected:
            misses.append(
                {
                    "sample_id": row.get(
                        "sample_id"
                    ),
                    "position": position,
                    "num_windows": n_windows,
                    "selected_indices": (
                        selected_indices
                    ),
                    "payload_preview": (
                        payload[:300]
                    ),
                    "text_length_chars": len(
                        safe(
                            row.get("text")
                        )
                    ),
                }
            )

    source_summary = {}

    for source, values in sorted(
        by_source.items()
    ):
        arr = np.asarray(
            values,
            dtype=np.int64,
        )

        source_summary[
            source
        ] = {
            "records": int(
                len(arr)
            ),
            "mean_windows": float(
                arr.mean()
            ),
            "median_windows": float(
                np.median(arr)
            ),
            "p95_windows": percentile(
                arr,
                95,
            ),
            "max_windows": int(
                arr.max()
            ),
            "multi_window_records": int(
                (
                    arr > 1
                ).sum()
            ),
        }

    positions = {}

    for position, counts in sorted(
        position_stats.items()
    ):
        total = counts[
            "total"
        ]

        positions[
            position
        ] = {
            "total": total,

            "all_windows_hit": (
                counts[
                    "all_windows_hit"
                ]
            ),

            "all_windows_coverage": (
                counts[
                    "all_windows_hit"
                ]
                / total
                if total
                else 0.0
            ),

            "three_windows_hit": (
                counts[
                    "three_windows_hit"
                ]
            ),

            "three_windows_coverage": (
                counts[
                    "three_windows_hit"
                ]
                / total
                if total
                else 0.0
            ),
        }

    result = {
        "split": split,

        "records": len(
            rows
        ),

        "mean_windows_per_record": (
            float(
                np.mean(
                    window_counts
                )
            )
        ),

        "median_windows_per_record": (
            float(
                np.median(
                    window_counts
                )
            )
        ),

        "p95_windows_per_record": (
            percentile(
                window_counts,
                95,
            )
        ),

        "max_windows_per_record": (
            int(
                max(
                    window_counts
                )
            )
        ),

        "multi_window_records": (
            long_documents
        ),

        "records_using_three_views": (
            three_window_documents
        ),

        "by_source": (
            source_summary
        ),

        "bipia_attack_records_with_payload": (
            bipia_total
        ),

        "bipia_payload_all_window_hits": (
            full_coverage
        ),

        "bipia_payload_all_window_coverage": (
            full_coverage
            / bipia_total
            if bipia_total
            else 0.0
        ),

        "bipia_payload_three_window_hits": (
            selected_coverage
        ),

        "bipia_payload_three_window_coverage": (
            selected_coverage
            / bipia_total
            if bipia_total
            else 0.0
        ),

        "bipia_by_attack_position": (
            positions
        ),

        "three_window_misses": (
            misses[:100]
        ),
    }

    return result


def print_result(
    result,
):
    print()
    print("=" * 90)
    print(
        result[
            "split"
        ].upper()
    )
    print("=" * 90)

    print(
        "Records:",
        result["records"]
    )

    print(
        "Mean windows/record:",
        f"{result['mean_windows_per_record']:.3f}"
    )

    print(
        "Median windows/record:",
        result[
            "median_windows_per_record"
        ]
    )

    print(
        "P95 windows/record:",
        result[
            "p95_windows_per_record"
        ]
    )

    print(
        "Max windows/record:",
        result[
            "max_windows_per_record"
        ]
    )

    print(
        "Multi-window records:",
        result[
            "multi_window_records"
        ]
    )

    print()
    print("BIPIA PAYLOAD COVERAGE")
    print("-" * 90)

    print(
        "BIPIA attacks audited:",
        result[
            "bipia_attack_records_with_payload"
        ]
    )

    print(
        "All windows coverage:",
        f"{result['bipia_payload_all_window_coverage']:.4f}"
    )

    print(
        "3-view coverage:",
        f"{result['bipia_payload_three_window_coverage']:.4f}"
    )

    print()
    print("BY ATTACK POSITION")
    print("-" * 90)

    for position, values in (
        result[
            "bipia_by_attack_position"
        ].items()
    ):
        print()

        print(position)

        print(
            "  total:",
            values["total"]
        )

        print(
            "  all-window coverage:",
            f"{values['all_windows_coverage']:.4f}"
        )

        print(
            "  three-window coverage:",
            f"{values['three_windows_coverage']:.4f}"
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
            "  records:",
            values["records"]
        )

        print(
            "  mean windows:",
            f"{values['mean_windows']:.3f}"
        )

        print(
            "  p95 windows:",
            values["p95_windows"]
        )

        print(
            "  max windows:",
            values["max_windows"]
        )

        print(
            "  multi-window:",
            values[
                "multi_window_records"
            ]
        )


def main():
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    report = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "header_token_budget": (
            HEADER_TOKEN_BUDGET
        ),
        "stride": STRIDE,
        "selection_strategy": (
            "all windows when <=3; "
            "otherwise first + middle + last"
        ),
    }

    for split, path in (
        FILES.items()
    ):
        rows = read_jsonl(
            path
        )

        result = analyze(
            split,
            rows,
            tokenizer,
        )

        report[
            split
        ] = result

        print_result(
            result
        )

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

    print()
    print(
        "Report:",
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
