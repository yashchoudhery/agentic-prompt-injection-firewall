from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.detectors.deberta_detector import (
    DebertaPromptInjectionDetector,
)
from src.detectors.rule_detector import (
    PromptInjectionRuleDetector,
)
from src.firewall.risk_fusion import (
    PromptInjectionRiskFusion,
)


DATASETS = {
    "bipia": (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "bipia"
        / "bipia_external_test.jsonl"
    ),
    "injecagent": (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "injecagent"
        / "injecagent_external_test.jsonl"
    ),
}

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "external"
)


def load_rows(
    path: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:

    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            if not bool(
                row.get(
                    "is_attack",
                    False,
                )
            ):
                raise RuntimeError(
                    f"External attack benchmark "
                    f"contains benign row: "
                    f"{row.get('sample_id')}"
                )

            rows.append(row)

            if (
                limit is not None
                and len(rows) >= limit
            ):
                break

    return rows


def percentile(
    values: list[float],
    q: float,
) -> float:

    if not values:
        return 0.0

    values = sorted(values)

    index = int(
        round(
            (len(values) - 1)
            * q
        )
    )

    return float(
        values[index]
    )


def blank_group() -> dict[str, int]:

    return {
        "total": 0,
        "neural_detected": 0,
        "neural_high_risk": 0,
        "rule_detected": 0,
        "fused_detected": 0,
    }


def update_group(
    groups: dict[
        str,
        dict[
            str,
            dict[str, int]
        ]
    ],
    dimension: str,
    value: Any,
    *,
    neural_detected: bool,
    neural_high_risk: bool,
    rule_detected: bool,
    fused_detected: bool,
) -> None:

    value = str(
        value
        if value is not None
        else "unknown"
    )

    stats = groups[
        dimension
    ][
        value
    ]

    stats[
        "total"
    ] += 1

    stats[
        "neural_detected"
    ] += int(
        neural_detected
    )

    stats[
        "neural_high_risk"
    ] += int(
        neural_high_risk
    )

    stats[
        "rule_detected"
    ] += int(
        rule_detected
    )

    stats[
        "fused_detected"
    ] += int(
        fused_detected
    )


def finalize_groups(
    groups: dict[
        str,
        dict[
            str,
            dict[str, int]
        ]
    ]
) -> dict[str, Any]:

    output = {}

    for dimension, values in groups.items():

        output[
            dimension
        ] = {}

        for value, stats in values.items():

            total = stats[
                "total"
            ]

            output[
                dimension
            ][
                value
            ] = {
                **stats,

                "neural_recall": round(
                    stats[
                        "neural_detected"
                    ]
                    / total,
                    6,
                ),

                "neural_high_risk_rate": round(
                    stats[
                        "neural_high_risk"
                    ]
                    / total,
                    6,
                ),

                "rule_recall": round(
                    stats[
                        "rule_detected"
                    ]
                    / total,
                    6,
                ),

                "fused_recall": round(
                    stats[
                        "fused_detected"
                    ]
                    / total,
                    6,
                ),

                "fused_miss_rate": round(
                    1.0
                    - (
                        stats[
                            "fused_detected"
                        ]
                        / total
                    ),
                    6,
                ),
            }

    return output


def evaluate_dataset(
    *,
    name: str,
    rows: list[dict[str, Any]],
    neural: DebertaPromptInjectionDetector,
    rules: PromptInjectionRuleDetector,
    fusion_policy: PromptInjectionRiskFusion,
    doc_chunk_size: int,
) -> dict[str, Any]:

    channel_cfg = (
        neural.config
        .get(
            "channel_thresholds",
            {}
        )
        .get(
            "external_content",
            {}
        )
    )

    review_threshold = float(
        channel_cfg.get(
            "review_threshold",
            0.70,
        )
    )

    high_threshold = float(
        channel_cfg.get(
            "high_risk_threshold",
            0.90,
        )
    )

    total = len(rows)

    neural_detected_count = 0
    neural_high_count = 0
    rule_detected_count = 0
    fused_detected_count = 0

    action_counts = Counter()
    risk_counts = Counter()

    neural_probabilities = []

    groups = defaultdict(
        lambda: defaultdict(
            blank_group
        )
    )

    missed_ids = []

    started = time.perf_counter()

    for chunk_start in range(
        0,
        total,
        doc_chunk_size,
    ):

        chunk = rows[
            chunk_start:
            chunk_start
            + doc_chunk_size
        ]

        # ----------------------------------------------------
        # Exact runtime-safe representation used by detector
        # ----------------------------------------------------

        headers = [
            neural._build_header(
                input_channel=(
                    "external_content"
                ),
                user_instruction=(
                    row.get(
                        "user_instruction"
                    )
                ),
            )
            for row in chunk
        ]

        contents = [
            str(
                row.get("text")
                or row.get(
                    "poisoned_content"
                )
                or ""
            )
            for row in chunk
        ]

        encoded = neural.tokenizer(
            headers,
            contents,
            truncation="only_second",
            max_length=(
                neural.max_length
            ),
            stride=(
                neural.stride
            ),
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )

        mapping = list(
            encoded[
                "overflow_to_sample_mapping"
            ]
        )

        windows = []

        for i in range(
            len(
                encoded[
                    "input_ids"
                ]
            )
        ):

            item = {
                "input_ids": (
                    encoded[
                        "input_ids"
                    ][i]
                ),
                "attention_mask": (
                    encoded[
                        "attention_mask"
                    ][i]
                ),
            }

            if (
                "token_type_ids"
                in encoded
            ):
                item[
                    "token_type_ids"
                ] = encoded[
                    "token_type_ids"
                ][i]

            windows.append(
                item
            )

        window_scores = (
            neural._score_windows(
                windows
            )
        )

        document_scores = [
            0.0
            for _ in chunk
        ]

        for (
            sample_index,
            score,
        ) in zip(
            mapping,
            window_scores,
        ):

            document_scores[
                sample_index
            ] = max(
                document_scores[
                    sample_index
                ],
                float(score),
            )

        # ----------------------------------------------------
        # Rules + final policy
        # ----------------------------------------------------

        for row, probability in zip(
            chunk,
            document_scores,
        ):

            neural_probabilities.append(
                probability
            )

            if (
                probability
                >= high_threshold
            ):
                neural_decision = (
                    "HIGH_RISK"
                )

            elif (
                probability
                >= review_threshold
            ):
                neural_decision = (
                    "SECURITY_REVIEW"
                )

            else:
                neural_decision = (
                    "ALLOW"
                )

            neural_detected = (
                neural_decision
                != "ALLOW"
            )

            neural_high = (
                neural_decision
                == "HIGH_RISK"
            )

            rule_result = (
                rules.analyze(
                    text=str(
                        row.get(
                            "text",
                            ""
                        )
                    ),
                    input_channel=(
                        "external_content"
                    ),
                    source_type=(
                        row.get(
                            "source_type"
                        )
                    ),
                    debug=False,
                )
            )

            rule_detected = (
                float(
                    rule_result[
                        "rule_score"
                    ]
                )
                >= 0.70
            )

            neural_result = {
                "decision": (
                    neural_decision
                ),
                "attack_probability": (
                    probability
                ),
            }

            (
                action,
                risk_level,
                _,
            ) = (
                fusion_policy._final_policy(
                    input_channel=(
                        "external_content"
                    ),
                    neural_result=(
                        neural_result
                    ),
                    rule_result=(
                        rule_result
                    ),
                )
            )

            fused_detected = (
                action != "ALLOW"
            )

            neural_detected_count += int(
                neural_detected
            )

            neural_high_count += int(
                neural_high
            )

            rule_detected_count += int(
                rule_detected
            )

            fused_detected_count += int(
                fused_detected
            )

            action_counts[
                action
            ] += 1

            risk_counts[
                risk_level
            ] += 1

            if not fused_detected:

                if len(
                    missed_ids
                ) < 100:
                    missed_ids.append(
                        row.get(
                            "sample_id"
                        )
                    )

            # ------------------------------------------------
            # Group breakdowns
            # ------------------------------------------------

            update_group(
                groups,
                "source_type",
                row.get(
                    "source_type"
                ),
                neural_detected=(
                    neural_detected
                ),
                neural_high_risk=(
                    neural_high
                ),
                rule_detected=(
                    rule_detected
                ),
                fused_detected=(
                    fused_detected
                ),
            )

            update_group(
                groups,
                "attack_position",
                row.get(
                    "attack_position"
                ),
                neural_detected=(
                    neural_detected
                ),
                neural_high_risk=(
                    neural_high
                ),
                rule_detected=(
                    rule_detected
                ),
                fused_detected=(
                    fused_detected
                ),
            )

            update_group(
                groups,
                "original_label",
                row.get(
                    "original_label"
                ),
                neural_detected=(
                    neural_detected
                ),
                neural_high_risk=(
                    neural_high
                ),
                rule_detected=(
                    rule_detected
                ),
                fused_detected=(
                    fused_detected
                ),
            )

            for attack_type in row.get(
                "attack_types",
                []
            ):

                update_group(
                    groups,
                    "attack_type",
                    attack_type,
                    neural_detected=(
                        neural_detected
                    ),
                    neural_high_risk=(
                        neural_high
                    ),
                    rule_detected=(
                        rule_detected
                    ),
                    fused_detected=(
                        fused_detected
                    ),
                )

            metadata = row.get(
                "metadata",
                {}
            )

            if name == "injecagent":

                update_group(
                    groups,
                    "attack_family",
                    metadata.get(
                        "attack_family"
                    ),
                    neural_detected=(
                        neural_detected
                    ),
                    neural_high_risk=(
                        neural_high
                    ),
                    rule_detected=(
                        rule_detected
                    ),
                    fused_detected=(
                        fused_detected
                    ),
                )

                update_group(
                    groups,
                    "variant",
                    metadata.get(
                        "variant"
                    ),
                    neural_detected=(
                        neural_detected
                    ),
                    neural_high_risk=(
                        neural_high
                    ),
                    rule_detected=(
                        rule_detected
                    ),
                    fused_detected=(
                        fused_detected
                    ),
                )

        done = min(
            chunk_start
            + len(chunk),
            total,
        )

        print(
            f"[{name}] "
            f"{done}/{total} "
            f"({done / total:.1%})",
            flush=True,
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    missed_count = (
        total
        - fused_detected_count
    )

    report = {
        "dataset": name,

        "evaluation_type": (
            "untouched_external_attack_only"
        ),

        "total_attacks": total,

        "benign_samples": 0,

        "metrics_supported": [
            "attack_recall",
            "miss_rate",
            "false_negatives",
            "per_family_recall",
        ],

        "metrics_not_supported": [
            "precision",
            "false_positive_rate",
            "specificity",
        ],

        "thresholds": {
            "external_review": (
                review_threshold
            ),
            "external_high_risk": (
                high_threshold
            ),
            "rule_detection": 0.70,
        },

        "neural": {
            "detected": (
                neural_detected_count
            ),

            "missed": (
                total
                - neural_detected_count
            ),

            "recall": round(
                neural_detected_count
                / total,
                6,
            ),

            "miss_rate": round(
                1.0
                - (
                    neural_detected_count
                    / total
                ),
                6,
            ),

            "high_risk_count": (
                neural_high_count
            ),

            "high_risk_rate": round(
                neural_high_count
                / total,
                6,
            ),
        },

        "rules": {
            "detected": (
                rule_detected_count
            ),

            "missed": (
                total
                - rule_detected_count
            ),

            "recall": round(
                rule_detected_count
                / total,
                6,
            ),
        },

        "fused_firewall": {
            "detected": (
                fused_detected_count
            ),

            "false_negatives": (
                missed_count
            ),

            "recall": round(
                fused_detected_count
                / total,
                6,
            ),

            "miss_rate": round(
                missed_count
                / total,
                6,
            ),
        },

        "action_distribution": dict(
            action_counts
        ),

        "risk_distribution": dict(
            risk_counts
        ),

        "probability_summary": {
            "mean": round(
                statistics.mean(
                    neural_probabilities
                ),
                6,
            ),

            "median": round(
                statistics.median(
                    neural_probabilities
                ),
                6,
            ),

            "p05": round(
                percentile(
                    neural_probabilities,
                    0.05,
                ),
                6,
            ),

            "p95": round(
                percentile(
                    neural_probabilities,
                    0.95,
                ),
                6,
            ),

            "min": round(
                min(
                    neural_probabilities
                ),
                6,
            ),

            "max": round(
                max(
                    neural_probabilities
                ),
                6,
            ),
        },

        "groups": (
            finalize_groups(
                groups
            )
        ),

        "first_missed_sample_ids": (
            missed_ids
        ),

        "elapsed_seconds": round(
            elapsed,
            3,
        ),

        "documents_per_second": round(
            total / elapsed,
            3,
        ),
    }

    return report


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "bipia",
            "injecagent",
        ],
        default=[
            "bipia",
            "injecagent",
        ],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("EXTERNAL ATTACK BENCHMARK")
    print("=" * 100)

    print(
        "Device:",
        args.device
    )

    print(
        "Inference batch size:",
        args.batch_size
    )

    print(
        "Document chunk size:",
        args.chunk_size
    )

    print(
        "Limit:",
        args.limit
    )

    neural = (
        DebertaPromptInjectionDetector(
            device=args.device,
            batch_size=args.batch_size,
        )
    )

    rules = (
        PromptInjectionRuleDetector()
    )

    # We need only the deterministic final policy method.
    # __init__ is intentionally skipped so a second
    # DeBERTa model is NOT loaded.
    fusion_policy = object.__new__(
        PromptInjectionRiskFusion
    )

    all_reports = {}

    for name in args.datasets:

        rows = load_rows(
            DATASETS[name],
            limit=args.limit,
        )

        print()
        print(
            f"Evaluating {name}: "
            f"{len(rows)} attacks"
        )

        report = evaluate_dataset(
            name=name,
            rows=rows,
            neural=neural,
            rules=rules,
            fusion_policy=(
                fusion_policy
            ),
            doc_chunk_size=(
                args.chunk_size
            ),
        )

        all_reports[
            name
        ] = report

        print()
        print("-" * 100)

        print(
            f"{name.upper()} RESULTS"
        )

        print("-" * 100)

        print(
            "Neural recall:",
            f"{report['neural']['recall']:.4%}"
        )

        print(
            "Rule recall:",
            f"{report['rules']['recall']:.4%}"
        )

        print(
            "Fused firewall recall:",
            f"{report['fused_firewall']['recall']:.4%}"
        )

        print(
            "False negatives:",
            report[
                "fused_firewall"
            ][
                "false_negatives"
            ]
        )

        print(
            "Actions:",
            report[
                "action_distribution"
            ]
        )

    output = {
        "benchmark_scope": (
            "attack-only untouched external "
            "BIPIA and InjecAgent sets"
        ),

        "important_note": (
            "These external sets contain no benign "
            "samples. Precision and false-positive "
            "rate are therefore intentionally not "
            "reported."
        ),

        "reports": (
            all_reports
        ),
    }

    if args.limit is None:

        filename = (
            "external_attack_benchmark_report.json"
        )

    else:

        filename = (
            f"external_attack_benchmark_smoke_"
            f"{args.limit}.json"
        )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.write("\n")

    print()
    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()
