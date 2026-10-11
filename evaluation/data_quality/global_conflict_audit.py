from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


ATTACK_FILES = {
    "direct_attacks": (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "unified"
        / "unified_direct_attacks.jsonl"
    ),

    "bipia_indirect": (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "bipia"
        / "bipia_train_balanced.jsonl"
    ),
}


BENIGN_FILES = {
    "dolly_ordinary": (
        PROJECT_ROOT
        / "data"
        / "interim"
        / "dolly"
        / "dolly_benign_refined.jsonl"
    ),

    "dolly_hard_negative": (
        PROJECT_ROOT
        / "data"
        / "hard_negatives"
        / "dolly_true_hard_negatives.jsonl"
    ),

    "bipia_benign": (
        PROJECT_ROOT
        / "data"
        / "hard_negatives"
        / "bipia_clean_contexts_dedup_balanced.jsonl"
    ),
}


REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "global_conflict_audit_report.json"
)


EXACT_CONFLICT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "global_exact_conflicts.jsonl"
)


NEAR_CONFLICT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "global_near_conflicts.jsonl"
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


def normalize_basic(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_aggressive(
    text: str,
) -> str:

    text = normalize_basic(
        text
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def load_group(
    files: dict[str, Path],
    expected_attack: bool,
) -> list[dict[str, Any]]:

    output = []

    for source_name, path in files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        rows = read_jsonl(
            path
        )

        for row in rows:

            actual = row.get(
                "is_attack"
            )

            if actual is not expected_attack:

                raise ValueError(
                    f"{source_name}: unexpected "
                    f"is_attack={actual} "
                    f"for {row.get('sample_id')}"
                )

            text = str(
                row.get(
                    "text",
                    "",
                )
                or ""
            )

            output.append(
                {
                    "source_group": source_name,

                    "sample_id": row.get(
                        "sample_id"
                    ),

                    "text": text,

                    "basic": normalize_basic(
                        text
                    ),

                    "aggressive": (
                        normalize_aggressive(
                            text
                        )
                    ),

                    "record": row,
                }
            )

    return output


def duplicate_stats(
    records: list[dict[str, Any]],
    key_name: str,
) -> dict[str, Any]:

    counter = Counter(
        row[key_name]
        for row in records
        if row[key_name]
    )

    duplicate_groups = {
        key: count
        for key, count
        in counter.items()
        if count > 1
    }

    duplicate_rows = sum(
        count - 1
        for count
        in duplicate_groups.values()
    )

    return {
        "unique": len(counter),

        "duplicate_groups": len(
            duplicate_groups
        ),

        "duplicate_rows": (
            duplicate_rows
        ),
    }


def build_index(
    records: list[dict[str, Any]],
    key_name: str,
):

    index = defaultdict(list)

    for row in records:

        key = row[key_name]

        if key:
            index[key].append(
                row
            )

    return index


def write_jsonl(
    rows: list[dict[str, Any]],
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

        for row in rows:

            json.dump(
                row,
                f,
                ensure_ascii=False,
            )

            f.write("\n")


def token_signature(
    text: str,
) -> tuple[str, ...]:

    tokens = normalize_aggressive(
        text
    ).split()

    return tuple(
        sorted(
            set(tokens)
        )
    )


def main() -> None:

    attacks = load_group(
        ATTACK_FILES,
        expected_attack=True,
    )

    benign = load_group(
        BENIGN_FILES,
        expected_attack=False,
    )

    print("=" * 80)
    print("GLOBAL ATTACK vs BENIGN CONFLICT AUDIT")
    print("=" * 80)

    print(
        "Attack records:",
        len(attacks),
    )

    print(
        "Benign records:",
        len(benign),
    )

    # ------------------------------------------------
    # Internal duplicate analysis
    # ------------------------------------------------

    attack_basic_stats = (
        duplicate_stats(
            attacks,
            "basic",
        )
    )

    benign_basic_stats = (
        duplicate_stats(
            benign,
            "basic",
        )
    )

    attack_aggressive_stats = (
        duplicate_stats(
            attacks,
            "aggressive",
        )
    )

    benign_aggressive_stats = (
        duplicate_stats(
            benign,
            "aggressive",
        )
    )

    # ------------------------------------------------
    # Exact normalized cross-label conflicts
    # ------------------------------------------------

    attack_index = build_index(
        attacks,
        "basic",
    )

    benign_index = build_index(
        benign,
        "basic",
    )

    exact_shared = (
        set(attack_index)
        & set(benign_index)
    )

    exact_conflicts = []

    for text in sorted(
        exact_shared
    ):

        for attack_row in (
            attack_index[text]
        ):

            for benign_row in (
                benign_index[text]
            ):

                exact_conflicts.append(
                    {
                        "normalized_text": (
                            text
                        ),

                        "attack_sample_id": (
                            attack_row[
                                "sample_id"
                            ]
                        ),

                        "attack_source": (
                            attack_row[
                                "source_group"
                            ]
                        ),

                        "benign_sample_id": (
                            benign_row[
                                "sample_id"
                            ]
                        ),

                        "benign_source": (
                            benign_row[
                                "source_group"
                            ]
                        ),

                        "text_preview": (
                            attack_row[
                                "text"
                            ][:1000]
                        ),
                    }
                )

    # ------------------------------------------------
    # Aggressive normalized cross-label conflicts
    # ------------------------------------------------

    attack_aggressive_index = (
        build_index(
            attacks,
            "aggressive",
        )
    )

    benign_aggressive_index = (
        build_index(
            benign,
            "aggressive",
        )
    )

    aggressive_shared = (
        set(
            attack_aggressive_index
        )
        & set(
            benign_aggressive_index
        )
    )

    # ------------------------------------------------
    # Lightweight near-duplicate search
    #
    # Only compare texts that share a signature prefix
    # to avoid O(N^2) over ~36k records.
    # ------------------------------------------------

    benign_buckets = defaultdict(
        list
    )

    for row in benign:

        tokens = token_signature(
            row["text"]
        )

        if not tokens:
            continue

        bucket_key = (
            len(tokens) // 5,
            tokens[:3],
        )

        benign_buckets[
            bucket_key
        ].append(
            row
        )

    near_conflicts = []

    for attack_row in attacks:

        attack_tokens = (
            token_signature(
                attack_row["text"]
            )
        )

        if not attack_tokens:
            continue

        bucket_key = (
            len(attack_tokens) // 5,
            attack_tokens[:3],
        )

        candidates = (
            benign_buckets.get(
                bucket_key,
                [],
            )
        )

        for benign_row in candidates:

            a = attack_row[
                "aggressive"
            ]

            b = benign_row[
                "aggressive"
            ]

            if not a or not b:
                continue

            # Skip already exact matches.
            if a == b:
                continue

            ratio = SequenceMatcher(
                None,
                a,
                b,
            ).ratio()

            if ratio >= 0.92:

                near_conflicts.append(
                    {
                        "similarity": round(
                            ratio,
                            4,
                        ),

                        "attack_sample_id": (
                            attack_row[
                                "sample_id"
                            ]
                        ),

                        "attack_source": (
                            attack_row[
                                "source_group"
                            ]
                        ),

                        "benign_sample_id": (
                            benign_row[
                                "sample_id"
                            ]
                        ),

                        "benign_source": (
                            benign_row[
                                "source_group"
                            ]
                        ),

                        "attack_preview": (
                            attack_row[
                                "text"
                            ][:800]
                        ),

                        "benign_preview": (
                            benign_row[
                                "text"
                            ][:800]
                        ),
                    }
                )

    near_conflicts.sort(
        key=lambda x: x[
            "similarity"
        ],
        reverse=True,
    )

    # Keep report manageable.
    near_conflicts = (
        near_conflicts[:500]
    )

    write_jsonl(
        exact_conflicts,
        EXACT_CONFLICT_FILE,
    )

    write_jsonl(
        near_conflicts,
        NEAR_CONFLICT_FILE,
    )

    source_attack_counts = Counter(
        row[
            "source_group"
        ]
        for row in attacks
    )

    source_benign_counts = Counter(
        row[
            "source_group"
        ]
        for row in benign
    )

    report = {
        "attack_records": (
            len(attacks)
        ),

        "benign_records": (
            len(benign)
        ),

        "total_records": (
            len(attacks)
            + len(benign)
        ),

        "attack_source_counts": dict(
            sorted(
                source_attack_counts.items()
            )
        ),

        "benign_source_counts": dict(
            sorted(
                source_benign_counts.items()
            )
        ),

        "attack_basic_duplicate_stats": (
            attack_basic_stats
        ),

        "benign_basic_duplicate_stats": (
            benign_basic_stats
        ),

        "attack_aggressive_duplicate_stats": (
            attack_aggressive_stats
        ),

        "benign_aggressive_duplicate_stats": (
            benign_aggressive_stats
        ),

        "exact_cross_label_normalized_keys": (
            len(exact_shared)
        ),

        "exact_cross_label_pairs": (
            len(exact_conflicts)
        ),

        "aggressive_cross_label_keys": (
            len(aggressive_shared)
        ),

        "near_conflict_candidates_ge_0_92": (
            len(near_conflicts)
        ),

        "near_conflict_threshold": (
            0.92
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

    print()
    print("SOURCE COUNTS")
    print("-" * 80)

    print(
        "Attack:",
        dict(
            source_attack_counts
        )
    )

    print(
        "Benign:",
        dict(
            source_benign_counts
        )
    )

    print()
    print("INTERNAL DUPLICATES")
    print("-" * 80)

    print(
        "Attack basic:",
        attack_basic_stats,
    )

    print(
        "Benign basic:",
        benign_basic_stats,
    )

    print(
        "Attack aggressive:",
        attack_aggressive_stats,
    )

    print(
        "Benign aggressive:",
        benign_aggressive_stats,
    )

    print()
    print("CROSS-LABEL CONFLICTS")
    print("-" * 80)

    print(
        "Exact normalized keys:",
        len(exact_shared),
    )

    print(
        "Exact conflict pairs:",
        len(exact_conflicts),
    )

    print(
        "Aggressive normalized keys:",
        len(aggressive_shared),
    )

    print(
        "Near conflicts >= 0.92:",
        len(near_conflicts),
    )

    print()
    print(
        "Report:",
        REPORT_FILE,
    )

    print(
        "Exact conflicts:",
        EXACT_CONFLICT_FILE,
    )

    print(
        "Near conflicts:",
        NEAR_CONFLICT_FILE,
    )


if __name__ == "__main__":
    main()
