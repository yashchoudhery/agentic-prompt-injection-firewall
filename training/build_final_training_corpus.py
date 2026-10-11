from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ALL_OUTPUT = (
    OUTPUT_DIR
    / "final_corpus.jsonl"
)

TRAIN_OUTPUT = (
    OUTPUT_DIR
    / "train.jsonl"
)

VALIDATION_OUTPUT = (
    OUTPUT_DIR
    / "validation.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "final_corpus_report.json"
)


VALIDATION_RATIO = 0.15

# Deterministic split.
SPLIT_SEED = "accenture-prompt-protector-v1"


# ================================================================
# Basic utilities
# ================================================================

def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    rows = []

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
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON: "
                    f"{path}:{line_number}"
                ) from exc

    return rows


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


def short_hash(
    text: str,
    length: int = 16,
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:length]


# ================================================================
# Initial leakage-group assignment
# ================================================================

def get_initial_group(
    row: dict[str, Any],
    source_group: str,
) -> str:

    metadata = (
        row.get("metadata")
        or {}
    )

    # ------------------------------------------------------------
    # BIPIA:
    #
    # Clean and poisoned versions of the exact same original
    # context MUST share one leakage group.
    # ------------------------------------------------------------

    if source_group in {
        "bipia_indirect",
        "bipia_benign",
    }:

        source_task = metadata.get(
            "bipia_source_task"
        )

        context_index = metadata.get(
            "bipia_context_index"
        )

        if (
            source_task is None
            or context_index is None
        ):

            raise ValueError(
                "Missing BIPIA grouping metadata "
                f"for sample {row.get('sample_id')}"
            )

        return (
            "bipia_train_context::"
            f"{source_task}::"
            f"{context_index}"
        )

    # ------------------------------------------------------------
    # Dolly:
    # each unique Dolly example is independent.
    # ------------------------------------------------------------

    if source_group.startswith(
        "dolly_"
    ):

        return (
            "dolly::"
            + str(
                row.get(
                    "sample_id"
                )
            )
        )

    # ------------------------------------------------------------
    # Direct attacks:
    #
    # unified_direct_attacks was already cleaned, so group by
    # normalized text.
    # ------------------------------------------------------------

    text = normalize_basic(
        row.get(
            "text",
            "",
        )
    )

    return (
        "direct_attack::"
        + short_hash(text)
    )


# ================================================================
# Union-Find
#
# Required because an exact duplicate attack may appear in multiple
# original groups. If we simply deduplicate the record without
# joining those groups, a related record could leak across splits.
# ================================================================

class UnionFind:

    def __init__(self):

        self.parent = {}

    def add(
        self,
        item: str,
    ) -> None:

        if item not in self.parent:
            self.parent[item] = item

    def find(
        self,
        item: str,
    ) -> str:

        self.add(item)

        if (
            self.parent[item]
            != item
        ):

            self.parent[item] = (
                self.find(
                    self.parent[item]
                )
            )

        return self.parent[item]

    def union(
        self,
        a: str,
        b: str,
    ) -> None:

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        # Stable deterministic root.
        if root_a < root_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_a] = root_b


# ================================================================
# Load records
# ================================================================

def load_sources(
    files: dict[str, Path],
    expected_attack: bool,
) -> list[dict[str, Any]]:

    output = []

    for source_group, path in (
        files.items()
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        rows = read_jsonl(
            path
        )

        for row in rows:

            if (
                row.get("is_attack")
                is not expected_attack
            ):

                raise ValueError(
                    f"Unexpected label in "
                    f"{source_group}: "
                    f"{row.get('sample_id')}"
                )

            row = copy.deepcopy(
                row
            )

            row[
                "_source_group"
            ] = source_group

            row[
                "_basic_normalized"
            ] = normalize_basic(
                row.get(
                    "text",
                    "",
                )
            )

            if not row[
                "_basic_normalized"
            ]:

                raise ValueError(
                    "Empty text after normalization: "
                    f"{row.get('sample_id')}"
                )

            row[
                "_initial_group"
            ] = get_initial_group(
                row,
                source_group,
            )

            output.append(
                row
            )

    return output


# ================================================================
# Merge duplicate attack rows
# ================================================================

def merge_attack_records(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    int,
    int,
]:

    grouped = defaultdict(list)

    for row in records:

        grouped[
            row["_basic_normalized"]
        ].append(
            row
        )

    merged = []

    duplicate_groups = 0
    duplicate_rows_removed = 0

    for normalized_text, rows in (
        grouped.items()
    ):

        if len(rows) > 1:

            duplicate_groups += 1

            duplicate_rows_removed += (
                len(rows) - 1
            )

        # Prefer richest record as representative.
        rows = sorted(
            rows,
            key=lambda r: (
                len(
                    r.get(
                        "attack_types",
                        [],
                    )
                ),
                len(
                    str(
                        r.get(
                            "text",
                            "",
                        )
                    )
                ),
            ),
            reverse=True,
        )

        canonical = copy.deepcopy(
            rows[0]
        )

        all_attack_types = sorted(
            {
                attack_type
                for row in rows
                for attack_type in (
                    row.get(
                        "attack_types",
                        []
                    )
                    or []
                )
            }
        )

        sample_ids = [
            row.get(
                "sample_id"
            )
            for row in rows
        ]

        source_groups = sorted(
            {
                row[
                    "_source_group"
                ]
                for row in rows
            }
        )

        initial_groups = sorted(
            {
                row[
                    "_initial_group"
                ]
                for row in rows
            }
        )

        dataset_sources = sorted(
            {
                str(
                    row.get(
                        "dataset_source"
                    )
                )
                for row in rows
            }
        )

        canonical[
            "attack_types"
        ] = all_attack_types

        metadata = copy.deepcopy(
            canonical.get(
                "metadata"
            )
            or {}
        )

        metadata[
            "global_duplicate_count"
        ] = len(rows)

        metadata[
            "merged_sample_ids"
        ] = sample_ids

        metadata[
            "merged_source_groups"
        ] = source_groups

        metadata[
            "merged_dataset_sources"
        ] = dataset_sources

        metadata[
            "merged_initial_groups"
        ] = initial_groups

        canonical[
            "metadata"
        ] = metadata

        # Keep all group membership so Union-Find
        # can connect them later.
        canonical[
            "_all_initial_groups"
        ] = initial_groups

        merged.append(
            canonical
        )

    return (
        merged,
        duplicate_groups,
        duplicate_rows_removed,
    )


# ================================================================
# Create leakage components
# ================================================================

def assign_component_groups(
    attacks: list[dict[str, Any]],
    benign: list[dict[str, Any]],
) -> None:

    uf = UnionFind()

    # Add every original group.
    for row in attacks + benign:

        uf.add(
            row[
                "_initial_group"
            ]
        )

    # Exact attack duplicates may connect multiple
    # original groups.
    for row in attacks:

        groups = row.get(
            "_all_initial_groups",
            [
                row[
                    "_initial_group"
                ]
            ],
        )

        if not groups:
            continue

        first = groups[0]

        for other in groups[1:]:

            uf.union(
                first,
                other,
            )

    # Assign stable component IDs.
    for row in attacks + benign:

        root = uf.find(
            row[
                "_initial_group"
            ]
        )

        row[
            "_component_group"
        ] = (
            "group_"
            + short_hash(
                root,
                20,
            )
        )


# ================================================================
# Deterministic group-aware split
# ================================================================

def assign_split(
    group_id: str,
) -> str:

    raw = (
        SPLIT_SEED
        + "::"
        + group_id
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    number = int(
        digest[:12],
        16,
    )

    fraction = (
        number
        / float(
            16 ** 12
        )
    )

    if fraction < VALIDATION_RATIO:
        return "validation"

    return "train"


# ================================================================
# Finalize rows
# ================================================================

def finalize_record(
    row: dict[str, Any],
) -> dict[str, Any]:

    output = copy.deepcopy(
        row
    )

    component_group = (
        output.pop(
            "_component_group"
        )
    )

    source_group = (
        output.pop(
            "_source_group"
        )
    )

    output.pop(
        "_basic_normalized",
        None,
    )

    output.pop(
        "_initial_group",
        None,
    )

    output.pop(
        "_all_initial_groups",
        None,
    )

    split = assign_split(
        component_group
    )

    output[
        "split"
    ] = split

    metadata = copy.deepcopy(
        output.get(
            "metadata"
        )
        or {}
    )

    metadata[
        "training_source_group"
    ] = source_group

    metadata[
        "leakage_group_id"
    ] = component_group

    output[
        "metadata"
    ] = metadata

    return output


# ================================================================
# Leakage verification
# ================================================================

def verify_no_group_leakage(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:

    train_groups = {
        row[
            "metadata"
        ][
            "leakage_group_id"
        ]
        for row in train
    }

    validation_groups = {
        row[
            "metadata"
        ][
            "leakage_group_id"
        ]
        for row in validation
    }

    overlap = (
        train_groups
        & validation_groups
    )

    if overlap:

        raise RuntimeError(
            "GROUP LEAKAGE DETECTED: "
            f"{len(overlap)} groups"
        )


def verify_no_exact_text_leakage(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:

    train_texts = {
        normalize_basic(
            row["text"]
        )
        for row in train
    }

    validation_texts = {
        normalize_basic(
            row["text"]
        )
        for row in validation
    }

    overlap = (
        train_texts
        & validation_texts
    )

    if overlap:

        raise RuntimeError(
            "EXACT TEXT LEAKAGE DETECTED: "
            f"{len(overlap)} normalized texts"
        )


# ================================================================
# Main
# ================================================================

def main() -> None:

    attacks_raw = load_sources(
        ATTACK_FILES,
        expected_attack=True,
    )

    benign = load_sources(
        BENIGN_FILES,
        expected_attack=False,
    )

    (
        attacks,
        duplicate_groups,
        duplicate_rows_removed,
    ) = merge_attack_records(
        attacks_raw
    )

    assign_component_groups(
        attacks,
        benign,
    )

    final_records = [
        finalize_record(row)
        for row in (
            attacks
            + benign
        )
    ]

    train = [
        row
        for row in final_records
        if row["split"]
        == "train"
    ]

    validation = [
        row
        for row in final_records
        if row["split"]
        == "validation"
    ]

    verify_no_group_leakage(
        train,
        validation,
    )

    verify_no_exact_text_leakage(
        train,
        validation,
    )

    # Stable ordering for reproducibility.
    final_records.sort(
        key=lambda r: (
            r["split"],
            str(
                r.get(
                    "sample_id"
                )
            ),
        )
    )

    train.sort(
        key=lambda r: str(
            r.get(
                "sample_id"
            )
        )
    )

    validation.sort(
        key=lambda r: str(
            r.get(
                "sample_id"
            )
        )
    )

    write_jsonl(
        final_records,
        ALL_OUTPUT,
    )

    write_jsonl(
        train,
        TRAIN_OUTPUT,
    )

    write_jsonl(
        validation,
        VALIDATION_OUTPUT,
    )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    def label_counts(rows):

        counts = Counter(
            "attack"
            if row[
                "is_attack"
            ]
            else "benign"
            for row in rows
        )

        return dict(
            sorted(
                counts.items()
            )
        )

    def source_counts(rows):

        counts = Counter(
            row[
                "metadata"
            ][
                "training_source_group"
            ]
            for row in rows
        )

        return dict(
            sorted(
                counts.items()
            )
        )

    train_attack_types = Counter()
    validation_attack_types = Counter()

    for row in train:

        for attack_type in (
            row.get(
                "attack_types",
                []
            )
            or []
        ):

            train_attack_types[
                attack_type
            ] += 1

    for row in validation:

        for attack_type in (
            row.get(
                "attack_types",
                []
            )
            or []
        ):

            validation_attack_types[
                attack_type
            ] += 1

    train_groups = {
        row["metadata"][
            "leakage_group_id"
        ]
        for row in train
    }

    validation_groups = {
        row["metadata"][
            "leakage_group_id"
        ]
        for row in validation
    }

    report = {
        "raw_attack_records": (
            len(attacks_raw)
        ),

        "raw_benign_records": (
            len(benign)
        ),

        "attack_duplicate_groups": (
            duplicate_groups
        ),

        "attack_duplicate_rows_removed": (
            duplicate_rows_removed
        ),

        "final_unique_attack_records": (
            len(attacks)
        ),

        "final_unique_benign_records": (
            len(benign)
        ),

        "final_total_records": (
            len(final_records)
        ),

        "validation_ratio_requested": (
            VALIDATION_RATIO
        ),

        "train_records": (
            len(train)
        ),

        "validation_records": (
            len(validation)
        ),

        "train_ratio_actual": (
            len(train)
            / len(final_records)
        ),

        "validation_ratio_actual": (
            len(validation)
            / len(final_records)
        ),

        "train_label_counts": (
            label_counts(
                train
            )
        ),

        "validation_label_counts": (
            label_counts(
                validation
            )
        ),

        "train_source_counts": (
            source_counts(
                train
            )
        ),

        "validation_source_counts": (
            source_counts(
                validation
            )
        ),

        "train_attack_type_counts": dict(
            sorted(
                train_attack_types.items()
            )
        ),

        "validation_attack_type_counts": dict(
            sorted(
                validation_attack_types.items()
            )
        ),

        "train_leakage_groups": (
            len(train_groups)
        ),

        "validation_leakage_groups": (
            len(validation_groups)
        ),

        "group_overlap": (
            len(
                train_groups
                & validation_groups
            )
        ),

        "exact_normalized_text_overlap": 0,

        "split_seed": (
            SPLIT_SEED
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

    print("=" * 80)
    print(
        "FINAL TRAINING CORPUS COMPLETE"
    )
    print("=" * 80)

    print()
    print("INPUT")

    print(
        "Raw attacks             :",
        len(attacks_raw),
    )

    print(
        "Raw benign              :",
        len(benign),
    )

    print()
    print("GLOBAL ATTACK DEDUP")

    print(
        "Duplicate groups        :",
        duplicate_groups,
    )

    print(
        "Duplicate rows removed  :",
        duplicate_rows_removed,
    )

    print(
        "Unique attacks          :",
        len(attacks),
    )

    print()
    print("FINAL CORPUS")

    print(
        "Attack                  :",
        len(attacks),
    )

    print(
        "Benign                  :",
        len(benign),
    )

    print(
        "Total                   :",
        len(final_records),
    )

    print()
    print("SPLIT")

    print(
        "Train                   :",
        len(train),
        label_counts(train),
    )

    print(
        "Validation              :",
        len(validation),
        label_counts(validation),
    )

    print()

    print(
        "Actual validation ratio :",
        f"{len(validation)/len(final_records):.4f}",
    )

    print()
    print("LEAKAGE CHECK")

    print(
        "Group overlap           :",
        len(
            train_groups
            & validation_groups
        ),
    )

    print(
        "Exact text overlap      :",
        0,
    )

    print()
    print(
        "Train output:",
        TRAIN_OUTPUT,
    )

    print(
        "Validation output:",
        VALIDATION_OUTPUT,
    )

    print(
        "Full corpus:",
        ALL_OUTPUT,
    )

    print(
        "Report:",
        REPORT_FILE,
    )


if __name__ == "__main__":
    main()
