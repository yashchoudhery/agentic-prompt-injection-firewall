from __future__ import annotations

import hashlib
import json
import random
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
    / "interim"
    / "bipia"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "bipia_conversion_report.json"
)


POSITIONS = [
    "start",
    "middle",
    "end",
]

TRAIN_CONTEXTS_PER_ATTACK = 3

RANDOM_SEED = 2026


def read_json(
    path: Path,
) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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


def flatten_attacks(
    data: dict[str, list[str]],
) -> list[dict[str, Any]]:

    output = []

    for category, payloads in data.items():

        for index, payload in enumerate(
            payloads
        ):

            output.append(
                {
                    "category": category,
                    "payload_index": index,
                    "payload": str(payload),
                }
            )

    return output


def stringify_context(
    source_name: str,
    record: dict[str, Any],
) -> str:

    context = record.get(
        "context",
        "",
    )

    if isinstance(context, list):
        return "\n".join(
            str(item)
            for item in context
        )

    return str(context)


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

    # CodeQA does not provide a simple
    # question field in the raw context.
    # Do not invent one.
    return None


def insert_attack(
    context: str,
    attack: str,
    position: str,
    seed: int,
) -> str:

    if position == "start":
        return (
            attack
            + "\n"
            + context
        )

    if position == "end":
        return (
            context
            + "\n"
            + attack
        )

    if position != "middle":
        raise ValueError(
            f"Unknown position: {position}"
        )

    lines = context.splitlines()

    if len(lines) <= 1:

        midpoint = max(
            0,
            len(context) // 2,
        )

        return (
            context[:midpoint]
            + "\n"
            + attack
            + "\n"
            + context[midpoint:]
        )

    rng = random.Random(
        seed
    )

    insertion_index = rng.randint(
        1,
        len(lines) - 1,
    )

    poisoned_lines = (
        lines[:insertion_index]
        + [attack]
        + lines[insertion_index:]
    )

    return "\n".join(
        poisoned_lines
    )


def stable_seed(
    *values: Any,
) -> int:

    raw = "|".join(
        str(value)
        for value in values
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


def make_sample_id(
    split: str,
    source_name: str,
    attack_category: str,
    attack_index: int,
    context_index: int,
    position: str,
) -> str:

    raw = (
        f"{split}|"
        f"{source_name}|"
        f"{attack_category}|"
        f"{attack_index}|"
        f"{context_index}|"
        f"{position}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"bipia_{split}_{digest}"
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


def build_record(
    *,
    split: str,
    source_name: str,
    context_record: dict[str, Any],
    context_index: int,
    attack_record: dict[str, Any],
    attack_index: int,
    position: str,
) -> dict[str, Any]:

    external_content = (
        stringify_context(
            source_name,
            context_record,
        )
    )

    attack_payload = (
        attack_record[
            "payload"
        ]
    )

    seed = stable_seed(
        split,
        source_name,
        attack_record[
            "category"
        ],
        attack_index,
        context_index,
        position,
    )

    poisoned_content = (
        insert_attack(
            external_content,
            attack_payload,
            position,
            seed,
        )
    )

    user_instruction = (
        get_user_instruction(
            source_name,
            context_record,
        )
    )

    attack_types = [
        "indirect_prompt_injection",
        "instruction_override",
    ]

    sample_id = make_sample_id(
        split,
        source_name,
        attack_record[
            "category"
        ],
        attack_index,
        context_index,
        position,
    )

    metadata = {
        "bipia_attack_category": (
            attack_record[
                "category"
            ]
        ),

        "bipia_attack_payload_index": (
            attack_record[
                "payload_index"
            ]
        ),

        "bipia_context_index": (
            context_index
        ),

        "bipia_source_task": (
            source_name
        ),

        "official_bipia_split": (
            split
        ),

        "ideal_answer": (
            context_record.get(
                "ideal"
            )
        ),
    }

    # Preserve code-specific source fields.
    if source_name == "code":

        metadata[
            "original_error"
        ] = context_record.get(
            "error"
        )

        metadata[
            "original_code"
        ] = context_record.get(
            "code"
        )

        metadata[
            "context_url"
        ] = context_record.get(
            "context_url"
        )

    return {
        "sample_id": sample_id,

        # Primary detector input for this
        # indirect-injection example.
        "text": poisoned_content,

        "normalized_text": None,

        "is_attack": True,

        "attack_types": (
            attack_types
        ),

        "source_type": (
            source_type_for(
                source_name
            )
        ),

        "trust_level": "untrusted",

        "directness": "indirect",

        # Payload itself is plain text.
        # Output-format instructions such as
        # Base64 do NOT imply encoded input.
        "encoding_type": "plain",

        "language": "en",

        "malicious_spans": [
            attack_payload
        ],

        "requested_actions": [],

        "target_data_types": [],

        "severity": "high",

        "dataset_source": "bipia",

        "original_label": (
            attack_record[
                "category"
            ]
        ),

        "split": (
            "train"
            if split == "train"
            else "external_test"
        ),

        "metadata": metadata,

        # Schema v2 provenance fields.
        "user_instruction": (
            user_instruction
        ),

        "external_content": (
            external_content
        ),

        "attack_payload": (
            attack_payload
        ),

        "poisoned_content": (
            poisoned_content
        ),

        "attack_position": (
            position
        ),

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

            "attack_origin": (
                "external_content"
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


def choose_context_indexes(
    *,
    source_name: str,
    attack_index: int,
    context_count: int,
    count: int,
) -> list[int]:

    if count >= context_count:
        return list(
            range(context_count)
        )

    seed = stable_seed(
        RANDOM_SEED,
        source_name,
        attack_index,
    )

    rng = random.Random(
        seed
    )

    return sorted(
        rng.sample(
            range(context_count),
            count,
        )
    )


def generate_balanced_train(
    text_attacks: list[
        dict[str, Any]
    ],
    code_attacks: list[
        dict[str, Any]
    ],
    contexts: dict[
        str,
        list[dict[str, Any]]
    ],
) -> list[dict[str, Any]]:

    records = []

    # ----------------------------------
    # Text attacks -> Email + Table
    # ----------------------------------

    for attack_index, attack in enumerate(
        text_attacks
    ):

        for source_name in [
            "email",
            "table",
        ]:

            source_contexts = (
                contexts[
                    source_name
                ]
            )

            indexes = (
                choose_context_indexes(
                    source_name=source_name,
                    attack_index=attack_index,
                    context_count=len(
                        source_contexts
                    ),
                    count=(
                        TRAIN_CONTEXTS_PER_ATTACK
                    ),
                )
            )

            for context_index in indexes:

                context_record = (
                    source_contexts[
                        context_index
                    ]
                )

                for position in POSITIONS:

                    records.append(
                        build_record(
                            split="train",
                            source_name=source_name,
                            context_record=(
                                context_record
                            ),
                            context_index=(
                                context_index
                            ),
                            attack_record=attack,
                            attack_index=(
                                attack_index
                            ),
                            position=position,
                        )
                    )

    # ----------------------------------
    # Code attacks -> CodeQA
    # ----------------------------------

    for attack_index, attack in enumerate(
        code_attacks
    ):

        source_contexts = (
            contexts["code"]
        )

        indexes = (
            choose_context_indexes(
                source_name="code",
                attack_index=attack_index,
                context_count=len(
                    source_contexts
                ),
                count=(
                    TRAIN_CONTEXTS_PER_ATTACK
                ),
            )
        )

        for context_index in indexes:

            context_record = (
                source_contexts[
                    context_index
                ]
            )

            for position in POSITIONS:

                records.append(
                    build_record(
                        split="train",
                        source_name="code",
                        context_record=(
                            context_record
                        ),
                        context_index=(
                            context_index
                        ),
                        attack_record=attack,
                        attack_index=(
                            attack_index
                        ),
                        position=position,
                    )
                )

    return records


def generate_full_test(
    text_attacks: list[
        dict[str, Any]
    ],
    code_attacks: list[
        dict[str, Any]
    ],
    contexts: dict[
        str,
        list[dict[str, Any]]
    ],
) -> list[dict[str, Any]]:

    records = []

    # ----------------------------------
    # Text attacks -> Email + Table
    # ----------------------------------

    for attack_index, attack in enumerate(
        text_attacks
    ):

        for source_name in [
            "email",
            "table",
        ]:

            for context_index, context_record in enumerate(
                contexts[source_name]
            ):

                for position in POSITIONS:

                    records.append(
                        build_record(
                            split="test",
                            source_name=source_name,
                            context_record=(
                                context_record
                            ),
                            context_index=(
                                context_index
                            ),
                            attack_record=attack,
                            attack_index=(
                                attack_index
                            ),
                            position=position,
                        )
                    )

    # ----------------------------------
    # Code attacks -> CodeQA
    # ----------------------------------

    for attack_index, attack in enumerate(
        code_attacks
    ):

        for context_index, context_record in enumerate(
            contexts["code"]
        ):

            for position in POSITIONS:

                records.append(
                    build_record(
                        split="test",
                        source_name="code",
                        context_record=(
                            context_record
                        ),
                        context_index=(
                            context_index
                        ),
                        attack_record=attack,
                        attack_index=(
                            attack_index
                        ),
                        position=position,
                    )
                )

    return records


def count_source_types(
    records: list[dict[str, Any]],
) -> dict[str, int]:

    counts = Counter(
        record["source_type"]
        for record in records
    )

    return dict(
        sorted(
            counts.items()
        )
    )


def count_positions(
    records: list[dict[str, Any]],
) -> dict[str, int]:

    counts = Counter(
        record["attack_position"]
        for record in records
    )

    return dict(
        sorted(
            counts.items()
        )
    )


def main() -> None:

    train_text_attacks = flatten_attacks(
        read_json(
            BIPIA_ROOT
            / "text_attack_train.json"
        )
    )

    test_text_attacks = flatten_attacks(
        read_json(
            BIPIA_ROOT
            / "text_attack_test.json"
        )
    )

    train_code_attacks = flatten_attacks(
        read_json(
            BIPIA_ROOT
            / "code_attack_train.json"
        )
    )

    test_code_attacks = flatten_attacks(
        read_json(
            BIPIA_ROOT
            / "code_attack_test.json"
        )
    )

    train_contexts = {
        "email": read_jsonl(
            BIPIA_ROOT
            / "email"
            / "train.jsonl"
        ),

        "table": read_jsonl(
            BIPIA_ROOT
            / "table"
            / "train.jsonl"
        ),

        "code": read_jsonl(
            BIPIA_ROOT
            / "code"
            / "train.jsonl"
        ),
    }

    test_contexts = {
        "email": read_jsonl(
            BIPIA_ROOT
            / "email"
            / "test.jsonl"
        ),

        "table": read_jsonl(
            BIPIA_ROOT
            / "table"
            / "test.jsonl"
        ),

        "code": read_jsonl(
            BIPIA_ROOT
            / "code"
            / "test.jsonl"
        ),
    }

    train_records = (
        generate_balanced_train(
            train_text_attacks,
            train_code_attacks,
            train_contexts,
        )
    )

    test_records = (
        generate_full_test(
            test_text_attacks,
            test_code_attacks,
            test_contexts,
        )
    )

    train_output = (
        OUTPUT_DIR
        / "bipia_train_balanced.jsonl"
    )

    test_output = (
        OUTPUT_DIR
        / "bipia_external_test.jsonl"
    )

    write_jsonl(
        train_records,
        train_output,
    )

    write_jsonl(
        test_records,
        test_output,
    )

    report = {
        "attack_payloads": {
            "text_train": len(
                train_text_attacks
            ),

            "text_test": len(
                test_text_attacks
            ),

            "code_train": len(
                train_code_attacks
            ),

            "code_test": len(
                test_code_attacks
            ),
        },

        "context_counts": {
            "train": {
                key: len(value)
                for key, value
                in train_contexts.items()
            },

            "test": {
                key: len(value)
                for key, value
                in test_contexts.items()
            },
        },

        "balanced_train_records": (
            len(train_records)
        ),

        "external_test_records": (
            len(test_records)
        ),

        "train_source_type_counts": (
            count_source_types(
                train_records
            )
        ),

        "test_source_type_counts": (
            count_source_types(
                test_records
            )
        ),

        "train_position_counts": (
            count_positions(
                train_records
            )
        ),

        "test_position_counts": (
            count_positions(
                test_records
            )
        ),

        "attack_types": [
            "indirect_prompt_injection",
            "instruction_override",
        ],

        "training_sampling": {
            "contexts_per_attack_per_source": (
                TRAIN_CONTEXTS_PER_ATTACK
            ),

            "positions": POSITIONS,

            "random_seed": (
                RANDOM_SEED
            ),
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

        f.write("\n")

    print("=" * 72)
    print("BIPIA conversion complete")
    print("=" * 72)

    print(
        "Text train attacks      :",
        len(train_text_attacks),
    )

    print(
        "Text test attacks       :",
        len(test_text_attacks),
    )

    print(
        "Code train attacks      :",
        len(train_code_attacks),
    )

    print(
        "Code test attacks       :",
        len(test_code_attacks),
    )

    print()

    print(
        "Balanced train records  :",
        len(train_records),
    )

    print(
        "External test records   :",
        len(test_records),
    )

    print()

    print(
        "Train source counts     :",
        count_source_types(
            train_records
        ),
    )

    print(
        "Test source counts      :",
        count_source_types(
            test_records
        ),
    )

    print()

    print(
        "Train position counts   :",
        count_positions(
            train_records
        ),
    )

    print(
        "Test position counts    :",
        count_positions(
            test_records
        ),
    )

    print()

    print(
        "Train output:",
        train_output,
    )

    print(
        "Test output:",
        test_output,
    )

    print(
        "Report:",
        REPORT_FILE,
    )


if __name__ == "__main__":
    main()
