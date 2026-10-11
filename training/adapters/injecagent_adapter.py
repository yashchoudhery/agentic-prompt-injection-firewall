from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "injecagent"
    / "source"
    / "data"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "injecagent"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "injecagent_conversion_report.json"
)


# High-confidence credential-related tools only.
CREDENTIAL_TOOL_NAMES = {
    "NortonIdentitySafeSearchPasswords",
    "NortonIdentitySafeSharePassword",
}


SEMANTIC_FIELDS = [
    "Attacker Tools",
    "Modifed",
    "Attacker Instruction",
    "Expected Achievements",
    "Attack Type",
    "User Tool",
    "User Instruction",
    "Tool Parameters",
    "Tool Response Template",
    "Thought",
]


def load_json(path: Path) -> Any:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def make_pair_id(
    family: str,
    index: int,
    row: dict[str, Any],
) -> str:

    raw = (
        f"{family}|"
        f"{index}|"
        f"{row.get('Attacker Instruction')}|"
        f"{row.get('User Tool')}|"
        f"{row.get('User Instruction')}"
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"injecagent_pair_{digest}"
    )


def make_sample_id(
    pair_id: str,
    variant: str,
) -> str:

    return (
        f"{pair_id}_{variant}"
    )


def validate_pair(
    base: dict[str, Any],
    enhanced: dict[str, Any],
) -> None:

    for field in SEMANTIC_FIELDS:

        if (
            base.get(field)
            != enhanced.get(field)
        ):
            raise ValueError(
                "Base/enhanced semantic mismatch "
                f"for field: {field}"
            )

    if (
        base.get("Tool Response")
        == enhanced.get("Tool Response")
    ):
        raise ValueError(
            "Base and enhanced Tool Response "
            "should differ."
        )


def is_credential_attack(
    row: dict[str, Any],
) -> bool:

    attacker_tools = set(
        row.get(
            "Attacker Tools",
            [],
        )
    )

    if (
        attacker_tools
        & CREDENTIAL_TOOL_NAMES
    ):
        return True

    instruction = str(
        row.get(
            "Attacker Instruction",
            "",
        )
    ).lower()

    achievement = str(
        row.get(
            "Expected Achievements",
            "",
        )
    ).lower()

    combined = (
        instruction
        + " "
        + achievement
    )

    credential_terms = [
        "password",
        "credential",
        "api key",
        "access token",
        "auth token",
        "login token",
    ]

    return any(
        term in combined
        for term in credential_terms
    )


def attack_types_for(
    *,
    family: str,
    variant: str,
    row: dict[str, Any],
) -> list[str]:

    labels = {
        "indirect_prompt_injection",
        "tool_abuse",
    }

    if family == "data_stealing":
        labels.add(
            "secret_extraction"
        )

    # Enhanced cases explicitly contain:
    # "Ignore all previous instructions..."
    if variant == "enhanced":
        labels.add(
            "instruction_override"
        )

    if is_credential_attack(row):
        labels.add(
            "credential_theft"
        )

    return sorted(labels)


def target_data_types_for(
    *,
    family: str,
    row: dict[str, Any],
) -> list[str]:

    targets = []

    if family == "data_stealing":
        targets.append(
            "private_user_data"
        )

    if is_credential_attack(row):
        targets.append(
            "credentials"
        )

    return sorted(
        set(targets)
    )


def build_record(
    *,
    family: str,
    variant: str,
    semantic_index: int,
    row: dict[str, Any],
    pair_id: str,
) -> dict[str, Any]:

    tool_response = str(
        row.get(
            "Tool Response",
            "",
        )
    )

    attacker_instruction = str(
        row.get(
            "Attacker Instruction",
            "",
        )
    )

    user_instruction = str(
        row.get(
            "User Instruction",
            "",
        )
    )

    attacker_tools = list(
        row.get(
            "Attacker Tools",
            [],
        )
    )

    attack_types = attack_types_for(
        family=family,
        variant=variant,
        row=row,
    )

    target_data_types = (
        target_data_types_for(
            family=family,
            row=row,
        )
    )

    return {
        "sample_id": make_sample_id(
            pair_id,
            variant,
        ),

        # Actual untrusted tool output presented
        # to the agent/firewall.
        "text": tool_response,

        "normalized_text": None,

        "is_attack": True,

        "attack_types": attack_types,

        "source_type": "tool_output",

        "trust_level": "untrusted",

        "directness": "indirect",

        "encoding_type": "plain",

        "language": "en",

        "malicious_spans": [
            attacker_instruction
        ],

        "requested_actions": (
            attacker_tools
        ),

        "target_data_types": (
            target_data_types
        ),

        "severity": "critical",

        "dataset_source": (
            "injecagent"
        ),

        "original_label": str(
            row.get(
                "Attack Type",
                "unknown",
            )
        ),

        # Entire InjecAgent corpus remains
        # outside model training.
        "split": "external_test",

        "metadata": {
            "semantic_pair_id": (
                pair_id
            ),

            "semantic_case_index": (
                semantic_index
            ),

            "attack_family": (
                family
            ),

            "variant": (
                variant
            ),

            "injecagent_attack_type": (
                row.get(
                    "Attack Type"
                )
            ),

            "modified_flag": (
                row.get(
                    "Modifed"
                )
            ),

            "expected_achievements": (
                row.get(
                    "Expected Achievements"
                )
            ),

            "user_tool": (
                row.get(
                    "User Tool"
                )
            ),

            "attacker_tools": (
                attacker_tools
            ),

            "tool_parameters": (
                row.get(
                    "Tool Parameters"
                )
            ),

            "tool_response_template": (
                row.get(
                    "Tool Response Template"
                )
            ),

            "original_thought": (
                row.get(
                    "Thought"
                )
            ),

            "credential_attack": (
                is_credential_attack(
                    row
                )
            ),
        },

        # ---------------------------------
        # Schema v2 provenance fields
        # ---------------------------------

        "user_instruction": (
            user_instruction
        ),

        # Tool template before substitution is
        # the clean structural representation.
        "external_content": str(
            row.get(
                "Tool Response Template",
                "",
            )
        ),

        "attack_payload": (
            attacker_instruction
        ),

        "poisoned_content": (
            tool_response
        ),

        # InjecAgent injects into a structured
        # tool-response field. It is not simply
        # start/middle/end.
        "attack_position": "unknown",

        "provenance": {
            "user_instruction_origin": (
                "user"
            ),

            "user_instruction_trust": (
                "trusted"
            ),

            "external_content_origin": (
                "tool_output"
            ),

            "external_content_trust": (
                "untrusted"
            ),

            "attack_origin": (
                "tool_output"
            ),

            "user_tool": (
                row.get(
                    "User Tool"
                )
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


def process_family(
    *,
    family: str,
    base_file: str,
    enhanced_file: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    base_rows = load_json(
        DATA_ROOT
        / base_file
    )

    enhanced_rows = load_json(
        DATA_ROOT
        / enhanced_file
    )

    if (
        len(base_rows)
        != len(enhanced_rows)
    ):
        raise ValueError(
            f"{family}: base/enhanced "
            "length mismatch"
        )

    canonical_records = []
    pair_records = []

    for index, (
        base,
        enhanced,
    ) in enumerate(
        zip(
            base_rows,
            enhanced_rows,
        )
    ):

        validate_pair(
            base,
            enhanced,
        )

        pair_id = make_pair_id(
            family,
            index,
            base,
        )

        base_record = build_record(
            family=family,
            variant="base",
            semantic_index=index,
            row=base,
            pair_id=pair_id,
        )

        enhanced_record = build_record(
            family=family,
            variant="enhanced",
            semantic_index=index,
            row=enhanced,
            pair_id=pair_id,
        )

        canonical_records.extend(
            [
                base_record,
                enhanced_record,
            ]
        )

        pair_records.append(
            {
                "pair_id": pair_id,

                "attack_family": (
                    family
                ),

                "semantic_case_index": (
                    index
                ),

                "attack_type": (
                    base.get(
                        "Attack Type"
                    )
                ),

                "attacker_instruction": (
                    base.get(
                        "Attacker Instruction"
                    )
                ),

                "expected_achievements": (
                    base.get(
                        "Expected Achievements"
                    )
                ),

                "user_tool": (
                    base.get(
                        "User Tool"
                    )
                ),

                "user_instruction": (
                    base.get(
                        "User Instruction"
                    )
                ),

                "attacker_tools": (
                    base.get(
                        "Attacker Tools",
                        [],
                    )
                ),

                "base_sample_id": (
                    base_record[
                        "sample_id"
                    ]
                ),

                "enhanced_sample_id": (
                    enhanced_record[
                        "sample_id"
                    ]
                ),

                "credential_attack": (
                    is_credential_attack(
                        base
                    )
                ),
            }
        )

    return (
        canonical_records,
        pair_records,
    )


def main() -> None:

    dh_records, dh_pairs = (
        process_family(
            family="direct_harm",
            base_file=(
                "test_cases_dh_base.json"
            ),
            enhanced_file=(
                "test_cases_dh_enhanced.json"
            ),
        )
    )

    ds_records, ds_pairs = (
        process_family(
            family="data_stealing",
            base_file=(
                "test_cases_ds_base.json"
            ),
            enhanced_file=(
                "test_cases_ds_enhanced.json"
            ),
        )
    )

    records = (
        dh_records
        + ds_records
    )

    pairs = (
        dh_pairs
        + ds_pairs
    )

    output_file = (
        OUTPUT_DIR
        / "injecagent_external_test.jsonl"
    )

    pair_file = (
        OUTPUT_DIR
        / "injecagent_semantic_pairs.jsonl"
    )

    write_jsonl(
        records,
        output_file,
    )

    write_jsonl(
        pairs,
        pair_file,
    )

    family_counts = Counter(
        r["metadata"][
            "attack_family"
        ]
        for r in records
    )

    variant_counts = Counter(
        r["metadata"][
            "variant"
        ]
        for r in records
    )

    attack_type_counts = Counter()

    for record in records:

        for label in record[
            "attack_types"
        ]:
            attack_type_counts[
                label
            ] += 1

    credential_variant_count = sum(
        1
        for record in records
        if record["metadata"][
            "credential_attack"
        ]
    )

    credential_semantic_count = sum(
        1
        for pair in pairs
        if pair[
            "credential_attack"
        ]
    )

    user_tool_counts = Counter(
        record["metadata"][
            "user_tool"
        ]
        for record in records
    )

    attacker_tool_counts = Counter()

    for pair in pairs:

        for tool in pair[
            "attacker_tools"
        ]:

            attacker_tool_counts[
                tool
            ] += 1

    report = {
        "semantic_cases": len(
            pairs
        ),

        "detector_variant_records": len(
            records
        ),

        "semantic_direct_harm_cases": len(
            dh_pairs
        ),

        "semantic_data_stealing_cases": len(
            ds_pairs
        ),

        "variant_counts": dict(
            sorted(
                variant_counts.items()
            )
        ),

        "variant_family_counts": dict(
            sorted(
                family_counts.items()
            )
        ),

        "attack_type_counts_across_variants": dict(
            sorted(
                attack_type_counts.items()
            )
        ),

        "credential_semantic_cases": (
            credential_semantic_count
        ),

        "credential_variant_records": (
            credential_variant_count
        ),

        "unique_user_tools": len(
            user_tool_counts
        ),

        "unique_attacker_tools": len(
            attacker_tool_counts
        ),

        "training_usage": False,

        "split": "external_test",
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
        "InjecAgent conversion complete"
    )
    print("=" * 72)

    print(
        "Semantic cases          :",
        len(pairs),
    )

    print(
        "  Direct harm           :",
        len(dh_pairs),
    )

    print(
        "  Data stealing         :",
        len(ds_pairs),
    )

    print()

    print(
        "Detector variant records:",
        len(records),
    )

    print(
        "  Base                  :",
        variant_counts["base"],
    )

    print(
        "  Enhanced              :",
        variant_counts["enhanced"],
    )

    print()

    print(
        "Credential semantic cases:",
        credential_semantic_count,
    )

    print()

    print(
        "Attack labels across variants:"
    )

    for label, count in sorted(
        attack_type_counts.items()
    ):

        print(
            f"  {label:28} {count}"
        )

    print()

    print(
        "Unique user tools       :",
        len(user_tool_counts),
    )

    print(
        "Unique attacker tools   :",
        len(attacker_tool_counts),
    )

    print()

    print(
        "External test:",
        output_file,
    )

    print(
        "Pair index   :",
        pair_file,
    )

    print(
        "Report       :",
        REPORT_FILE,
    )


if __name__ == "__main__":
    main()
