from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


FIELDS = [
    "source_type",
    "trust_level",
    "directness",
    "encoding_type",
]


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
        return "<NONE>"

    value = str(value).strip()

    return (
        value
        if value
        else "<EMPTY>"
    )


def main():

    for split, path in FILES.items():

        rows = read_jsonl(path)

        print()
        print("=" * 90)
        print(split.upper())
        print("=" * 90)

        print(
            "Rows:",
            len(rows)
        )

        # --------------------------------------------------------
        # Basic scalar fields
        # --------------------------------------------------------

        for field in FIELDS:

            counts = defaultdict(
                Counter
            )

            for row in rows:

                label = (
                    "attack"
                    if row["is_attack"]
                    else "benign"
                )

                value = safe(
                    row.get(field)
                )

                counts[value][
                    label
                ] += 1

            print()
            print("-" * 90)
            print(
                "FIELD:",
                field
            )
            print("-" * 90)

            for value in sorted(
                counts
            ):

                attack = counts[
                    value
                ]["attack"]

                benign = counts[
                    value
                ]["benign"]

                total = (
                    attack
                    + benign
                )

                attack_rate = (
                    attack / total
                    if total
                    else 0
                )

                print(
                    f"{value:30} "
                    f"attack={attack:6d} "
                    f"benign={benign:6d} "
                    f"attack_rate={attack_rate:.4f}"
                )

        # --------------------------------------------------------
        # Presence of structured fields
        # --------------------------------------------------------

        presence_fields = [
            "user_instruction",
            "external_content",
            "attack_payload",
            "poisoned_content",
            "attack_position",
        ]

        print()
        print("=" * 90)
        print("FIELD PRESENCE BY LABEL")
        print("=" * 90)

        for field in presence_fields:

            stats = Counter()

            for row in rows:

                label = (
                    "attack"
                    if row["is_attack"]
                    else "benign"
                )

                value = row.get(
                    field
                )

                present = bool(
                    str(value).strip()
                ) if value is not None else False

                key = (
                    label,
                    "present"
                    if present
                    else "missing",
                )

                stats[key] += 1

            print()
            print(field)

            print(
                "  attack present :",
                stats[
                    (
                        "attack",
                        "present",
                    )
                ]
            )

            print(
                "  attack missing :",
                stats[
                    (
                        "attack",
                        "missing",
                    )
                ]
            )

            print(
                "  benign present :",
                stats[
                    (
                        "benign",
                        "present",
                    )
                ]
            )

            print(
                "  benign missing :",
                stats[
                    (
                        "benign",
                        "missing",
                    )
                ]
            )

        # --------------------------------------------------------
        # BIPIA-specific audit
        # --------------------------------------------------------

        print()
        print("=" * 90)
        print("BIPIA ONLY")
        print("=" * 90)

        bipia_rows = [
            row
            for row in rows
            if row[
                "metadata"
            ].get(
                "training_source_group"
            )
            in {
                "bipia_indirect",
                "bipia_benign",
            }
        ]

        print(
            "BIPIA rows:",
            len(bipia_rows)
        )

        for field in [
            "source_type",
            "trust_level",
            "directness",
        ]:

            print()
            print(
                "BIPIA FIELD:",
                field
            )

            counts = defaultdict(
                Counter
            )

            for row in bipia_rows:

                label = (
                    "attack"
                    if row[
                        "is_attack"
                    ]
                    else "benign"
                )

                value = safe(
                    row.get(field)
                )

                counts[
                    value
                ][label] += 1

            for value in sorted(
                counts
            ):

                print(
                    f"  {value:25} "
                    f"attack={counts[value]['attack']:5d} "
                    f"benign={counts[value]['benign']:5d}"
                )

        print()
        print("BIPIA structured-field presence")

        for field in [
            "user_instruction",
            "external_content",
        ]:

            counts = Counter()

            for row in bipia_rows:

                label = (
                    "attack"
                    if row[
                        "is_attack"
                    ]
                    else "benign"
                )

                value = row.get(
                    field
                )

                present = bool(
                    str(value).strip()
                ) if value is not None else False

                counts[
                    (
                        label,
                        present,
                    )
                ] += 1

            print()
            print(field)

            print(
                "  attack present:",
                counts[
                    (
                        "attack",
                        True,
                    )
                ]
            )

            print(
                "  attack missing:",
                counts[
                    (
                        "attack",
                        False,
                    )
                ]
            )

            print(
                "  benign present:",
                counts[
                    (
                        "benign",
                        True,
                    )
                ]
            )

            print(
                "  benign missing:",
                counts[
                    (
                        "benign",
                        False,
                    )
                ]
            )


if __name__ == "__main__":
    main()
