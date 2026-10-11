from __future__ import annotations

import bz2
import json
import statistics
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tensor_trust"
    / "source"
    / "raw-data"
)

FILES = {
    "v1": ROOT / "v1" / "raw_dump_defenses.jsonl.bz2",
    "v2": ROOT / "v2" / "raw_dump_defenses.jsonl.bz2",
}


def normalize(text: str) -> str:
    return " ".join(
        str(text)
        .lower()
        .split()
    )


def percentile(values, p):
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
        return float(values[lower])

    weight = k - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def load(path: Path):

    rows = []

    with bz2.open(
        path,
        "rt",
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


def detect_text_fields(row):

    candidates = []

    for key, value in row.items():

        if isinstance(value, str):

            if any(
                term in key.lower()
                for term in [
                    "defense",
                    "prompt",
                    "instruction",
                    "opening",
                    "closing",
                ]
            ):
                candidates.append(
                    (
                        key,
                        value,
                    )
                )

    return candidates


def main():

    for version, path in FILES.items():

        print()
        print("=" * 90)
        print(
            f"Tensor Trust defenses — {version}"
        )
        print("=" * 90)

        rows = load(path)

        print(
            "Rows:",
            len(rows)
        )

        print()

        if not rows:
            continue

        print(
            "FIELDS:",
            list(rows[0].keys())
        )

        print()

        print("FIRST RECORD:")

        print(
            json.dumps(
                rows[0],
                ensure_ascii=False,
                indent=2,
            )[:5000]
        )

        # -----------------------------------
        # Inspect candidate textual fields
        # -----------------------------------

        field_counter = Counter()

        for row in rows:

            for key, value in detect_text_fields(
                row
            ):

                if value.strip():
                    field_counter[key] += 1

        print()
        print(
            "POSSIBLE DEFENSE TEXT FIELDS:"
        )

        for field, count in (
            field_counter.most_common()
        ):
            print(
                f"{field:35} {count}"
            )

        # -----------------------------------
        # Analyze opening + closing defense
        # if present.
        # -----------------------------------

        texts = []

        for row in rows:

            opening = str(
                row.get(
                    "opening_defense",
                    "",
                )
                or ""
            ).strip()

            closing = str(
                row.get(
                    "closing_defense",
                    "",
                )
                or ""
            ).strip()

            combined = "\n".join(
                part
                for part in [
                    opening,
                    closing,
                ]
                if part
            ).strip()

            if combined:
                texts.append(
                    combined
                )

        print()
        print(
            "Combined non-empty defenses:",
            len(texts)
        )

        if not texts:
            continue

        normalized = [
            normalize(text)
            for text in texts
        ]

        unique = len(
            set(normalized)
        )

        print(
            "Unique normalized defenses:",
            unique
        )

        print(
            "Duplicate rows estimate:",
            len(texts) - unique
        )

        char_lengths = [
            len(text)
            for text in texts
        ]

        word_lengths = [
            len(text.split())
            for text in texts
        ]

        print()
        print("Character length")

        print(
            "  min    :",
            min(char_lengths)
        )

        print(
            "  median :",
            f"{statistics.median(char_lengths):.2f}"
        )

        print(
            "  mean   :",
            f"{statistics.mean(char_lengths):.2f}"
        )

        print(
            "  p95    :",
            f"{percentile(char_lengths, .95):.2f}"
        )

        print(
            "  p99    :",
            f"{percentile(char_lengths, .99):.2f}"
        )

        print(
            "  max    :",
            max(char_lengths)
        )

        print()
        print("Word length")

        print(
            "  median :",
            f"{statistics.median(word_lengths):.2f}"
        )

        print(
            "  p95    :",
            f"{percentile(word_lengths, .95):.2f}"
        )

        print(
            "  max    :",
            max(word_lengths)
        )

        # -----------------------------------
        # Look for attack-looking language
        # -----------------------------------

        indicators = [
            "ignore previous",
            "ignore all previous",
            "secret",
            "password",
            "access code",
            "system prompt",
            "do not reveal",
            "never reveal",
            "instructions",
        ]

        print()
        print(
            "ATTACK-LIKE TERMS INSIDE DEFENSES"
        )

        lowered = [
            text.lower()
            for text in texts
        ]

        for term in indicators:

            count = sum(
                term in text
                for text in lowered
            )

            print(
                f"{term:25} {count}"
            )

        print()
        print(
            "FIRST 5 DEFENSE EXAMPLES"
        )

        for i, text in enumerate(
            texts[:5],
            start=1,
        ):

            print()
            print(
                f"{i}.",
                text[:700]
                .replace(
                    "\n",
                    "\\n"
                )
            )


if __name__ == "__main__":
    main()
