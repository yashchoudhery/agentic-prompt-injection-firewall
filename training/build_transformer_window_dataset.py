from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformer_windows"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data_quality"
    / "transformer_window_dataset_report.json"
)


MAX_LENGTH = 512
HEADER_TOKEN_BUDGET = 96
STRIDE = 128

EXTERNAL_SOURCE_TYPES = {
    "email",
    "table",
    "source_code",
    "web",
    "pdf",
    "markdown",
    "html",
    "word",
    "api",
    "ocr",
}


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

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


def write_jsonl(
    rows,
    path,
):

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


def safe(value):

    if value is None:
        return ""

    return str(value).strip()


def is_external(
    row,
):

    source_type = safe(
        row.get(
            "source_type"
        )
    ).lower()

    return (
        source_type
        in EXTERNAL_SOURCE_TYPES
    )


def build_header(
    row,
    tokenizer,
):

    # ------------------------------------------------------------
    # Runtime-safe provenance representation.
    #
    # IMPORTANT:
    # Direct user-message samples DO NOT use the dataset's
    # user_instruction field because its missingness was correlated
    # with the label in our corpus.
    # ------------------------------------------------------------

    if not is_external(row):

        header = (
            "[INPUT_CHANNEL]\n"
            "USER_MESSAGE"
        )

    else:

        user_task = safe(
            row.get(
                "user_instruction"
            )
        )

        header = (
            "[INPUT_CHANNEL]\n"
            "EXTERNAL_CONTENT\n\n"
            "[USER_TASK]\n"
            f"{user_task or '<UNKNOWN>'}"
        )

    header_ids = tokenizer(
        header,
        add_special_tokens=False,
        truncation=True,
        max_length=HEADER_TOKEN_BUDGET,
    )["input_ids"]

    return tokenizer.decode(
        header_ids,
        skip_special_tokens=True,
    )


def tokenize_windows(
    row,
    tokenizer,
):

    header = build_header(
        row,
        tokenizer,
    )

    content = safe(
        row.get(
            "text"
        )
    )

    encoded = tokenizer(
        header,
        content,
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
    )

    windows = []

    n = len(
        encoded[
            "input_ids"
        ]
    )

    for index in range(n):

        sequence_ids = (
            encoded.sequence_ids(
                index
            )
        )

        offsets = (
            encoded[
                "offset_mapping"
            ][index]
        )

        # Offsets belonging specifically to sequence B,
        # i.e. the raw content being inspected.
        content_offsets = [
            offset
            for offset, sequence_id
            in zip(
                offsets,
                sequence_ids,
            )
            if (
                sequence_id == 1
                and offset != (0, 0)
            )
        ]

        if content_offsets:

            content_char_start = min(
                offset[0]
                for offset
                in content_offsets
            )

            content_char_end = max(
                offset[1]
                for offset
                in content_offsets
            )

        else:

            content_char_start = 0
            content_char_end = 0

        item = {
            "input_ids": (
                encoded[
                    "input_ids"
                ][index]
            ),

            "attention_mask": (
                encoded[
                    "attention_mask"
                ][index]
            ),

            "_content_char_start": (
                content_char_start
            ),

            "_content_char_end": (
                content_char_end
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
            ][index]

        windows.append(
            item
        )

    return windows


def select_indices(
    n_windows,
):

    # Use every overflow window.
    #
    # Our corpus averages only ~1.05 windows/document,
    # so this adds little average cost while avoiding
    # position-based blind spots.
    return list(
        range(
            n_windows
        )
    )


def contains_subsequence(
    sequence,
    subsequence,
):

    if not subsequence:
        return False

    if (
        len(subsequence)
        > len(sequence)
    ):
        return False

    limit = (
        len(sequence)
        - len(subsequence)
        + 1
    )

    for i in range(limit):

        if (
            sequence[
                i:
                i + len(subsequence)
            ]
            == subsequence
        ):

            return True

    return False


def make_window_rows(
    split,
    documents,
    tokenizer,
):

    output = []

    document_counts = Counter()
    local_labels = Counter()
    source_counts = Counter()
    source_local_labels = (
        defaultdict(
            Counter
        )
    )

    bipia_attack_documents = 0
    bipia_attack_documents_covered = 0

    bipia_positive_windows = 0
    bipia_local_negative_windows = 0

    max_selected_windows = 0

    for document in documents:

        source_group = (
            document[
                "metadata"
            ].get(
                "training_source_group",
                "unknown",
            )
        )

        windows = tokenize_windows(
            document,
            tokenizer,
        )

        selected = select_indices(
            len(windows)
        )

        max_selected_windows = max(
            max_selected_windows,
            len(selected),
        )

        document_label = (
            1
            if document[
                "is_attack"
            ]
            else 0
        )

        payload_spans = []

        is_bipia_attack = (
            source_group
            == "bipia_indirect"
            and document_label == 1
        )

        if is_bipia_attack:

            payload = safe(
                document.get(
                    "attack_payload"
                )
            )

            if not payload:

                raise RuntimeError(
                    "Missing BIPIA attack payload: "
                    f"{document.get('sample_id')}"
                )

            content_text = safe(
                document.get(
                    "text"
                )
            )

            # Find every literal occurrence because a payload
            # could theoretically appear more than once.
            search_from = 0

            while True:

                payload_start = (
                    content_text.find(
                        payload,
                        search_from,
                    )
                )

                if payload_start < 0:
                    break

                payload_end = (
                    payload_start
                    + len(payload)
                )

                payload_spans.append(
                    (
                        payload_start,
                        payload_end,
                    )
                )

                search_from = (
                    payload_start
                    + 1
                )

            if not payload_spans:

                raise RuntimeError(
                    "BIPIA payload not found "
                    "literally in content: "
                    f"{document.get('sample_id')}"
                )

            bipia_attack_documents += 1

        positive_seen = False

        for selected_rank, window_index in enumerate(
            selected
        ):

            window = windows[
                window_index
            ]

            # ----------------------------------------------------
            # Local window label
            # ----------------------------------------------------

            if is_bipia_attack:

                window_start = (
                    window[
                        "_content_char_start"
                    ]
                )

                window_end = (
                    window[
                        "_content_char_end"
                    ]
                )

                # Character-span overlap is robust to
                # SentencePiece tokenization differences.
                has_payload = any(
                    (
                        window_start
                        < payload_end
                        and window_end
                        > payload_start
                    )
                    for (
                        payload_start,
                        payload_end,
                    )
                    in payload_spans
                )

                local_label = (
                    1
                    if has_payload
                    else 0
                )

                if has_payload:

                    positive_seen = True
                    bipia_positive_windows += 1

                else:

                    # Useful local negative:
                    # same attacked document, but this particular
                    # window contains no injected payload.
                    bipia_local_negative_windows += 1

            else:

                local_label = (
                    document_label
                )

            record = {
                "window_id": (
                    f"{document.get('sample_id')}"
                    f"__w{window_index}"
                ),

                "document_id": (
                    document.get(
                        "sample_id"
                    )
                ),

                "split": split,

                "source_group": (
                    source_group
                ),

                "source_type": (
                    document.get(
                        "source_type"
                    )
                ),

                "document_label": (
                    document_label
                ),

                "label": (
                    local_label
                ),

                "window_index": (
                    window_index
                ),

                "selected_rank": (
                    selected_rank
                ),

                "total_token_windows": (
                    len(windows)
                ),

                "selected_window_count": (
                    len(selected)
                ),

                "content_char_start": (
                    window[
                        "_content_char_start"
                    ]
                ),

                "content_char_end": (
                    window[
                        "_content_char_end"
                    ]
                ),

                "input_ids": (
                    window[
                        "input_ids"
                    ]
                ),

                "attention_mask": (
                    window[
                        "attention_mask"
                    ]
                ),
            }

            if (
                "token_type_ids"
                in window
            ):

                record[
                    "token_type_ids"
                ] = window[
                    "token_type_ids"
                ]

            output.append(
                record
            )

            local_labels[
                local_label
            ] += 1

            source_counts[
                source_group
            ] += 1

            source_local_labels[
                source_group
            ][
                local_label
            ] += 1

        if is_bipia_attack:

            if not positive_seen:

                raise RuntimeError(
                    "Selected windows missed "
                    "BIPIA payload: "
                    f"{document.get('sample_id')}"
                )

            bipia_attack_documents_covered += 1

        document_counts[
            document_label
        ] += 1

    report = {
        "documents": len(
            documents
        ),

        "document_labels": {
            "benign": (
                document_counts[0]
            ),

            "attack": (
                document_counts[1]
            ),
        },

        "windows": len(
            output
        ),

        "local_window_labels": {
            "benign": (
                local_labels[0]
            ),

            "attack": (
                local_labels[1]
            ),
        },

        "max_selected_windows_per_document": (
            max_selected_windows
        ),

        "window_source_counts": dict(
            sorted(
                source_counts.items()
            )
        ),

        "window_source_local_labels": {
            source: {
                "benign": counts[0],
                "attack": counts[1],
            }
            for source, counts
            in sorted(
                source_local_labels.items()
            )
        },

        "bipia_attack_documents": (
            bipia_attack_documents
        ),

        "bipia_attack_documents_covered": (
            bipia_attack_documents_covered
        ),

        "bipia_positive_windows": (
            bipia_positive_windows
        ),

        "bipia_local_negative_windows": (
            bipia_local_negative_windows
        ),
    }

    return (
        output,
        report,
    )


def main():

    print(
        "Loading tokenizer:",
        MODEL_NAME
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    final_report = {
        "model_name": MODEL_NAME,

        "max_length": MAX_LENGTH,

        "stride": STRIDE,

        "header_token_budget": (
            HEADER_TOKEN_BUDGET
        ),

        "representation": (
            "runtime_safe_provenance_v2"
        ),

        "policy": {
            "direct_user_message": (
                "channel marker + raw content; "
                "do not use dataset user_instruction"
            ),

            "external_content": (
                "external-content marker + "
                "runtime user task + raw content"
            ),

            "window_selection": (
                "all overflow windows"
            ),

            "bipia_window_labels": (
                "positive only when selected window "
                "contains gold payload; non-payload "
                "windows become local negatives"
            ),
        },
    }

    for split, path in (
        FILES.items()
    ):

        documents = read_jsonl(
            path
        )

        (
            windows,
            report,
        ) = make_window_rows(
            split,
            documents,
            tokenizer,
        )

        output_path = (
            OUTPUT_DIR
            / f"{split}_windows.jsonl"
        )

        write_jsonl(
            windows,
            output_path,
        )

        final_report[
            split
        ] = report

        print()
        print("=" * 80)
        print(
            split.upper()
        )
        print("=" * 80)

        print(
            "Documents:",
            report[
                "documents"
            ]
        )

        print(
            "Document labels:",
            report[
                "document_labels"
            ]
        )

        print(
            "Windows:",
            report[
                "windows"
            ]
        )

        print(
            "Local window labels:",
            report[
                "local_window_labels"
            ]
        )

        print(
            "Max selected windows/document:",
            report[
                "max_selected_windows_per_document"
            ]
        )

        print()

        print(
            "BIPIA attack documents:",
            report[
                "bipia_attack_documents"
            ]
        )

        print(
            "BIPIA covered:",
            report[
                "bipia_attack_documents_covered"
            ]
        )

        print(
            "BIPIA positive windows:",
            report[
                "bipia_positive_windows"
            ]
        )

        print(
            "BIPIA local negative windows:",
            report[
                "bipia_local_negative_windows"
            ]
        )

        print()
        print(
            "Source/local-label counts:"
        )

        for source, counts in (
            report[
                "window_source_local_labels"
            ].items()
        ):

            print(
                f"  {source:25} "
                f"benign={counts['benign']:6d} "
                f"attack={counts['attack']:6d}"
            )

        print()
        print(
            "Output:",
            output_path
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
            final_report,
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
