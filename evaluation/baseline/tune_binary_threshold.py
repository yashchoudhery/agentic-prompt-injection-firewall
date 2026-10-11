from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "baseline_tfidf"
    / "binary_tfidf_logreg.joblib"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "baseline"
    / "binary_threshold_sweep.json"
)


def load_validation():

    texts = []
    labels = []
    rows = []

    with VALIDATION_FILE.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            texts.append(
                str(
                    row.get(
                        "text",
                        ""
                    )
                )
            )

            labels.append(
                1
                if row["is_attack"]
                else 0
            )

            rows.append(row)

    return (
        texts,
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        rows,
    )


def metrics(
    y_true,
    probabilities,
    threshold,
):

    y_pred = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    fnr = (
        fn / (fn + tp)
        if fn + tp
        else 0.0
    )

    return {
        "threshold": float(
            threshold
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "fpr": float(
            fpr
        ),

        "fnr": float(
            fnr
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():

    bundle = joblib.load(
        MODEL_FILE
    )

    features = bundle[
        "features"
    ]

    classifier = bundle[
        "classifier"
    ]

    (
        texts,
        y,
        rows,
    ) = load_validation()

    X = features.transform(
        texts
    )

    probabilities = (
        classifier.predict_proba(
            X
        )[:, 1]
    )

    thresholds = np.arange(
        0.05,
        0.951,
        0.05,
    )

    results = []

    print("=" * 105)
    print(
        "THRESHOLD  PRECISION  RECALL     F1         FPR        FNR        FP     FN"
    )
    print("=" * 105)

    for threshold in thresholds:

        m = metrics(
            y,
            probabilities,
            threshold,
        )

        results.append(m)

        print(
            f"{threshold:9.2f}  "
            f"{m['precision']:9.4f}  "
            f"{m['recall']:8.4f}  "
            f"{m['f1']:8.4f}  "
            f"{m['fpr']:9.4f}  "
            f"{m['fnr']:9.4f}  "
            f"{m['fp']:6d} "
            f"{m['fn']:6d}"
        )

    # ------------------------------------------------------------
    # BIPIA-only threshold analysis
    # ------------------------------------------------------------

    bipia_indices = [
        i
        for i, row in enumerate(rows)
        if row["metadata"].get(
            "training_source_group"
        ) in {
            "bipia_indirect",
            "bipia_benign",
        }
    ]

    bipia_y = y[
        bipia_indices
    ]

    bipia_prob = probabilities[
        bipia_indices
    ]

    bipia_results = []

    print()
    print("=" * 105)
    print("BIPIA-ONLY")
    print("=" * 105)

    print(
        "THRESHOLD  PRECISION  RECALL     F1         FPR        FNR        FP     FN"
    )

    for threshold in thresholds:

        m = metrics(
            bipia_y,
            bipia_prob,
            threshold,
        )

        bipia_results.append(m)

        print(
            f"{threshold:9.2f}  "
            f"{m['precision']:9.4f}  "
            f"{m['recall']:8.4f}  "
            f"{m['f1']:8.4f}  "
            f"{m['fpr']:9.4f}  "
            f"{m['fnr']:9.4f}  "
            f"{m['fp']:6d} "
            f"{m['fn']:6d}"
        )

    # ------------------------------------------------------------
    # Find candidate thresholds
    # ------------------------------------------------------------

    recall_99 = [
        r
        for r in results
        if r["recall"] >= 0.99
    ]

    recall_995 = [
        r
        for r in results
        if r["recall"] >= 0.995
    ]

    best_f1 = max(
        results,
        key=lambda x: x["f1"],
    )

    output = {
        "overall": results,

        "bipia_only": (
            bipia_results
        ),

        "best_f1_threshold": (
            best_f1
        ),

        "best_threshold_with_recall_ge_0_99": (
            max(
                recall_99,
                key=lambda x: x[
                    "threshold"
                ],
            )
            if recall_99
            else None
        ),

        "best_threshold_with_recall_ge_0_995": (
            max(
                recall_995,
                key=lambda x: x[
                    "threshold"
                ],
            )
            if recall_995
            else None
        ),
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

        f.write("\n")

    print()
    print("=" * 105)
    print("CANDIDATE THRESHOLDS")
    print("=" * 105)

    print(
        "Best F1:",
        output[
            "best_f1_threshold"
        ]
    )

    print()

    print(
        "Recall >= 99%:",
        output[
            "best_threshold_with_recall_ge_0_99"
        ]
    )

    print()

    print(
        "Recall >= 99.5%:",
        output[
            "best_threshold_with_recall_ge_0_995"
        ]
    )

    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
