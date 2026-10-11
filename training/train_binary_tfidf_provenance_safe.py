from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.jsonl"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.jsonl"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "baseline_tfidf_provenance_safe"
)

MODEL_FILE = (
    MODEL_DIR
    / "binary_tfidf_provenance_safe.joblib"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "baseline"
    / "binary_tfidf_provenance_safe_report.json"
)


def safe(value):

    if value is None:
        return ""

    return str(value).strip()


def build_input(row):

    source_type = safe(
        row.get(
            "source_type"
        )
    )

    user_instruction = safe(
        row.get(
            "user_instruction"
        )
    )

    external_content = safe(
        row.get(
            "external_content"
        )
    )

    raw_text = safe(
        row.get(
            "text"
        )
    )

    return "\n".join(
        [
            "[SOURCE_TYPE]",
            source_type
            if source_type
            else "<UNKNOWN>",

            "[USER_INSTRUCTION]",
            user_instruction
            if user_instruction
            else "<NONE>",

            "[EXTERNAL_CONTENT]",
            external_content
            if external_content
            else "<NONE>",

            "[CONTENT_TO_INSPECT]",
            raw_text,
        ]
    )


def load_data(path):

    texts = []
    labels = []
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(
                line
            )

            texts.append(
                build_input(
                    row
                )
            )

            labels.append(
                1
                if row[
                    "is_attack"
                ]
                else 0
            )

            rows.append(
                row
            )

    return (
        texts,
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        rows,
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
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

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "tp": int(tp),
    }


def main():

    (
        train_texts,
        y_train,
        train_rows,
    ) = load_data(
        TRAIN_FILE
    )

    (
        validation_texts,
        y_validation,
        validation_rows,
    ) = load_data(
        VALIDATION_FILE
    )

    print("=" * 80)
    print(
        "LEAKAGE-SAFE PROVENANCE BASELINE"
    )
    print("=" * 80)

    print(
        "Train:",
        len(train_texts)
    )

    print(
        "Validation:",
        len(validation_texts)
    )

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(
                        1,
                        2,
                    ),
                    min_df=2,
                    max_df=0.995,
                    sublinear_tf=True,
                    max_features=140_000,
                    strip_accents="unicode",
                ),
            ),

            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(
                        3,
                        5,
                    ),
                    min_df=2,
                    max_features=180_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    print()
    print(
        "Building features..."
    )

    X_train = (
        features.fit_transform(
            train_texts
        )
    )

    X_validation = (
        features.transform(
            validation_texts
        )
    )

    print(
        "Train feature matrix:",
        X_train.shape
    )

    classifier = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=2026,
    )

    print()
    print("Training...")

    classifier.fit(
        X_train,
        y_train,
    )

    probabilities = (
        classifier.predict_proba(
            X_validation
        )[:, 1]
    )

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    overall_results = []

    print()
    print("=" * 100)
    print(
        "OVERALL THRESHOLD RESULTS"
    )
    print("=" * 100)

    for threshold in thresholds:

        result = calculate_metrics(
            y_validation,
            probabilities,
            threshold,
        )

        overall_results.append(
            result
        )

        print(
            f"{threshold:.2f} "
            f"precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f} "
            f"f1={result['f1']:.4f} "
            f"fpr={result['fpr']:.4f} "
            f"FP={result['fp']} "
            f"FN={result['fn']}"
        )

    # ----------------------------------------------------------
    # BIPIA only
    # ----------------------------------------------------------

    bipia_indices = [
        i
        for i, row in enumerate(
            validation_rows
        )
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

    bipia_y = (
        y_validation[
            bipia_indices
        ]
    )

    bipia_prob = (
        probabilities[
            bipia_indices
        ]
    )

    bipia_results = []

    print()
    print("=" * 100)
    print(
        "BIPIA THRESHOLD RESULTS"
    )
    print("=" * 100)

    for threshold in thresholds:

        result = calculate_metrics(
            bipia_y,
            bipia_prob,
            threshold,
        )

        bipia_results.append(
            result
        )

        print(
            f"{threshold:.2f} "
            f"precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f} "
            f"f1={result['f1']:.4f} "
            f"fpr={result['fpr']:.4f} "
            f"FP={result['fp']} "
            f"FN={result['fn']}"
        )

    # ----------------------------------------------------------
    # Per-source @ 0.50
    # ----------------------------------------------------------

    threshold = 0.50

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    print()
    print("=" * 80)
    print(
        "OVERALL @ 0.50"
    )
    print("=" * 80)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    per_source = {}

    sources = sorted(
        {
            row[
                "metadata"
            ].get(
                "training_source_group"
            )
            for row
            in validation_rows
        }
    )

    print()
    print("=" * 80)
    print(
        "PER SOURCE @ 0.50"
    )
    print("=" * 80)

    for source in sources:

        indices = [
            i
            for i, row
            in enumerate(
                validation_rows
            )
            if row[
                "metadata"
            ].get(
                "training_source_group"
            )
            == source
        ]

        y_true = (
            y_validation[
                indices
            ]
        )

        probs = (
            probabilities[
                indices
            ]
        )

        pred = (
            probs
            >= threshold
        ).astype(
            np.int64
        )

        acc = accuracy_score(
            y_true,
            pred,
        )

        if int(
            y_true.sum()
        ) == 0:

            source_recall = None

            source_fpr = float(
                pred.sum()
                / len(pred)
            )

        else:

            source_recall = float(
                recall_score(
                    y_true,
                    pred,
                    zero_division=0,
                )
            )

            source_fpr = None

        per_source[
            source
        ] = {
            "records": len(
                indices
            ),

            "accuracy": float(
                acc
            ),

            "recall": (
                source_recall
            ),

            "fpr": (
                source_fpr
            ),

            "mean_attack_probability": float(
                probs.mean()
            ),
        }

        print()
        print(source)

        print(
            f"  records                 : "
            f"{len(indices)}"
        )

        print(
            f"  accuracy                : "
            f"{acc:.4f}"
        )

        if source_recall is not None:

            print(
                f"  recall                  : "
                f"{source_recall:.4f}"
            )

        if source_fpr is not None:

            print(
                f"  fpr                     : "
                f"{source_fpr:.4f}"
            )

        print(
            f"  mean_attack_probability : "
            f"{probs.mean():.4f}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "features": features,

            "classifier": (
                classifier
            ),

            "allowed_fields": [
                "source_type",
                "user_instruction",
                "external_content",
                "text",
            ],

            "input_representation": (
                "provenance_safe_v1"
            ),
        },
        MODEL_FILE,
    )

    report = {
        "model": (
            "tfidf_provenance_safe_v1"
        ),

        "allowed_input_fields": [
            "source_type",
            "user_instruction",
            "external_content",
            "text",
        ],

        "explicitly_excluded_fields": [
            "directness",
            "encoding_type",
            "attack_payload",
            "poisoned_content",
            "attack_position",
            "attack_types",
            "severity",
        ],

        "overall_threshold_results": (
            overall_results
        ),

        "bipia_threshold_results": (
            bipia_results
        ),

        "metrics_at_0_50": {
            "accuracy": float(
                accuracy
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

            "roc_auc": float(
                roc_auc
            ),

            "pr_auc": float(
                pr_auc
            ),
        },

        "per_source_at_0_50": (
            per_source
        ),

        "feature_count": int(
            X_train.shape[1]
        ),

        "model_file": str(
            MODEL_FILE
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
            indent=2,
        )

        f.write("\n")

    print()
    print(
        "Model:",
        MODEL_FILE
    )

    print(
        "Report:",
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
