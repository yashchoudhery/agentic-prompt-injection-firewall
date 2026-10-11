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
    classification_report,
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
    / "baseline_tfidf"
)

MODEL_FILE = (
    MODEL_DIR
    / "binary_tfidf_logreg.joblib"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "baseline"
    / "binary_tfidf_validation_report.json"
)


def load_jsonl(path: Path):

    texts = []
    labels = []
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            text = str(
                row.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            if not text:
                continue

            label = (
                1
                if row["is_attack"]
                else 0
            )

            texts.append(text)
            labels.append(label)
            rows.append(row)

    return (
        texts,
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        rows,
    )


def main():

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    (
        train_texts,
        y_train,
        train_rows,
    ) = load_jsonl(
        TRAIN_FILE
    )

    (
        validation_texts,
        y_validation,
        validation_rows,
    ) = load_jsonl(
        VALIDATION_FILE
    )

    print(
        "Train:",
        len(train_texts)
    )

    print(
        "Validation:",
        len(validation_texts)
    )

    print()

    print(
        "Train attacks:",
        int(y_train.sum())
    )

    print(
        "Train benign:",
        int(
            len(y_train)
            - y_train.sum()
        )
    )

    print(
        "Validation attacks:",
        int(y_validation.sum())
    )

    print(
        "Validation benign:",
        int(
            len(y_validation)
            - y_validation.sum()
        )
    )

    # ============================================================
    # Features
    # ============================================================

    word_vectorizer = (
        TfidfVectorizer(
            lowercase=True,

            ngram_range=(
                1,
                2,
            ),

            min_df=2,

            max_df=0.995,

            sublinear_tf=True,

            max_features=120_000,

            strip_accents="unicode",
        )
    )

    char_vectorizer = (
        TfidfVectorizer(
            analyzer="char_wb",

            lowercase=True,

            ngram_range=(
                3,
                5,
            ),

            min_df=2,

            max_features=150_000,

            sublinear_tf=True,
        )
    )

    features = FeatureUnion(
        [
            (
                "word",
                word_vectorizer,
            ),
            (
                "char",
                char_vectorizer,
            ),
        ]
    )

    print()
    print("=" * 80)
    print("BUILDING TF-IDF FEATURES")
    print("=" * 80)

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

    print(
        "Validation feature matrix:",
        X_validation.shape
    )

    # ============================================================
    # Classifier
    # ============================================================

    print()
    print("=" * 80)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 80)

    classifier = (
        LogisticRegression(
            C=2.0,

            max_iter=2000,

            class_weight="balanced",

            solver="liblinear",

            random_state=2026,
        )
    )

    classifier.fit(
        X_train,
        y_train,
    )

    # ============================================================
    # Validation
    # ============================================================

    probabilities = (
        classifier.predict_proba(
            X_validation
        )[:, 1]
    )

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

    cm = confusion_matrix(
        y_validation,
        predictions,
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    fpr = (
        fp
        / (
            fp
            + tn
        )
        if (
            fp
            + tn
        )
        else 0.0
    )

    fnr = (
        fn
        / (
            fn
            + tp
        )
        if (
            fn
            + tp
        )
        else 0.0
    )

    print()
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print()

    print(
        "TN:",
        int(tn)
    )

    print(
        "FP:",
        int(fp)
    )

    print(
        "FN:",
        int(fn)
    )

    print(
        "TP:",
        int(tp)
    )

    print()

    print(
        f"False Positive Rate: {fpr:.4f}"
    )

    print(
        f"False Negative Rate: {fnr:.4f}"
    )

    print()

    print(
        classification_report(
            y_validation,
            predictions,
            target_names=[
                "benign",
                "attack",
            ],
            digits=4,
        )
    )

    # ============================================================
    # Per-source validation performance
    # ============================================================

    source_stats = {}

    source_names = sorted(
        {
            row["metadata"][
                "training_source_group"
            ]
            for row
            in validation_rows
        }
    )

    print()
    print("=" * 80)
    print("PER-SOURCE VALIDATION")
    print("=" * 80)

    for source in source_names:

        indices = [
            i
            for i, row
            in enumerate(
                validation_rows
            )
            if row[
                "metadata"
            ][
                "training_source_group"
            ]
            == source
        ]

        y_true = (
            y_validation[
                indices
            ]
        )

        y_pred = (
            predictions[
                indices
            ]
        )

        probs = (
            probabilities[
                indices
            ]
        )

        src_accuracy = (
            accuracy_score(
                y_true,
                y_pred,
            )
        )

        src_f1 = (
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
            if len(
                set(
                    y_true.tolist()
                )
            ) > 1
            or int(
                y_true.sum()
            ) > 0
            else 0.0
        )

        source_stats[
            source
        ] = {
            "records": len(
                indices
            ),

            "attack_records": int(
                y_true.sum()
            ),

            "benign_records": int(
                len(y_true)
                - y_true.sum()
            ),

            "accuracy": float(
                src_accuracy
            ),

            "f1_attack": float(
                src_f1
            ),

            "mean_attack_probability": float(
                np.mean(
                    probs
                )
            ),
        }

        print()
        print(
            source
        )

        print(
            "  rows      :",
            len(indices)
        )

        print(
            "  accuracy  :",
            f"{src_accuracy:.4f}"
        )

        print(
            "  attack F1 :",
            f"{src_f1:.4f}"
        )

    # ============================================================
    # Save hardest errors
    # ============================================================

    false_positives = []
    false_negatives = []

    for index, row in enumerate(
        validation_rows
    ):

        actual = int(
            y_validation[
                index
            ]
        )

        predicted = int(
            predictions[
                index
            ]
        )

        probability = float(
            probabilities[
                index
            ]
        )

        error_record = {
            "sample_id": row.get(
                "sample_id"
            ),

            "actual": (
                "attack"
                if actual
                else "benign"
            ),

            "predicted": (
                "attack"
                if predicted
                else "benign"
            ),

            "attack_probability": (
                probability
            ),

            "source": row[
                "metadata"
            ][
                "training_source_group"
            ],

            "attack_types": row.get(
                "attack_types",
                [],
            ),

            "text": row.get(
                "text",
                "",
            )[:3000],
        }

        if (
            actual == 0
            and predicted == 1
        ):

            false_positives.append(
                error_record
            )

        elif (
            actual == 1
            and predicted == 0
        ):

            false_negatives.append(
                error_record
            )

    false_positives.sort(
        key=lambda x: x[
            "attack_probability"
        ],
        reverse=True,
    )

    false_negatives.sort(
        key=lambda x: x[
            "attack_probability"
        ],
    )

    # ============================================================
    # Save model
    # ============================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "features": features,

            "classifier": classifier,

            "threshold": threshold,

            "label_mapping": {
                0: "benign",
                1: "attack",
            },
        },
        MODEL_FILE,
    )

    report = {
        "model": (
            "tfidf_word_char_logistic_regression"
        ),

        "threshold": (
            threshold
        ),

        "train_records": (
            len(train_texts)
        ),

        "validation_records": (
            len(validation_texts)
        ),

        "metrics": {
            "accuracy": float(
                accuracy
            ),

            "precision_attack": float(
                precision
            ),

            "recall_attack": float(
                recall
            ),

            "f1_attack": float(
                f1
            ),

            "roc_auc": float(
                roc_auc
            ),

            "pr_auc": float(
                pr_auc
            ),

            "false_positive_rate": (
                float(fpr)
            ),

            "false_negative_rate": (
                float(fnr)
            ),
        },

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },

        "feature_shape": {
            "train_rows": int(
                X_train.shape[0]
            ),

            "features": int(
                X_train.shape[1]
            ),
        },

        "per_source": (
            source_stats
        ),

        "false_positives": (
            false_positives[:100]
        ),

        "false_negatives": (
            false_negatives[:100]
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
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(
        "Model:",
        MODEL_FILE
    )

    print(
        "Report:",
        REPORT_FILE
    )

    print(
        "False positives:",
        len(
            false_positives
        )
    )

    print(
        "False negatives:",
        len(
            false_negatives
        )
    )


if __name__ == "__main__":
    main()
