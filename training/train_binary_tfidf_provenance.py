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
    / "baseline_tfidf_provenance"
)

MODEL_FILE = (
    MODEL_DIR
    / "binary_tfidf_provenance_logreg.joblib"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "baseline"
    / "binary_tfidf_provenance_validation_report.json"
)


def safe(value):
    if value is None:
        return ""
    return str(value).strip()


def build_structured_text(row):

    source_type = safe(
        row.get("source_type")
    )

    trust_level = safe(
        row.get("trust_level")
    )

    directness = safe(
        row.get("directness")
    )

    user_instruction = safe(
        row.get("user_instruction")
    )

    external_content = safe(
        row.get("external_content")
    )

    raw_text = safe(
        row.get("text")
    )

    parts = [
        "[SOURCE_TYPE]",
        source_type,

        "[TRUST_LEVEL]",
        trust_level,

        "[DIRECTNESS]",
        directness,

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

    return "\n".join(parts)


def load_jsonl(path):

    texts = []
    labels = []
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            texts.append(
                build_structured_text(
                    row
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


def evaluate_source(
    source,
    rows,
    y_true,
    probabilities,
    threshold,
):

    indices = [
        i
        for i, row
        in enumerate(rows)
        if row[
            "metadata"
        ].get(
            "training_source_group"
        )
        == source
    ]

    if not indices:
        return None

    truth = y_true[
        indices
    ]

    probs = probabilities[
        indices
    ]

    pred = (
        probs >= threshold
    ).astype(
        np.int64
    )

    accuracy = accuracy_score(
        truth,
        pred,
    )

    if len(set(
        truth.tolist()
    )) > 1:

        tn, fp, fn, tp = (
            confusion_matrix(
                truth,
                pred,
                labels=[0, 1],
            ).ravel()
        )

    elif truth[0] == 0:

        tn = int(
            len(truth)
        )

        fp = int(
            pred.sum()
        )

        fn = 0
        tp = 0

        tn = (
            len(truth)
            - fp
        )

    else:

        tn = 0
        fp = 0

        fn = int(
            (
                pred == 0
            ).sum()
        )

        tp = int(
            (
                pred == 1
            ).sum()
        )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else None
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else None
    )

    return {
        "records": len(
            indices
        ),

        "accuracy": float(
            accuracy
        ),

        "recall": (
            float(recall)
            if recall is not None
            else None
        ),

        "fpr": (
            float(fpr)
            if fpr is not None
            else None
        ),

        "mean_attack_probability": (
            float(
                probs.mean()
            )
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

    print("=" * 80)
    print("PROVENANCE-AWARE TF-IDF")
    print("=" * 80)

    print(
        "Train:",
        len(train_texts)
    )

    print(
        "Validation:",
        len(validation_texts)
    )

    word_vectorizer = (
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.995,
            sublinear_tf=True,
            max_features=140_000,
            strip_accents="unicode",
        )
    )

    char_vectorizer = (
        TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=2,
            max_features=180_000,
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
    print("Building features...")

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

    classifier = (
        LogisticRegression(
            C=2.0,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=2026,
        )
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

    threshold_results = []

    print()
    print("=" * 100)
    print(
        "THRESHOLD  PRECISION  RECALL    F1       FPR      FP     FN"
    )
    print("=" * 100)

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(
            np.int64
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

        tn, fp, fn, tp = (
            confusion_matrix(
                y_validation,
                predictions,
                labels=[0, 1],
            ).ravel()
        )

        fpr = (
            fp / (fp + tn)
        )

        result = {
            "threshold": threshold,
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

        threshold_results.append(
            result
        )

        print(
            f"{threshold:9.2f} "
            f"{precision:10.4f} "
            f"{recall:8.4f} "
            f"{f1:8.4f} "
            f"{fpr:8.4f} "
            f"{fp:6d} "
            f"{fn:6d}"
        )

    # ============================================================
    # Use 0.50 for direct apples-to-apples baseline comparison.
    # ============================================================

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

    tn, fp, fn, tp = (
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    print()
    print("=" * 80)
    print("0.50 VALIDATION RESULTS")
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

    print()

    print(
        "TN FP FN TP:",
        tn,
        fp,
        fn,
        tp,
    )

    sources = sorted(
        {
            row["metadata"].get(
                "training_source_group"
            )
            for row
            in validation_rows
        }
    )

    per_source = {}

    print()
    print("=" * 80)
    print("PER SOURCE @ 0.50")
    print("=" * 80)

    for source in sources:

        result = evaluate_source(
            source,
            validation_rows,
            y_validation,
            probabilities,
            threshold,
        )

        per_source[
            source
        ] = result

        print()
        print(source)

        for key, value in (
            result.items()
        ):

            if isinstance(
                value,
                float,
            ):

                print(
                    f"  {key:25}: "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"  {key:25}: "
                    f"{value}"
                )

    # ============================================================
    # Dedicated BIPIA aggregate
    # ============================================================

    bipia_indices = [
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

    bipia_probs = (
        probabilities[
            bipia_indices
        ]
    )

    bipia_results = []

    print()
    print("=" * 100)
    print("BIPIA THRESHOLD COMPARISON")
    print("=" * 100)

    for threshold in thresholds:

        pred = (
            bipia_probs
            >= threshold
        ).astype(
            np.int64
        )

        tn_b, fp_b, fn_b, tp_b = (
            confusion_matrix(
                bipia_y,
                pred,
                labels=[0, 1],
            ).ravel()
        )

        precision_b = (
            precision_score(
                bipia_y,
                pred,
                zero_division=0,
            )
        )

        recall_b = (
            recall_score(
                bipia_y,
                pred,
                zero_division=0,
            )
        )

        f1_b = (
            f1_score(
                bipia_y,
                pred,
                zero_division=0,
            )
        )

        fpr_b = (
            fp_b / (
                fp_b + tn_b
            )
        )

        result = {
            "threshold": threshold,
            "precision": float(
                precision_b
            ),
            "recall": float(
                recall_b
            ),
            "f1": float(
                f1_b
            ),
            "fpr": float(
                fpr_b
            ),
            "tn": int(tn_b),
            "fp": int(fp_b),
            "fn": int(fn_b),
            "tp": int(tp_b),
        }

        bipia_results.append(
            result
        )

        print(
            f"{threshold:.2f} "
            f"precision={precision_b:.4f} "
            f"recall={recall_b:.4f} "
            f"f1={f1_b:.4f} "
            f"fpr={fpr_b:.4f} "
            f"FP={fp_b} "
            f"FN={fn_b}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "features": features,
            "classifier": classifier,
            "input_representation": (
                "provenance_structured_v1"
            ),
        },
        MODEL_FILE,
    )

    report = {
        "model": (
            "tfidf_word_char_logreg_provenance_v1"
        ),

        "input_representation": (
            "provenance_structured_v1"
        ),

        "overall_threshold_results": (
            threshold_results
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
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
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
    print("Model:", MODEL_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
