from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from torch.optim import AdamW
from torch.utils.data import (
    Dataset,
    DataLoader,
)

from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "microsoft/deberta-v3-small"

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformer_windows"
    / "train_windows.jsonl"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transformer_windows"
    / "validation_windows.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "deberta_v3_small_window_v1"
)

BEST_MODEL_DIR = (
    OUTPUT_DIR
    / "best"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "transformer"
    / "deberta_v3_small_window_v1.json"
)


SEED = 2026

EPOCHS = 2

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 8

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

MAX_GRAD_NORM = 1.0

THRESHOLD = 0.50


# Hard cases receive more training importance.
SOURCE_WEIGHTS = {
    "bipia_indirect": 3.0,
    "bipia_benign": 3.0,

    "dolly_hard_negative": 2.0,

    "direct_attacks": 1.0,
    "dolly_ordinary": 1.0,
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data
# ============================================================

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


class WindowDataset(Dataset):

    def __init__(
        self,
        rows,
    ):

        self.rows = rows

    def __len__(self):

        return len(
            self.rows
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.rows[
            index
        ]

        item = {
            "input_ids": row[
                "input_ids"
            ],

            "attention_mask": row[
                "attention_mask"
            ],

            "label": int(
                row["label"]
            ),

            "row_index": index,

            "sample_weight": float(
                SOURCE_WEIGHTS.get(
                    row[
                        "source_group"
                    ],
                    1.0,
                )
            ),
        }

        if (
            "token_type_ids"
            in row
        ):

            item[
                "token_type_ids"
            ] = row[
                "token_type_ids"
            ]

        return item


class WindowCollator:

    def __init__(
        self,
        tokenizer,
    ):

        self.tokenizer = tokenizer

    def __call__(
        self,
        batch,
    ):

        model_features = []

        labels = []
        weights = []
        row_indices = []

        for item in batch:

            feature = {
                "input_ids": (
                    item[
                        "input_ids"
                    ]
                ),

                "attention_mask": (
                    item[
                        "attention_mask"
                    ]
                ),
            }

            if (
                "token_type_ids"
                in item
            ):

                feature[
                    "token_type_ids"
                ] = item[
                    "token_type_ids"
                ]

            model_features.append(
                feature
            )

            labels.append(
                item["label"]
            )

            weights.append(
                item[
                    "sample_weight"
                ]
            )

            row_indices.append(
                item[
                    "row_index"
                ]
            )

        padded = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )

        padded[
            "labels"
        ] = torch.tensor(
            labels,
            dtype=torch.long,
        )

        padded[
            "sample_weight"
        ] = torch.tensor(
            weights,
            dtype=torch.float32,
        )

        padded[
            "row_index"
        ] = torch.tensor(
            row_indices,
            dtype=torch.long,
        )

        return padded


# ============================================================
# Metrics
# ============================================================

def binary_metrics(
    labels,
    probabilities,
    threshold=0.50,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
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

    result = {
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

    if len(
        np.unique(
            labels
        )
    ) == 2:

        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

        result[
            "pr_auc"
        ] = float(
            average_precision_score(
                labels,
                probabilities,
            )
        )

    return result


def document_aggregate(
    rows,
    probabilities,
):

    grouped = defaultdict(
        list
    )

    document_info = {}

    for row, probability in zip(
        rows,
        probabilities,
    ):

        document_id = row[
            "document_id"
        ]

        grouped[
            document_id
        ].append(
            float(
                probability
            )
        )

        document_info[
            document_id
        ] = {
            "label": int(
                row[
                    "document_label"
                ]
            ),

            "source_group": row[
                "source_group"
            ],
        }

    documents = []

    for document_id, probs in (
        grouped.items()
    ):

        info = document_info[
            document_id
        ]

        documents.append(
            {
                "document_id": (
                    document_id
                ),

                "label": (
                    info["label"]
                ),

                "source_group": (
                    info[
                        "source_group"
                    ]
                ),

                # Firewall policy:
                # one dangerous window is enough.
                "probability": float(
                    max(probs)
                ),

                "window_count": len(
                    probs
                ),
            }
        )

    return documents


def evaluate_documents(
    documents,
):

    labels = [
        x["label"]
        for x in documents
    ]

    probabilities = [
        x["probability"]
        for x in documents
    ]

    overall = binary_metrics(
        labels,
        probabilities,
        THRESHOLD,
    )

    bipia = [
        x
        for x in documents
        if x[
            "source_group"
        ]
        in {
            "bipia_indirect",
            "bipia_benign",
        }
    ]

    bipia_labels = [
        x["label"]
        for x in bipia
    ]

    bipia_probs = [
        x["probability"]
        for x in bipia
    ]

    bipia_metrics = binary_metrics(
        bipia_labels,
        bipia_probs,
        THRESHOLD,
    )

    per_source = {}

    sources = sorted(
        {
            x[
                "source_group"
            ]
            for x
            in documents
        }
    )

    for source in sources:

        subset = [
            x
            for x
            in documents
            if x[
                "source_group"
            ]
            == source
        ]

        source_labels = [
            x["label"]
            for x in subset
        ]

        source_probs = [
            x["probability"]
            for x in subset
        ]

        predictions = (
            np.asarray(
                source_probs
            )
            >= THRESHOLD
        ).astype(
            np.int64
        )

        entry = {
            "records": len(
                subset
            ),

            "accuracy": float(
                accuracy_score(
                    source_labels,
                    predictions,
                )
            ),

            "mean_attack_probability": (
                float(
                    np.mean(
                        source_probs
                    )
                )
            ),
        }

        if (
            sum(source_labels)
            == 0
        ):

            entry[
                "fpr"
            ] = float(
                predictions.mean()
            )

        else:

            entry[
                "recall"
            ] = float(
                recall_score(
                    source_labels,
                    predictions,
                    zero_division=0,
                )
            )

        per_source[
            source
        ] = entry

    return {
        "overall": overall,

        "bipia": bipia_metrics,

        "per_source": (
            per_source
        ),
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    rows,
    device,
):

    model.eval()

    all_probabilities = np.zeros(
        len(rows),
        dtype=np.float64,
    )

    all_labels = np.zeros(
        len(rows),
        dtype=np.int64,
    )

    total_loss = 0.0
    total_examples = 0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for batch in progress:

        labels = batch.pop(
            "labels"
        ).to(
            device
        )

        # Not used during validation.
        batch.pop(
            "sample_weight"
        )

        row_indices = batch.pop(
            "row_index"
        ).cpu().numpy()

        model_inputs = {
            key: value.to(
                device
            )
            for key, value
            in batch.items()
        }

        outputs = model(
            **model_inputs
        )

        logits = outputs.logits

        loss = F.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

        labels_np = (
            labels
            .detach()
            .cpu()
            .numpy()
        )

        all_probabilities[
            row_indices
        ] = probabilities

        all_labels[
            row_indices
        ] = labels_np

        total_loss += float(
            loss.item()
        )

        total_examples += len(
            labels_np
        )

    window_metrics = (
        binary_metrics(
            all_labels,
            all_probabilities,
            THRESHOLD,
        )
    )

    documents = document_aggregate(
        rows,
        all_probabilities,
    )

    document_metrics = (
        evaluate_documents(
            documents
        )
    )

    return {
        "validation_loss": (
            total_loss
            / max(
                total_examples,
                1,
            )
        ),

        "window_metrics": (
            window_metrics
        ),

        "document_metrics": (
            document_metrics
        ),
    }


# ============================================================
# Printing
# ============================================================

def print_eval(
    epoch,
    result,
):

    print()
    print("=" * 90)
    print(
        f"EPOCH {epoch} VALIDATION"
    )
    print("=" * 90)

    print(
        "Validation loss:",
        f"{result['validation_loss']:.6f}"
    )

    window = result[
        "window_metrics"
    ]

    print()
    print("WINDOW LEVEL @ 0.50")

    print(
        f"  precision : "
        f"{window['precision']:.4f}"
    )

    print(
        f"  recall    : "
        f"{window['recall']:.4f}"
    )

    print(
        f"  F1        : "
        f"{window['f1']:.4f}"
    )

    print(
        f"  FPR       : "
        f"{window['fpr']:.4f}"
    )

    document = result[
        "document_metrics"
    ][
        "overall"
    ]

    print()
    print(
        "DOCUMENT LEVEL @ 0.50"
    )

    print(
        f"  precision : "
        f"{document['precision']:.4f}"
    )

    print(
        f"  recall    : "
        f"{document['recall']:.4f}"
    )

    print(
        f"  F1        : "
        f"{document['f1']:.4f}"
    )

    print(
        f"  FPR       : "
        f"{document['fpr']:.4f}"
    )

    bipia = result[
        "document_metrics"
    ][
        "bipia"
    ]

    print()
    print(
        "BIPIA DOCUMENT LEVEL @ 0.50"
    )

    print(
        f"  precision : "
        f"{bipia['precision']:.4f}"
    )

    print(
        f"  recall    : "
        f"{bipia['recall']:.4f}"
    )

    print(
        f"  F1        : "
        f"{bipia['f1']:.4f}"
    )

    print(
        f"  FPR       : "
        f"{bipia['fpr']:.4f}"
    )

    print(
        f"  FP        : "
        f"{bipia['fp']}"
    )

    print(
        f"  FN        : "
        f"{bipia['fn']}"
    )

    print()
    print(
        "PER SOURCE"
    )

    for source, metrics in (
        result[
            "document_metrics"
        ][
            "per_source"
        ].items()
    ):

        print()

        print(
            f"  {source}"
        )

        print(
            f"    records   : "
            f"{metrics['records']}"
        )

        print(
            f"    accuracy  : "
            f"{metrics['accuracy']:.4f}"
        )

        if (
            "recall"
            in metrics
        ):

            print(
                f"    recall    : "
                f"{metrics['recall']:.4f}"
            )

        if (
            "fpr"
            in metrics
        ):

            print(
                f"    FPR       : "
                f"{metrics['fpr']:.4f}"
            )

        print(
            f"    mean prob : "
            f"{metrics['mean_attack_probability']:.4f}"
        )


# ============================================================
# Main training
# ============================================================

def main():

    set_seed(
        SEED
    )

    if torch.backends.mps.is_available():

        device = torch.device(
            "mps"
        )

    elif torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print("=" * 90)
    print(
        "DEBERTA V3 SMALL WINDOW CLASSIFIER"
    )
    print("=" * 90)

    print(
        "Device:",
        device
    )

    print(
        "Precision: float32"
    )

    train_rows = read_jsonl(
        TRAIN_FILE
    )

    validation_rows = read_jsonl(
        VALIDATION_FILE
    )

    print(
        "Train windows:",
        len(train_rows)
    )

    print(
        "Validation windows:",
        len(validation_rows)
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = WindowDataset(
        train_rows
    )

    validation_dataset = WindowDataset(
        validation_rows
    )

    collator = WindowCollator(
        tokenizer
    )

    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    id2label = {
        0: "BENIGN",
        1: "ATTACK",
    }

    label2id = {
        "BENIGN": 0,
        "ATTACK": 1,
    }

    print()
    print(
        "Loading:",
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
    )

    # Reduce activation memory.
    if hasattr(
        model,
        "gradient_checkpointing_enable",
    ):

        model.gradient_checkpointing_enable()

        print(
            "Gradient checkpointing: ON"
        )

    model.to(
        device
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer_steps_per_epoch = (
        math.ceil(
            len(train_loader)
            / GRADIENT_ACCUMULATION_STEPS
        )
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * EPOCHS
    )

    warmup_steps = int(
        total_optimizer_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(
                total_optimizer_steps
            ),
        )
    )

    print()
    print(
        "Train batch:",
        TRAIN_BATCH_SIZE
    )

    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "Effective batch:",
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Optimizer steps:",
        total_optimizer_steps
    )

    print(
        "Warmup steps:",
        warmup_steps
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []

    best_bipia_f1 = -1.0
    best_overall_f1 = -1.0
    best_epoch = None

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        seen_batches = 0

        progress = tqdm(
            enumerate(
                train_loader,
                start=1,
            ),
            total=len(
                train_loader
            ),
            desc=(
                f"Epoch {epoch}/{EPOCHS}"
            ),
        )

        for step, batch in progress:

            labels = batch.pop(
                "labels"
            ).to(
                device
            )

            sample_weight = (
                batch.pop(
                    "sample_weight"
                ).to(
                    device
                )
            )

            batch.pop(
                "row_index"
            )

            model_inputs = {
                key: value.to(
                    device
                )
                for key, value
                in batch.items()
            }

            outputs = model(
                **model_inputs
            )

            per_sample_loss = (
                F.cross_entropy(
                    outputs.logits,
                    labels,
                    reduction="none",
                )
            )

            loss = (
                per_sample_loss
                * sample_weight
            ).sum() / (
                sample_weight.sum()
                + 1e-8
            )

            raw_loss = float(
                loss.detach().item()
            )

            (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            ).backward()

            should_update = (
                step
                % GRADIENT_ACCUMULATION_STEPS
                == 0
                or step
                == len(train_loader)
            )

            if should_update:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += (
                raw_loss
            )

            seen_batches += 1

            if (
                step % 100 == 0
                or step
                == len(train_loader)
            ):

                progress.set_postfix(
                    loss=(
                        running_loss
                        / seen_batches
                    ),

                    lr=(
                        scheduler
                        .get_last_lr()[0]
                    ),
                )

        training_loss = (
            running_loss
            / max(
                seen_batches,
                1,
            )
        )

        print()
        print(
            f"Epoch {epoch} "
            f"training loss: "
            f"{training_loss:.6f}"
        )

        result = evaluate_model(
            model,
            validation_loader,
            validation_rows,
            device,
        )

        result[
            "epoch"
        ] = epoch

        result[
            "training_loss"
        ] = training_loss

        history.append(
            result
        )

        print_eval(
            epoch,
            result,
        )

        bipia_f1 = (
            result[
                "document_metrics"
            ][
                "bipia"
            ][
                "f1"
            ]
        )

        overall_f1 = (
            result[
                "document_metrics"
            ][
                "overall"
            ][
                "f1"
            ]
        )

        is_better = (
            bipia_f1
            > best_bipia_f1
            or (
                abs(
                    bipia_f1
                    - best_bipia_f1
                )
                < 1e-12
                and overall_f1
                > best_overall_f1
            )
        )

        if is_better:

            best_bipia_f1 = (
                bipia_f1
            )

            best_overall_f1 = (
                overall_f1
            )

            best_epoch = epoch

            BEST_MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save_pretrained(
                BEST_MODEL_DIR
            )

            tokenizer.save_pretrained(
                BEST_MODEL_DIR
            )

            print()
            print(
                "Saved new best model:",
                BEST_MODEL_DIR
            )

        # Helpful on Apple unified memory.
        if (
            device.type
            == "mps"
        ):

            torch.mps.empty_cache()

    report = {
        "model_name": MODEL_NAME,

        "device": str(
            device
        ),

        "precision": "float32",

        "epochs": EPOCHS,

        "train_batch_size": (
            TRAIN_BATCH_SIZE
        ),

        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),

        "effective_batch_size": (
            TRAIN_BATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        ),

        "learning_rate": (
            LEARNING_RATE
        ),

        "weight_decay": (
            WEIGHT_DECAY
        ),

        "warmup_ratio": (
            WARMUP_RATIO
        ),

        "threshold": (
            THRESHOLD
        ),

        "document_aggregation": (
            "max_window_probability"
        ),

        "source_weights": (
            SOURCE_WEIGHTS
        ),

        "best_epoch": (
            best_epoch
        ),

        "best_bipia_f1": (
            best_bipia_f1
        ),

        "best_overall_f1": (
            best_overall_f1
        ),

        "history": history,

        "best_model_dir": str(
            BEST_MODEL_DIR
        ),
    }

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
    print("=" * 90)

    print(
        "TRAINING COMPLETE"
    )

    print("=" * 90)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best BIPIA F1:",
        f"{best_bipia_f1:.4f}"
    )

    print(
        "Best overall F1:",
        f"{best_overall_f1:.4f}"
    )

    print(
        "Best model:",
        BEST_MODEL_DIR
    )

    print(
        "Report:",
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
