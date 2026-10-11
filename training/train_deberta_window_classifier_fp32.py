from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

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

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# ============================================================
# PATHS
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


# ============================================================
# TRAINING CONFIG
# ============================================================

SEED = 2026

EPOCHS = 3

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64

GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.06

MAX_GRAD_NORM = 1.0

THRESHOLD = 0.50


# Give hard indirect cases more influence.
SOURCE_WEIGHTS = {
    "bipia_indirect": 1.0,
    "bipia_benign": 1.0,
    "dolly_hard_negative": 1.0,
    "direct_attacks": 1.0,
    "dolly_ordinary": 1.0,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
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

    def __init__(self, rows):

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

        result = {
            "input_ids": (
                row[
                    "input_ids"
                ]
            ),

            "attention_mask": (
                row[
                    "attention_mask"
                ]
            ),

            "label": int(
                row["label"]
            ),

            "row_index": (
                index
            ),

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

            result[
                "token_type_ids"
            ] = row[
                "token_type_ids"
            ]

        return result


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

        features = []

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

            features.append(
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
            features,
            padding=True,

            # Better Tensor Core utilization.
            pad_to_multiple_of=8,

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
# METRICS
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

    result = {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),

        "fpr": float(
            fp / (fp + tn)
            if fp + tn
            else 0.0
        ),

        "fnr": float(
            fn / (fn + tp)
            if fn + tp
            else 0.0
        ),

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "tp": int(tp),
    }

    if len(
        np.unique(labels)
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

    probabilities_by_doc = (
        defaultdict(list)
    )

    document_info = {}

    for row, probability in zip(
        rows,
        probabilities,
    ):

        document_id = row[
            "document_id"
        ]

        probabilities_by_doc[
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

            "source_group": (
                row[
                    "source_group"
                ]
            ),
        }

    documents = []

    for document_id, probs in (
        probabilities_by_doc.items()
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
                    info[
                        "label"
                    ]
                ),

                "source_group": (
                    info[
                        "source_group"
                    ]
                ),

                # Firewall logic:
                # any dangerous window can
                # make the document dangerous.
                "probability": float(
                    max(probs)
                ),

                "window_count": (
                    len(probs)
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
        for x
        in documents
        if x[
            "source_group"
        ]
        in {
            "bipia_indirect",
            "bipia_benign",
        }
    ]

    bipia_metrics = (
        binary_metrics(
            [
                x["label"]
                for x
                in bipia
            ],

            [
                x[
                    "probability"
                ]
                for x
                in bipia
            ],

            THRESHOLD,
        )
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

        source_labels = np.asarray(
            [
                x["label"]
                for x
                in subset
            ],
            dtype=np.int64,
        )

        source_probs = np.asarray(
            [
                x[
                    "probability"
                ]
                for x
                in subset
            ],
            dtype=np.float64,
        )

        predictions = (
            source_probs
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

            "mean_attack_probability": float(
                np.mean(
                    source_probs
                )
            ),
        }

        if (
            source_labels.sum()
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

        "bipia": (
            bipia_metrics
        ),

        "per_source": (
            per_source
        ),
    }


# ============================================================
# VALIDATION
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

        labels = (
            batch.pop(
                "labels"
            )
            .to(
                device,
                non_blocking=True,
            )
        )

        batch.pop(
            "sample_weight"
        )

        row_indices = (
            batch.pop(
                "row_index"
            )
            .numpy()
        )

        inputs = {
            key: value.to(
                device,
                non_blocking=True,
            )
            for key, value
            in batch.items()
        }

        logits = model(
            **inputs
        ).logits

        if not torch.isfinite(
            logits
        ).all():

            raise RuntimeError(
                "Non-finite validation logits detected."
            )

        loss = F.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite validation "
                "loss detected."
            )

        probabilities = (
            torch.softmax(
                logits.float(),
                dim=-1,
            )[:, 1]
            .cpu()
            .numpy()
        )

        labels_np = (
            labels
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

        total_examples += (
            len(labels_np)
        )

    windows = binary_metrics(
        all_labels,
        all_probabilities,
        THRESHOLD,
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
        "validation_loss": float(
            total_loss
            / max(
                total_examples,
                1,
            )
        ),

        "window_metrics": (
            windows
        ),

        "document_metrics": (
            document_metrics
        ),
    }


# ============================================================
# PRINTING
# ============================================================

def print_metrics(
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

    windows = result[
        "window_metrics"
    ]

    print()
    print(
        "WINDOW LEVEL @ 0.50"
    )

    print(
        f"Precision: "
        f"{windows['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{windows['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{windows['f1']:.4f}"
    )

    print(
        f"FPR      : "
        f"{windows['fpr']:.4f}"
    )

    overall = result[
        "document_metrics"
    ][
        "overall"
    ]

    print()
    print(
        "DOCUMENT LEVEL @ 0.50"
    )

    print(
        f"Precision: "
        f"{overall['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{overall['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{overall['f1']:.4f}"
    )

    print(
        f"FPR      : "
        f"{overall['fpr']:.4f}"
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
        f"Precision: "
        f"{bipia['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{bipia['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{bipia['f1']:.4f}"
    )

    print(
        f"FPR      : "
        f"{bipia['fpr']:.4f}"
    )

    print(
        f"FP       : "
        f"{bipia['fp']}"
    )

    print(
        f"FN       : "
        f"{bipia['fn']}"
    )

    print()
    print("PER SOURCE")

    for source, metrics in (
        result[
            "document_metrics"
        ][
            "per_source"
        ].items()
    ):

        print()

        print(
            source
        )

        print(
            "  records:",
            metrics[
                "records"
            ]
        )

        print(
            "  accuracy:",
            f"{metrics['accuracy']:.4f}"
        )

        if (
            "recall"
            in metrics
        ):

            print(
                "  recall:",
                f"{metrics['recall']:.4f}"
            )

        if (
            "fpr"
            in metrics
        ):

            print(
                "  FPR:",
                f"{metrics['fpr']:.4f}"
            )

        print(
            "  mean probability:",
            f"{metrics['mean_attack_probability']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    # --------------------------------------------------------
    # HARD REQUIREMENT: CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "Do NOT train this script on CPU."
        )

    device = torch.device(
        "cuda:0"
    )

    gpu_name = (
        torch.cuda
        .get_device_name(0)
    )

    gpu_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print("=" * 90)
    print(
        "DEBERTA V3 SMALL — CUDA/FP32"
    )
    print("=" * 90)

    print(
        "Device:",
        device
    )

    print(
        "GPU:",
        gpu_name
    )

    print(
        "GPU VRAM:",
        f"{gpu_memory:.2f} GB"
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported()
    )



    # Faster Tensor Core operations.
    torch.set_float32_matmul_precision(
        "high"
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_rows = read_jsonl(
        TRAIN_FILE
    )

    validation_rows = (
        read_jsonl(
            VALIDATION_FILE
        )
    )

    print()

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


    train_dataset = (
        WindowDataset(
            train_rows
        )
    )

    validation_dataset = (
        WindowDataset(
            validation_rows
        )
    )

    collator = (
        WindowCollator(
            tokenizer
        )
    )


    train_loader = DataLoader(
        train_dataset,

        batch_size=(
            TRAIN_BATCH_SIZE
        ),

        shuffle=True,

        collate_fn=(
            collator
        ),

        num_workers=2,

        pin_memory=True,

        persistent_workers=True,
    )


    validation_loader = (
        DataLoader(
            validation_dataset,

            batch_size=(
                EVAL_BATCH_SIZE
            ),

            shuffle=False,

            collate_fn=(
                collator
            ),

            num_workers=2,

            pin_memory=True,

            persistent_workers=True,
        )
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        "Loading model:",
        MODEL_NAME
    )


    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,

            dtype=torch.float32,

            num_labels=2,

            id2label={
                0: "BENIGN",
                1: "ATTACK",
            },

            label2id={
                "BENIGN": 0,
                "ATTACK": 1,
            },
        )
    )


    # We have ~95 GB VRAM.
    # Disable checkpointing for speed.
    if hasattr(
        model,
        "gradient_checkpointing_disable",
    ):

        model.gradient_checkpointing_disable()


    model.to(
        device=device,
        dtype=torch.float32,
    )

    print(
        "Model dtype:",
        next(model.parameters()).dtype
    )


    print(
        "Gradient checkpointing: OFF"
    )

    print(
        "Training precision: FP32"
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    no_decay = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
    )


    optimizer_groups = [
        {
            "params": [
                p
                for name, p
                in model.named_parameters()
                if (
                    p.requires_grad
                    and not any(
                        term in name
                        for term
                        in no_decay
                    )
                )
            ],

            "weight_decay": (
                WEIGHT_DECAY
            ),
        },

        {
            "params": [
                p
                for name, p
                in model.named_parameters()
                if (
                    p.requires_grad
                    and any(
                        term in name
                        for term
                        in no_decay
                    )
                )
            ],

            "weight_decay": 0.0,
        },
    ]


    try:

        optimizer = torch.optim.AdamW(
            optimizer_groups,

            lr=LEARNING_RATE,

            fused=True,
        )

        print(
            "Optimizer: fused AdamW"
        )

    except Exception:

        optimizer = torch.optim.AdamW(
            optimizer_groups,

            lr=LEARNING_RATE,
        )

        print(
            "Optimizer: standard AdamW"
        )


    steps_per_epoch = math.ceil(
        len(train_loader)
        / GRADIENT_ACCUMULATION_STEPS
    )

    total_steps = (
        steps_per_epoch
        * EPOCHS
    )

    warmup_steps = int(
        total_steps
        * WARMUP_RATIO
    )


    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,

            num_warmup_steps=(
                warmup_steps
            ),

            num_training_steps=(
                total_steps
            ),
        )
    )


    print()

    print(
        "Train batch:",
        TRAIN_BATCH_SIZE
    )

    print(
        "Eval batch:",
        EVAL_BATCH_SIZE
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
        total_steps
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


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history = []

    best_bipia_f1 = -1.0

    best_overall_f1 = -1.0

    best_epoch = None


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

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

            labels = (
                batch.pop(
                    "labels"
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )


            weights = (
                batch.pop(
                    "sample_weight"
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )


            batch.pop(
                "row_index"
            )


            inputs = {
                key: value.to(
                    device,
                    non_blocking=True,
                )

                for key, value
                in batch.items()
            }


            # -----------------------------------------------
            # CUDA BF16 forward pass
            # -----------------------------------------------

            logits = model(
                **inputs
            ).logits

            if not torch.isfinite(
                logits
            ).all():

                raise RuntimeError(
                    f"Non-finite logits detected "
                    f"at epoch={epoch}, step={step}"
                )


            # Loss calculation in FP32 for stability.
            per_sample_loss = (
                F.cross_entropy(
                    logits,
                    labels,
                    reduction="none",
                )
            )


            loss = (
                per_sample_loss
                * weights
            ).sum() / (
                weights.sum()
                + 1e-8
            )


            # -----------------------------------------------
            # HARD STOP ON NAN / INF
            # -----------------------------------------------

            if not torch.isfinite(
                loss
            ):

                print()
                print(
                    "NON-FINITE LOSS DETECTED"
                )

                print(
                    "Epoch:",
                    epoch
                )

                print(
                    "Step:",
                    step
                )

                print(
                    "Loss:",
                    loss
                )

                print(
                    "Logits finite:",
                    bool(
                        torch.isfinite(
                            logits
                        ).all()
                    )
                )

                raise RuntimeError(
                    "Training aborted because "
                    "loss became NaN/Inf."
                )


            (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            ).backward()


            should_update = (
                step
                % GRADIENT_ACCUMULATION_STEPS
                == 0

                or

                step
                == len(
                    train_loader
                )
            )


            if should_update:

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        model.parameters(),
                        MAX_GRAD_NORM,
                    )
                )


                if not torch.isfinite(
                    grad_norm
                ):

                    raise RuntimeError(
                        "Gradient norm became "
                        "NaN/Inf."
                    )


                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )


            running_loss += float(
                loss.detach().item()
            )

            seen_batches += 1


            if (
                step % 20 == 0
                or step
                == len(
                    train_loader
                )
            ):

                allocated = (
                    torch.cuda
                    .memory_allocated()
                    / 1024**3
                )

                reserved = (
                    torch.cuda
                    .memory_reserved()
                    / 1024**3
                )


                progress.set_postfix(
                    loss=(
                        running_loss
                        / seen_batches
                    ),

                    lr=(
                        scheduler
                        .get_last_lr()[0]
                    ),

                    gpu_GB=(
                        f"{allocated:.1f}"
                    ),

                    reserved_GB=(
                        f"{reserved:.1f}"
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


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

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
        ] = (
            training_loss
        )


        history.append(
            result
        )


        print_metrics(
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


        better = (
            bipia_f1
            > best_bipia_f1

            or

            (
                abs(
                    bipia_f1
                    - best_bipia_f1
                )
                < 1e-12

                and

                overall_f1
                > best_overall_f1
            )
        )


        if better:

            best_bipia_f1 = (
                bipia_f1
            )

            best_overall_f1 = (
                overall_f1
            )

            best_epoch = (
                epoch
            )


            BEST_MODEL_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )


            model.save_pretrained(
                BEST_MODEL_DIR,
                safe_serialization=True,
            )


            tokenizer.save_pretrained(
                BEST_MODEL_DIR
            )


            print()

            print(
                "Saved new best model:",
                BEST_MODEL_DIR
            )


    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "model_name": (
            MODEL_NAME
        ),

        "gpu": (
            gpu_name
        ),

        "gpu_memory_gb": (
            gpu_memory
        ),

        "precision": (
            "pure_float32"
        ),

        "epochs": (
            EPOCHS
        ),

        "train_batch_size": (
            TRAIN_BATCH_SIZE
        ),

        "eval_batch_size": (
            EVAL_BATCH_SIZE
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

        "history": (
            history
        ),

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
