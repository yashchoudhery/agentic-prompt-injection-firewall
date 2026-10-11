from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "deberta_v3_small_window_v1"
    / "best"
)

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "deberta_firewall.json"
)

# Must match the representation used to create
# the transformer training windows.
HEADER_TOKEN_BUDGET = 96


class DebertaPromptInjectionDetector:
    """
    Binary prompt-injection detector.

    Runtime-safe inputs:
      - user_message:
            direct user-supplied prompt/message

      - external_content:
            untrusted content retrieved from email, web,
            PDF, HTML, code, API, OCR, tables, etc.

    The detector never receives training-only fields such as:
      - attack_payload
      - attack_position
      - attack_types
      - poisoned_content
      - severity
    """

    VALID_CHANNELS = {
        "user_message",
        "external_content",
    }

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str = "auto",
        batch_size: int = 8,
    ):
        self.model_dir = Path(model_dir)
        self.config_path = Path(config_path)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {self.config_path}"
            )

        with self.config_path.open(
            encoding="utf-8"
        ) as f:
            self.config = json.load(f)

        self.threshold = float(
            self.config["threshold"]
        )

        self.high_confidence_threshold = float(
            self.config[
                "high_confidence_threshold"
            ]
        )

        self.max_length = int(
            self.config.get(
                "max_length",
                512,
            )
        )

        self.stride = int(
            self.config.get(
                "stride",
                128,
            )
        )

        self.batch_size = int(
            batch_size
        )

        self.device = self._resolve_device(
            device
        )

        print(
            f"[DeBERTa Firewall] Loading tokenizer: "
            f"{self.model_dir}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
        )

        print(
            f"[DeBERTa Firewall] Loading model on "
            f"{self.device}"
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.model_dir,
                local_files_only=True,
                dtype=torch.float32,
            )
        )

        self.model.to(
            device=self.device,
            dtype=torch.float32,
        )

        self.model.eval()

        print(
            "[DeBERTa Firewall] READY"
        )

        print(
            "[DeBERTa Firewall] "
            "channel-aware thresholds enabled"
        )

        for channel, thresholds in (
            self.config
            .get(
                "channel_thresholds",
                {}
            )
            .items()
        ):
            print(
                f"[DeBERTa Firewall] "
                f"{channel}: "
                f"review="
                f"{thresholds['review_threshold']} "
                f"high_risk="
                f"{thresholds['high_risk_threshold']}"
            )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    @staticmethod
    def _resolve_device(
        requested: str,
    ) -> torch.device:

        requested = requested.lower()

        if requested == "cpu":
            return torch.device("cpu")

        if requested == "mps":

            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS requested but unavailable."
                )

            return torch.device("mps")

        if requested == "cuda":

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but unavailable."
                )

            return torch.device("cuda:0")

        if requested != "auto":
            raise ValueError(
                "device must be one of: "
                "auto, cpu, mps, cuda"
            )

        if torch.cuda.is_available():
            return torch.device("cuda:0")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    # --------------------------------------------------------
    # Runtime-safe representation
    # --------------------------------------------------------

    def _build_header(
        self,
        input_channel: str,
        user_instruction: str | None,
    ) -> str:

        if (
            input_channel
            not in self.VALID_CHANNELS
        ):
            raise ValueError(
                f"Invalid input_channel={input_channel}. "
                f"Expected one of "
                f"{sorted(self.VALID_CHANNELS)}"
            )

        if input_channel == "user_message":

            # Important:
            # Direct messages intentionally do not include
            # user_instruction because that would recreate
            # the metadata leakage discovered earlier.
            header = (
                "[INPUT_CHANNEL]\n"
                "USER_MESSAGE"
            )

        else:

            task = (
                str(user_instruction).strip()
                if user_instruction
                else "<UNKNOWN>"
            )

            header = (
                "[INPUT_CHANNEL]\n"
                "EXTERNAL_CONTENT\n\n"
                "[USER_TASK]\n"
                f"{task}"
            )

        # Match training preprocessing exactly.
        header_ids = self.tokenizer(
            header,
            add_special_tokens=False,
            truncation=True,
            max_length=HEADER_TOKEN_BUDGET,
        )["input_ids"]

        return self.tokenizer.decode(
            header_ids,
            skip_special_tokens=True,
        )

    # --------------------------------------------------------
    # Windowing
    # --------------------------------------------------------

    def _make_windows(
        self,
        text: str,
        input_channel: str,
        user_instruction: str | None,
    ) -> list[dict[str, Any]]:

        content = str(
            text or ""
        ).strip()

        if not content:
            raise ValueError(
                "text cannot be empty"
            )

        header = self._build_header(
            input_channel=input_channel,
            user_instruction=user_instruction,
        )

        encoded = self.tokenizer(
            header,
            content,
            truncation="only_second",
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )

        windows = []

        total = len(
            encoded["input_ids"]
        )

        for i in range(total):

            item = {
                "input_ids": encoded[
                    "input_ids"
                ][i],

                "attention_mask": encoded[
                    "attention_mask"
                ][i],
            }

            if (
                "token_type_ids"
                in encoded
            ):
                item[
                    "token_type_ids"
                ] = encoded[
                    "token_type_ids"
                ][i]

            windows.append(item)

        return windows

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------

    def _score_windows_once(
        self,
        windows: list[dict[str, Any]],
    ) -> list[float]:

        scores = []

        with torch.inference_mode():

            for start in range(
                0,
                len(windows),
                self.batch_size,
            ):

                batch_windows = windows[
                    start:
                    start + self.batch_size
                ]

                batch = self.tokenizer.pad(
                    batch_windows,
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )

                model_inputs = {
                    key: value.to(
                        self.device
                    )
                    for key, value
                    in batch.items()
                }

                logits = self.model(
                    **model_inputs
                ).logits

                if not torch.isfinite(
                    logits
                ).all():

                    raise RuntimeError(
                        "Model produced "
                        "non-finite logits."
                    )

                probabilities = (
                    torch.softmax(
                        logits,
                        dim=-1,
                    )[:, 1]
                    .detach()
                    .cpu()
                    .tolist()
                )

                scores.extend(
                    float(p)
                    for p
                    in probabilities
                )

        return scores

    def _score_windows(
        self,
        windows: list[dict[str, Any]],
    ) -> list[float]:

        try:

            return self._score_windows_once(
                windows
            )

        except Exception as exc:

            # The original training instability was on
            # Apple MPS. Inference normally works, but for
            # reliability we fall back to CPU automatically
            # if MPS ever produces an error/non-finite result.
            if self.device.type != "mps":
                raise

            print(
                "[DeBERTa Firewall] "
                "MPS inference failed; "
                "falling back to CPU."
            )

            print(
                "[DeBERTa Firewall] "
                f"Reason: {exc}"
            )

            self.device = torch.device(
                "cpu"
            )

            self.model.to(
                device=self.device,
                dtype=torch.float32,
            )

            self.model.eval()

            return self._score_windows_once(
                windows
            )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    def _decision(
        self,
        score: float,
        input_channel: str,
    ) -> tuple[str, str]:

        channel_cfg = (
            self.config
            .get(
                "channel_thresholds",
                {}
            )
            .get(
                input_channel,
                {}
            )
        )

        review_threshold = float(
            channel_cfg.get(
                "review_threshold",
                self.threshold,
            )
        )

        high_threshold = float(
            channel_cfg.get(
                "high_risk_threshold",
                self.high_confidence_threshold,
            )
        )

        if score < review_threshold:

            return (
                "ALLOW",
                "LOW",
            )

        if score < high_threshold:

            return (
                "SECURITY_REVIEW",
                "MEDIUM",
            )

        return (
            "HIGH_RISK",
            "HIGH",
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def analyze(
        self,
        text: str,
        input_channel: str = "user_message",
        user_instruction: str | None = None,
        source_type: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:

        windows = self._make_windows(
            text=text,
            input_channel=input_channel,
            user_instruction=user_instruction,
        )

        scores = self._score_windows(
            windows
        )

        if not scores:
            raise RuntimeError(
                "No model windows were generated."
            )

        attack_probability = float(
            max(scores)
        )

        highest_window = int(
            max(
                range(len(scores)),
                key=lambda i: scores[i],
            )
        )

        decision, risk_level = (
            self._decision(
                attack_probability,
                input_channel,
            )
        )

        result = {
            "is_suspicious": bool(
                decision
                != "ALLOW"
            ),

            "attack_probability": round(
                attack_probability,
                6,
            ),

            "risk_level": (
                risk_level
            ),

            "decision": (
                decision
            ),

            "threshold": float(
                self.config
                .get(
                    "channel_thresholds",
                    {}
                )
                .get(
                    input_channel,
                    {}
                )
                .get(
                    "review_threshold",
                    self.threshold,
                )
            ),

            "high_confidence_threshold": float(
                self.config
                .get(
                    "channel_thresholds",
                    {}
                )
                .get(
                    input_channel,
                    {}
                )
                .get(
                    "high_risk_threshold",
                    self.high_confidence_threshold,
                )
            ),

            "input_channel": (
                input_channel
            ),

            "source_type": (
                source_type
            ),

            "window_count": (
                len(scores)
            ),

            "highest_risk_window": (
                highest_window
            ),

            "device": str(
                self.device
            ),

            "model": self.config[
                "model_name"
            ],
        }

        if debug:

            result[
                "window_scores"
            ] = [
                round(
                    score,
                    6,
                )
                for score
                in scores
            ]

        return result


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Local DeBERTa prompt-injection detector"
        )
    )

    parser.add_argument(
        "--text",
        required=True,
    )

    parser.add_argument(
        "--channel",
        default="user_message",
        choices=[
            "user_message",
            "external_content",
        ],
    )

    parser.add_argument(
        "--user-task",
        default=None,
    )

    parser.add_argument(
        "--source-type",
        default=None,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    detector = (
        DebertaPromptInjectionDetector(
            device=args.device
        )
    )

    result = detector.analyze(
        text=args.text,
        input_channel=args.channel,
        user_instruction=args.user_task,
        source_type=args.source_type,
        debug=args.debug,
    )

    print()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
