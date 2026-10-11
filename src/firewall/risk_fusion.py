from __future__ import annotations

import argparse
import json
from typing import Any

from src.detectors.deberta_detector import (
    DebertaPromptInjectionDetector,
)
from src.detectors.rule_detector import (
    PromptInjectionRuleDetector,
)


class PromptInjectionRiskFusion:
    """
    Combines:

      1. DeBERTa semantic detector
      2. Deterministic A01-A09 rules
      3. Source/provenance trust

    Final policy actions:
      ALLOW
      MONITOR
      SANITIZE
      QUARANTINE
      APPROVAL
      BLOCK
    """

    VALID_TRUST_LEVELS = {
        "trusted",
        "unknown",
        "untrusted",
    }

    def __init__(
        self,
        device: str = "auto",
    ):
        print(
            "[Risk Fusion] Initializing detectors..."
        )

        self.neural = (
            DebertaPromptInjectionDetector(
                device=device
            )
        )

        self.rules = (
            PromptInjectionRuleDetector()
        )

        print(
            "[Risk Fusion] READY"
        )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    @staticmethod
    def _default_trust(
        input_channel: str,
    ) -> str:

        if input_channel == "external_content":
            return "untrusted"

        return "unknown"

    @staticmethod
    def _provenance_score(
        trust_level: str,
    ) -> float:

        return {
            "trusted": 0.0,
            "unknown": 0.5,
            "untrusted": 1.0,
        }[trust_level]

    # --------------------------------------------------------
    # Neural signal normalization
    # --------------------------------------------------------

    @staticmethod
    def _neural_signal(
        neural_decision: str,
    ) -> float:

        return {
            "ALLOW": 0.0,
            "SECURITY_REVIEW": 0.5,
            "HIGH_RISK": 1.0,
        }.get(
            neural_decision,
            0.0,
        )

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    def _final_policy(
        self,
        *,
        input_channel: str,
        neural_result: dict[str, Any],
        rule_result: dict[str, Any],
    ) -> tuple[str, str, list[str]]:

        reasons: list[str] = []

        neural_decision = neural_result[
            "decision"
        ]

        neural_probability = float(
            neural_result[
                "attack_probability"
            ]
        )

        rule_score = float(
            rule_result[
                "rule_score"
            ]
        )

        attack_ids = {
            item["attack_id"]
            for item
            in rule_result[
                "attack_types"
            ]
        }

        # ----------------------------------------------------
        # Tool abuse / credential extraction
        # ----------------------------------------------------

        if (
            {"A04", "A05"}
            & attack_ids
        ):

            reasons.append(
                "tool_or_credential_security_signal"
            )

            if (
                neural_decision
                == "HIGH_RISK"
            ):
                reasons.append(
                    "neural_high_risk_confirmation"
                )

                return (
                    "BLOCK",
                    "CRITICAL",
                    reasons,
                )

            return (
                "APPROVAL",
                "HIGH",
                reasons,
            )

        # ----------------------------------------------------
        # Strong indirect prompt injection
        # ----------------------------------------------------

        if (
            input_channel
            == "external_content"
            and "A09" in attack_ids
            and rule_score >= 0.85
        ):

            reasons.append(
                "indirect_prompt_injection"
            )

            if (
                neural_decision
                == "HIGH_RISK"
            ):
                reasons.append(
                    "neural_high_risk_confirmation"
                )

            return (
                "QUARANTINE",
                "HIGH",
                reasons,
            )

        # ----------------------------------------------------
        # Both detectors strongly agree
        # ----------------------------------------------------

        if (
            neural_decision
            == "HIGH_RISK"
            and rule_score >= 0.70
        ):

            reasons.extend(
                [
                    "neural_high_risk",
                    "rule_detector_confirmation",
                ]
            )

            return (
                "BLOCK",
                "CRITICAL",
                reasons,
            )

        # ----------------------------------------------------
        # Neural-only high risk
        # ----------------------------------------------------

        if (
            neural_decision
            == "HIGH_RISK"
        ):

            reasons.append(
                "neural_high_risk_without_rule_confirmation"
            )

            return (
                "QUARANTINE",
                "HIGH",
                reasons,
            )

        # ----------------------------------------------------
        # Strong rule signal without neural confirmation
        # ----------------------------------------------------

        if rule_score >= 0.70:

            reasons.append(
                "strong_rule_signal"
            )

            if (
                input_channel
                == "external_content"
            ):

                return (
                    "SANITIZE",
                    "MEDIUM",
                    reasons,
                )

            return (
                "MONITOR",
                "MEDIUM",
                reasons,
            )

        # ----------------------------------------------------
        # Neural uncertainty / review region
        # ----------------------------------------------------

        if (
            neural_decision
            == "SECURITY_REVIEW"
        ):

            reasons.append(
                "neural_uncertainty"
            )

            return (
                "MONITOR",
                "MEDIUM",
                reasons,
            )

        # ----------------------------------------------------
        # Low risk
        # ----------------------------------------------------

        if (
            neural_probability
            > 0
        ):
            reasons.append(
                "below_operational_threshold"
            )

        return (
            "ALLOW",
            "LOW",
            reasons,
        )

    # --------------------------------------------------------
    # Public analyze
    # --------------------------------------------------------

    def analyze(
        self,
        text: str,
        *,
        input_channel: str = "user_message",
        user_instruction: str | None = None,
        source_type: str | None = None,
        trust_level: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:

        if trust_level is None:
            trust_level = (
                self._default_trust(
                    input_channel
                )
            )

        if (
            trust_level
            not in self.VALID_TRUST_LEVELS
        ):
            raise ValueError(
                "trust_level must be one of: "
                "trusted, unknown, untrusted"
            )

        # ----------------------------------------------------
        # Run both detectors
        # ----------------------------------------------------

        neural = self.neural.analyze(
            text=text,
            input_channel=input_channel,
            user_instruction=user_instruction,
            source_type=source_type,
            debug=debug,
        )

        rules = self.rules.analyze(
            text=text,
            input_channel=input_channel,
            source_type=source_type,
            debug=debug,
        )

        # ----------------------------------------------------
        # Fusion score
        #
        # We use normalized detector state rather than raw
        # probability alone because operational thresholds
        # differ for user_message vs external_content.
        # ----------------------------------------------------

        neural_signal = self._neural_signal(
            neural[
                "decision"
            ]
        )

        rule_signal = float(
            rules[
                "rule_score"
            ]
        )

        provenance_signal = (
            self._provenance_score(
                trust_level
            )
        )

        fused_score = (
            0.55 * neural_signal
            + 0.35 * rule_signal
            + 0.10 * provenance_signal
        )

        fused_score = min(
            max(
                fused_score,
                0.0,
            ),
            1.0,
        )

        action, risk_level, reasons = (
            self._final_policy(
                input_channel=input_channel,
                neural_result=neural,
                rule_result=rules,
            )
        )

        attack_types = [
            item["attack_type"]
            for item
            in rules[
                "attack_types"
            ]
        ]

        attack_ids = [
            item["attack_id"]
            for item
            in rules[
                "attack_types"
            ]
        ]

        safe_to_forward = (
            action == "ALLOW"
        )

        result = {
            "decision": action,

            "risk_level": risk_level,

            "fused_risk_score": round(
                fused_score,
                6,
            ),

            "safe_to_forward": (
                safe_to_forward
            ),

            "requires_intervention": (
                not safe_to_forward
            ),

            "attack_probability": (
                neural[
                    "attack_probability"
                ]
            ),

            "neural_decision": (
                neural[
                    "decision"
                ]
            ),

            "rule_score": (
                rules[
                    "rule_score"
                ]
            ),

            "attack_ids": (
                attack_ids
            ),

            "attack_types": (
                attack_types
            ),

            "policy_reasons": (
                reasons
            ),

            "provenance": {
                "input_channel": (
                    input_channel
                ),
                "source_type": (
                    source_type
                ),
                "trust_level": (
                    trust_level
                ),
            },

            "model": neural[
                "model"
            ],
        }

        if debug:

            result[
                "detectors"
            ] = {
                "neural": neural,
                "rules": rules,
            }

            result[
                "fusion_components"
            ] = {
                "neural_signal": round(
                    neural_signal,
                    4,
                ),

                "rule_signal": round(
                    rule_signal,
                    4,
                ),

                "provenance_signal": round(
                    provenance_signal,
                    4,
                ),

                "weights": {
                    "neural": 0.55,
                    "rules": 0.35,
                    "provenance": 0.10,
                },
            }

        return result


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prompt Injection Firewall "
            "risk-fusion engine"
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
        "--trust-level",
        default=None,
        choices=[
            "trusted",
            "unknown",
            "untrusted",
        ],
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

    engine = (
        PromptInjectionRiskFusion(
            device=args.device
        )
    )

    result = engine.analyze(
        text=args.text,
        input_channel=args.channel,
        user_instruction=args.user_task,
        source_type=args.source_type,
        trust_level=args.trust_level,
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
