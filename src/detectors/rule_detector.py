from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any


ATTACK_NAMES = {
    "A01": "instruction_override",
    "A02": "role_change",
    "A03": "secret_extraction",
    "A04": "tool_abuse",
    "A05": "credential_theft",
    "A06": "context_poisoning",
    "A07": "multi_step_jailbreak",
    "A08": "encoded_instruction",
    "A09": "indirect_prompt_injection",
}


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: str
    score: float


RULES: dict[str, list[Rule]] = {

    # --------------------------------------------------------
    # A01 — instruction override
    # --------------------------------------------------------
    "A01": [
        Rule(
            "ignore_previous",
            r"\bignore\s+(?:all\s+)?(?:previous|prior|earlier|above)\s+"
            r"(?:instructions?|prompts?|rules?|directions?)\b",
            1.00,
        ),
        Rule(
            "disregard_previous",
            r"\bdisregard\s+(?:all\s+)?(?:previous|prior|earlier|above)\s+"
            r"(?:instructions?|prompts?|rules?)\b",
            1.00,
        ),
        Rule(
            "override_instructions",
            r"\boverride\s+(?:the\s+)?(?:system|developer|previous|existing)?\s*"
            r"(?:instructions?|rules?|policy|policies)\b",
            0.95,
        ),
        Rule(
            "new_instructions",
            r"\b(?:your\s+)?new\s+(?:instructions?|rules?)\s+(?:are|is)\b",
            0.90,
        ),
        Rule(
            "do_not_follow",
            r"\bdo\s+not\s+follow\s+(?:the\s+)?(?:previous|above|system|developer)"
            r"\s+(?:instructions?|rules?)\b",
            1.00,
        ),
        Rule(
            "ignore_user_task",
            r"\b(?:ignore|disregard|override)\s+"
            r"(?:the\s+)?(?:user(?:'s)?|original|current)\s+"
            r"(?:task|request|goal|instruction|instructions)\b",
            1.00,
        ),
    ],

    # --------------------------------------------------------
    # A02 — role change
    # --------------------------------------------------------
    "A02": [
        Rule(
            "you_are_now",
            r"\byou\s+are\s+now\s+(?:a|an|the)\b",
            0.72,
        ),
        Rule(
            "act_as_unrestricted",
            r"\bact\s+as\s+(?:an?\s+)?(?:unrestricted|uncensored|unfiltered|"
            r"unlimited)\b",
            0.95,
        ),
        Rule(
            "developer_mode",
            r"\b(?:enable|enter|activate|switch\s+to)\s+developer\s+mode\b",
            0.95,
        ),
        Rule(
            "dan_jailbreak_role",
            r"\b(?:you\s+are|act\s+as|become)\s+(?:DAN|do\s+anything\s+now)\b",
            1.00,
        ),
    ],

    # --------------------------------------------------------
    # A03 — secret / prompt extraction
    # --------------------------------------------------------
    "A03": [
        Rule(
            "system_prompt",
            r"\b(?:reveal|show|print|display|output|repeat|expose|give\s+me)\b"
            r".{0,50}\b(?:system\s+prompt|system\s+message)\b",
            1.00,
        ),
        Rule(
            "hidden_instructions",
            r"\b(?:reveal|show|print|display|output|expose)\b"
            r".{0,60}\b(?:hidden|internal|private)\s+"
            r"(?:(?:system|developer)\s+)?instructions?\b",
            1.00,
        ),
        Rule(
            "developer_message",
            r"\b(?:reveal|show|print|output|expose)\b"
            r".{0,50}\bdeveloper\s+(?:message|instructions?|prompt)\b",
            1.00,
        ),
        Rule(
            "initial_prompt",
            r"\b(?:what|show|tell|reveal).{0,40}\b"
            r"(?:initial|original)\s+(?:prompt|instructions?)\b",
            0.90,
        ),
    ],

    # --------------------------------------------------------
    # A04 — unauthorized tool abuse
    # --------------------------------------------------------
    "A04": [
        Rule(
            "bypass_tool_permissions",
            r"\bbypass\s+(?:tool\s+)?(?:permissions?|authorization|restrictions?)\b",
            1.00,
        ),
        Rule(
            "unauthorized_tool",
            r"\b(?:execute|run|call|invoke|use)\b"
            r".{0,60}\b(?:unauthorized|without\s+(?:permission|approval))\b",
            0.95,
        ),
        Rule(
            "exfiltrate_with_tool",
            r"\b(?:use|call|invoke|run)\b.{0,50}\b(?:tool|shell|terminal|command|api)\b"
            r".{0,70}\b(?:exfiltrat|steal|send|upload)\w*\b",
            1.00,
        ),
    ],

    # --------------------------------------------------------
    # A05 — credentials / authentication secrets
    # --------------------------------------------------------
    "A05": [
        Rule(
            "credential_extraction",
            r"\b(?:reveal|show|print|extract|steal|send|expose|dump)\w*\b"
            r".{0,60}\b(?:passwords?|api[\s_-]?keys?|access[\s_-]?tokens?|"
            r"auth(?:entication)?[\s_-]?tokens?|credentials?|private[\s_-]?keys?)\b",
            1.00,
        ),
        Rule(
            "credential_files",
            r"\b(?:read|open|print|cat|dump|exfiltrat\w*)\b"
            r".{0,60}\b(?:\.env|credentials\.json|id_rsa|secret(?:s)?\.json)\b",
            0.95,
        ),
    ],

    # --------------------------------------------------------
    # A06 — context / memory poisoning
    # --------------------------------------------------------
    "A06": [
        Rule(
            "poison_memory",
            r"\b(?:store|save|write|add|insert|remember)\b"
            r".{0,50}\b(?:memory|context)\b"
            r".{0,80}\b(?:instead|override|replace|ignore|future)\b",
            0.85,
        ),
        Rule(
            "replace_context",
            r"\b(?:replace|overwrite|modify|poison)\b"
            r".{0,30}\b(?:memory|context|conversation\s+history)\b",
            0.95,
        ),
        Rule(
            "future_instruction",
            r"\bfrom\s+now\s+on\b.{0,80}\b"
            r"(?:ignore|override|always|must)\b",
            0.75,
        ),
    ],

    # --------------------------------------------------------
    # A07 — jailbreak / staged bypass
    # --------------------------------------------------------
    "A07": [
        Rule(
            "jailbreak_keyword",
            r"\b(?:jailbreak|jailbroken)\b",
            0.90,
        ),
        Rule(
            "no_restrictions",
            r"\b(?:without|ignore|remove|bypass)\s+"
            r"(?:any\s+)?(?:restrictions?|safety\s+rules?|guardrails?|limitations?)\b",
            0.95,
        ),
        Rule(
            "hypothetical_bypass",
            r"\b(?:hypothetical|fictional|roleplay)\b"
            r".{0,80}\b(?:ignore|bypass|no\s+restrictions?|unrestricted)\b",
            0.80,
        ),
        Rule(
            "split_attack",
            r"\b(?:split|divide|break)\b.{0,50}\b"
            r"(?:instructions?|request|task)\b.{0,60}\b"
            r"(?:filters?|detector|guardrail|safety)\b",
            0.90,
        ),
    ],

    # --------------------------------------------------------
    # A08 — encoded / obfuscated instructions
    # --------------------------------------------------------
    "A08": [
        Rule(
            "decode_execute",
            r"\b(?:decode|decrypt|deobfuscate)\b"
            r".{0,80}\b(?:execute|follow|obey|run|instruction|command)\w*\b",
            0.95,
        ),
        Rule(
            "base64_instruction",
            r"\bbase64\b.{0,80}\b(?:decode|execute|follow|instruction|command)\w*\b",
            0.90,
        ),
        Rule(
            "hex_instruction",
            r"\b(?:hex|hexadecimal)\b.{0,80}\b"
            r"(?:decode|execute|follow|instruction|command)\w*\b",
            0.90,
        ),
    ],
}


BASE64_BLOB = re.compile(
    r"(?<![A-Za-z0-9+/])"
    r"(?:[A-Za-z0-9+/]{4}){10,}"
    r"(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"
    r"(?![A-Za-z0-9+/])"
)

HEX_BLOB = re.compile(
    r"(?<![A-Fa-f0-9])"
    r"(?:[A-Fa-f0-9]{2}){20,}"
    r"(?![A-Fa-f0-9])"
)


class PromptInjectionRuleDetector:

    def __init__(self):
        self.compiled: dict[
            str,
            list[tuple[Rule, re.Pattern[str]]]
        ] = {}

        for attack_id, rules in RULES.items():

            self.compiled[attack_id] = [
                (
                    rule,
                    re.compile(
                        rule.pattern,
                        re.IGNORECASE | re.DOTALL,
                    ),
                )
                for rule in rules
            ]

    @staticmethod
    def _snippet(
        text: str,
        start: int,
        end: int,
        radius: int = 50,
    ) -> str:

        left = max(
            0,
            start - radius,
        )

        right = min(
            len(text),
            end + radius,
        )

        return (
            text[left:right]
            .replace("\n", " ")
            .strip()
        )

    @staticmethod
    def _looks_decodable_base64(
        value: str,
    ) -> bool:

        try:
            padding = (
                4 - len(value) % 4
            ) % 4

            decoded = base64.b64decode(
                value + "=" * padding,
                validate=True,
            )

            if not decoded:
                return False

            printable = sum(
                32 <= b <= 126
                or b in (9, 10, 13)
                for b in decoded
            )

            return (
                printable / len(decoded)
                >= 0.75
            )

        except (
            ValueError,
            binascii.Error,
        ):
            return False

    def analyze(
        self,
        text: str,
        input_channel: str = "user_message",
        source_type: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:

        text = str(
            text or ""
        )

        if not text.strip():

            return {
                "is_suspicious": False,
                "rule_score": 0.0,
                "attack_types": [],
                "matches": [],
                "input_channel": input_channel,
                "source_type": source_type,
            }

        matches = []
        category_scores = {
            attack_id: 0.0
            for attack_id
            in ATTACK_NAMES
        }

        # ----------------------------------------------------
        # Regex rules A01–A08
        # ----------------------------------------------------

        for (
            attack_id,
            rules,
        ) in self.compiled.items():

            for rule, pattern in rules:

                match = pattern.search(
                    text
                )

                if not match:
                    continue

                category_scores[
                    attack_id
                ] = max(
                    category_scores[
                        attack_id
                    ],
                    rule.score,
                )

                matches.append(
                    {
                        "attack_id": attack_id,
                        "attack_type": ATTACK_NAMES[
                            attack_id
                        ],
                        "rule": rule.name,
                        "score": rule.score,
                        "snippet": self._snippet(
                            text,
                            match.start(),
                            match.end(),
                        ),
                    }
                )

        # ----------------------------------------------------
        # A08 — actual encoded blobs
        # ----------------------------------------------------

        for match in BASE64_BLOB.finditer(
            text
        ):

            blob = match.group(0)

            if self._looks_decodable_base64(
                blob
            ):

                score = 0.60

                category_scores[
                    "A08"
                ] = max(
                    category_scores[
                        "A08"
                    ],
                    score,
                )

                matches.append(
                    {
                        "attack_id": "A08",
                        "attack_type": ATTACK_NAMES[
                            "A08"
                        ],
                        "rule": "base64_blob",
                        "score": score,
                        "snippet": self._snippet(
                            text,
                            match.start(),
                            match.end(),
                        ),
                    }
                )

                break

        if HEX_BLOB.search(
            text
        ):

            category_scores[
                "A08"
            ] = max(
                category_scores[
                    "A08"
                ],
                0.55,
            )

            match = HEX_BLOB.search(
                text
            )

            matches.append(
                {
                    "attack_id": "A08",
                    "attack_type": ATTACK_NAMES[
                        "A08"
                    ],
                    "rule": "hex_blob",
                    "score": 0.55,
                    "snippet": self._snippet(
                        text,
                        match.start(),
                        match.end(),
                    ),
                }
            )

        # ----------------------------------------------------
        # A09 — indirect prompt injection
        #
        # Only meaningful when content comes from an
        # untrusted/external source.
        # ----------------------------------------------------

        non_a09_scores = [
            score
            for attack_id, score
            in category_scores.items()
            if attack_id != "A09"
        ]

        strongest_instruction_signal = max(
            non_a09_scores,
            default=0.0,
        )

        if (
            input_channel
            == "external_content"
            and strongest_instruction_signal
            >= 0.70
        ):

            a09_score = max(
                0.85,
                strongest_instruction_signal,
            )

            category_scores[
                "A09"
            ] = min(
                a09_score,
                1.0,
            )

            matches.append(
                {
                    "attack_id": "A09",
                    "attack_type": ATTACK_NAMES[
                        "A09"
                    ],
                    "rule": (
                        "instruction_signal_"
                        "inside_external_content"
                    ),
                    "score": category_scores[
                        "A09"
                    ],
                    "snippet": (
                        "External content contains "
                        "instruction-like attack signals."
                    ),
                }
            )

        # ----------------------------------------------------
        # Final rule score
        # ----------------------------------------------------

        rule_score = max(
            category_scores.values(),
            default=0.0,
        )

        attack_types = [
            {
                "attack_id": attack_id,
                "attack_type": ATTACK_NAMES[
                    attack_id
                ],
                "score": round(
                    score,
                    4,
                ),
            }
            for attack_id, score
            in category_scores.items()
            if score > 0
        ]

        result = {
            "is_suspicious": bool(
                rule_score >= 0.70
            ),

            "rule_score": round(
                rule_score,
                4,
            ),

            "attack_types": (
                attack_types
            ),

            "input_channel": (
                input_channel
            ),

            "source_type": (
                source_type
            ),

            "match_count": len(
                matches
            ),
        }

        if debug:
            result[
                "matches"
            ] = matches

            result[
                "category_scores"
            ] = {
                attack_id: round(
                    score,
                    4,
                )
                for attack_id, score
                in category_scores.items()
            }

        return result


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Deterministic prompt-injection "
            "rule detector"
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
        "--source-type",
        default=None,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    detector = (
        PromptInjectionRuleDetector()
    )

    result = detector.analyze(
        text=args.text,
        input_channel=args.channel,
        source_type=args.source_type,
        debug=args.debug,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
