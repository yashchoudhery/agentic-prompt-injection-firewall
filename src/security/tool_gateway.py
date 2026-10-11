from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "tool_policy.json"
)


class ToolSecurityGateway:
    """
    Deterministic authorization layer placed between
    the agent and every external tool.

    Important:
    this component AUTHORIZES tool calls.
    It does not execute them.
    """

    BLOCKING_FIREWALL_DECISIONS = {
        "BLOCK",
        "QUARANTINE",
    }

    VALID_TRUST_LEVELS = {
        "trusted",
        "unknown",
        "untrusted",
    }

    def __init__(
        self,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
    ):
        self.policy_path = Path(
            policy_path
        )

        with self.policy_path.open(
            encoding="utf-8"
        ) as f:
            self.policy = json.load(f)

        self.sensitive_names = {
            x.lower()
            for x
            in self.policy[
                "sensitive_argument_names"
            ]
        }

        self.side_effect_actions = {
            x.lower()
            for x
            in self.policy[
                "side_effect_actions"
            ]
        }

    # --------------------------------------------------------
    # Sensitive argument inspection
    # --------------------------------------------------------

    def _find_sensitive_keys(
        self,
        value: Any,
        prefix: str = "",
    ) -> list[str]:

        findings: list[str] = []

        if isinstance(
            value,
            dict,
        ):

            for key, child in value.items():

                key_text = str(
                    key
                ).lower()

                path = (
                    f"{prefix}.{key}"
                    if prefix
                    else str(key)
                )

                normalized = (
                    key_text
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                if any(
                    sensitive
                    in normalized
                    for sensitive
                    in self.sensitive_names
                ):
                    findings.append(
                        path
                    )

                findings.extend(
                    self._find_sensitive_keys(
                        child,
                        path,
                    )
                )

        elif isinstance(
            value,
            list,
        ):

            for i, child in enumerate(
                value
            ):

                findings.extend(
                    self._find_sensitive_keys(
                        child,
                        f"{prefix}[{i}]",
                    )
                )

        return findings

    # --------------------------------------------------------
    # Base policy
    # --------------------------------------------------------

    def _base_policy(
        self,
        tool_name: str,
        action: str,
    ) -> str:

        tools = self.policy.get(
            "tools",
            {},
        )

        tool = tools.get(
            tool_name,
        )

        if tool is None:
            return self.policy.get(
                "default_policy",
                "BLOCK",
            )

        return tool.get(
            action,
            self.policy.get(
                "default_policy",
                "BLOCK",
            ),
        )

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    def authorize(
        self,
        *,
        tool_name: str,
        action: str,
        arguments: dict[str, Any] | None = None,
        firewall_decision: str = "ALLOW",
        trust_level: str = "unknown",
        user_approved: bool = False,
    ) -> dict[str, Any]:

        tool_name = (
            str(tool_name)
            .strip()
            .lower()
        )

        action = (
            str(action)
            .strip()
            .lower()
        )

        firewall_decision = (
            str(firewall_decision)
            .strip()
            .upper()
        )

        trust_level = (
            str(trust_level)
            .strip()
            .lower()
        )

        arguments = (
            arguments
            if arguments is not None
            else {}
        )

        if (
            trust_level
            not in self.VALID_TRUST_LEVELS
        ):
            raise ValueError(
                "trust_level must be "
                "trusted, unknown, or untrusted"
            )

        reasons: list[str] = []

        # ----------------------------------------------------
        # Rule 1:
        # BLOCK/QUARANTINE content can never initiate tools.
        # ----------------------------------------------------

        if (
            firewall_decision
            in self.BLOCKING_FIREWALL_DECISIONS
        ):

            return {
                "authorization": "BLOCK",
                "authorized": False,
                "requires_approval": False,
                "tool_name": tool_name,
                "action": action,
                "reasons": [
                    "firewall_content_not_authorized"
                ],
            }

        # ----------------------------------------------------
        # Rule 2:
        # unknown tools/actions are denied by default.
        # ----------------------------------------------------

        base_policy = self._base_policy(
            tool_name,
            action,
        )

        if base_policy == "BLOCK":

            return {
                "authorization": "BLOCK",
                "authorized": False,
                "requires_approval": False,
                "tool_name": tool_name,
                "action": action,
                "reasons": [
                    "tool_or_action_not_allowlisted"
                ],
            }

        # ----------------------------------------------------
        # Rule 3:
        # inspect arguments for secrets.
        # ----------------------------------------------------

        sensitive_keys = (
            self._find_sensitive_keys(
                arguments
            )
        )

        if sensitive_keys:

            reasons.append(
                "sensitive_arguments_detected"
            )

            # Never automatically send/expose secrets.
            if (
                action
                in {
                    "send",
                    "submit",
                    "publish",
                    "upload",
                    "transfer",
                }
            ):

                return {
                    "authorization": "BLOCK",
                    "authorized": False,
                    "requires_approval": False,
                    "tool_name": tool_name,
                    "action": action,
                    "reasons": [
                        "potential_secret_exfiltration"
                    ],
                    "sensitive_argument_paths": (
                        sensitive_keys
                    ),
                }

        # ----------------------------------------------------
        # Rule 4:
        # untrusted content cannot autonomously cause
        # side effects.
        # ----------------------------------------------------

        if (
            trust_level == "untrusted"
            and action
            in self.side_effect_actions
        ):

            reasons.append(
                "untrusted_source_side_effect"
            )

            if not user_approved:

                return {
                    "authorization": "APPROVAL",
                    "authorized": False,
                    "requires_approval": True,
                    "tool_name": tool_name,
                    "action": action,
                    "reasons": reasons,
                    "sensitive_argument_paths": (
                        sensitive_keys
                    ),
                }

        # ----------------------------------------------------
        # Rule 5:
        # policy explicitly requires approval.
        # ----------------------------------------------------

        if (
            base_policy == "APPROVAL"
            and not user_approved
        ):

            reasons.append(
                "explicit_user_approval_required"
            )

            return {
                "authorization": "APPROVAL",
                "authorized": False,
                "requires_approval": True,
                "tool_name": tool_name,
                "action": action,
                "reasons": reasons,
                "sensitive_argument_paths": (
                    sensitive_keys
                ),
            }

        # ----------------------------------------------------
        # Approved
        # ----------------------------------------------------

        if user_approved:
            reasons.append(
                "explicit_user_approval_present"
            )

        if not reasons:
            reasons.append(
                "allowlisted_low_risk_action"
            )

        return {
            "authorization": "ALLOW",
            "authorized": True,
            "requires_approval": False,
            "tool_name": tool_name,
            "action": action,
            "reasons": reasons,
            "sensitive_argument_paths": (
                sensitive_keys
            ),
        }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tool",
        required=True,
    )

    parser.add_argument(
        "--action",
        required=True,
    )

    parser.add_argument(
        "--firewall-decision",
        default="ALLOW",
    )

    parser.add_argument(
        "--trust-level",
        default="unknown",
        choices=[
            "trusted",
            "unknown",
            "untrusted",
        ],
    )

    parser.add_argument(
        "--arguments",
        default="{}",
    )

    parser.add_argument(
        "--approved",
        action="store_true",
    )

    args = parser.parse_args()

    gateway = ToolSecurityGateway()

    arguments = json.loads(
        args.arguments
    )

    result = gateway.authorize(
        tool_name=args.tool,
        action=args.action,
        arguments=arguments,
        firewall_decision=(
            args.firewall_decision
        ),
        trust_level=args.trust_level,
        user_approved=args.approved,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
