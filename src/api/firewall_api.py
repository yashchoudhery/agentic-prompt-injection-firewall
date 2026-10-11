from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.firewall.risk_fusion import (
    PromptInjectionRiskFusion,
)
from src.security.tool_gateway import (
    ToolSecurityGateway,
)


# ============================================================
# Global runtime
# ============================================================

ENGINE: PromptInjectionRiskFusion | None = None
TOOL_GATEWAY: ToolSecurityGateway | None = None

START_TIME = time.time()


# ============================================================
# Analyze schemas
# ============================================================

class AnalyzeRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )

    input_channel: Literal[
        "user_message",
        "external_content",
    ] = "user_message"

    user_instruction: str | None = None

    source_type: str | None = None

    trust_level: Literal[
        "trusted",
        "unknown",
        "untrusted",
    ] | None = None

    debug: bool = False


# ============================================================
# Tool authorization schemas
# ============================================================

class ToolAuthorizationRequest(BaseModel):

    tool_name: str = Field(
        ...,
        min_length=1,
    )

    action: str = Field(
        ...,
        min_length=1,
    )

    arguments: dict[str, Any] = Field(
        default_factory=dict,
    )

    firewall_decision: str = "ALLOW"

    trust_level: Literal[
        "trusted",
        "unknown",
        "untrusted",
    ] = "unknown"

    user_approved: bool = False


# ============================================================
# End-to-end secure action
# ============================================================

class SecureActionRequest(BaseModel):

    # Original content
    text: str = Field(
        ...,
        min_length=1,
    )

    input_channel: Literal[
        "user_message",
        "external_content",
    ] = "user_message"

    user_instruction: str | None = None

    source_type: str | None = None

    trust_level: Literal[
        "trusted",
        "unknown",
        "untrusted",
    ] | None = None

    # Proposed agent tool call
    tool_name: str = Field(
        ...,
        min_length=1,
    )

    action: str = Field(
        ...,
        min_length=1,
    )

    arguments: dict[str, Any] = Field(
        default_factory=dict,
    )

    user_approved: bool = False

    debug: bool = False


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global ENGINE
    global TOOL_GATEWAY

    device = os.getenv(
        "FIREWALL_DEVICE",
        "auto",
    )

    print(
        "[Firewall API] "
        f"Starting with device={device}"
    )

    ENGINE = (
        PromptInjectionRiskFusion(
            device=device
        )
    )

    TOOL_GATEWAY = (
        ToolSecurityGateway()
    )

    print(
        "[Firewall API] "
        "Tool Security Gateway READY"
    )

    print(
        "[Firewall API] READY"
    )

    yield

    print(
        "[Firewall API] shutting down"
    )


# ============================================================
# App
# ============================================================

app = FastAPI(
    title=(
        "Agentic Prompt Injection Firewall"
    ),
    description=(
        "Defense-in-depth security gateway for "
        "LLMs and autonomous agents."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    ready = (
        ENGINE is not None
        and TOOL_GATEWAY is not None
    )

    return {
        "status": (
            "ready"
            if ready
            else "starting"
        ),

        "service": (
            "prompt-injection-firewall"
        ),

        "version": "1.1.0",

        "model": (
            "deberta_v3_small_window_v1"
        ),

        "components": {
            "neural_detector": (
                "ready"
                if ENGINE is not None
                else "starting"
            ),

            "rule_detector": (
                "ready"
                if ENGINE is not None
                else "starting"
            ),

            "risk_fusion": (
                "ready"
                if ENGINE is not None
                else "starting"
            ),

            "tool_gateway": (
                "ready"
                if TOOL_GATEWAY is not None
                else "starting"
            ),
        },

        "uptime_seconds": round(
            time.time()
            - START_TIME,
            2,
        ),
    }


# ============================================================
# Firewall analysis
# ============================================================

@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
):

    if ENGINE is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Firewall engine "
                "is not ready."
            ),
        )

    start = time.perf_counter()

    try:

        result = ENGINE.analyze(
            text=request.text,
            input_channel=(
                request.input_channel
            ),
            user_instruction=(
                request.user_instruction
            ),
            source_type=(
                request.source_type
            ),
            trust_level=(
                request.trust_level
            ),
            debug=request.debug,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Firewall analysis failed: "
                f"{exc}"
            ),
        ) from exc

    result["latency_ms"] = round(
        (
            time.perf_counter()
            - start
        )
        * 1000,
        3,
    )

    return result


# ============================================================
# Standalone tool authorization
# ============================================================

@app.post("/authorize-tool")
def authorize_tool(
    request: ToolAuthorizationRequest,
):

    if TOOL_GATEWAY is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Tool Security Gateway "
                "is not ready."
            ),
        )

    try:

        return TOOL_GATEWAY.authorize(
            tool_name=(
                request.tool_name
            ),
            action=(
                request.action
            ),
            arguments=(
                request.arguments
            ),
            firewall_decision=(
                request.firewall_decision
            ),
            trust_level=(
                request.trust_level
            ),
            user_approved=(
                request.user_approved
            ),
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


# ============================================================
# End-to-end secure agent action
# ============================================================

@app.post("/secure-action")
def secure_action(
    request: SecureActionRequest,
):

    if (
        ENGINE is None
        or TOOL_GATEWAY is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Security pipeline "
                "is not ready."
            ),
        )

    start = time.perf_counter()

    # --------------------------------------------------------
    # 1. Inspect the originating content
    # --------------------------------------------------------

    firewall = ENGINE.analyze(
        text=request.text,
        input_channel=(
            request.input_channel
        ),
        user_instruction=(
            request.user_instruction
        ),
        source_type=(
            request.source_type
        ),
        trust_level=(
            request.trust_level
        ),
        debug=request.debug,
    )

    effective_trust = (
        request.trust_level
        or firewall[
            "provenance"
        ][
            "trust_level"
        ]
    )

    # --------------------------------------------------------
    # 2. Independently authorize the proposed tool call
    # --------------------------------------------------------

    tool = TOOL_GATEWAY.authorize(
        tool_name=request.tool_name,
        action=request.action,
        arguments=request.arguments,
        firewall_decision=(
            firewall[
                "decision"
            ]
        ),
        trust_level=(
            effective_trust
        ),
        user_approved=(
            request.user_approved
        ),
    )

    # --------------------------------------------------------
    # 3. Final execution permission
    # --------------------------------------------------------

    execute_allowed = bool(
        firewall[
            "safe_to_forward"
        ]
        and tool[
            "authorized"
        ]
    )

    if execute_allowed:

        final_decision = (
            "EXECUTION_ALLOWED"
        )

    elif (
        tool.get(
            "requires_approval",
            False,
        )
    ):

        final_decision = (
            "USER_APPROVAL_REQUIRED"
        )

    else:

        final_decision = (
            "EXECUTION_BLOCKED"
        )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    return {
        "final_decision": (
            final_decision
        ),

        "execute_allowed": (
            execute_allowed
        ),

        "firewall": (
            firewall
        ),

        "tool_authorization": (
            tool
        ),

        "security_invariant": (
            "Tool execution requires both "
            "firewall approval and deterministic "
            "tool authorization."
        ),

        "latency_ms": round(
            latency_ms,
            3,
        ),
    }


# ============================================================
# Policy information
# ============================================================

@app.get("/policy")
def policy():

    return {
        "firewall_actions": [
            "ALLOW",
            "MONITOR",
            "SANITIZE",
            "QUARANTINE",
            "APPROVAL",
            "BLOCK",
        ],

        "tool_actions": [
            "ALLOW",
            "APPROVAL",
            "BLOCK",
        ],

        "attack_taxonomy": {
            "A01": (
                "instruction_override"
            ),
            "A02": (
                "role_change"
            ),
            "A03": (
                "secret_extraction"
            ),
            "A04": (
                "tool_abuse"
            ),
            "A05": (
                "credential_theft"
            ),
            "A06": (
                "context_poisoning"
            ),
            "A07": (
                "multi_step_jailbreak"
            ),
            "A08": (
                "encoded_instruction"
            ),
            "A09": (
                "indirect_prompt_injection"
            ),
        },

        "security_principles": [
            (
                "Untrusted content is never "
                "treated as authoritative instructions."
            ),
            (
                "Detection alone never authorizes "
                "a tool call."
            ),
            (
                "Unknown tools and actions are "
                "denied by default."
            ),
            (
                "Side-effecting actions can require "
                "explicit user approval."
            ),
            (
                "Potential secret exfiltration is "
                "blocked independently of detector output."
            ),
        ],
    }
