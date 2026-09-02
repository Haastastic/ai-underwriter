"""Choose the LLM client the API uses to render adverse-action notices.

Real Anthropic client when ANTHROPIC_API_KEY is set; otherwise a
deterministic stub so the whole API -- and its test suite -- runs offline.
The stub composes the notice from the same fixed reason statements the real
prompt would carry, and marks itself clearly so a stubbed notice is never
mistaken for a generated one.
"""

from __future__ import annotations

import os
import re

from src.llm.client import AnthropicClient, LLMClient

_NUMBERED = re.compile(r"^\d+\.\s+(.*)$")

STUB_PREFIX = "[DRAFT - generated without a language model]"


class StubLLMClient:
    """Offline stand-in for :class:`~src.llm.client.LLMClient`."""

    def complete(self, *, system: str, user: str, model: str) -> str:
        reasons = [
            m.group(1)
            for line in user.splitlines()
            if (m := _NUMBERED.match(line.strip()))
        ]
        bullets = "\n".join(f"- {r}" for r in reasons)
        return (
            f"{STUB_PREFIX}\n"
            "Your application for credit was denied. The specific reasons are:\n"
            f"{bullets}\n"
            "You have the right to a statement of the specific reasons for this "
            "decision."
        )


def build_llm_client() -> tuple[LLMClient, str]:
    """Return (client, provider_label) based on the environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient(), "anthropic"
    return StubLLMClient(), "stub"
