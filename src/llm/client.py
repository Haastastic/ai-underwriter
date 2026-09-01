"""LLM client seam for the adverse-action layer.

The rest of the layer depends only on the :class:`LLMClient` protocol --
``complete(system, user, model) -> str`` -- so the provider can be swapped
(Claude -> GPT -> a local model) and tests can inject a fake without the
``anthropic`` package or an API key.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Per the claude-api skill: default to the most capable model and let the
# caller downgrade. Adverse-action prose is short, so cost is low regardless.
DEFAULT_MODEL = "claude-opus-5"


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, model: str) -> str:
        """Return the model's text response to a system + user prompt."""


class AnthropicClient:
    """Thin adapter over the Anthropic Messages API.

    The underlying ``anthropic.Anthropic`` client is imported lazily and can
    be injected, so importing this module never requires the package.
    """

    def __init__(self, client=None, *, max_tokens: int = 1024) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, *, system: str, user: str, model: str) -> str:
        response = self._ensure_client().messages.create(
            model=model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


def build_default_client() -> LLMClient:
    return AnthropicClient()
