"""LLM layer: structured SHAP explanation -> ECOA-style adverse-action text.

The LLM only renders prose. It never sees the model or its score and never
influences the decision -- see :func:`generate_adverse_action`.
"""

from src.llm.adverse_action import (
    AdverseActionResult,
    generate_adverse_action,
    result_to_json,
)
from src.llm.client import (
    DEFAULT_MODEL,
    AnthropicClient,
    LLMClient,
    build_default_client,
)
from src.llm.reasons import AGE_DERIVED_FEATURES, FEATURE_REASONS, select_reasons

__all__ = [
    "generate_adverse_action",
    "AdverseActionResult",
    "result_to_json",
    "select_reasons",
    "FEATURE_REASONS",
    "AGE_DERIVED_FEATURES",
    "LLMClient",
    "AnthropicClient",
    "build_default_client",
    "DEFAULT_MODEL",
]
