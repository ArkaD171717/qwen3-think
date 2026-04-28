"""Thinking session manager for Qwen3.6: backend normalization,
sampling parameter swap, and 128K context budget guard."""

from .backends import (
    BaseBackend,
    DashScopeBackend,
    LlamaCppBackend,
    OpenAIBackend,
    SGLangBackend,
    VLLMBackend,
    detect_backend,
    get_backend,
)
from .budget import BudgetManager, BudgetStatus, estimate_tokens
from .router import ComplexityRouter, LLMClassifier, RuleBasedClassifier
from .sampling import SamplingManager
from .session import ThinkingSession
from .types import (
    NON_THINKING_SAMPLING,
    THINKING_SAMPLING,
    Backend,
    BudgetAction,
    Complexity,
    Message,
    RouterDecision,
    SamplingConfig,
    ThinkingMode,
)

__version__ = "0.1.1"

__all__ = [
    "ThinkingSession",
    "BaseBackend",
    "VLLMBackend",
    "SGLangBackend",
    "OpenAIBackend",
    "DashScopeBackend",
    "LlamaCppBackend",
    "get_backend",
    "detect_backend",
    "BudgetManager",
    "BudgetStatus",
    "estimate_tokens",
    "ComplexityRouter",
    "RuleBasedClassifier",
    "LLMClassifier",
    "SamplingManager",
    "THINKING_SAMPLING",
    "NON_THINKING_SAMPLING",
    "Backend",
    "BudgetAction",
    "Complexity",
    "Message",
    "RouterDecision",
    "SamplingConfig",
    "ThinkingMode",
]
