"""
qwen-think types: Enums, dataclasses, and type definitions for
managing Qwen3.6 thinking state across sessions, backends, and frameworks.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Backend(str, enum.Enum):
    """Supported inference backends for Qwen3.6."""

    VLLM = "vllm"
    SGLANG = "sglang"  # Same nesting as vLLM
    DASHSCOPE = "dashscope"  # Alibaba Cloud Model Studio
    LLAMACPP = "llamacpp"
    OPENAI = "openai"  # Generic OpenAI-compatible (treated like vLLM)


class ThinkingMode(str, enum.Enum):
    """Thinking mode for the current request / session."""

    THINK = "think"  # enable_thinking=True  (default for Qwen3.6)
    NO_THINK = "no_think"  # enable_thinking=False


class Complexity(str, enum.Enum):
    """Query complexity classification used by the router."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class BudgetAction(str, enum.Enum):
    """Action taken by the budget manager when context is running low."""

    OK = "ok"  # Plenty of headroom
    WARN = "warn"  # Approaching 128K threshold
    COMPRESS = "compress"  # Auto-compressed older messages
    REFUSE = "refuse"  # Below minimum, refuse to continue


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters that must stay in sync with the thinking mode."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
        }


THINKING_SAMPLING = SamplingConfig(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)

NON_THINKING_SAMPLING = SamplingConfig(
    temperature=0.7,
    top_p=0.80,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)


@dataclass
class BackendPayload:
    """Normalized payload ready to be sent to a specific backend.

    The ``extra_body`` dict is backend-specific and already formatted
    for the target (nested for vLLM/SGLang, top-level for DashScope).
    """

    enable_thinking: bool = True
    preserve_thinking: bool = True
    extra_body: Dict[str, Any] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BudgetStatus:
    """Context token budget status returned by the budget manager."""

    total_tokens: int = 0
    used_tokens: int = 0
    available_tokens: int = 0
    min_context: int = 128_000  # 128K -- Alibaba's recommended minimum
    action: BudgetAction = BudgetAction.OK
    message: str = ""

    @property
    def usage_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.used_tokens / self.total_tokens

    @property
    def is_below_minimum(self) -> bool:
        return self.available_tokens < self.min_context


@dataclass
class Message:
    """A single conversation message with optional thinking content."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    thinking_content: Optional[str] = None
    token_count: int = 0

    def to_openai_dict(self, *, include_thinking: bool = False) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role, "content": self.content}
        if include_thinking and self.thinking_content is not None:
            msg["reasoning_content"] = self.thinking_content
        return msg


@dataclass
class RouterDecision:
    """Output of the complexity router for a given query."""

    complexity: Complexity
    mode: ThinkingMode
    preserve_thinking: bool
    sampling: SamplingConfig
    confidence: float = 1.0  # 0.0-1.0 confidence of classification
    reasoning: str = ""  # Why this decision was made
