from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..types import Backend, BackendPayload, ThinkingMode
from .base import BaseBackend

if TYPE_CHECKING:
    from ..sampling import SamplingManager


class VLLMBackend(BaseBackend):

    def __init__(
        self,
        backend: Backend = Backend.VLLM,
        detect_patterns: Optional[List[str]] = None,
        detect_fallback: float = 0.3,
        sampling_manager: Optional["SamplingManager"] = None,
    ) -> None:
        super().__init__(sampling_manager)
        self.backend = backend
        self._detect_patterns = (
            detect_patterns
            if detect_patterns is not None
            else [
                r"vllm",
                r":8000",  # vLLM default port
            ]
        )
        self._detect_fallback = detect_fallback

    def build_payload(
        self,
        mode: ThinkingMode,
        preserve_thinking: bool = True,
        sampling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BackendPayload:
        enable_thinking = mode == ThinkingMode.THINK

        extra_body: Dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        }

        if preserve_thinking:
            extra_body["chat_template_kwargs"]["preserve_thinking"] = True

        if "extra_body" in kwargs:
            for k, v in kwargs["extra_body"].items():
                if k == "chat_template_kwargs":
                    extra_body["chat_template_kwargs"].update(v)
                else:
                    extra_body[k] = v

        sampling_params = self._common_sampling(mode, sampling)

        warnings: list[str] = []

        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if "/no_think" in content or "/think" in content:
                    warnings.append(
                        "Qwen3.6 does not support /think or /no_think prompt "
                        "prefixes. Use enable_thinking parameter instead. "
                        "The /think and /no_think switches were removed from "
                        "Qwen3.6 (they only worked in Qwen3)."
                    )
                    break

        if preserve_thinking and enable_thinking:
            warnings.append(
                "preserve_thinking + vLLM prefix caching is an untested "
                "combination. If you see degraded reasoning across turns, "
                "try --no-enable-prefix-caching on your vLLM server."
            )

        return BackendPayload(
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
            extra_body=extra_body,
            sampling=sampling_params,
            warnings=warnings,
        )

    def detect(self, base_url: Optional[str] = None) -> float:
        if base_url is None:
            return 0.0

        url_lower = base_url.lower()
        for pattern in self._detect_patterns:
            if re.search(pattern, url_lower):
                return 0.6

        if self._detect_fallback > 0.0 and "/v1" in url_lower:
            return self._detect_fallback

        return 0.0


def SGLangBackend(
    sampling_manager: Optional["SamplingManager"] = None,
) -> VLLMBackend:
    return VLLMBackend(
        backend=Backend.SGLANG,
        detect_patterns=[r"sglang", r":30000"],
        sampling_manager=sampling_manager,
    )


def OpenAIBackend(
    sampling_manager: Optional["SamplingManager"] = None,
) -> VLLMBackend:
    return VLLMBackend(
        backend=Backend.OPENAI,
        detect_patterns=[],
        detect_fallback=0.0,
        sampling_manager=sampling_manager,
    )
