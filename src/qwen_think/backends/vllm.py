"""vLLM and SGLang backend: nested chat_template_kwargs format.

Always explicitly sets enable_thinking to fix vLLM semantic router #858
(field removal instead of setting false).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..types import Backend, BackendPayload, ThinkingMode
from .base import BaseBackend


class VLLMBackend(BaseBackend):
    """Backend normalization for vLLM and SGLang."""

    backend = Backend.VLLM

    _VLLM_PATTERNS = [
        r"vllm",
        r":8000",  # vLLM default port
    ]

    def build_payload(
        self,
        mode: ThinkingMode,
        preserve_thinking: bool = True,
        sampling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BackendPayload:
        """Build payload with enable_thinking nested in chat_template_kwargs."""
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
        """Score 0.0-1.0 for how likely this URL is a vLLM server.

        0.6 = keyword/port match, 0.3 = generic /v1 endpoint.
        DashScope scores 0.9 so it wins when both could match.
        """
        if base_url is None:
            return 0.0

        url_lower = base_url.lower()
        for pattern in self._VLLM_PATTERNS:
            if re.search(pattern, url_lower):
                return 0.6

        if "/v1" in url_lower:
            return 0.3

        return 0.0


class SGLangBackend(VLLMBackend):
    """Same nested format as vLLM, different detection patterns."""

    backend = Backend.SGLANG

    _VLLM_PATTERNS = [
        r"sglang",
        r":30000",  # SGLang default port
    ]


class OpenAIBackend(VLLMBackend):
    """Generic OpenAI-compatible servers. Same payload shape as vLLM."""

    backend = Backend.OPENAI

    def detect(self, base_url: Optional[str] = None) -> float:
        return 0.0
