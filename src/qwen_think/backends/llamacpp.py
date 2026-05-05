"""llama.cpp backend: server-side only thinking flag (no per-request control)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..types import Backend, BackendPayload, ThinkingMode
from .base import BaseBackend

if TYPE_CHECKING:
    from ..sampling import SamplingManager


class LlamaCppBackend(BaseBackend):
    """Backend normalization for llama.cpp (server-side flag only)."""

    backend = Backend.LLAMACPP

    _LLAMACPP_PATTERNS = [
        r"llama",
    ]

    def __init__(
        self,
        server_enable_thinking: Optional[bool] = None,
        sampling_manager: Optional["SamplingManager"] = None,
    ) -> None:
        super().__init__(sampling_manager)
        self.server_enable_thinking = server_enable_thinking

    def build_payload(
        self,
        mode: ThinkingMode,
        preserve_thinking: bool = True,
        sampling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BackendPayload:
        """Build payload; warns if requested mode differs from server config."""
        enable_thinking = mode == ThinkingMode.THINK

        warnings: list[str] = []

        effective_thinking = self.server_enable_thinking
        if effective_thinking is None:
            effective_thinking = True

        if enable_thinking != effective_thinking:
            warnings.append(
                f"llama.cpp does NOT support per-request thinking control. "
                f"Requested enable_thinking={enable_thinking}, but the "
                f"server was started with enable_thinking={effective_thinking}. "
                f"The server's configuration will be used. "
                f"To change this, restart the server with: "
                f"{self.get_startup_command(enable_thinking, preserve_thinking)}"
            )

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

        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if "/no_think" in content or "/think" in content:
                    warnings.append(
                        "Qwen3.6 does not support /think or /no_think prompt "
                        "prefixes. Use enable_thinking parameter instead."
                    )
                    break

        if not enable_thinking:
            warnings.append(
                "Known issue (llama.cpp #20182): enable_thinking=false via "
                "--chat-template-kwargs may not reliably disable thinking for "
                "Qwen3.5/3.6. Workaround: use --reasoning-budget 0 combined "
                "with --chat-template-kwargs '{\"enable_thinking\": false}'"
            )

        return BackendPayload(
            enable_thinking=effective_thinking,  # Use effective, not requested
            preserve_thinking=preserve_thinking,
            extra_body=extra_body,
            sampling=sampling_params,
            warnings=warnings,
        )

    def detect(self, base_url: Optional[str] = None) -> float:
        if base_url is None:
            return 0.0

        url_lower = base_url.lower()
        for pattern in self._LLAMACPP_PATTERNS:
            if re.search(pattern, url_lower):
                return 0.5

        return 0.0

    @staticmethod
    def get_startup_command(
        enable_thinking: bool = True,
        preserve_thinking: bool = True,
        model_path: str = "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        port: int = 8080,
        context_length: int = 131072,
    ) -> str:
        kwargs_dict: Dict[str, Any] = {
            "enable_thinking": enable_thinking,
        }
        if preserve_thinking:
            kwargs_dict["preserve_thinking"] = True

        import json

        kwargs_json = json.dumps(kwargs_dict)

        cmd_parts = [
            "llama-server",
            f"-m {model_path}",
            f"--port {port}",
            f"--ctx-size {context_length}",
            f"--chat-template-kwargs '{kwargs_json}'",
            "--jinja",  # Required for tool calling & thinking support
        ]

        if not enable_thinking:
            cmd_parts.append("--reasoning-budget 0")

        return " \\\n  ".join(cmd_parts)
