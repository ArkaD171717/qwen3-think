"""Sampling parameter swap on mode toggle."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .types import (
    NON_THINKING_SAMPLING,
    THINKING_SAMPLING,
    SamplingConfig,
    ThinkingMode,
)

logger = logging.getLogger("qwen-think.sampling")


class SamplingManager:
    """Ensures sampling params always match the active thinking mode."""

    def __init__(
        self,
        thinking: Optional[SamplingConfig] = None,
        non_thinking: Optional[SamplingConfig] = None,
    ) -> None:
        self.thinking = thinking or THINKING_SAMPLING
        self.non_thinking = non_thinking or NON_THINKING_SAMPLING

    def get_config(self, mode: ThinkingMode) -> SamplingConfig:
        if mode == ThinkingMode.THINK:
            return self.thinking
        return self.non_thinking

    def get_params(self, mode: ThinkingMode) -> Dict[str, Any]:
        return self.get_config(mode).to_dict()

    def swap_params(
        self,
        current_mode: ThinkingMode,
        target_mode: ThinkingMode,
        current_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        if current_mode == target_mode:
            return current_params

        target_sampling = self.get_params(target_mode)
        return {**current_params, **target_sampling}

    def validate_params(
        self, mode: ThinkingMode, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        expected = self.get_params(mode)
        mismatches: Dict[str, Any] = {}

        for key, expected_val in expected.items():
            actual_val = params.get(key)
            if actual_val is not None and actual_val != expected_val:
                mismatches[key] = {"expected": expected_val, "actual": actual_val}

        if mismatches:
            logger.warning(
                "Sampling params don't match %s mode defaults: %s. "
                "User values will be kept.",
                mode.value,
                ", ".join(
                    f"{k}={v['actual']} (expected {v['expected']})"
                    for k, v in mismatches.items()
                ),
            )

        return {
            "valid": len(mismatches) == 0,
            "mismatches": mismatches,
            # User values take precedence; defaults fill gaps only
            "merged": {**expected, **params},
        }
