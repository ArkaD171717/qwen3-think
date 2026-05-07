from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..types import Backend, BackendPayload, ThinkingMode

if TYPE_CHECKING:
    from ..sampling import SamplingManager


class BaseBackend(ABC):
    backend: Backend

    def __init__(self, sampling_manager: Optional["SamplingManager"] = None) -> None:
        self._sampling_manager = sampling_manager

    @abstractmethod
    def build_payload(
        self,
        mode: ThinkingMode,
        preserve_thinking: bool = True,
        sampling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BackendPayload: ...

    @abstractmethod
    def detect(self, base_url: Optional[str] = None) -> float: ...

    def _common_sampling(
        self,
        mode: ThinkingMode,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if sampling is not None:
            return sampling
        if self._sampling_manager is None:
            from ..sampling import SamplingManager

            self._sampling_manager = SamplingManager()
        return self._sampling_manager.get_params(mode)
