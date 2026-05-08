from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..models import PageContext

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class StepResult:
    ok: bool
    message: str = ""
    error: Optional[Exception] = None


class PipelineStep(ABC):
    name: str = "step"

    @abstractmethod
    def run(
        self,
        ctx: PageContext,
        params: Any,
        progress: Optional[ProgressCallback] = None,
    ) -> StepResult:
        ...
