"""
Abstract base class for all OmniSense pipelines.
Every pipeline must implement load() and run().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omnisense.utils.logger import log


class BasePipeline(ABC):
    """
    All pipelines inherit from this.
    Enforces a consistent load → run lifecycle and provides
    shared logging and error handling.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._loaded = False
        log.info(f"{self.__class__.__name__} initialised on device={device}")

    @abstractmethod
    def load(self) -> None:
        """Download / initialise model weights. Called once before run()."""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute the pipeline. Returns a structured result dict."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if not self._loaded:
            log.info(f"Auto-loading {self.__class__.__name__}…")
            self.load()
            self._loaded = True
        return self.run(*args, **kwargs)
