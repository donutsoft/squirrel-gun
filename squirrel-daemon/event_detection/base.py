from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time


@dataclass
class DetectionEvent:
    ts: float
    rect: Optional[Tuple[int, int, int, int]] = None
    center: Optional[Tuple[float, float]] = None
    u: Optional[float] = None
    v: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class DetectionResult:
    frame: Any  # typically a numpy ndarray (opencv BGR)
    events: List[DetectionEvent]
    metrics: Dict[str, Any]


class EventDetector:
    """Abstract EventDetector API."""

    def configure(self, **kwargs: Any) -> None:
        """Update configuration parameters for the detector."""
        raise NotImplementedError

    def set_zone(self, zone: Optional[Sequence[float]]) -> None:
        """Set an optional normalized motion zone (x, y, w, h) in [0,1]."""
        raise NotImplementedError

    def get_zone(self) -> Optional[Tuple[float, float, float, float]]:
        raise NotImplementedError

    def suppress(self, duration_sec: float) -> None:
        """Temporarily suppress event emission for a duration."""
        raise NotImplementedError

    def enabled(self) -> bool:
        raise NotImplementedError

    def info(self, frame_size: Tuple[int, int]) -> Dict[str, Any]:
        """Return a dictionary describing the current detection state/metrics."""
        raise NotImplementedError

    def config(self) -> Dict[str, Any]:
        """Return a dictionary describing the current configuration."""
        raise NotImplementedError

    def reset_metrics(self) -> None:
        """Reset accumulated counters/peaks."""
        raise NotImplementedError

    def process(self, frame: Any, now_ts: Optional[float] = None) -> DetectionResult:
        """Process a frame, possibly drawing overlays, and return events + metrics."""
        raise NotImplementedError

