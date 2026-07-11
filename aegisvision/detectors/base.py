"""The single contract every detector must follow.

This is the key to scaling: motion, face, gun, knife, fire — every one of them
is just a `Detector`. Add a feature = add a new class here. Remove a feature =
stop registering it in the pipeline. Nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    """A single thing a detector found."""
    label: str                     # e.g. "motion", "face", "gun"
    confidence: float = 1.0        # 0.0 - 1.0
    box: tuple = None              # (x, y, w, h) if the detector localises it
    extra: dict = field(default_factory=dict)  # anything detector-specific


class Detector(ABC):
    """Base class for every detection module.

    Implement `process(frame)` and return a list of Detection objects
    (empty list = found nothing). That's the whole contract.
    """

    name: str = "detector"

    @abstractmethod
    def process(self, frame) -> list["Detection"]:
        raise NotImplementedError
