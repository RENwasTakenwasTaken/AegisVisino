"""The alert output contract.

Mirror image of the Detector abstraction: detectors decide WHAT happened,
Alerters decide WHAT TO DO about it. Console print, save a snapshot, sound the
12V siren, push to the mobile app — each is just an Alerter. Add an output =
add a class. Remove one = stop registering it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Alert:
    event_type: str            # e.g. "concealed_person", "gun", "fire"
    message: str               # human-readable summary
    frame: object = None       # the video frame at alert time (for snapshots)
    box: tuple = None          # (x, y, w, h) of what triggered it
    meta: dict = field(default_factory=dict)


class Alerter(ABC):
    """One output channel. Implement send(alert)."""

    name: str = "alerter"

    @abstractmethod
    def send(self, alert: "Alert") -> None:
        raise NotImplementedError
