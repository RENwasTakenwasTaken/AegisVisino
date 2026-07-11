"""Fans one alert out to every registered output, with per-event cooldown.

Cooldown matters: a masked person standing in view for 10 seconds must not
fire 100 alerts. We rate-limit per event_type, so "concealed_person" and a
future "fire" alert have independent timers.
"""

import time

from .base import Alert


class AlertManager:
    def __init__(self, alerters, cooldown_seconds=5.0):
        self.alerters = list(alerters)
        self.cooldown = cooldown_seconds
        self._last_fired = {}  # event_type -> timestamp

    def fire(self, event_type, message, frame=None, box=None, meta=None):
        """Raise an alert, unless this event_type is still in cooldown."""
        now = time.time()
        last = self._last_fired.get(event_type, 0.0)
        if now - last < self.cooldown:
            return False
        self._last_fired[event_type] = now

        alert = Alert(event_type=event_type, message=message,
                      frame=frame, box=box, meta=meta or {})
        for alerter in self.alerters:
            alerter.send(alert)
        return True
