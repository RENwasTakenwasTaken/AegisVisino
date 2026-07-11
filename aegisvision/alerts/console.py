"""Simplest alert output: print to the console."""

from .base import Alerter, Alert


class ConsoleAlerter(Alerter):
    name = "console"

    def send(self, alert: Alert) -> None:
        print(f"[ALERT:{alert.event_type}] {alert.message}")
