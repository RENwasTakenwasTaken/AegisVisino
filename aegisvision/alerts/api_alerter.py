"""Alert output that pushes to the on-board API server for the mobile app.

Just another Alerter — the AlertManager fans every alert to it, and it hands
the alert to the AlertServer, which the phone polls over Wi-Fi.
"""

from .base import Alerter, Alert


class ApiAlerter(Alerter):
    name = "api"

    def __init__(self, server):
        self.server = server

    def send(self, alert: Alert) -> None:
        self.server.add(alert.event_type, alert.message, frame=alert.frame)
