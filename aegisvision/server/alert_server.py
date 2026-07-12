"""A tiny HTTP server on the board that the mobile app polls over Wi-Fi.

Holds the most recent alerts in memory and exposes them as JSON, plus the
snapshot image for each. The app just GETs /alerts every few seconds.

Endpoints:
    GET /health            -> {"status": "ok"}
    GET /alerts            -> [ {id, type, message, time, image_url}, ... ]
    GET /snapshot/<id>.jpg -> the evidence image for that alert

Deliberately simple (polling, in-memory) for the first increment. WebSocket
push, persistence, and auth are natural later steps and won't change the app's
alert model.
"""

import threading
import time
from collections import deque

import cv2


class AlertServer:
    def __init__(self, config):
        self.cfg = config
        self._alerts = deque(maxlen=config.max_alerts)  # newest first
        self._images = {}                                # id -> jpeg bytes
        self._next_id = 1
        self._lock = threading.Lock()
        self._app = self._build_app()

    # Called by ApiAlerter whenever an alert fires.
    def add(self, event_type, message, frame=None):
        with self._lock:
            aid = self._next_id
            self._next_id += 1
            record = {
                "id": aid,
                "type": event_type,
                "message": message,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_url": None,
            }
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    self._images[aid] = buf.tobytes()
                    record["image_url"] = f"/snapshot/{aid}.jpg"
                    # Drop images that fell out of the alert window.
                    live_ids = {a["id"] for a in self._alerts} | {aid}
                    for old in list(self._images):
                        if old not in live_ids:
                            del self._images[old]
            self._alerts.appendleft(record)

    def _build_app(self):
        from flask import Flask, jsonify, Response

        app = Flask(__name__)

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.get("/alerts")
        def alerts():
            with self._lock:
                return jsonify(list(self._alerts))

        @app.get("/snapshot/<int:aid>.jpg")
        def snapshot(aid):
            with self._lock:
                img = self._images.get(aid)
            if img is None:
                return ("not found", 404)
            return Response(img, mimetype="image/jpeg")

        return app

    def start(self):
        """Run the server in a background thread so the pipeline keeps going."""
        thread = threading.Thread(
            target=self._app.run,
            kwargs={"host": self.cfg.host, "port": self.cfg.port,
                    "threaded": True, "use_reloader": False},
            daemon=True,
        )
        thread.start()
        print(f"[SERVER] alert API on http://{self.cfg.host}:{self.cfg.port}  "
              f"(phone connects here)")
