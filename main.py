"""Entry point.

Run:  python main.py
Press 'q' in the video window to quit.
"""

from aegisvision.config import AppConfig
from aegisvision.pipeline import SecurityPipeline


def main():
    # Change camera_source to an RTSP URL to test against a real IP/CCTV camera:
    #   config = AppConfig(camera_source="rtsp://user:pass@192.168.1.10:554/stream")
    config = AppConfig(camera_source=0)
    SecurityPipeline(config).run()


if __name__ == "__main__":
    main()
