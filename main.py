"""
All settings are here.
"""

from aegisvision.config import AppConfig, InsightFaceConfig, QualityConfig
from aegisvision.pipeline import SecurityPipeline


def main():
    # Change camera_source to an RTSP URL to test against a real IP/CCTV camera:
    #   config = AppConfig(camera_source="rtsp://user:pass@192.168.1.10:554/stream")

    camera_index = int(input("Enter camera: "))

    config = AppConfig(
        camera_source=camera_index,
        quality=QualityConfig(min_face_quality=20.0),
        alert_on_concealed=True,
        insightface=InsightFaceConfig(model_name="buffalo_sc", det_size=320)
    )
    SecurityPipeline(config).run()


if __name__ == "__main__":
    main()
