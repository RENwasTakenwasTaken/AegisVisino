"""
All settings are here.
"""

import sys

from aegisvision.config import (
    AppConfig, InsightFaceConfig, QualityConfig, OcclusionConfig,
)
from aegisvision.pipeline import SecurityPipeline

args = sys.argv

def main():
    # Change camera_source to an RTSP URL to test against a real IP/CCTV camera:
    #   config = AppConfig(camera_source="rtsp://user:pass@192.168.1.10:554/stream")

    camera_index = int(input("Enter camera: "))

    config = AppConfig(
        camera_source=camera_index,
        quality=QualityConfig(
            min_face_quality=20.0,  # below this = too blurry to register as a person
            debug=True if "qualitydebug" in args else False,
        ),
        occlusion=OcclusionConfig(
            skin_ratio_threshold=0.30,  # lower face below this skin ratio = covered
            debug=True if "occlusiondebug" in args else False,
        ),
        alert_on_concealed=True if "conceal" in args else False,
        insightface=InsightFaceConfig(model_name="buffalo_sc", det_size=320)
    )
    SecurityPipeline(config).run()


if __name__ == "__main__":
    main()
