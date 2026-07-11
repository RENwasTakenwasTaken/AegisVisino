"""Central configuration.

Keeping all tunables in one place means we never hunt through the code to
change a threshold. Later this can be loaded from a JSON/YAML file so the
consumer device can be configured without touching code.
"""

from dataclasses import dataclass


@dataclass
class MotionConfig:
    # Minimum changed-pixel area (in pixels) to count as "real" motion.
    # Filters out sensor noise / tiny flickers.
    min_area: int = 1500
    # How much a pixel must change (0-255) to be considered "different".
    diff_threshold: int = 25
    # Gaussian blur kernel — smooths noise before differencing. Must be odd.
    blur_kernel: int = 21


@dataclass
class FaceConfig:
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: int = 60  # ignore faces smaller than this many pixels


@dataclass
class FaceStoreConfig:
    folder: str = "people"       # where cropped faces are saved
    padding: float = 0.2         # extra margin around the face box (fraction)
    cooldown_seconds: float = 2  # min gap between saves, so we don't flood disk


@dataclass
class AppConfig:
    camera_source: object = 0        # 0 = default webcam; or an RTSP URL string
    process_every_n_frames: int = 2  # skip frames to save CPU (1 = every frame)
    show_window: bool = True         # set False on a headless device
    save_faces: bool = True          # crop + store detected faces to disk
    motion: MotionConfig = None
    face: FaceConfig = None
    face_store: FaceStoreConfig = None

    def __post_init__(self):
        # Give each sub-config a default instance if the caller didn't pass one.
        self.motion = self.motion or MotionConfig()
        self.face = self.face or FaceConfig()
        self.face_store = self.face_store or FaceStoreConfig()
