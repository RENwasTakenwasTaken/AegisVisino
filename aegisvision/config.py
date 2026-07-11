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
    # Haar-engine settings (only used when face_engine == "haar").
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: int = 60  # ignore faces smaller than this many pixels


@dataclass
class InsightFaceConfig:
    # "buffalo_l" = best accuracy (laptop). On the Uno Q use the lighter
    # "buffalo_sc" or "buffalo_s" to keep it fast.
    model_name: str = "buffalo_l"
    det_size: int = 640          # detection input size; smaller = faster
    min_score: float = 0.5       # ignore low-confidence face detections
    ctx_id: int = -1             # -1 = CPU, 0+ = GPU
    providers: tuple = ("CPUExecutionProvider",)


@dataclass
class QualityConfig:
    # Reject a face whose box OR any landmark is within this many pixels of the
    # frame border — that's a face cut off by the camera edge.
    edge_margin_px: int = 8
    # A cleanly-detected face box is roughly square. A half-cut face is not.
    min_aspect: float = 0.6      # width / height lower bound
    max_aspect: float = 1.6      # width / height upper bound
    # Require all 5 InsightFace landmarks to be inside the frame.
    require_landmarks_inside: bool = True

    # --- enrollment quality gate ---------------------------------------
    # Minimum face quality (ArcFace embedding magnitude). A blurry / half-turned
    # / low-quality face scores LOW here — and would also get a distorted
    # embedding that looks like a NEW person. So we drop faces below this so
    # they're never registered as an identity.
    # 0.0 = disabled. Calibrate with debug=True (watch the q= values).
    min_face_quality: float = 0.0
    debug: bool = False


@dataclass
class OcclusionConfig:
    # Region-based "covered face" detector (nose/mouth skin analysis). This is
    # the sole concealment signal — a masked/covered lower face has a low skin
    # ratio, which raises the concealed-person alert.
    enabled: bool = True
    # Below this fraction of skin pixels in the lower face -> considered covered.
    # Calibrate with debug=True: watch skin_ratio for a bare face vs a masked one.
    skin_ratio_threshold: float = 0.30
    debug: bool = False


@dataclass
class FaceStoreConfig:
    folder: str = "people"           # where cropped faces are saved
    padding: float = 0.2             # extra margin around the face box (fraction)
    # Cosine-similarity cutoff for "same person". Higher = stricter.
    # ~0.5 is a good start for buffalo models; raise to reduce false matches.
    recognition_threshold: float = 0.5
    # Keep several embeddings per person so identity is robust to angle/lighting.
    max_samples_per_person: int = 5


@dataclass
class TamperConfig:
    # Detect a covered / blinded / defocused camera.
    enabled: bool = True
    max_std: float = 12.0        # below this pixel std = flat/uniform image
    max_laplacian: float = 30.0  # below this = little edge detail (blur/cover)
    # The flat condition must persist this many processed frames before alerting
    # (stops a brief close-up from triggering it).
    sustained_frames: int = 15
    debug: bool = False


@dataclass
class AlertConfig:
    cooldown_seconds: float = 5.0   # min gap between repeats of the same alert
    snapshot: bool = True           # save an evidence image per alert
    snapshot_folder: str = "alerts"


@dataclass
class AppConfig:
    camera_source: object = 0        # 0 = default webcam; or an RTSP URL string
    process_every_n_frames: int = 2  # skip frames to save CPU (1 = every frame)
    show_window: bool = True         # set False on a headless device
    save_faces: bool = True          # crop + store detected faces to disk
    motion_gating: bool = True       # False = always run face detection (no motion gate)
    face_engine: str = "insightface"  # "insightface" (embeddings) or "haar" (light)
    alert_on_concealed: bool = True  # fire an alert when a masked/covered face appears
    alert_on_tamper: bool = True     # fire an alert when the camera is covered
    motion: MotionConfig = None
    face: FaceConfig = None
    insightface: InsightFaceConfig = None
    quality: QualityConfig = None
    occlusion: OcclusionConfig = None
    tamper: TamperConfig = None
    face_store: FaceStoreConfig = None
    alert: AlertConfig = None

    def __post_init__(self):
        # Give each sub-config a default instance if the caller didn't pass one.
        self.motion = self.motion or MotionConfig()
        self.face = self.face or FaceConfig()
        self.insightface = self.insightface or InsightFaceConfig()
        self.quality = self.quality or QualityConfig()
        self.occlusion = self.occlusion or OcclusionConfig()
        self.tamper = self.tamper or TamperConfig()
        self.face_store = self.face_store or FaceStoreConfig()
        self.alert = self.alert or AlertConfig()
