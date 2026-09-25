"""MediaPipe FaceMesh detector using the Tasks API (mediapipe >= 0.10.30)."""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _ensure_model():
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.exists(_MODEL_PATH):
        print(f"[INFO] Downloading face_landmarker model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[INFO] Download complete.")


class FaceMeshDetector:
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

    def __init__(self, max_faces=1, det_conf=0.5, track_conf=0.5):
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=det_conf,
            min_face_presence_confidence=track_conf,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame):
        """Detect face landmarks. Returns (468, 3) numpy array or None."""
        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]
        return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks])
