"""MediaPipe FaceMesh detector with legacy and Tasks API support."""
import numpy as np

# Try legacy API first, fall back to Tasks API
_USE_LEGACY = True
try:
    import mediapipe as mp
    # Check if solutions is available
    _ = mp.solutions.face_mesh
except (AttributeError, ImportError):
    _USE_LEGACY = False

if _USE_LEGACY:
    import mediapipe as mp

    class FaceMeshDetector:
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

        def __init__(self, max_faces=1, det_conf=0.5, track_conf=0.5):
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )

        def detect(self, frame):
            results = self.face_mesh.process(frame[:, :, ::-1])  # BGR -> RGB
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks.landmark])

else:
    # New Tasks API (mediapipe >= 0.10.14)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe import solutions
    import urllib.request
    import os

    _MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

    def _ensure_model():
        if not os.path.exists(_MODEL_PATH):
            print(f"[INFO] Downloading face_landmarker model...")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print(f"[INFO] Model saved to {_MODEL_PATH}")

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
            rgb = frame[:, :, ::-1].copy()  # BGR -> RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            if not result.face_landmarks:
                return None
            landmarks = result.face_landmarks[0]
            h, w = frame.shape[:2]
            return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks])
