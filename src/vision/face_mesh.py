"""MediaPipe FaceMesh detector."""
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]
    def __init__(self, max_faces=1, det_conf=0.5, track_conf=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=max_faces, refine_landmarks=True, min_detection_confidence=det_conf, min_tracking_confidence=track_conf)
    def detect(self, frame):
        results = self.face_mesh.process(frame[:, :, ::-1])
        if not results.multi_face_landmarks: return None
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks.landmark])