"""Feature extraction buffer with per-session baseline normalization.

Computes 10 features per frame:
  0. ear       — average Eye Aspect Ratio
  1. gaze      — horizontal gaze deviation (iris-based)
  2. roll      — head roll angle (degrees)
  3. pitch     — head pitch angle (degrees)
  4. yaw       — head yaw angle (degrees)
  5. perclos   — Percentage of Eye Closure over sliding window
  6. blink_rate — blinks per minute (rolling 60-frame window)
  7. ear_var   — EAR variance over last 30 frames
  8. ear_min   — EAR minimum over last 30 frames
  9. gaze_var  — gaze variance over last 30 frames
"""
import math
import numpy as np

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_C = 33
RIGHT_EYE_C = 263

EAR_THRESHOLD = 0.21
BASELINE_FRAMES = 330
SEQ_LEN = 200
PERCLOS_WINDOW = 300


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ear(landmarks, indices):
    p = landmarks[indices][:, :2]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def compute_gaze(landmarks):
    """Horizontal gaze deviation from iris landmarks.
    Returns ~0.5 when looking straight, <0.5 left, >0.5 right."""
    if landmarks.shape[0] < 478:
        return 0.5  # no iris landmarks available

    try:
        lc = landmarks[LEFT_IRIS][:, :2].mean(axis=0)
        rc = landmarks[RIGHT_IRIS][:, :2].mean(axis=0)

        ll = landmarks[362][0]; lr = landmarks[263][0]
        rl = landmarks[33][0];  rr = landmarks[133][0]

        l_ratio = (lc[0] - ll) / (lr - ll + 1e-6)
        r_ratio = (rc[0] - rl) / (rr - rl + 1e-6)
        return (l_ratio + r_ratio) / 2.0
    except (IndexError, ValueError):
        return 0.5


def compute_head_pose(landmarks, w, h):
    """Geometric head pose estimation -> (roll, pitch, yaw) in degrees."""
    nose = landmarks[NOSE_TIP][:2]
    chin = landmarks[CHIN][:2]
    le = landmarks[LEFT_EYE_C][:2]
    re = landmarks[RIGHT_EYE_C][:2]

    eye_vec = re - le
    roll = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))

    face_h = _dist(tuple(le), tuple(chin))
    nose_chin = _dist(tuple(nose), tuple(chin))
    pitch = (nose_chin / (face_h + 1e-6) - 0.5) * 90

    eye_mid_x = (le[0] + re[0]) / 2
    face_w = _dist(tuple(le), tuple(re))
    yaw = ((nose[0] - eye_mid_x) / (face_w + 1e-6)) * 90

    return roll, pitch, yaw


class FeatureBuffer:
    def __init__(self, baseline_frames=BASELINE_FRAMES, seq_len=SEQ_LEN,
                 perclos_window=PERCLOS_WINDOW, ear_threshold=EAR_THRESHOLD):
        self.baseline_frames = baseline_frames
        self.seq_len = seq_len
        self.perclos_window = perclos_window
        self.ear_threshold = ear_threshold
        self.raw_buffer = []
        self.norm_buffer = []
        self.baseline_ready = False
        self.baseline_mean = None
        self.baseline_std = None
        self.blink_times = []
        self.prev_ear = None
        self.in_blink = False
        self.perclos_win = []

    def _extract_raw(self, landmarks, w, h):
        ear_l = compute_ear(landmarks, LEFT_EYE)
        ear_r = compute_ear(landmarks, RIGHT_EYE)
        ear = (ear_l + ear_r) / 2.0

        gaze = compute_gaze(landmarks)
        roll, pitch, yaw = compute_head_pose(landmarks, w, h)

        # PERCLOS
        self.perclos_win.append(1 if ear < self.ear_threshold else 0)
        if len(self.perclos_win) > self.perclos_window:
            self.perclos_win.pop(0)
        perclos = np.mean(self.perclos_win) if self.perclos_win else 0.0

        # Blink detection
        frame_idx = len(self.raw_buffer)
        if self.prev_ear is not None:
            thr = self.ear_threshold + 0.01
            if self.prev_ear >= thr and ear < thr:
                self.in_blink = True
            elif self.in_blink and ear >= thr:
                self.blink_times.append(frame_idx)
                self.in_blink = False
        self.prev_ear = ear

        # Trim old blink times (keep last 300 frames worth)
        cutoff = max(0, frame_idx - 300)
        self.blink_times = [t for t in self.blink_times if t >= cutoff]

        recent_blinks = sum(1 for t in self.blink_times if frame_idx - t < 60)
        blink_rate = recent_blinks  # per ~60 frames

        # Rolling stats (last 30 frames)
        if len(self.raw_buffer) >= 30:
            recent = np.array(self.raw_buffer[-30:])
            ear_var = float(np.var(recent[:, 0]))
            ear_min = float(np.min(recent[:, 0]))
            gaze_var = float(np.var(recent[:, 1]))
        else:
            ear_var = 0.0
            ear_min = ear
            gaze_var = 0.0

        return np.array([ear, gaze, roll, pitch, yaw,
                         perclos, blink_rate, ear_var, ear_min, gaze_var],
                        dtype=np.float32)

    def update(self, landmarks, w, h):
        """Process one frame. Returns (seq_len, 10) array or None."""
        raw = self._extract_raw(landmarks, w, h)
        self.raw_buffer.append(raw)

        if not self.baseline_ready:
            if len(self.raw_buffer) >= self.baseline_frames:
                arr = np.array(self.raw_buffer[:self.baseline_frames])
                self.baseline_mean = arr.mean(axis=0)
                self.baseline_std = arr.std(axis=0) + 1e-8
                self.baseline_ready = True
                # Normalize all buffered frames
                for r in self.raw_buffer:
                    self.norm_buffer.append((r - self.baseline_mean) / self.baseline_std)
                print(f"[INFO] Calibration complete. Baseline EAR={self.baseline_mean[0]:.3f}, "
                      f"Gaze={self.baseline_mean[1]:.3f}, "
                      f"Pitch={self.baseline_mean[3]:.1f}")
            return None

        norm = (raw - self.baseline_mean) / self.baseline_std
        self.norm_buffer.append(norm)

        if len(self.norm_buffer) >= self.seq_len:
            return np.array(self.norm_buffer[-self.seq_len:], dtype=np.float32)
        return None

    @property
    def calibrating(self):
        return not self.baseline_ready

    @property
    def calibration_progress(self):
        return min(len(self.raw_buffer) / self.baseline_frames, 1.0)

    @property
    def raw_features(self):
        return self.raw_buffer[-1] if self.raw_buffer else np.zeros(10)
