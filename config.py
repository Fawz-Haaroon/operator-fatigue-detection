"""Configuration for the fatigue detection system."""

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# --- MediaPipe FaceMesh ---
MAX_NUM_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- EAR ---
EAR_THRESHOLD = 0.21

# --- BiLSTM+Attention Model (SARATHI-V2) ---
MODEL_PATH = "models/bilstm_sarathi.pth"
SEQ_LEN = 200
BASELINE_FRAMES = 330
PERCLOS_WINDOW = 300
INFERENCE_EVERY = 1          # run every frame (EMA handles smoothing)
DROWSY_THRESHOLD = 0.5

# --- Alert ---
ALERT_SMOOTHING = 8
ALERT_COOLDOWN = 4

# --- Severity (model probability thresholds) ---
MILD_THRESHOLD = 0.50
MODERATE_THRESHOLD = 0.65
SEVERE_THRESHOLD = 0.80

# --- PERCLOS ---
PERCLOS_THRESHOLD = 0.40

# --- Head Pose ---
HEAD_NOD_PITCH_THRESHOLD = 20

# --- MAR (compat) ---
MAR_THRESHOLD = 0.7

# --- Dashboard ---
DASHBOARD_URL = ""
DASHBOARD_SESSION_ID = ""
PUSH_INTERVAL = 1.0
ALERT_SOUND_PATH = "assets/alert.wav"
