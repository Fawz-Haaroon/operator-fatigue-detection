"""Main entry point — opens webcam and starts fatigue detection.

Run with --diag to show a detailed diagnostic overlay with all signal values.
"""
import cv2
import sys
import time
from src.vision.capture import CameraCapture
from src.vision.face_mesh import FaceMeshDetector
from src.features.ear import compute_ear
from src.features.mar import compute_mar
from src.features.perclos import PERCLOSTracker
from src.features.head_pose import HeadPoseEstimator
from src.model.predictor import FatiguePredictor
from src.alerting.alert_manager import AlertManager
from src.api.dashboard_client import DashboardClient
import config


def main():
    diag_mode = "--diag" in sys.argv

    camera = CameraCapture(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    face_mesh = FaceMeshDetector(
        config.MAX_NUM_FACES,
        config.MIN_DETECTION_CONFIDENCE,
        config.MIN_TRACKING_CONFIDENCE,
    )
    perclos_tracker = PERCLOSTracker(config.PERCLOS_WINDOW, config.FPS)
    head_pose = HeadPoseEstimator(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    predictor = FatiguePredictor(
        config.MODEL_PATH, config.SEQUENCE_LENGTH,
        config.INPUT_FEATURES, config.HIDDEN_SIZE, config.NUM_LAYERS,
    )
    alert_manager = AlertManager(config.ALERT_SMOOTHING, config.ALERT_COOLDOWN)
    dashboard = DashboardClient(
        config.DASHBOARD_URL, config.DASHBOARD_SESSION_ID, config.PUSH_INTERVAL,
    )

    # Resizable window
    cv2.namedWindow("Fatigue Monitor", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Fatigue Monitor", 960, 720)

    print("[INFO] Fatigue detection started. Press 'q' to quit.")
    if diag_mode:
        print("[INFO] Diagnostic overlay enabled (--diag)")

    frame_count = 0
    fps_start = time.time()
    actual_fps = 0.0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

            # FPS measurement
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                actual_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            landmarks = face_mesh.detect(frame)
            if landmarks is None:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Fatigue Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Feature extraction
            ear_left, ear_right = compute_ear(landmarks)
            ear = (ear_left + ear_right) / 2.0
            mar = compute_mar(landmarks)
            pitch, yaw, roll = head_pose.estimate(landmarks)
            eyes_closed = ear < config.EAR_THRESHOLD
            perclos = perclos_tracker.update(eyes_closed)
            blink_rate = perclos_tracker.blink_rate

            # Fatigue scoring
            fatigue_score = predictor.predict(
                ear_left, ear_right, mar, perclos, pitch, yaw, roll, blink_rate,
            )
            alert = alert_manager.update(fatigue_score, perclos, mar, pitch)

            # === Display ===
            if diag_mode:
                _draw_diag(frame, ear_left, ear_right, ear, mar, perclos,
                           blink_rate, pitch, yaw, roll, eyes_closed,
                           fatigue_score, actual_fps, perclos_tracker, alert)
            else:
                _draw_normal(frame, ear, mar, perclos, blink_rate,
                             fatigue_score, alert)

            # Dashboard push
            if alert:
                dashboard.send_alert(fatigue_score, perclos, blink_rate, alert)

            cv2.imshow("Fatigue Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        dashboard.close()
        print("[INFO] Fatigue detection stopped.")


def _draw_normal(frame, ear, mar, perclos, blink_rate, score, alert):
    """Standard HUD overlay."""
    color = ((0, 255, 0) if score < config.MILD_THRESHOLD
             else (0, 165, 255) if score < config.MODERATE_THRESHOLD
             else (0, 0, 255))
    cv2.putText(frame, f"Score: {score:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"PERCLOS: {perclos:.2f} | Blinks: {blink_rate:.0f}/min",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if alert:
        cv2.putText(frame, f"ALERT: {alert['severity']}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


def _draw_diag(frame, ear_l, ear_r, ear, mar, perclos, blink_rate,
               pitch, yaw, roll, eyes_closed, score, fps, tracker, alert):
    """Detailed diagnostic overlay showing every signal."""
    h, w = frame.shape[:2]

    # Semi-transparent background for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    white = (255, 255, 255)
    gray = (160, 160, 160)
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 200, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 25
    dy = 22

    def line(label, value, color=white):
        nonlocal y
        cv2.putText(frame, f"{label}: {value}", (10, y), font, 0.45, color, 1)
        y += dy

    # Header
    cv2.putText(frame, "DIAGNOSTIC", (10, y), font, 0.6, yellow, 2)
    y += dy + 5

    # FPS
    line("FPS (actual)", f"{fps:.1f}", green if fps > 20 else red)
    line("Samples in window", f"{tracker.sample_count}")
    line("Effective FPS", f"{tracker.effective_fps:.1f}")
    y += 5

    # EAR
    cv2.putText(frame, "--- EAR ---", (10, y), font, 0.45, yellow, 1)
    y += dy
    ear_color = red if eyes_closed else green
    line("EAR left", f"{ear_l:.4f}", ear_color)
    line("EAR right", f"{ear_r:.4f}", ear_color)
    line("EAR avg", f"{ear:.4f}", ear_color)
    line("Threshold", f"{config.EAR_THRESHOLD}")
    line("Eyes closed", f"{eyes_closed}", red if eyes_closed else green)
    y += 5

    # MAR
    cv2.putText(frame, "--- MAR ---", (10, y), font, 0.45, yellow, 1)
    y += dy
    mar_triggered = mar > config.MAR_THRESHOLD
    line("MAR", f"{mar:.4f}", red if mar_triggered else white)
    line("Threshold", f"{config.MAR_THRESHOLD}")
    line("Triggered", f"{mar_triggered}", red if mar_triggered else gray)
    y += 5

    # PERCLOS
    cv2.putText(frame, "--- PERCLOS ---", (10, y), font, 0.45, yellow, 1)
    y += dy
    perc_triggered = perclos > config.PERCLOS_THRESHOLD
    line("PERCLOS", f"{perclos:.4f}", red if perc_triggered else white)
    line("Threshold", f"{config.PERCLOS_THRESHOLD}")
    line("Blink rate", f"{blink_rate:.1f}/min")
    y += 5

    # Head pose
    cv2.putText(frame, "--- HEAD POSE ---", (10, y), font, 0.45, yellow, 1)
    y += dy
    pitch_triggered = abs(pitch) > config.HEAD_NOD_PITCH_THRESHOLD
    line("Pitch", f"{pitch:.1f}", red if pitch_triggered else white)
    line("Yaw", f"{yaw:.1f}")
    line("Roll", f"{roll:.1f}")
    line("Pitch thr", f"{config.HEAD_NOD_PITCH_THRESHOLD}")
    y += 5

    # Fatigue score breakdown
    cv2.putText(frame, "--- SCORE ---", (10, y), font, 0.45, yellow, 1)
    y += dy
    # Show which rules fired
    ear_contrib = 0.30 if ear < config.EAR_THRESHOLD else 0.0
    perc_contrib = 0.30 if perclos > config.PERCLOS_THRESHOLD else 0.0
    mar_contrib = 0.15 if mar > config.MAR_THRESHOLD else 0.0
    pitch_contrib = 0.15 if abs(pitch) > config.HEAD_NOD_PITCH_THRESHOLD else 0.0
    br_contrib = 0.10 if blink_rate < 8 else 0.0

    line(f"EAR<{config.EAR_THRESHOLD}", f"+{ear_contrib:.2f}", red if ear_contrib > 0 else gray)
    line(f"PERCLOS>{config.PERCLOS_THRESHOLD}", f"+{perc_contrib:.2f}", red if perc_contrib > 0 else gray)
    line(f"MAR>{config.MAR_THRESHOLD}", f"+{mar_contrib:.2f}", red if mar_contrib > 0 else gray)
    line(f"pitch>{config.HEAD_NOD_PITCH_THRESHOLD}", f"+{pitch_contrib:.2f}", red if pitch_contrib > 0 else gray)
    line(f"blinks<8", f"+{br_contrib:.2f}", red if br_contrib > 0 else gray)
    y += 5

    score_color = green if score < 0.25 else yellow if score < 0.7 else red
    cv2.putText(frame, f"SCORE: {score:.2f}", (10, y), font, 0.7, score_color, 2)
    y += dy + 5

    if alert:
        cv2.putText(frame, f"ALERT: {alert['severity']}", (10, y),
                    font, 0.7, red, 2)


if __name__ == "__main__":
    main()
