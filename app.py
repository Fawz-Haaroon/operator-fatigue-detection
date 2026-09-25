"""Operator Fatigue Detection — real-time webcam monitoring.

Uses SARATHI-V2's BiLSTM+Attention model with 10-feature extraction
and per-session baseline normalization.

Usage:
  python app.py           # normal mode
  python app.py --diag    # diagnostic overlay
"""
import cv2
import sys
import time
import numpy as np
from src.vision.capture import CameraCapture
from src.vision.face_mesh import FaceMeshDetector
from src.features.feature_buffer import FeatureBuffer
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
    feat_buffer = FeatureBuffer(
        baseline_frames=config.BASELINE_FRAMES,
        seq_len=config.SEQ_LEN,
        perclos_window=config.PERCLOS_WINDOW,
        ear_threshold=config.EAR_THRESHOLD,
    )
    predictor = FatiguePredictor(config.MODEL_PATH, device="cpu")
    alert_manager = AlertManager(config.ALERT_SMOOTHING, config.ALERT_COOLDOWN)
    dashboard = DashboardClient(
        config.DASHBOARD_URL, config.DASHBOARD_SESSION_ID, config.PUSH_INTERVAL,
    )

    cv2.namedWindow("Fatigue Monitor", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Fatigue Monitor", 960, 720)

    print("[INFO] Fatigue detection started. Press 'q' to quit.")
    if predictor.model_type == "none":
        print("[WARN] No model — using rule-based fallback. Run: python setup_model.py")

    frame_count = 0
    fps_start = time.time()
    actual_fps = 0.0

    # Smoothed display probability (EMA for snappy feel)
    display_prob = 0.0
    display_label = "CALIBRATING"
    model_prob = 0.0
    ema_alpha = 0.3  # higher = more responsive to new readings
    no_face_frames = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                actual_fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            landmarks = face_mesh.detect(frame)
            if landmarks is None:
                no_face_frames += 1
                if no_face_frames > 30:  # ~1 second with no face
                    display_prob = display_prob * 0.95  # slowly decay
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Fatigue Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            no_face_frames = 0
            h, w = frame.shape[:2]

            # Feed frame to feature buffer
            sequence = feat_buffer.update(landmarks, w, h)
            raw = feat_buffer.raw_features

            # Calibration phase
            if feat_buffer.calibrating:
                progress = feat_buffer.calibration_progress
                _draw_calibration(frame, progress, raw, actual_fps, diag_mode)
                cv2.imshow("Fatigue Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # Run model inference every frame if sequence is available
            if sequence is not None:
                if predictor.model is not None:
                    label, prob = predictor.predict_sequence(sequence)
                    if label is not None:
                        model_prob = prob
                else:
                    model_prob = FatiguePredictor.rule_based_score(
                        raw[0], raw[5], raw[1], raw[3], raw[6],
                    )

            # EMA smoothing for display (fast response, no jitter)
            display_prob = ema_alpha * model_prob + (1 - ema_alpha) * display_prob
            display_label = "DROWSY" if display_prob > 0.5 else "ALERT"

            # Alert check
            alert = None
            if display_prob > 0.5:
                alert = alert_manager.update(display_prob, raw[5], 0.0, raw[3])

            # Draw
            if diag_mode:
                _draw_diag(frame, raw, display_label, display_prob, model_prob,
                           actual_fps, feat_buffer, predictor.model_type, alert)
            else:
                _draw_normal(frame, display_label, display_prob, raw, alert)

            if alert:
                dashboard.send_alert(display_prob, raw[5], raw[6], alert)

            cv2.imshow("Fatigue Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        dashboard.close()
        print("[INFO] Fatigue detection stopped.")


def _draw_calibration(frame, progress, raw, fps, diag):
    h, w = frame.shape[:2]
    bar_w = 300
    bar_h = 24
    bx = (w - bar_w) // 2
    by = h // 2

    # Background box
    cv2.rectangle(frame, (bx - 60, by - 60), (bx + bar_w + 80, by + 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (bx - 60, by - 60), (bx + bar_w + 80, by + 50), (0, 200, 255), 1)

    cv2.putText(frame, "CALIBRATING", (bx + 70, by - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, "Sit normally, look at camera",
                (bx + 30, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Progress bar
    cv2.rectangle(frame, (bx, by + 5), (bx + bar_w, by + 5 + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * progress)
    color = (0, 200, 255) if progress < 1.0 else (0, 255, 0)
    cv2.rectangle(frame, (bx, by + 5), (bx + fill, by + 5 + bar_h), color, -1)
    cv2.rectangle(frame, (bx, by + 5), (bx + bar_w, by + 5 + bar_h), (200, 200, 200), 1)

    pct = f"{progress * 100:.0f}%"
    cv2.putText(frame, pct, (bx + bar_w + 10, by + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if diag:
        cv2.putText(frame, f"EAR: {raw[0]:.3f}  Gaze: {raw[1]:.3f}  FPS: {fps:.1f}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


def _draw_normal(frame, label, prob, raw, alert):
    # Score bar at top
    h, w = frame.shape[:2]
    bar_h = 6
    bar_fill = int(w * min(prob, 1.0))
    bar_color = (0, 255, 0) if prob < 0.5 else (0, 165, 255) if prob < 0.7 else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (bar_fill, bar_h), bar_color, -1)
    cv2.rectangle(frame, (0, 0), (w, bar_h), (60, 60, 60), 1)

    color = (0, 255, 0) if label == "ALERT" else (0, 0, 255)
    cv2.putText(frame, f"{label}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"{prob:.0%}", (20 + 180, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.putText(frame, f"EAR: {raw[0]:.2f}  PERCLOS: {raw[5]:.2f}  Blinks: {raw[6]:.0f}/min",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    if alert:
        # Flash effect for alerts
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"ALERT: {alert['severity']} — {alert['trigger']}",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _draw_diag(frame, raw, label, prob, model_raw, fps, buffer, model_type, alert):
    h, w = frame.shape[:2]
    panel_w = 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    white = (255, 255, 255)
    yellow = (0, 200, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    gray = (140, 140, 140)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    dy = 18

    def line(lbl, val, col=white):
        nonlocal y
        cv2.putText(frame, f"{lbl}: {val}", (8, y), font, 0.38, col, 1)
        y += dy

    def header(txt):
        nonlocal y
        y += 4
        cv2.putText(frame, txt, (8, y), font, 0.38, yellow, 1)
        y += dy

    cv2.putText(frame, "DIAGNOSTIC", (8, y), font, 0.5, yellow, 2)
    y += dy + 3

    line("Model", model_type, green if model_type != "none" else red)
    line("FPS", f"{fps:.1f}")
    line("Norm samples", f"{len(buffer.norm_buffer)}")
    line("Seq ready", "Yes" if len(buffer.norm_buffer) >= buffer.seq_len else "No")

    header("FEATURES (raw)")
    names = ["EAR", "Gaze", "Roll", "Pitch", "Yaw",
             "PERCLOS", "Blink/min", "EAR var", "EAR min", "Gaze var"]
    for i, name in enumerate(names):
        val = raw[i]
        col = white
        if name == "EAR" and val < 0.21:
            col = red
        elif name == "PERCLOS" and val > 0.4:
            col = red
        elif name == "Pitch" and abs(val) > 20:
            col = (0, 165, 255)
        line(name, f"{val:.4f}", col)

    header("INFERENCE")
    line("Model raw", f"{model_raw:.4f}", gray)
    score_color = green if label == "ALERT" else red
    cv2.putText(frame, f"{label}: {prob:.1%}", (8, y), font, 0.55, score_color, 2)
    y += dy + 5

    # Mini probability bar
    bar_x = 8
    bar_w = panel_w - 16
    bar_h2 = 10
    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h2), (60, 60, 60), -1)
    fill = int(bar_w * min(prob, 1.0))
    bar_c = (0, 255, 0) if prob < 0.5 else (0, 165, 255) if prob < 0.7 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, y), (bar_x + fill, y + bar_h2), bar_c, -1)
    # Threshold marker at 50%
    mid = bar_x + bar_w // 2
    cv2.line(frame, (mid, y), (mid, y + bar_h2), (255, 255, 255), 1)
    y += bar_h2 + dy

    if alert:
        cv2.putText(frame, f"ALERT: {alert['severity']}", (8, y), font, 0.55, red, 2)


if __name__ == "__main__":
    main()
