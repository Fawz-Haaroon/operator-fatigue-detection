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
    last_label = "CALIBRATING"
    last_prob = 0.0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

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

            h, w = frame.shape[:2]
            sequence = feat_buffer.update(landmarks, w, h)
            raw = feat_buffer.raw_features

            if feat_buffer.calibrating:
                progress = feat_buffer.calibration_progress
                _draw_calibration(frame, progress, raw, actual_fps, diag_mode)
                cv2.imshow("Fatigue Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            if sequence is not None and len(feat_buffer.norm_buffer) % config.INFERENCE_EVERY == 0:
                if predictor.model is not None:
                    label, prob = predictor.predict_sequence(sequence)
                    if label is not None:
                        last_label = label
                        last_prob = prob
                else:
                    score = FatiguePredictor.rule_based_score(
                        raw[0], raw[5], raw[1], raw[3], raw[6],
                    )
                    last_prob = score
                    last_label = "DROWSY" if score > 0.5 else "ALERT"

            alert = None
            if last_label == "DROWSY" and last_prob > 0.5:
                alert = alert_manager.update(last_prob, raw[5], 0.0, raw[3])

            if diag_mode:
                _draw_diag(frame, raw, last_label, last_prob, actual_fps,
                           feat_buffer, predictor.model_type, alert)
            else:
                _draw_normal(frame, last_label, last_prob, raw, alert)

            if alert:
                dashboard.send_alert(last_prob, raw[5], raw[6], alert)

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
    bar_h = 20
    bx = (w - bar_w) // 2
    by = h // 2

    cv2.putText(frame, "CALIBRATING - sit normally, look at camera",
                (bx - 40, by - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (100, 100, 100), -1)
    fill = int(bar_w * progress)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), (0, 200, 255), -1)
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (255, 255, 255), 1)
    pct = f"{progress * 100:.0f}%"
    cv2.putText(frame, pct, (bx + bar_w + 10, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if diag:
        cv2.putText(frame, f"EAR: {raw[0]:.3f}  Gaze: {raw[1]:.3f}  FPS: {fps:.1f}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


def _draw_normal(frame, label, prob, raw, alert):
    color = (0, 255, 0) if label == "ALERT" else (0, 0, 255)
    cv2.putText(frame, f"{label} ({prob:.2f})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"EAR: {raw[0]:.2f} | PERCLOS: {raw[5]:.2f} | Blinks: {raw[6]:.0f}/min",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    if alert:
        cv2.putText(frame, f"ALERT: {alert['severity']}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)


def _draw_diag(frame, raw, label, prob, fps, buffer, model_type, alert):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    white = (255, 255, 255)
    yellow = (0, 200, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25
    dy = 20

    def line(lbl, val, col=white):
        nonlocal y
        cv2.putText(frame, f"{lbl}: {val}", (10, y), font, 0.4, col, 1)
        y += dy

    cv2.putText(frame, "DIAGNOSTIC", (10, y), font, 0.55, yellow, 2)
    y += dy + 5

    line("Model", model_type, green if model_type != "none" else red)
    line("FPS", f"{fps:.1f}")
    line("Norm samples", f"{len(buffer.norm_buffer)}")
    y += 5

    cv2.putText(frame, "--- RAW FEATURES ---", (10, y), font, 0.4, yellow, 1)
    y += dy
    names = ["EAR", "Gaze", "Roll", "Pitch", "Yaw",
             "PERCLOS", "Blink/min", "EAR var", "EAR min", "Gaze var"]
    for i, name in enumerate(names):
        val = raw[i]
        col = white
        if name == "EAR" and val < 0.21:
            col = red
        elif name == "PERCLOS" and val > 0.4:
            col = red
        line(name, f"{val:.4f}", col)

    y += 5
    cv2.putText(frame, "--- INFERENCE ---", (10, y), font, 0.4, yellow, 1)
    y += dy
    score_color = green if label == "ALERT" else red
    cv2.putText(frame, f"{label}: {prob:.3f}", (10, y), font, 0.6, score_color, 2)
    y += dy + 5
    if alert:
        cv2.putText(frame, f"ALERT: {alert['severity']}", (10, y), font, 0.6, red, 2)


if __name__ == "__main__":
    main()
