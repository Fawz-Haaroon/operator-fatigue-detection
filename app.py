"""Main entry point."""
import cv2
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
    camera = CameraCapture(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    face_mesh = FaceMeshDetector(config.MAX_NUM_FACES, config.MIN_DETECTION_CONFIDENCE, config.MIN_TRACKING_CONFIDENCE)
    perclos_tracker = PERCLOSTracker(config.PERCLOS_WINDOW, config.FPS)
    head_pose = HeadPoseEstimator(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    predictor = FatiguePredictor(config.MODEL_PATH, config.SEQUENCE_LENGTH, config.INPUT_FEATURES, config.HIDDEN_SIZE, config.NUM_LAYERS)
    alert_manager = AlertManager(config.ALERT_SMOOTHING, config.ALERT_COOLDOWN)
    dashboard = DashboardClient(config.DASHBOARD_URL, config.DASHBOARD_SESSION_ID, config.PUSH_INTERVAL)
    print("[INFO] Fatigue detection started. Press q to quit.")
    try:
        while True:
            frame = camera.read()
            if frame is None: break
            landmarks = face_mesh.detect(frame)
            if landmarks is None:
                cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Fatigue Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue
            ear_left, ear_right = compute_ear(landmarks)
            ear = (ear_left + ear_right) / 2.0
            mar = compute_mar(landmarks)
            pitch, yaw, roll = head_pose.estimate(landmarks)
            eyes_closed = ear < config.EAR_THRESHOLD
            perclos = perclos_tracker.update(eyes_closed)
            blink_rate = perclos_tracker.blink_rate
            fatigue_score = predictor.predict(ear_left, ear_right, mar, perclos, pitch, yaw, roll, blink_rate)
            alert = alert_manager.update(fatigue_score, perclos, mar, pitch)
            color = (0, 255, 0) if fatigue_score < config.MILD_THRESHOLD else (0, 165, 255) if fatigue_score < config.MODERATE_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"Score: {fatigue_score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"EAR: {ear:.2f} | MAR: {mar:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"PERCLOS: {perclos:.2f} | Blinks: {blink_rate}/min", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if alert:
                cv2.putText(frame, f"ALERT: {alert['severity']}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                dashboard.send_alert(fatigue_score, perclos, blink_rate, alert)
            cv2.imshow("Fatigue Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        dashboard.close()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()