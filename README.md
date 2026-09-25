# Operator Fatigue Detection System

Real-time drowsiness/fatigue detection combining the best of SafeSteer, SARATHI-V2, driver-drowsiness-system-FYP, and DriverBehaviorDetectionSystem.

## Features
- **MediaPipe FaceMesh** landmark detection (468 points)
- **EAR** (Eye Aspect Ratio) for blink detection
- **MAR** (Mouth Aspect Ratio) for yawn detection
- **PERCLOS** (Percentage of Eye Closure) over a sliding window
- **Head Pose** estimation (pitch/yaw/roll)
- **BiLSTM + Attention** model for fatigue classification
- **Smoothed alerting** with escalation levels (Mild → Moderate → Severe)
- **FastAPI** server pushing readings to a remote dashboard

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

## Structure
```
├── src/vision/          # Camera + MediaPipe FaceMesh
├── src/features/        # EAR, MAR, PERCLOS, head pose
├── src/model/           # BiLSTM + attention network
├── src/training/        # Training loop
├── src/alerting/        # Alert smoothing + escalation
├── src/api/             # FastAPI + dashboard client
├── dashboard/           # Local web UI
├── models/              # Saved checkpoints
├── config.py            # All thresholds
├── app.py               # Entry point
└── Dockerfile
```

## Configuration
Edit `config.py` to tune: EAR_THRESHOLD, MAR_THRESHOLD, PERCLOS_WINDOW, ALERT_SMOOTHING, severity thresholds, DASHBOARD_URL.

## Training
```bash
python -m src.training.train --data dataset/ --epochs 50
```

## License
MIT