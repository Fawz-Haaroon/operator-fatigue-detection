# Operator Fatigue Detection System

Real-time drowsiness/fatigue detection using **MediaPipe FaceMesh** + **BiLSTM+Attention** neural network (trained weights from SARATHI-V2).

## Quick Start

```bash
git clone https://github.com/Fawz-Haaroon/operator-fatigue-detection.git
cd operator-fatigue-detection
pip install -r requirements.txt

# Download trained model + MediaPipe model
python setup_model.py

# Run
python app.py            # normal mode
python app.py --diag     # with diagnostic overlay
```

## How It Works

1. **Camera** → frames at 640x480
2. **MediaPipe FaceMesh** → 478 facial landmarks (including iris)
3. **Feature extraction** → 10 features: EAR, gaze, roll/pitch/yaw, PERCLOS, blink rate, EAR variance/min, gaze variance
4. **Calibration** → first ~11 seconds build your personal baseline (sit normally, look at camera)
5. **Z-score normalization** → features normalized against baseline for person-independent detection
6. **BiLSTM+Attention** → 200-frame sequence → drowsiness probability (0-1)
7. **Alerting** → severity escalation with smoothing and cooldown

## Model

**Architecture**: BiLSTM+Attention (10 features, 128 hidden, 2 bidirectional LSTM layers, LayerNorm, attention, GELU)

Trained on real labeled drowsiness data. Outputs sigmoid probability: >0.5 = drowsy.

## Configuration

Edit `config.py` — key settings:
- `BASELINE_FRAMES`: Calibration duration (default: 330 = ~11s at 30fps)
- `SEQ_LEN`: Model input window (default: 200 frames)
- `INFERENCE_EVERY`: Model runs every N frames (default: 10)

## Camera Setup (phone via scrcpy)

```bash
scrcpy --v4l2-sink=/dev/video0 --video-source=camera --camera-facing=front \
  --video-codec=h264 --camera-size=1920x1080 --no-audio --no-window --no-playback
```

## Project Structure

```
src/vision/         Camera + MediaPipe FaceMesh (Tasks API)
src/features/       Feature buffer + baseline normalization
src/model/          BiLSTM+Attention network + predictor
src/alerting/       Alert smoothing + escalation
src/api/            Dashboard client
models/             Downloaded model weights
config.py           All settings
app.py              Main entry point
setup_model.py      Model downloader
```
