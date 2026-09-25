"""Download trained models. Run once after cloning:  python setup_model.py"""
import os
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

DOWNLOADS = [
    ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
     os.path.join(MODEL_DIR, "face_landmarker.task"), "face_landmarker.task"),
    ("https://github.com/Fawz-Haaroon/SARATHI-V2/raw/main/models/bilstm_sarathi.pth",
     os.path.join(MODEL_DIR, "bilstm_sarathi.pth"), "bilstm_sarathi.pth"),
]

def download(url, path, name):
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"[OK] {name} already exists ({size:,} bytes)")
        return
    print(f"[INFO] Downloading {name} ...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    size = os.path.getsize(path)
    print(f"[OK] {name} downloaded ({size:,} bytes)")

if __name__ == "__main__":
    for url, path, name in DOWNLOADS:
        download(url, path, name)
    print("\nAll models ready. Run: python app.py")
