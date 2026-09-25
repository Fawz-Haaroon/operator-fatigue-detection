"""Camera capture wrapper."""
import cv2

class CameraCapture:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open camera {index}")
    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    def release(self): self.cap.release()