"""Camera capture wrapper with explicit resize for v4l2loopback."""
import cv2


class CameraCapture:
    def __init__(self, index=0, width=640, height=480):
        self.target_width = width
        self.target_height = height
        self.cap = cv2.VideoCapture(index)
        # Try to set resolution — v4l2loopback may ignore this
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        # Read actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.needs_resize = (actual_w != width or actual_h != height)
        if self.needs_resize:
            print(f"[INFO] Camera provides {actual_w}x{actual_h}, will resize to {width}x{height}")
        else:
            print(f"[INFO] Camera provides {actual_w}x{actual_h}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        if self.needs_resize:
            frame = cv2.resize(frame, (self.target_width, self.target_height))
        return frame

    def release(self):
        self.cap.release()
