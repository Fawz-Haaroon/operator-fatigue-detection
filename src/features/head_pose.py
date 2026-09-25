"""Head pose estimation."""
import numpy as np, cv2

class HeadPoseEstimator:
    def __init__(self, fw, fh):
        f = fw
        c = (fw / 2, fh / 2)
        self.cam = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros((4, 1))
        self.model = np.array([(0,0,0),(0,-330,-65),(-225,170,-135),(225,170,-135),(-150,-150,-125),(150,-150,-125)], dtype=np.float64)
    def estimate(self, lm):
        pts = np.array([lm[1][:2], lm[152][:2], lm[263][:2], lm[33][:2], lm[61][:2], lm[291][:2]], dtype=np.float64)
        _, rv, _ = cv2.solvePnP(self.model, pts, self.cam, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        rm, _ = cv2.Rodrigues(rv)
        a, _, _, _, _, _ = cv2.RQDecomp3x3(rm)
        return a[0], a[1], a[2]