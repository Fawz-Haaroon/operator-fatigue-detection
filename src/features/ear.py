"""Eye Aspect Ratio."""
import numpy as np
from scipy.spatial import distance

def compute_ear(landmarks):
    left = _ear(landmarks[[362, 385, 387, 263, 373, 380]])
    right = _ear(landmarks[[33, 160, 158, 133, 153, 144]])
    return left, right

def _ear(pts):
    p = pts[:, :2]
    v1 = distance.euclidean(p[1], p[5])
    v2 = distance.euclidean(p[2], p[4])
    h = distance.euclidean(p[0], p[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0