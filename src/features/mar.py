"""Mouth Aspect Ratio."""
import numpy as np
from scipy.spatial import distance

def compute_mar(landmarks):
    pts = landmarks[[61, 291, 39, 181, 0, 17, 269, 405]][:, :2]
    v1 = distance.euclidean(pts[2], pts[6])
    v2 = distance.euclidean(pts[3], pts[7])
    v3 = distance.euclidean(pts[4], pts[5])
    h = distance.euclidean(pts[0], pts[1])
    return (v1 + v2 + v3) / (2.0 * h) if h > 0 else 0.0