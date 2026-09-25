"""Test EAR."""
import numpy as np
from src.features.ear import compute_ear

def test_open_eyes():
    lm = np.zeros((468, 3))
    lm[362]=[100,50,0]; lm[385]=[110,40,0]; lm[387]=[120,40,0]; lm[263]=[130,50,0]; lm[373]=[120,60,0]; lm[380]=[110,60,0]
    lm[33]=[200,50,0]; lm[160]=[210,40,0]; lm[158]=[220,40,0]; lm[133]=[230,50,0]; lm[153]=[220,60,0]; lm[144]=[210,60,0]
    l, r = compute_ear(lm)
    assert l > 0.2 and r > 0.2