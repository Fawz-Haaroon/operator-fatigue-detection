"""Fatigue predictor."""
import os, collections
try:
    import torch
    from src.model.network import FatigueBiLSTM
    HAS_TORCH = True
except ImportError: HAS_TORCH = False

class FatiguePredictor:
    def __init__(self, path, seq=90, inp=8, hs=128, nl=2):
        self.seq = seq
        self.buf = collections.deque(maxlen=seq)
        self.model = None
        if HAS_TORCH and os.path.exists(path):
            self.model = FatigueBiLSTM(inp, hs, nl)
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
            self.model.eval()
            print(f"[INFO] Loaded model from {path}")
        else: print("[WARN] No model — using rule-based fallback")
    def predict(self, el, er, mar, perclos, pitch, yaw, roll, br):
        self.buf.append([el, er, mar, perclos, pitch, yaw, roll, br])
        if self.model and len(self.buf) >= self.seq:
            with torch.no_grad():
                x = torch.tensor([list(self.buf)], dtype=torch.float32)
                return max(0.0, min(1.0, self.model(x).item()))
        return self._fb(el, er, mar, perclos, pitch, br)
    def _fb(self, el, er, mar, perclos, pitch, br):
        s = 0.0
        if (el+er)/2 < 0.21: s += 0.3
        if perclos > 0.4: s += 0.3
        if mar > 0.6: s += 0.15
        if abs(pitch) > 25: s += 0.15
        if br < 8: s += 0.1
        return min(1.0, s)