"""Fatigue predictor — loads trained SARATHI BiLSTM+Attention model."""
import os

try:
    import torch
    from src.model.network import BiLSTM_Attention
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class FatiguePredictor:
    def __init__(self, model_path, device="cpu"):
        self.model = None
        self.device = device
        self.model_type = "none"

        if not HAS_TORCH:
            print("[WARN] PyTorch not available — using rule-based fallback")
            return

        if os.path.exists(model_path):
            try:
                self.model = BiLSTM_Attention(n_features=10, hidden=128, n_layers=2, dropout=0.5)
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.model_type = "bilstm_attention"
                print(f"[INFO] Loaded BiLSTM+Attention model from {model_path}")
            except Exception as e:
                print(f"[WARN] Failed to load model: {e}")
                self.model = None
        else:
            print(f"[WARN] No model at {model_path} — run: python setup_model.py")

    def predict_sequence(self, sequence_np):
        if self.model is None:
            return None, None
        with torch.no_grad():
            x = torch.FloatTensor(sequence_np).unsqueeze(0).to(self.device)
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()
        label = "DROWSY" if prob > 0.5 else "ALERT"
        return label, prob

    @staticmethod
    def rule_based_score(ear, perclos, gaze, pitch, blink_rate):
        score = 0.0
        if ear < 0.21:
            score += 0.30
        if perclos > 0.40:
            score += 0.30
        if abs(pitch) > 20:
            score += 0.15
        if blink_rate < 5:
            score += 0.10
        if blink_rate > 25:
            score += 0.10
        return min(1.0, score)
