"""Alert smoothing and escalation."""
import time
import config

class AlertManager:
    def __init__(self, smoothing=15, cooldown=5.0):
        self.smoothing = smoothing
        self.cooldown = cooldown
        self._count = 0
        self._last = 0.0
        self._yawns = 0
    def update(self, score, perclos, mar, pitch):
        if score < config.MILD_THRESHOLD:
            self._count = max(0, self._count - 1)
            return None
        self._count += 1
        if mar > config.MAR_THRESHOLD: self._yawns += 1
        if self._count < self.smoothing: return None
        now = time.time()
        if now - self._last < self.cooldown: return None
        self._last = now; self._count = 0
        sev = "Severe" if score >= config.SEVERE_THRESHOLD else "Moderate" if score >= config.MODERATE_THRESHOLD else "Mild"
        triggers = []
        if perclos > config.PERCLOS_THRESHOLD: triggers.append(f"PERCLOS {perclos:.2f}")
        if abs(pitch) > config.HEAD_NOD_PITCH_THRESHOLD: triggers.append("head nod")
        if mar > config.MAR_THRESHOLD: triggers.append("yawn detected")
        if not triggers: triggers.append("sustained low EAR")
        return {"severity": sev, "score": score, "trigger": " + ".join(triggers), "perclos": perclos}