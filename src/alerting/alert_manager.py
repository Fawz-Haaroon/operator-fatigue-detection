"""Alert smoothing and escalation.

The BiLSTM model outputs a drowsiness probability. This manager:
1. Requires sustained high probability (smoothing) before firing
2. Enforces cooldown between alerts
3. Assigns severity based on model probability
"""
import time


# Thresholds (model probability-based)
MILD_THRESHOLD = 0.50
MODERATE_THRESHOLD = 0.65
SEVERE_THRESHOLD = 0.80
PERCLOS_ALERT = 0.40
HEAD_NOD_PITCH = 20


class AlertManager:
    def __init__(self, smoothing=15, cooldown=5.0):
        self.smoothing = smoothing
        self.cooldown = cooldown
        self._count = 0
        self._last = 0.0

    def update(self, score, perclos, mar, pitch):
        """Check if an alert should fire.

        Args:
            score: model drowsiness probability (0-1)
            perclos: current PERCLOS value
            mar: unused (kept for API compat, pass 0.0)
            pitch: head pitch in degrees

        Returns:
            Alert dict or None
        """
        if score < MILD_THRESHOLD:
            self._count = max(0, self._count - 1)
            return None

        self._count += 1

        if self._count < self.smoothing:
            return None

        now = time.time()
        if now - self._last < self.cooldown:
            return None

        self._last = now
        self._count = 0

        # Severity from model probability
        if score >= SEVERE_THRESHOLD:
            sev = "Severe"
        elif score >= MODERATE_THRESHOLD:
            sev = "Moderate"
        else:
            sev = "Mild"

        # Build trigger description
        triggers = []
        if perclos > PERCLOS_ALERT:
            triggers.append(f"PERCLOS {perclos:.2f}")
        if abs(pitch) > HEAD_NOD_PITCH:
            triggers.append("head nod")
        if score >= SEVERE_THRESHOLD:
            triggers.append("high drowsiness score")
        if not triggers:
            triggers.append("sustained drowsiness")

        return {
            "severity": sev,
            "score": score,
            "trigger": " + ".join(triggers),
            "perclos": perclos,
        }
