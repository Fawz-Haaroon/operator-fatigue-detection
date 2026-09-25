"""Push readings to Zite dashboard."""
import requests

class DashboardClient:
    def __init__(self, url, session_id, interval=1.0):
        self.url = url; self.sid = session_id
    def send_alert(self, score, perclos, blink_rate, alert):
        if not self.url: return
        try: requests.post(self.url, json={"sessionId": self.sid, "fatigueScore": round(score, 3), "perclos": round(perclos, 3), "blinkRate": round(blink_rate, 1), "trigger": alert.get("trigger", ""), "durationSeconds": 0}, timeout=2)
        except Exception as e: print(f"[WARN] Dashboard push failed: {e}")
    def close(self): pass