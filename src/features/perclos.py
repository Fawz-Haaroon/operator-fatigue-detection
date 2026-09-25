"""PERCLOS tracker."""
import collections, time

class PERCLOSTracker:
    def __init__(self, window_seconds=60.0, fps=30):
        self.window = window_seconds
        self.fps = fps
        self.history = collections.deque()
        self.blink_count = 0
        self._prev_closed = False
    def update(self, eyes_closed):
        now = time.time()
        self.history.append((now, eyes_closed))
        if eyes_closed and not self._prev_closed: self.blink_count += 1
        self._prev_closed = eyes_closed
        cutoff = now - self.window
        while self.history and self.history[0][0] < cutoff: self.history.popleft()
        if not self.history: return 0.0
        return sum(1 for _, c in self.history if c) / len(self.history)
    @property
    def blink_rate(self):
        if len(self.history) < 2: return 0.0
        elapsed = self.history[-1][0] - self.history[0][0]
        return (self.blink_count / elapsed) * 60 if elapsed > 1 else 0.0