"""PERCLOS — Percentage of Eye Closure over a sliding time window.

PERCLOS is defined as the fraction of time the eyes are closed (EAR below
threshold) over a rolling window. This implementation stores per-sample
timestamps so it works correctly regardless of actual processing FPS.

Blink counting: a blink is an open->closed transition. To keep blink_rate
accurate over the window, we store each blink timestamp separately and
discard those outside the window — NOT a cumulative counter.
"""
import collections
import time


class PERCLOSTracker:
    def __init__(self, window_seconds: float = 60.0, fps: int = 30):
        self.window = window_seconds
        self.fps = fps
        # (timestamp, eyes_closed_bool)
        self.history: collections.deque = collections.deque()
        # Blink timestamps (open->closed transitions) within the window
        self._blink_times: collections.deque = collections.deque()
        self._prev_closed = False

    def update(self, eyes_closed: bool) -> float:
        now = time.time()
        self.history.append((now, eyes_closed))

        # Detect blink: transition from open to closed
        if eyes_closed and not self._prev_closed:
            self._blink_times.append(now)
        self._prev_closed = eyes_closed

        # Trim samples outside the window
        cutoff = now - self.window
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        if not self.history:
            return 0.0

        closed_count = sum(1 for _, c in self.history if c)
        return closed_count / len(self.history)

    @property
    def blink_rate(self) -> float:
        """Blinks per minute, computed over the current window."""
        if len(self.history) < 2:
            return 0.0
        elapsed = self.history[-1][0] - self.history[0][0]
        if elapsed < 1.0:
            return 0.0
        # Use actual blink count within the window, scaled to per-minute
        return (len(self._blink_times) / elapsed) * 60.0

    @property
    def sample_count(self) -> int:
        """Number of samples currently in the window."""
        return len(self.history)

    @property
    def effective_fps(self) -> float:
        """Actual samples per second over the window."""
        if len(self.history) < 2:
            return 0.0
        elapsed = self.history[-1][0] - self.history[0][0]
        return len(self.history) / elapsed if elapsed > 0 else 0.0
