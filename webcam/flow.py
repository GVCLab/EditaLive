import math
from collections import deque

FIRST_CHUNK = 9
CHUNK = 12
# Maximum generated chunks awaiting playback.
OUTPUT_LEAD = 3


class FrameBuffer:
    def __init__(self, capacity=2 * CHUNK):
        self.capacity = capacity
        self.frames = deque()
        self.last_id = -1
        self.last_capture = None
        self.source_end = None

    def append(self, header, payload):
        frame_id = header.get("frame_id")
        stamp = header.get("capture_ms")
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id <= self.last_id:
            raise ValueError("Frame IDs must increase.")
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("Invalid capture timestamp.")
        if self.last_capture is not None and stamp <= self.last_capture:
            raise ValueError("Capture timestamps must increase.")
        if len(self.frames) >= self.capacity:
            raise ValueError("Input credits exceeded.")
        self.frames.append((frame_id, float(stamp), payload))
        self.last_id, self.last_capture = frame_id, float(stamp)

    def take(self, needed, fps, window):
        """Sample a bounded source window into one fixed-size model chunk."""
        if len(self.frames) < needed:
            return None
        count = min(len(self.frames), max(needed, min(window, needed * 3 // 2)))
        candidates = [self.frames.popleft() for _ in range(count)]
        indices = [int(i * count / (needed - 1)) for i in range(needed - 1)] + [count - 1]
        chosen = [candidates[i] for i in indices]
        start = self.source_end
        if start is None:
            start = candidates[0][1] - 1000.0 / fps
        end = candidates[-1][1]
        self.source_end = end
        return dict(frames=chosen, consumed=count, ratio=count / needed,
                    source_start=start, source_end=end)


class CompletionRate:
    def __init__(self, window=5.0):
        self.window = window
        self.samples = deque()
        self.frames = 0

    def update(self, count, completed_at, started_at=None):
        if not self.samples:
            self.samples.append((started_at if started_at is not None else completed_at, 0))
        self.frames += count
        self.samples.append((completed_at, self.frames))
        while len(self.samples) > 2 and self.samples[1][0] <= completed_at - self.window:
            self.samples.popleft()
        start, frames = self.samples[0]
        elapsed = completed_at - start
        return (self.frames - frames) / elapsed if elapsed > 0 else 0.0
