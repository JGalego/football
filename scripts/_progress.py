"""Tiny progress reporter shared by download_data.py and prepare_data.py.

Prints plain lines (no carriage returns or ANSI codes) so it reads well in Docker build logs and CI, where output is not a terminal.
"""

from __future__ import annotations

import sys
import threading
import time


def line_buffered() -> None:
    """Flush every line immediately. Python buffers stdout when it is not a terminal, which makes a long build look frozen."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass


def human_time(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class Progress:
    """`update()` after each finished item; prints at most one line every `every` seconds, plus the last one.

    Safe to call from several threads.
    """

    def __init__(self, label: str, total: int, every: float = 5.0) -> None:
        self.label, self.total, self.every = label, total, every
        self.done = self.bytes = 0
        self.t0 = self.last = time.monotonic()
        self.lock = threading.Lock()
        print(f"{label}: {total:,} items", flush=True)

    def update(self, n: int = 1, nbytes: int = 0, note: str = "") -> None:
        with self.lock:
            self.done += n
            self.bytes += nbytes
            now = time.monotonic()
            if now - self.last < self.every and self.done < self.total:
                return
            self.last = now
            elapsed = now - self.t0
            rate = self.done / elapsed if elapsed else 0.0
            eta = (self.total - self.done) / rate if rate else 0.0
            parts = [f"[{self.done:>{len(str(self.total))},}/{self.total:,} {self.done / max(self.total, 1):4.0%}]", f"{human_time(elapsed)} elapsed"]
            if self.done < self.total:
                parts.append(f"ETA {human_time(eta)}")
            if self.bytes:
                parts.append(f"{self.bytes / 1e6:,.0f} MB" + (f" at {self.bytes / 1e6 / elapsed:.1f} MB/s" if elapsed else ""))
            if note:
                parts.append(note)
            print("  " + parts[0] + " " + " · ".join(parts[1:]), flush=True)
