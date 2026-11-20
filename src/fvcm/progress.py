"""Log-friendly progress reporting: one plain line per wall-clock interval,
rather than tqdm's carriage-return redraws, which are unreadable in a job log.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")

DEFAULT_INTERVAL = 30.0     # seconds between lines for an outer loop
NESTED_INTERVAL = 120.0     # inner loops report more rarely


def fmt_duration(seconds: float) -> str:
    """Compact duration: 45s, 2m14s, 1h03m."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def track(
    items: Sequence[T] | Iterable[T],
    desc: str,
    *,
    total: int | None = None,
    stage: str = "",
    enabled: bool = True,
    interval: float = DEFAULT_INTERVAL,
    label=None,
    stream=None,
) -> Iterator[T]:
    """Iterate `items`, printing a progress line every `interval` seconds.

    `label` maps an item to a short string shown in parentheses.
    """
    stream = stream or sys.stdout
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None

    prefix = f"[{stage}] " if stage else ""
    start = last = time.monotonic()
    n = 0

    def emit(item, final: bool = False) -> None:
        now = time.monotonic()
        elapsed = now - start
        rate = n / elapsed if elapsed > 0 else 0.0
        parts = [f"{prefix}{desc:<12s}"]
        if total:
            parts.append(f"{n:>5d}/{total:<5d} {100 * n / total:>3.0f}%")
        else:
            parts.append(f"{n:>5d}")
        parts.append(f"elapsed {fmt_duration(elapsed):>7s}")
        if total and n and not final:
            parts.append(f"eta {fmt_duration((total - n) / rate):>7s}")
        elif final:
            parts.append("done" + " " * 7)
        if rate:
            per_min = rate * 60
            parts.append(f"rate {per_min:5.1f}/min" if per_min < 600
                         else f"rate {rate:5.1f}/s")
        if label is not None and item is not None:
            try:
                parts.append(f"({label(item)})")
            except Exception:  # noqa: BLE001
                pass
        print("  ".join(parts), file=stream, flush=True)

    if not enabled:
        yield from items
        return

    for item in items:
        if n == 0:
            emit(item)
        yield item
        n += 1
        now = time.monotonic()
        if now - last >= interval and (total is None or n < total):
            last = now
            emit(item)
    emit(None, final=True)


def nested(items, desc, **kw) -> Iterator:
    """`track` with the longer interval, for inner loops."""
    kw.setdefault("interval", NESTED_INTERVAL)
    return track(items, desc, **kw)


class Heartbeat:
    """Periodic 'still alive' line for a long stretch with no natural loop."""

    def __init__(self, desc: str, stage: str = "", interval: float = 60.0):
        self.desc, self.stage, self.interval = desc, stage, interval
        self.start = self.last = time.monotonic()

    def tick(self, note: str = "") -> None:
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.last = now
            prefix = f"[{self.stage}] " if self.stage else ""
            msg = f"{prefix}{self.desc}: still running after " \
                  f"{fmt_duration(now - self.start)}"
            print(f"{msg}  {note}".rstrip(), flush=True)
