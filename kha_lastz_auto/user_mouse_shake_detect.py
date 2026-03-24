"""
user_mouse_shake_detect.py
--------------------------
Pure, platform-agnostic detection of a **horizontal zoom-shake** gesture: move the
cursor left (decreasing x), then right (increasing x), within a short time window
— similar to macOS accessibility zoom.

Used by ``user_mouse_abort``. Tune and verify with::

    python test_fast_user_mouse.py
    python test_fast_user_mouse.py --synthetic-tests

``samples`` are ``(t_mono, x, y)`` with ``t_mono`` in seconds (e.g. ``time.perf_counter()``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ZoomShakeResult:
    """A matched gesture."""

    left_leg_px: float
    right_leg_px: float
    span_sec: float


@dataclass(frozen=True)
class ZoomShakeParams:
    """Thresholds for ``detect_zoom_shake``."""

    left_leg_px: float = 75.0
    right_leg_px: float = 75.0
    window_sec: float = 0.42
    max_downstroke_sec: float = 0.22
    max_upstroke_sec: float = 0.22
    # Reject "one long drag + tiny correction": require similar magnitude on both legs.
    # e.g. 0.4 means the shorter leg must be at least 40% of the longer leg.
    min_leg_balance_ratio: float = 0.4
    # Reject whole-screen motion mistaken for a shake; 0 disables.
    max_single_leg_px: float = 400.0


def detect_zoom_shake(
    samples: Sequence[tuple[float, float | int, float | int]],
    params: ZoomShakeParams,
) -> ZoomShakeResult | None:
    """Return a match if the x-trace shows left-then-right within ``params``; else None.

    Screen coordinates: smaller x is further left.
    """
    if len(samples) < 3:
        return None
    xs = [float(s[1]) for s in samples]
    ts = [float(s[0]) for s in samples]
    n = len(xs)
    t_span = ts[-1] - ts[0]
    if t_span <= 0.0 or t_span > params.window_sec:
        return None

    for im in range(1, n - 1):
        idx_high = max(range(im), key=lambda i: xs[i])
        peak_left = xs[idx_high]
        trough_x = xs[im]
        left_leg = peak_left - trough_x
        if left_leg < params.left_leg_px:
            continue
        if params.max_downstroke_sec > 0.0 and (ts[im] - ts[idx_high]) > params.max_downstroke_sec:
            continue

        suffix = xs[im + 1 :]
        if not suffix:
            continue
        peak_right = max(suffix)
        right_leg = peak_right - trough_x
        if right_leg < params.right_leg_px:
            continue

        idx_peak_r = im + 1 + suffix.index(peak_right)
        if params.max_upstroke_sec > 0.0 and (ts[idx_peak_r] - ts[im]) > params.max_upstroke_sec:
            continue

        lo, hi = (left_leg, right_leg) if left_leg <= right_leg else (right_leg, left_leg)
        if hi <= 0.0:
            continue
        balance = lo / hi
        if balance < params.min_leg_balance_ratio:
            continue

        if params.max_single_leg_px > 0.0:
            if left_leg > params.max_single_leg_px or right_leg > params.max_single_leg_px:
                continue

        return ZoomShakeResult(left_leg_px=left_leg, right_leg_px=right_leg, span_sec=t_span)
    return None


def run_synthetic_self_tests() -> None:
    """Run built-in cases; raises AssertionError on failure."""
    p = ZoomShakeParams(
        left_leg_px=70.0,
        right_leg_px=70.0,
        window_sec=0.5,
        max_downstroke_sec=0.3,
        max_upstroke_sec=0.3,
        min_leg_balance_ratio=0.38,
        max_single_leg_px=500.0,
    )
    t0 = 1000.0  # arbitrary mono base
    # Good: ~120px left then ~120px right, balanced
    good: list[tuple[float, float, float]] = []
    x = 500.0
    for i in range(12):
        good.append((t0 + i * 0.012, x, 100.0))
        x -= 10.0
    trough_x = x
    for j in range(12):
        good.append((t0 + 0.15 + j * 0.012, trough_x + (j + 1) * 10.0, 100.0))
    assert detect_zoom_shake(good, p) is not None, "expected balanced L→R shake to match"

    # Bad: one-direction drift ~1082px left then only ~95px right (user log false positive)
    bad: list[tuple[float, float, float]] = []
    x = 2000.0
    for i in range(40):
        bad.append((t0 + i * 0.01, x, 100.0))
        x -= 28.0
    for j in range(8):
        bad.append((t0 + 0.41 + j * 0.01, x + (j + 1) * 12.0, 100.0))
    assert detect_zoom_shake(bad, p) is None, "expected huge left + tiny right to NOT match"

    # max_single_leg_px: tiny cap should reject the same balanced path that matched ``p`` above
    p_tiny_cap = ZoomShakeParams(
        left_leg_px=60.0,
        right_leg_px=60.0,
        window_sec=0.5,
        max_downstroke_sec=0.3,
        max_upstroke_sec=0.3,
        min_leg_balance_ratio=0.35,
        max_single_leg_px=50.0,
    )
    assert detect_zoom_shake(good, p_tiny_cap) is None, "expected max_single_leg cap to reject"


if __name__ == "__main__":
    run_synthetic_self_tests()
    print("user_mouse_shake_detect: synthetic self-tests OK")
