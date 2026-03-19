"""
event_sleep.py
--------------
Handler for the ``sleep`` event type.

Waits ``duration_sec`` ± JITTER_SEC seconds (non-blocking — returns "running"
every tick until the randomised duration elapses, then advances the step).

The actual target duration is picked once per step activation (stored on the
runner as ``_sleep_target``) so it stays stable across ticks.

Constants
---------
JITTER_SEC : float
    Maximum random deviation applied to duration_sec in either direction.
    Actual sleep = duration_sec + uniform(-JITTER_SEC, +JITTER_SEC).
    Set to 0.0 to disable jitter.
"""

import time
import random
import logging

log = logging.getLogger("kha_lastz")

JITTER_SEC = 0.5   # ± seconds of random jitter added to every sleep


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``sleep`` event."""
    now      = time.time()
    base     = float(step.get("duration_sec", 0))

    # Pick a randomised target once per step activation, cache it on the runner.
    if getattr(runner, "_sleep_target", None) is None:
        jitter = random.uniform(-JITTER_SEC, JITTER_SEC) if JITTER_SEC > 0 else 0.0
        runner._sleep_target = max(0.0, base + jitter)
        log.info("[sleep] {} -> target={:.2f}s (base={:.1f}s jitter={:+.2f}s)".format(
            runner._step_label(step), runner._sleep_target, base, jitter))

    if now - runner.step_start_time >= runner._sleep_target:
        log.info("[sleep] {} -> true".format(runner._step_label(step)))
        runner._sleep_target = None
        runner._advance_step(True)
    return "running"
