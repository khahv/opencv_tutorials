"""
event_match_move.py
-------------------
Handler for the ``match_move`` event type.

Finds a template on screen and moves the mouse cursor to it (no click).
Advances on success; fails on timeout.
"""

import time
import logging

import pyautogui

log = logging.getLogger("kha_lastz")


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``match_move`` event."""
    now = time.time()

    template       = step.get("template")
    threshold      = step.get("threshold", 0.75)
    timeout_sec    = step.get("timeout_sec") or 999
    click_offset_x = step.get("click_offset_x") or 0.0
    click_offset_y = step.get("click_offset_y") or 0.0
    debug_click    = step.get("debug_click", False)
    debug_log      = step.get("debug_log", False)

    vision = runner.vision_cache.get(template)
    if not vision:
        runner._advance_step(True)
        return "running"

    points = vision.find(screenshot, threshold=threshold,
                         debug_mode="info" if (debug_click or debug_log) else None)
    if points:
        rc = points[0]
        cx, cy, mw, mh = (
            (rc[0], rc[1], rc[2], rc[3]) if len(rc) >= 4
            else (rc[0], rc[1], vision.needle_w, vision.needle_h)
        )
        raw_center = (cx, cy)
        center = [cx, cy]
        center[0] += int(click_offset_x * mw)
        center[1] += int(click_offset_y * mh)
        sx, sy = wincap.get_screen_position(tuple(center))

        if debug_log:
            log.info("[match_move] {} | raw=({},{}) needle=({}x{}) matched=({}x{}) "
                     "offset=({},{}) final=({},{}) screen=({},{})".format(
                runner._step_label(step), raw_center[0], raw_center[1],
                vision.needle_w, vision.needle_h, mw, mh,
                click_offset_x, click_offset_y,
                center[0], center[1], sx, sy))

        if debug_click and not getattr(runner, "_debug_click_saved", False):
            runner._debug_click_saved = True
            from bot_engine import _save_debug_image
            _save_debug_image(screenshot, raw_center, tuple(center),
                              mw, mh, "match_move", template)

        import adb_input as _adb_mod
        if _adb_mod.get_adb_input() is not None:
            log.debug("[match_move] ADB mode: skipping desktop mouse move (no hover)")
            runner._advance_step(True)
            return "running"

        pyautogui.moveTo(sx, sy)

        if debug_log:
            actual = pyautogui.position()
            log.info("[match_move] {} -> true | intended=({},{}) actual=({},{}) diff=({},{})".format(
                runner._step_label(step), sx, sy, actual.x, actual.y,
                actual.x - sx, actual.y - sy))

        runner._advance_step(True)
        return "running"

    if now - runner.step_start_time >= timeout_sec:
        log.info("[match_move] {} -> false (not found in {}s)".format(
            runner._step_label(step), timeout_sec))
        runner._advance_step(False)
    return "running"
