"""
event_click_position.py
-----------------------
Handler for the ``click_position`` event type.

Clicks a fixed screen position expressed as ratios of the game window size.

The target position can be resolved in three ways (priority order):
1. ``position_setting_key`` + ``positions`` map: look up a fn_settings value
   in a dict of key → [x, y] ratios.  Falls back to ``default`` key if the
   setting value is not in the map.
2. Explicit ``x`` / ``y`` (or legacy ``offset_x`` / ``offset_y``) ratios.
3. Hard-coded default (0.15, 0.15) when nothing else is specified.

After clicking, honours ``on_success_goto`` if present.
"""

import time
import logging

from pynput.mouse import Button, Controller

log = logging.getLogger("kha_lastz")
_mouse_ctrl = Controller()


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``click_position`` event."""
    ox, oy = None, None

    setting_key   = step.get("position_setting_key")
    positions_map = step.get("positions")  # {"8h": [0.75, 0.31], ...}

    if setting_key and positions_map and isinstance(positions_map, dict):
        val = runner._fn_setting(setting_key)
        key = (str(val).strip().lower() if val is not None else "") or \
              str(step.get("default", "")).strip().lower()
        if key in positions_map:
            pos = positions_map[key]
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                ox, oy = float(pos[0]), float(pos[1])
        if ox is None and "default" in step:
            default_key = str(step.get("default", "")).strip().lower()
            if default_key in positions_map:
                pos = positions_map[default_key]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    ox, oy = float(pos[0]), float(pos[1])

    if ox is None:
        ox = step.get("x", step.get("offset_x", 0.15))
        oy = step.get("y", step.get("offset_y", 0.15))

    px = int(wincap.w * ox)
    py = int(wincap.h * oy)
    sx, sy = wincap.get_screen_position((px, py))

    try:
        if not runner._safe_move(sx, sy, wincap, "click_position"):
            runner._advance_step(False)
            return "running"
        time.sleep(0.05)
    except Exception:
        pass

    if hasattr(wincap, "focus_window"):
        wincap.focus_window()

    _mouse_ctrl.press(Button.left)
    time.sleep(0.1)
    _mouse_ctrl.release(Button.left)

    if setting_key and positions_map and ox is not None:
        log.info("[click_position] {} -> true ({} -> x={}, y={})".format(
            runner._step_label(step),
            str(runner._fn_setting(setting_key) or step.get("default", "")).strip().lower(),
            round(ox, 2), round(oy, 2)))
    else:
        log.info("[click_position] {} -> true".format(runner._step_label(step)))

    on_success_goto = step.get("on_success_goto")
    if on_success_goto is not None:
        log.info("[click_position] {} -> success, goto {}".format(
            runner._step_label(step), on_success_goto))
        runner._goto_step(runner._resolve_goto(on_success_goto))
    else:
        runner._advance_step(True, step=step)
    return "running"
