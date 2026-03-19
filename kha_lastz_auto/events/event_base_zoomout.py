"""
event_base_zoomout.py
---------------------
Handler for the ``base_zoomout`` event type.

Delegates to ``zoom_helpers.do_base_zoomout`` which scrolls out to the base
view and confirms by looking for a template (e.g. the World button).

Advances the step on success; logs and retries (without scrolling) when the
World button is not yet visible.

Required YAML keys
------------------
template : str
    Template that confirms we are on the base/world screen.
world_button : str
    Template of the button to click to enter the world view.
"""

import logging

from zoom_helpers import do_base_zoomout

log = logging.getLogger("kha_lastz")


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``base_zoomout`` event."""
    template     = step.get("template")
    world_button = step.get("world_button")

    if not template or not world_button:
        log.error("[base_zoomout] requires 'template' and 'world_button'. Got template=%s, world_button=%s",
                  template, world_button)
        runner._advance_step(False)
        return "running"

    ok = do_base_zoomout(
        wincap, runner.vision_cache, log, template, world_button,
        screenshot=screenshot,
        threshold=float(step.get("threshold", 0.75)),
        scroll_times=int(step.get("scroll_times", 5)),
        scroll_interval_sec=float(step.get("scroll_interval_sec", 0.1)),
        roi_center_x=step.get("roi_center_x"),
        roi_center_y=step.get("roi_center_y"),
        roi_padding=float(step.get("roi_padding", 3.0)),
        log_prefix="[base_zoomout] ",
        debug_log=bool(step.get("debug_log", False)),
        match_color=bool(step.get("match_color", False)),
        color_tolerance=step.get("color_match_tolerance"),
    )

    if ok:
        runner._advance_step(True)
    else:
        log.info("[base_zoomout] no World visible -> retry (no scroll)")
    return "running"
