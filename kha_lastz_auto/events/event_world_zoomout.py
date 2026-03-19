"""
event_world_zoomout.py
----------------------
Handler for the ``world_zoomout`` event type.

Delegates to ``zoom_helpers.do_world_zoomout`` which scrolls out to the world
map view and confirms by locating a template.

Advances on success; retries on each tick until ``timeout_sec`` is reached.
Optionally saves a debug PNG of the ROI crop when ``debug_save: true``.

Required YAML keys
------------------
template : str
    Template that confirms the world view is visible.
world_button : str
    Template of the button used to reach the world view.
"""

import os
import time
import logging

import cv2 as cv

from zoom_helpers import do_world_zoomout, _roi_crop

log = logging.getLogger("kha_lastz")


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``world_zoomout`` event."""
    template     = step.get("template")
    world_button = step.get("world_button")

    if not template or not world_button:
        log.error("[world_zoomout] requires 'template' and 'world_button'. Got template=%s, world_button=%s",
                  template, world_button)
        runner._advance_step(False)
        return "running"

    timeout_sec = step.get("timeout_sec", 15)
    if getattr(runner, "_world_zoomout_start", None) is None:
        runner._world_zoomout_start = time.time()

    ok = do_world_zoomout(
        wincap, runner.vision_cache, log, template, world_button,
        screenshot=screenshot,
        threshold=float(step.get("threshold", 0.75)),
        scroll_times=int(step.get("scroll_times", 5)),
        scroll_interval_sec=float(step.get("scroll_interval_sec", 0.1)),
        roi_center_x=step.get("roi_center_x"),
        roi_center_y=step.get("roi_center_y"),
        roi_padding=float(step.get("roi_padding", 3.0)),
        log_prefix="[world_zoomout] ",
        debug_log=bool(step.get("debug_log", False)),
        match_color=bool(step.get("match_color", False)),
        color_tolerance=step.get("color_match_tolerance"),
    )

    if ok:
        runner._world_zoomout_start = None
        runner._advance_step(True)
        return "running"

    elapsed = time.time() - runner._world_zoomout_start
    if elapsed >= timeout_sec:
        log.warning("[world_zoomout] template not found after %.1fs (timeout) -> abort step", elapsed)
        runner._world_zoomout_start = None
        runner._advance_step(False)
        return "running"

    if step.get("debug_save"):
        try:
            vision = runner.vision_cache.get(template)
            search_img, _ = _roi_crop(screenshot, vision,
                                      step.get("roi_center_x"),
                                      step.get("roi_center_y"),
                                      float(step.get("roi_padding", 3.0)))
            if search_img is not None and search_img.size > 0:
                os.makedirs("debug", exist_ok=True)
                ts_str   = time.strftime("%Y%m%d_%H%M%S")
                tpl_name = os.path.splitext(os.path.basename(template))[0]
                path     = os.path.join("debug",
                    "world_zoomout_{}_{}_roi_not_found.png".format(tpl_name, ts_str))
                cv.imwrite(path, search_img)
                log.info("[world_zoomout] debug_save: ROI saved -> {}".format(path))
        except Exception as exc:
            log.info("[world_zoomout] debug_save failed: {}".format(exc))

    log.info("[world_zoomout] template not found (not on world) -> retry")
    return "running"
