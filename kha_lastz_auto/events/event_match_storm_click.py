"""
event_match_storm_click.py
--------------------------
Handler for the ``match_storm_click`` event type.

Locates a template on screen then hands off to FastClicker for a sustained
high-rate click storm at that position.

Flow
----
1. Search for template every tick until found (or timeout_sec elapsed).
2. On first find: install WindowClickGuard, start FastClicker.
3. While storm is running each subsequent tick:
   - Check timeout_sec → stop storm and advance step (True).
   - If FastClicker stopped due to offset_change_time → wait 0.2s, then:
       * If storm elapsed >= check_template_after: grab fresh screenshot; stop if
         template gone, otherwise restart FastClicker.
       * Otherwise: restart FastClicker without checking.
   - If check_template_frequence > 0 and storm elapsed >= check_template_after:
       check template every check_template_frequence seconds; stop if gone.
   - If position_refresh_sec elapsed → re-detect and reposition.
   - If close_ui_check fires (1 s timer) → dismiss any UI overlay.

YAML keys
---------
template : str
    Template to find.
threshold : float
    Match threshold (default 0.75).
timeout_sec : float
    Total wall-clock limit for the step. When reached, storm stops and step
    advances with success (True). Default 999.
offset_x / offset_y : float
    Random click offset as fraction of game window size.
guard_outside : bool
    Block clicks that leave the game window (default True).
position_refresh_sec : float
    Re-detect template every N seconds while storm runs (0 = never).
offset_change_time : float
    FastClicker restarts after this many seconds to vary the click position (default 1.0).
check_template_after : float
    Seconds after storm start before template-presence checks are enabled (default 0).
    Before this threshold the template is assumed present and checks are skipped.
check_template_frequence : float
    After check_template_after has elapsed, take a fresh screenshot and verify the
    template is still visible every this many seconds (0 = disabled).
corner : {offset_x, offset_y, every} or {x, y, every}
    Interleave a corner click every N clicks.
close_ui_check : bool
    While storm runs, dismiss UI overlays that obscure the target (default True).
close_ui_click_x / close_ui_click_y : float
    Ratio coords to click when dismissing UI (default 0.03 / 0.08).
close_ui_back_button : str
    Template for a back/close button to prefer when dismissing UI.
"""

import time
import logging

log = logging.getLogger("kha_lastz")


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``match_storm_click`` event."""
    now = time.time()

    template         = step.get("template")
    threshold        = float(step.get("threshold", 0.75))
    timeout_sec      = float(step.get("timeout_sec") or 999)
    offset_x         = int(wincap.w * float(step.get("offset_x") or 0))
    offset_y         = int(wincap.h * float(step.get("offset_y") or 0))
    guard_outside    = bool(step.get("guard_outside", True))
    pos_refresh             = float(step.get("position_refresh_sec") or 0)
    offset_change_time      = float(step.get("offset_change_time", 1.0))
    check_template_after    = float(step.get("check_template_after", 0))
    check_template_frequence = float(step.get("check_template_frequence", 0))
    close_ui_check          = bool(step.get("close_ui_check", True))
    close_ui_click_x    = float(step.get("close_ui_click_x", 0.03))
    close_ui_click_y    = float(step.get("close_ui_click_y", 0.08))
    close_ui_back_btn   = step.get("close_ui_back_button")

    _corner_cfg = step.get("corner")
    if _corner_cfg:
        if "offset_x" in _corner_cfg or "offset_y" in _corner_cfg:
            _cox = _corner_cfg.get("offset_x", 0.05)
            _coy = _corner_cfg.get("offset_y", 0.05)
            _corner_pos = wincap.get_screen_position(
                (int(wincap.w * _cox), int(wincap.h * _coy)))
        else:
            _corner_pos = (int(_corner_cfg["x"]), int(_corner_cfg["y"]))
    else:
        _corner_pos = None
    _corner_every = int((_corner_cfg or {}).get("every", 1000))

    elapsed = now - runner.step_start_time

    # ── Storm already running ─────────────────────────────────────────────────
    if runner._storm_clicker_active:
        # timeout_sec reached → stop storm, advance step normally
        if elapsed >= timeout_sec:
            _n = runner._fast_clicker.click_count
            _el = max(0.001, now - runner._storm_start_t)
            runner._fast_clicker.stop()
            runner._window_click_guard.stop()
            runner._storm_clicker_active = False
            log.info("[match_storm_click] {} -> timeout ({:.0f}s), storm stopped ({} clicks in {:.1f}s)".format(
                runner._step_label(step), timeout_sec, _n, _el))
            runner._advance_step(True, step=step)
            return "running"

        # FastClicker stopped due to offset_change_time — check template then restart
        if not runner._fast_clicker.is_running and runner._fast_clicker.offset_changed:
            if runner._storm_offset_restart_t is None:
                runner._storm_offset_restart_t = now
                log.debug("[match_storm_click] {} -> offset change, waiting 0.2s before restart".format(
                    runner._step_label(step)))
            elif now - runner._storm_offset_restart_t >= 0.2:
                runner._storm_offset_restart_t = None
                _storm_elapsed = now - runner._storm_start_t
                if _storm_elapsed >= check_template_after:
                    # Fresh screenshot check — stop if template gone
                    _fresh = wincap.get_screenshot()
                    _vision_chk = runner._get_vision(template)
                    _still_there = bool(_fresh is not None and _vision_chk
                                        and _vision_chk.find(_fresh, threshold=threshold))
                    if not _still_there:
                        _n = runner._fast_clicker.click_count
                        _el = max(0.001, now - runner._storm_start_t)
                        runner._fast_clicker.stop()
                        runner._window_click_guard.stop()
                        runner._storm_clicker_active = False
                        log.info("[match_storm_click] {} -> template gone at offset change, storm stopped "
                                 "({} clicks in {:.1f}s)".format(runner._step_label(step), _n, _el))
                        runner._advance_step(True, step=step)
                        return "running"
                runner._fast_clicker.start(**runner._storm_clicker_kwargs)
                log.info("[match_storm_click] {} -> FastClicker restarted after offset change".format(
                    runner._step_label(step)))
            return "running"

        # Position refresh
        if pos_refresh > 0:
            _ref_el = now - getattr(runner, "_storm_pos_refresh_t", runner._storm_start_t)
            if _ref_el >= pos_refresh:
                runner._storm_pos_refresh_t = now
                vision = runner._get_vision(template)
                pts = vision.find(screenshot, threshold=threshold) if vision else []
                if pts:
                    _cx, _cy = int(pts[0][0]), int(pts[0][1])
                    _sx, _sy = wincap.get_screen_position((_cx, _cy))
                    _rl, _rt = wincap.get_screen_position((0, 0))
                    _rr, _rb = _rl + wincap.w, _rt + wincap.h
                    if _rl <= _sx < _rr and _rt <= _sy < _rb:
                        runner._fast_clicker.stop()
                        runner._storm_offset_restart_t = None
                        runner._storm_clicker_kwargs = dict(
                            sx=_sx, sy=_sy, rate=0,
                            offset_x=offset_x, offset_y=offset_y,
                            corner_pos=_corner_pos, corner_every=_corner_every,
                            win_bounds=(_rl, _rt, _rr, _rb),
                            offset_change_time=offset_change_time,
                        )
                        runner._fast_clicker.start(**runner._storm_clicker_kwargs)
                        log.info("[match_storm_click] {} -> position refresh at ({},{})".format(
                            runner._step_label(step), _sx, _sy))
                    else:
                        log.warning("[match_storm_click] {} -> refresh target ({},{}) outside window".format(
                            runner._step_label(step), _sx, _sy))
                else:
                    _n = runner._fast_clicker.click_count
                    _el = max(0.001, now - runner._storm_start_t)
                    runner._fast_clicker.stop()
                    runner._window_click_guard.stop()
                    runner._storm_clicker_active = False
                    log.info("[match_storm_click] {} -> template gone, storm stopped ({} clicks in {:.1f}s)".format(
                        runner._step_label(step), _n, _el))
                    runner._advance_step(True, step=step)

        # Periodic template check (after check_template_after, every check_template_frequence s)
        if check_template_frequence > 0:
            _storm_elapsed = now - runner._storm_start_t
            if _storm_elapsed >= check_template_after:
                _last_chk = getattr(runner, "_storm_template_check_t", None)
                if _last_chk is None:
                    runner._storm_template_check_t = now
                elif now - _last_chk >= check_template_frequence:
                    runner._storm_template_check_t = now
                    _fresh = wincap.get_screenshot()
                    _vision_chk = runner._get_vision(template)
                    _still_there = bool(_fresh is not None and _vision_chk
                                        and _vision_chk.find(_fresh, threshold=threshold))
                    if not _still_there:
                        _n = runner._fast_clicker.click_count
                        _el = max(0.001, now - runner._storm_start_t)
                        runner._fast_clicker.stop()
                        runner._window_click_guard.stop()
                        runner._storm_clicker_active = False
                        log.info("[match_storm_click] {} -> template gone (periodic check), storm stopped "
                                 "({} clicks in {:.1f}s)".format(runner._step_label(step), _n, _el))
                        runner._advance_step(True, step=step)
                        return "running"

        # close_ui_check every 1 s
        if close_ui_check:
            _cui_el = now - getattr(runner, "_storm_cui_check_t", runner._storm_start_t)
            if _cui_el >= 1.0:
                runner._storm_cui_check_t = now
                _cui_vision  = runner._get_vision(template)
                _cui_visible = bool(_cui_vision and _cui_vision.find(screenshot, threshold=threshold))
                if not _cui_visible:
                    _dismissed = False
                    if close_ui_back_btn:
                        _vback = runner._get_vision(close_ui_back_btn)
                        if _vback:
                            _bpts = _vback.find(screenshot, threshold=0.75)
                            if _bpts:
                                _bsx, _bsy = wincap.get_screen_position(
                                    (int(_bpts[0][0]), int(_bpts[0][1])))
                                runner._safe_click(_bsx, _bsy, wincap, "storm close_ui back")
                                log.info("[match_storm_click] {} -> close_ui: BackButton clicked ({},{})".format(
                                    runner._step_label(step), _bsx, _bsy))
                                _dismissed = True
                    if not _dismissed:
                        _cpx = int(wincap.w * close_ui_click_x)
                        _cpy = int(wincap.h * close_ui_click_y)
                        _csx, _csy = wincap.get_screen_position((_cpx, _cpy))
                        runner._safe_click(_csx, _csy, wincap, "storm close_ui")
                        log.info("[match_storm_click] {} -> close_ui: dismissed at ({},{})".format(
                            runner._step_label(step), _csx, _csy))

        return "running"

    # ── Storm not started yet ─────────────────────────────────────────────────
    if elapsed >= timeout_sec:
        log.info("[match_storm_click] {} -> timeout ({:.0f}s), template never found".format(
            runner._step_label(step), timeout_sec))
        runner._advance_step(True, step=step)
        return "running"

    vision = runner._get_vision(template)
    if vision is None:
        log.warning("[match_storm_click] {} -> template not found: {}".format(
            runner._step_label(step), template))
        runner._advance_step(False, step=step)
        return "running"

    points = vision.find(screenshot, threshold=threshold)
    if not points:
        return "running"

    cx, cy = int(points[0][0]), int(points[0][1])
    sx, sy = wincap.get_screen_position((cx, cy))

    _win_left, _win_top = wincap.get_screen_position((0, 0))
    _win_right  = _win_left + wincap.w
    _win_bottom = _win_top  + wincap.h
    if not (_win_left <= sx < _win_right and _win_top <= sy < _win_bottom):
        log.warning("[match_storm_click] {} -> click target ({},{}) outside game window".format(
            runner._step_label(step), sx, sy))
        return "running"

    if guard_outside:
        runner._window_click_guard.start(_win_left, _win_top, _win_right, _win_bottom)

    runner._storm_start_t             = now
    runner._storm_pos_refresh_t       = now
    runner._storm_cui_check_t         = now
    runner._storm_offset_restart_t    = None
    runner._storm_template_check_t    = None
    runner._storm_clicker_active      = True
    runner._storm_clicker_kwargs = dict(
        sx=sx, sy=sy, rate=0,
        offset_x=offset_x, offset_y=offset_y,
        corner_pos=_corner_pos, corner_every=_corner_every,
        win_bounds=(_win_left, _win_top, _win_right, _win_bottom),
        offset_change_time=offset_change_time,
    )
    runner._fast_clicker.start(**runner._storm_clicker_kwargs)
    log.info("[match_storm_click] {} -> storm started at ({},{}) timeout_sec={} guard={}".format(
        runner._step_label(step), sx, sy, timeout_sec, guard_outside))
    return "running"
