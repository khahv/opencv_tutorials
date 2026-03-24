"""
event_match_storm_click.py
--------------------------
Handler for the ``match_storm_click`` event type.

Locates a template on screen then hands off to FastClicker for a sustained
high-rate click storm at that position.

Flow
----
1. Search for template every tick until found (or timeout_sec elapsed).
2. On first find: start FastClicker.
3. While storm is running each subsequent tick:
   - Check timeout_sec → stop storm and advance step (True).
   - If FastClicker stopped due to offset_change_time → wait offset_change_pause_sec, then:
       * If storm elapsed >= check_template_after: match on current tick screenshot (shared cache); stop if
         template gone, otherwise restart FastClicker.
       * Otherwise: restart FastClicker without checking.
   - If check_template_frequence > 0 and storm elapsed >= check_template_after:
       pause FastClicker, wait template_verify_settle_sec, then screenshot and verify every
       check_template_frequence seconds (first check on the first tick after the threshold);
       resume storm if still matched, else stop if gone.
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
position_refresh_sec : float
    Re-detect template every N seconds while storm runs (0 = never).
offset_change_time : float
    Seconds between **random** click-position changes (default 1.0). The timer is not reset by
    ``corner:`` restarts — only by a true phase rollover, ``position_refresh_sec``, or a new storm.
offset_change_pause_sec : float
    Idle time after FastClicker stops for an offset rollover before restarting (default 0.2).
storm_click_interval_sec : float
    Seconds to sleep after each storm click (default 0.1 ≈ 10/s). Mutually preferred over
    storm_click_max_rate.
storm_click_max_rate : float
    If set and storm_click_interval_sec is omitted, sleep = 1/rate seconds per click.
check_template_after : float
    Seconds after storm start before template-presence checks are enabled (default 0).
    Before this threshold the template is assumed present and checks are skipped.
check_template_frequence : float
    After check_template_after has elapsed, take a fresh screenshot and verify the
    template is still visible every this many seconds (0 = disabled).
template_verify_settle_sec : float
    After pausing the storm for a periodic template check (or before the offset-change
    screenshot when check_template_after applies), wait this many seconds so click
    effects clear before capturing and matching (default 0.25).
corner : {offset_x, offset_y, every, pause_sec} or {x, y, every, pause_sec}
    Every N clicks: FastClicker stops; after pause_sec (default 0.2) the runner
    sends one isolated corner click (via safe_click), then restarts FastClicker.
    pause_sec : float — wait after storm stop before corner click (seconds).
close_ui_check : bool
    While storm runs, dismiss UI overlays that obscure the target (default True).
    FastClicker is fully stopped before each dismiss click and restarted after.
close_ui_click_x / close_ui_click_y : float
    Ratio coords to click when dismissing UI (default 0.03 / 0.08).
close_ui_back_button : str
    Template for a back/close button to prefer when dismissing UI.
close_ui_check_interval_sec : float
    How often (in seconds) to run the close_ui check while storm runs (default 1.0, min 0.1).
debug_log : bool
    When True, log the best template match score (and threshold) for each periodic template verify
    and for the offset-change template check (after ``check_template_after``).
"""

import time
import logging

log = logging.getLogger("kha_lastz")


def _storm_click_interval_sec(step: dict) -> float:
    """Sleep duration after each storm click (seconds). Default 0.1; cap 0.001–10."""
    v = step.get("storm_click_interval_sec")
    if v is not None:
        return max(0.001, min(10.0, float(v)))
    r = step.get("storm_click_max_rate") or 0
    try:
        rf = float(r)
    except (TypeError, ValueError):
        rf = 0.0
    if rf > 0:
        return max(0.001, min(10.0, 1.0 / rf))
    return 0.1


def _clear_storm_periodic_verify(runner) -> None:
    """Reset periodic template-verify settle state (storm stop or step exit)."""
    runner._storm_periodic_verify_in_flight = False
    runner._storm_verify_settle_until = None


def _storm_resume_fast_clicker_kwargs(runner) -> dict:
    """Rebuild FastClicker kwargs after a pause so count, phase epoch, and pixel stay consistent."""
    _kw = dict(runner._storm_clicker_kwargs)
    _kw.pop("initial_click_count", None)
    _kw.pop("fixed_target_xy", None)
    _lt = runner._fast_clicker.last_target_xy
    if _lt is not None:
        _kw["fixed_target_xy"] = (int(_lt[0]), int(_lt[1]))
    _kw["initial_click_count"] = runner._fast_clicker.click_count
    return _kw


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``match_storm_click`` event."""
    now = time.time()

    template         = step.get("template")
    threshold        = float(step.get("threshold", 0.75))
    timeout_sec      = float(step.get("timeout_sec") or 15)
    offset_x         = int(wincap.w * float(step.get("offset_x") or 0))
    offset_y         = int(wincap.h * float(step.get("offset_y") or 0))
    pos_refresh             = float(step.get("position_refresh_sec") or 0)
    offset_change_time      = float(step.get("offset_change_time", 1.0))
    offset_change_pause_sec = max(0.0, float(step.get("offset_change_pause_sec", 0.2)))
    check_template_after    = float(step.get("check_template_after", 0))
    check_template_frequence = float(step.get("check_template_frequence", 0))
    template_verify_settle_sec = max(0.0, float(step.get("template_verify_settle_sec", 0.25)))
    close_ui_check              = bool(step.get("close_ui_check", True))
    close_ui_click_x            = float(step.get("close_ui_click_x", 0.03))
    close_ui_click_y            = float(step.get("close_ui_click_y", 0.08))
    close_ui_back_btn           = step.get("close_ui_back_button")
    close_ui_check_interval_sec = max(0.1, float(step.get("close_ui_check_interval_sec", 1.0)))
    debug_log                   = bool(step.get("debug_log", False))

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
    _corner_pause_sec = max(0.0, float((_corner_cfg or {}).get("pause_sec", 0.2)))
    _storm_click_sleep = _storm_click_interval_sec(step)

    elapsed = now - runner.step_start_time

    # ── Storm already running ─────────────────────────────────────────────────
    if runner._storm_clicker_active:
        _storm_elapsed = now - runner._storm_start_t

        # Countdown until check_template_after (wall clock from storm start); log every 10s
        if check_template_after > 0:
            _rem_cta = check_template_after - _storm_elapsed
            _cta_log_t = getattr(runner, "_storm_cta_log_t", None)
            if _rem_cta > 0:
                if _cta_log_t is None or now - _cta_log_t >= 10.0:
                    runner._storm_cta_log_t = now
                    log.info("[match_storm_click] {} -> check_template_after: {:.1f}s remaining "
                             "(threshold {:.1f}s from storm start)".format(
                                 runner._step_label(step), _rem_cta, check_template_after))
            elif not getattr(runner, "_storm_cta_armed_logged", False):
                runner._storm_cta_armed_logged = True
                log.info("[match_storm_click] {} -> check_template_after: threshold reached — "
                         "offset-change and periodic template checks apply from now".format(
                             runner._step_label(step)))

        # timeout_sec reached → stop storm, advance step normally
        if elapsed >= timeout_sec:
            _n = runner._fast_clicker.click_count
            _el = max(0.001, now - runner._storm_start_t)
            runner._fast_clicker.stop()
            runner._storm_clicker_active = False
            runner._storm_offset_restart_t = None
            runner._storm_corner_restart_t = None
            _clear_storm_periodic_verify(runner)
            log.info("[match_storm_click] {} -> timeout ({:.0f}s), storm stopped ({} clicks in {:.1f}s)".format(
                runner._step_label(step), timeout_sec, _n, _el))
            runner._advance_step(True, step=step)
            return "running"

        # Periodic template verify: pause storm, settle UI, then one fresh screenshot + match
        if getattr(runner, "_storm_periodic_verify_in_flight", False):
            _su = getattr(runner, "_storm_verify_settle_until", None)
            if _su is not None and now < _su:
                return "running"
            _clear_storm_periodic_verify(runner)
            _fresh = screenshot
            _vision_chk = runner._get_vision(template)
            _still_there = bool(_fresh is not None and _vision_chk
                                and _vision_chk.find(_fresh, threshold=threshold))
            if debug_log and _vision_chk is not None and _fresh is not None:
                _sc = _vision_chk.match_score(_fresh)
                log.info(
                    "[match_storm_click] {} -> periodic template verify: best_match_score={:.3f} "
                    "threshold={:.3f} found={}".format(
                        runner._step_label(step), _sc, threshold, _still_there))
            if not _still_there:
                _n = runner._fast_clicker.click_count
                _el = max(0.001, now - runner._storm_start_t)
                runner._fast_clicker.stop()
                runner._storm_clicker_active = False
                runner._storm_corner_restart_t = None
                log.info("[match_storm_click] {} -> template gone (periodic check after settle), storm stopped "
                         "({} clicks in {:.1f}s)".format(runner._step_label(step), _n, _el))
                runner._advance_step(True, step=step)
                return "running"
            _kw = _storm_resume_fast_clicker_kwargs(runner)
            runner._fast_clicker.start(**_kw)
            runner._storm_template_check_t = now
            log.info("[match_storm_click] {} -> periodic template still present, storm resumed".format(
                runner._step_label(step)))
            return "running"

        # FastClicker stopped due to offset_change_time — check template then restart
        if not runner._fast_clicker.is_running and runner._fast_clicker.offset_changed:
            if runner._storm_offset_restart_t is None:
                runner._storm_offset_restart_t = now
                log.debug("[match_storm_click] {} -> offset change, waiting {:.2f}s before restart".format(
                    runner._step_label(step), offset_change_pause_sec))
            elif now - runner._storm_offset_restart_t >= offset_change_pause_sec:
                runner._storm_offset_restart_t = None
                if _storm_elapsed >= check_template_after:
                    # Uses cached frame for this tick (main loop capture only; no extra grab)
                    _fresh = screenshot
                    _vision_chk = runner._get_vision(template)
                    _still_there = bool(_fresh is not None and _vision_chk
                                        and _vision_chk.find(_fresh, threshold=threshold))
                    if debug_log and _vision_chk is not None and _fresh is not None:
                        _sc = _vision_chk.match_score(_fresh)
                        log.info(
                            "[match_storm_click] {} -> offset-change template verify: best_match_score={:.3f} "
                            "threshold={:.3f} found={}".format(
                                runner._step_label(step), _sc, threshold, _still_there))
                    if not _still_there:
                        _n = runner._fast_clicker.click_count
                        _el = max(0.001, now - runner._storm_start_t)
                        runner._fast_clicker.stop()
                        runner._storm_clicker_active = False
                        runner._storm_corner_restart_t = None
                        _clear_storm_periodic_verify(runner)
                        log.info("[match_storm_click] {} -> template gone at offset change, storm stopped "
                                 "({} clicks in {:.1f}s)".format(runner._step_label(step), _n, _el))
                        runner._advance_step(True, step=step)
                        return "running"
                _kw_off = dict(runner._storm_clicker_kwargs)
                _kw_off.pop("initial_click_count", None)
                _kw_off.pop("fixed_target_xy", None)
                _new_ep = time.monotonic()
                _kw_off["offset_epoch_mono"] = _new_ep
                runner._storm_clicker_kwargs["offset_epoch_mono"] = _new_ep
                runner._storm_clicker_kwargs.pop("fixed_target_xy", None)
                runner._fast_clicker.start(**_kw_off)
                log.info("[match_storm_click] {} -> FastClicker restarted after offset change".format(
                    runner._step_label(step)))
            return "running"

        # FastClicker stopped for isolated corner click, then restart (see YAML corner:)
        if not runner._fast_clicker.is_running and runner._fast_clicker.corner_pause:
            if runner._storm_corner_restart_t is None:
                runner._storm_corner_restart_t = now
                log.debug("[match_storm_click] {} -> corner pause, waiting {:.2f}s before corner click".format(
                    runner._step_label(step), _corner_pause_sec))
            elif now - runner._storm_corner_restart_t >= _corner_pause_sec:
                runner._storm_corner_restart_t = None
                _cp = runner._storm_clicker_kwargs.get("corner_pos")
                _resume = runner._fast_clicker.click_count + 1
                _kw_c = dict(runner._storm_clicker_kwargs)
                _kw_c.pop("initial_click_count", None)
                _kw_c.pop("fixed_target_xy", None)
                _lt = runner._fast_clicker.last_target_xy
                if _lt is not None:
                    _kw_c["fixed_target_xy"] = (int(_lt[0]), int(_lt[1]))
                _kw_c["initial_click_count"] = _resume
                if _cp:
                    runner._safe_click(int(_cp[0]), int(_cp[1]), wincap, "storm corner focus")
                    log.info("[match_storm_click] {} -> corner focus at ({},{}) then storm resumes".format(
                        runner._step_label(step), int(_cp[0]), int(_cp[1])))
                else:
                    log.warning("[match_storm_click] {} -> corner_pause but corner_pos missing; resuming storm".format(
                        runner._step_label(step)))
                runner._fast_clicker.start(**_kw_c)
                log.info("[match_storm_click] {} -> FastClicker restarted after corner focus".format(
                    runner._step_label(step)))
            return "running"

        # Position refresh
        if pos_refresh > 0:
            _ref_el = now - getattr(runner, "_storm_pos_refresh_t", runner._storm_start_t)
            if _ref_el >= pos_refresh:
                runner._storm_pos_refresh_t = now
                _clear_storm_periodic_verify(runner)
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
                        runner._storm_corner_restart_t = None
                        runner._storm_clicker_kwargs = dict(
                            sx=_sx, sy=_sy, rate=0,
                            offset_x=offset_x, offset_y=offset_y,
                            corner_pos=_corner_pos, corner_every=_corner_every,
                            win_bounds=(_rl, _rt, _rr, _rb),
                            offset_change_time=offset_change_time,
                            offset_epoch_mono=time.monotonic(),
                            click_interval_sec=_storm_click_sleep,
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
                    runner._storm_clicker_active = False
                    runner._storm_corner_restart_t = None
                    _clear_storm_periodic_verify(runner)
                    log.info("[match_storm_click] {} -> template gone, storm stopped ({} clicks in {:.1f}s)".format(
                        runner._step_label(step), _n, _el))
                    runner._advance_step(True, step=step)

        # Periodic template check (after check_template_after, every check_template_frequence s)
        if check_template_frequence > 0 and _storm_elapsed >= check_template_after:
            _last_chk = getattr(runner, "_storm_template_check_t", None)
            _run_periodic = False
            if _last_chk is None:
                # First check on the first tick after check_template_after elapses
                _run_periodic = True
            elif now - _last_chk >= check_template_frequence:
                _run_periodic = True
            if _run_periodic:
                if runner._fast_clicker.is_running:
                    runner._fast_clicker.stop()
                runner._storm_periodic_verify_in_flight = True
                runner._storm_verify_settle_until = now + template_verify_settle_sec
                log.info("[match_storm_click] {} -> periodic template verify: storm paused {:.2f}s for UI settle".format(
                    runner._step_label(step), template_verify_settle_sec))
                return "running"

        # close_ui_check every close_ui_check_interval_sec
        _in_periodic_settle = bool(getattr(runner, "_storm_periodic_verify_in_flight", False))
        if close_ui_check and not _in_periodic_settle:
            _cui_el = now - getattr(runner, "_storm_cui_check_t", runner._storm_start_t)
            if _cui_el >= close_ui_check_interval_sec:
                runner._storm_cui_check_t = now
                _cui_vision  = runner._get_vision(template)
                _cui_visible = bool(_cui_vision and _cui_vision.find(screenshot, threshold=threshold))
                if not _cui_visible:
                    # Stop FastClicker before dismissing UI to avoid interference
                    _cui_was_running = runner._fast_clicker.is_running
                    if _cui_was_running:
                        runner._fast_clicker.stop()
                        log.debug("[match_storm_click] {} -> close_ui: FastClicker stopped before dismiss".format(
                            runner._step_label(step)))

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

                    # Restart FastClicker after dismiss
                    if _cui_was_running:
                        _kw_cui = _storm_resume_fast_clicker_kwargs(runner)
                        runner._fast_clicker.start(**_kw_cui)
                        log.debug("[match_storm_click] {} -> close_ui: FastClicker restarted after dismiss".format(
                            runner._step_label(step)))

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

    runner._storm_start_t             = now
    runner._storm_pos_refresh_t       = now
    runner._storm_cui_check_t         = now
    runner._storm_offset_restart_t    = None
    runner._storm_corner_restart_t    = None
    runner._storm_template_check_t    = None
    runner._storm_cta_log_t           = None
    runner._storm_cta_armed_logged  = False
    _clear_storm_periodic_verify(runner)
    runner._storm_clicker_active      = True
    runner._storm_clicker_kwargs = dict(
        sx=sx, sy=sy, rate=0,
        offset_x=offset_x, offset_y=offset_y,
        corner_pos=_corner_pos, corner_every=_corner_every,
        win_bounds=(_win_left, _win_top, _win_right, _win_bottom),
        offset_change_time=offset_change_time,
        offset_epoch_mono=time.monotonic(),
        click_interval_sec=_storm_click_sleep,
    )
    runner._fast_clicker.start(**runner._storm_clicker_kwargs)
    log.info("[match_storm_click] {} -> storm started at ({},{}) timeout_sec={}".format(
        runner._step_label(step), sx, sy, timeout_sec))
    return "running"
