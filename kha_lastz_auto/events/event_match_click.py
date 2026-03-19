"""
event_match_click.py
--------------------
Handler for the ``match_click`` event type.

Finds a template on screen and clicks it.  Supports a rich feature set:

- single template or template_array (try each in order with per-template timeout)
- ROI crop (roi_center_x/y or search_region) to restrict matching area
- one_shot vs. continuous clicking up to max_clicks at click_interval_sec
- click_storm_sec: hand off to FastClicker for a tight high-rate loop
- no_click: match-only without clicking
- track_tried: iterate through multiple matches, refresh when all tried
- dedup_enable: skip positions already clicked this cycle
- require_text_in_region: edge-density filter to skip non-text matches
- require_bright_region: mean-brightness filter
- exclude_template: reject matches that contain another template inside them
- match_color / color_match_tolerance: colour-aware matching
- click_offset_x/y, click_random_offset_x/y, match_center_x/y: fine-tune click position
- ocr_name_region: OCR a name region relative to each match (for truck logging)
- max_tries / on_max_tries_reach_goto: cycle counter guard
- cache_position: lock on first match and reuse for all subsequent ticks
"""

import os
import re
import time
import random
import logging
import datetime

import cv2 as cv
import pyautogui
from pynput.mouse import Button, Controller
from vision import get_global_scale
from ocr_utils import read_region_relative

log = logging.getLogger("kha_lastz")
_mouse_ctrl = Controller()


# ── Lazy imports from bot_engine to avoid circular dependency ─────────────────

def _get_save_debug_image():
    from bot_engine import _save_debug_image
    return _save_debug_image


def _get_crop_region_relative():
    from bot_engine import _crop_region_relative
    return _crop_region_relative


# ── Public entry point ────────────────────────────────────────────────────────

def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``match_click`` event.

    Parameters
    ----------
    step : dict       Parsed YAML step.
    screenshot        Current game screenshot (BGR).
    wincap            WindowCapture (provides .w, .h, .get_screen_position).
    runner            FunctionRunner (provides all state and helper methods).

    Returns
    -------
    str  Always ``"running"``.
    """
    now = time.time()
    _sidx = runner.step_index

    # ── max_tries visit counter ───────────────────────────────────────────────
    if runner._step_visit_start_times.get(_sidx) != runner.step_start_time:
        runner._step_visit_start_times[_sidx] = runner.step_start_time
        runner._step_visit_counts[_sidx] = runner._step_visit_counts.get(_sidx, 0) + 1

    _max_tries_v = step.get("max_tries")
    _on_max_goto = step.get("on_max_tries_reach_goto")
    if _max_tries_v is not None and _on_max_goto is not None:
        _visits = runner._step_visit_counts.get(_sidx, 0)
        if _visits > int(_max_tries_v):
            log.info("[match_click] {} -> max_tries ({}) reached ({} visits) -> goto {}".format(
                runner._step_label(step), _max_tries_v, _visits, _on_max_goto))
            runner._step_visit_counts[_sidx] = 0
            runner._step_dedup_positions.pop(_sidx, None)
            runner._goto_step(runner._resolve_goto(_on_max_goto))
            return "running"

    # ── Parameter extraction ──────────────────────────────────────────────────
    template       = step.get("template")
    template_array = step.get("template_array")
    threshold      = step.get("threshold", 0.75)
    ratio_test     = step.get("ratio_test")
    min_inliers    = step.get("min_inliers")
    one_shot       = step.get("one_shot", True)
    no_click       = bool(step.get("no_click", False))
    timeout_sec    = step.get("timeout_sec") or 999

    # ── template_array: try each template in order, each with its own timeout ─
    _tpls = template_array if isinstance(template_array, list) and template_array else None
    _tpl_overrides: dict = {}
    if _tpls:
        tpl_idx = int(getattr(runner, "_tpl_array_idx", 0) or 0)
        tpl_idx = max(0, tpl_idx)
        if tpl_idx >= len(_tpls):
            log.info("[match_click] {} -> false (all {} templates exhausted)".format(
                runner._step_label(step), len(_tpls)))
            runner._tpl_array_idx = 0
            runner._tpl_array_start_t = None
            runner._fail_step(step, "(all templates exhausted)")
            return "running"

        cur_tpl = _tpls[tpl_idx]
        _tpl_overrides = cur_tpl if isinstance(cur_tpl, dict) else {}
        cur_tpl_path = _tpl_overrides.get("template", cur_tpl) if _tpl_overrides else (cur_tpl or "")

        last_tpl = getattr(runner, "_tpl_array_last_tpl", None)
        if last_tpl != cur_tpl_path or getattr(runner, "_tpl_array_start_t", None) is None:
            runner._tpl_array_last_tpl = cur_tpl_path
            runner._tpl_array_start_t = time.time()
            runner._step_pos_cache = None

        template = cur_tpl_path
        runner.step_start_time = getattr(runner, "_tpl_array_start_t", now)

    refresh_sleep_sec      = step.get("refresh_sleep_sec", 1.0)
    max_clicks             = step.get("max_clicks") or 999999
    click_interval_sec     = step.get("click_interval_sec") or 0

    _ov_max_clicks = runner._fn_setting("max_clicks")
    if _ov_max_clicks is not None:
        try:
            max_clicks = int(_ov_max_clicks)
        except (ValueError, TypeError):
            pass
    _ov_interval = runner._fn_setting("click_interval_sec")
    if _ov_interval is not None:
        try:
            click_interval_sec = float(_ov_interval)
        except (ValueError, TypeError):
            pass

    click_random_offset   = step.get("click_random_offset") or 0
    click_random_offset_x = step.get("click_random_offset_x")
    click_random_offset_y = step.get("click_random_offset_y")
    click_offset_x        = step.get("click_offset_x") or 0.0
    click_offset_y        = step.get("click_offset_y") or 0.0
    match_center_x = float(_tpl_overrides.get("match_center_x", step.get("match_center_x") or 0.0))
    match_center_y = float(_tpl_overrides.get("match_center_y", step.get("match_center_y") or 0.0))

    click_storm_sec         = step.get("click_storm_sec") or 0.0
    click_storm_max_rate    = step.get("click_storm_max_rate") or 0
    click_storm_offset_x    = int(step.get("click_storm_offset_x") or 0)
    click_storm_offset_y    = int(step.get("click_storm_offset_y") or 0)
    click_storm_block_input = bool(step.get("click_storm_block_input", False))
    _corner_cfg = step.get("click_storm_corner")
    if _corner_cfg:
        if "offset_x" in _corner_cfg or "offset_y" in _corner_cfg:
            _cox = _corner_cfg.get("offset_x", 0.05)
            _coy = _corner_cfg.get("offset_y", 0.05)
            click_storm_corner = wincap.get_screen_position(
                (int(wincap.w * _cox), int(wincap.h * _coy)))
        else:
            click_storm_corner = (int(_corner_cfg["x"]), int(_corner_cfg["y"]))
    else:
        click_storm_corner = None
    click_storm_corner_every      = int((_corner_cfg or {}).get("every", 1000))
    position_refresh_interval     = float(step.get("position_refresh_interval") or 0)

    vision = runner._get_vision(template)
    if not vision:
        log.warning("[match_click] Step {} skipped: template NOT FOUND ({})".format(
            runner._step_label(step), template))
        runner._advance_step(True, step=step)
        return "running"

    debug_click            = step.get("debug_click", False)
    debug_log              = step.get("debug_log", False)
    match_color            = step.get("match_color", False)
    color_match_tolerance  = step.get("color_match_tolerance")
    log_all_scores         = step.get("log_all_scores", False)
    ocr_name_region        = step.get("ocr_name_region")
    cache_position         = step.get("cache_position", False) or step.get("cache_frames", 0) > 0

    # ── ROI crop ──────────────────────────────────────────────────────────────
    _roi_offset = (0, 0)
    _roi_cx     = step.get("roi_center_x")
    _roi_cy     = step.get("roi_center_y")
    _explicit_roi = step.get("search_region")
    if _explicit_roi:
        _sh, _sw = screenshot.shape[:2]
        _rx  = max(0, int(_explicit_roi.get("x", 0.0) * _sw))
        _ry  = max(0, int(_explicit_roi.get("y", 0.0) * _sh))
        _rx2 = min(_sw, _rx + max(1, int(_explicit_roi.get("w", 1.0) * _sw)))
        _ry2 = min(_sh, _ry + max(1, int(_explicit_roi.get("h", 1.0) * _sh)))
        search_img  = screenshot[_ry:_ry2, _rx:_rx2]
        _roi_offset = (_rx, _ry)
    elif _roi_cx is not None and _roi_cy is not None:
        _sh, _sw = screenshot.shape[:2]
        _scale   = get_global_scale()
        _nw_px   = max(1, int(vision.needle_w * _scale))
        _nh_px   = max(1, int(vision.needle_h * _scale))
        _padding = float(step.get("roi_padding", 3.0))
        _cx_px   = int(_roi_cx * _sw)
        _cy_px   = int(_roi_cy * _sh)
        _half_w  = int(_nw_px * _padding)
        _half_h  = int(_nh_px * _padding)
        _rx      = max(0, _cx_px - _half_w)
        _ry      = max(0, _cy_px - _half_h)
        _rx2     = min(_sw, _cx_px + _half_w)
        _ry2     = min(_sh, _cy_px + _half_h)
        search_img  = screenshot[_ry:_ry2, _rx:_rx2]
        _roi_offset = (_rx, _ry)
        if debug_log:
            log.info("[match_click] {} ROI: center=({:.2f},{:.2f}) crop=({},{})→({},{}) tpl={}×{}px".format(
                runner._step_label(step), _roi_cx, _roi_cy, _rx, _ry, _rx2, _ry2, _nw_px, _nh_px))
    else:
        search_img = screenshot

    # ── Template matching ─────────────────────────────────────────────────────
    _cached_pos = getattr(runner, "_step_pos_cache", None)
    if cache_position and _cached_pos is not None:
        points = [_cached_pos]
    else:
        dbg = "info" if (debug_click or debug_log) else None
        meta_list = None
        _use_multi = bool(step.get("track_tried", False))
        if match_color:
            result = vision.find(search_img, threshold=threshold, debug_mode=dbg,
                                 is_color=True, debug_log=debug_log,
                                 color_tolerance=color_match_tolerance,
                                 ratio_test=ratio_test, min_inliers=min_inliers,
                                 log_all_scores=log_all_scores,
                                 multi=_use_multi)
            points    = result[0] if isinstance(result, tuple) else result
            meta_list = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        else:
            points = vision.find(search_img, threshold=threshold, debug_mode=dbg,
                                 debug_log=debug_log,
                                 ratio_test=ratio_test, min_inliers=min_inliers,
                                 log_all_scores=log_all_scores)
        if points and _roi_offset != (0, 0):
            ox, oy = _roi_offset
            points = [(pt[0] + ox, pt[1] + oy) + tuple(pt[2:]) for pt in points]
        if points:
            runner._step_pos_cache = points[0]
        elif debug_log:
            log.info("[match_click] {} -> not found (elapsed={:.1f}s)".format(
                runner._step_label(step), now - runner.step_start_time))
        # Fallback debug save for YellowTruckSmall when color filter drops all
        if not points and debug_click and "YellowTruckSmall" in (template or ""):
            if not getattr(runner, "_debug_yellow_fallback_saved", False):
                _fallback = vision.find(screenshot, threshold=0.5, debug_mode=None)
                if _fallback:
                    runner._debug_yellow_fallback_saved = True
                    rc = _fallback[0]
                    mw, mh = (rc[2], rc[3]) if len(rc) >= 4 else (vision.needle_w, vision.needle_h)
                    rcenter = (rc[0] + int(click_offset_x * mw),
                               rc[1] + int(click_offset_y * mh))
                    _get_save_debug_image()(screenshot, rc[:2], rcenter, mw, mh,
                                           "match_click", template)
                    log.info("[match_click] {} -> 0 passed color filter, saved best match".format(
                        runner._step_label(step)))

    # template_array: per-template timeout → advance to next
    if not points and _tpls and (now - runner.step_start_time >= timeout_sec):
        tpl_idx = int(getattr(runner, "_tpl_array_idx", 0) or 0)
        cur_tpl = _tpls[tpl_idx] if 0 <= tpl_idx < len(_tpls) else None
        _cur_tpl_path = (cur_tpl.get("template", "") if isinstance(cur_tpl, dict) else cur_tpl) or ""
        tpl_name = os.path.splitext(os.path.basename(_cur_tpl_path))[0] if _cur_tpl_path else str(tpl_idx)
        log.info("[match_click] {} [{}] -> not found in {}s, trying next template".format(
            runner._step_label(step), tpl_name, timeout_sec))
        runner._tpl_array_idx = tpl_idx + 1
        runner._tpl_array_start_t = None
        runner._tpl_array_last_tpl = None
        runner._step_pos_cache = None
        return "running"

    # track_tried + 0 results: click refresh to unstick
    if not points and step.get("track_tried") and step.get("refresh_template"):
        _last_refresh = getattr(runner, "_last_zero_refresh_t", 0)
        if now - _last_refresh >= 2.0:
            _refresh_tpl = step.get("refresh_template")
            _v_ref = runner.vision_cache.get(_refresh_tpl) if _refresh_tpl else None
            if _v_ref:
                _ref_pts = _v_ref.find(screenshot, threshold=step.get("threshold", 0.75))
                if _ref_pts:
                    _rsx, _rsy = wincap.get_screen_position(tuple(_ref_pts[0]))
                    runner._safe_click(_rsx, _rsy, wincap, "match_click refresh")
                    log.info("[match_click] {} -> 0 trucks, clicked refresh, sleeping {:.1f}s".format(
                        runner._step_label(step), refresh_sleep_sec))
                    time.sleep(refresh_sleep_sec)
                    runner._tried_positions = []
                    runner._rtr_cache = {}
                    runner._rtr_page_logged = False
                    runner.step_start_time = time.time()
                    runner._last_zero_refresh_t = now
                    return "running"

    if points:
        # ── OCR player name ───────────────────────────────────────────────────
        _point_names = {}
        if ocr_name_region and len(ocr_name_region) == 4:
            _onr_x, _onr_y, _onr_w, _onr_h = ocr_name_region
            for _pt in points:
                cx, cy = _pt[0], _pt[1]
                mw, mh = (_pt[2], _pt[3]) if len(_pt) >= 4 else (vision.needle_w, vision.needle_h)
                _name = read_region_relative(screenshot, cx, cy, mw, mh,
                                             x=_onr_x, y=_onr_y, w=_onr_w, h=_onr_h)
                _point_names[tuple(_pt)] = (_name or "").strip()

        # ── require_text_in_region ────────────────────────────────────────────
        _rtr = step.get("require_text_in_region")
        _crop_rel = _get_crop_region_relative()
        if _rtr and points:
            _rtr_x        = _rtr.get("x", -1.0)
            _rtr_y        = _rtr.get("y", 0.0)
            _rtr_w        = _rtr.get("w", 2.0)
            _rtr_h        = _rtr.get("h", 0.8)
            _rtr_density  = _rtr.get("min_edge_density", 0.05)
            _rtr_canny_lo = _rtr.get("canny_low", 50)
            _rtr_canny_hi = _rtr.get("canny_high", 150)
            _rtr_debug    = _rtr.get("debug_save", False)
            _rtr_top_k    = _rtr.get("top_k")
            _rtr_key = frozenset((round(_p[0] / 10) * 10, round(_p[1] / 10) * 10) for _p in points)
            if _rtr_key in runner._rtr_cache:
                _rtr_passed, _rtr_density_map = runner._rtr_cache[_rtr_key]
                points = _rtr_passed
            else:
                _passed = []
                _rtr_density_map = {}
                _rtr_log_parts = []
                for _pt in points:
                    cx, cy = _pt[0], _pt[1]
                    mw, mh = (_pt[2], _pt[3]) if len(_pt) >= 4 else (vision.needle_w, vision.needle_h)
                    _crop = _crop_rel(screenshot, cx, cy, mw, mh,
                                     _rtr_x, _rtr_y, _rtr_w, _rtr_h)
                    if _crop is None:
                        continue
                    _gray  = cv.cvtColor(_crop, cv.COLOR_BGR2GRAY) if len(_crop.shape) == 3 else _crop
                    _edges = cv.Canny(_gray, _rtr_canny_lo, _rtr_canny_hi)
                    _density = float(_edges.sum()) / (255.0 * max(_edges.size, 1))
                    _ok = _density >= _rtr_density
                    _rtr_density_map[tuple(_pt)] = _density
                    _rtr_log_parts.append("({},{}) {}:{:.3f}".format(
                        _pt[0], _pt[1], "PASS" if _ok else "SKIP", _density))
                    if _rtr_debug:
                        _ts = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
                        _tag = "pass" if _ok else "skip"
                        _dbg_dir = os.path.join(os.path.dirname(__file__), "..", "debug_ocr")
                        os.makedirs(_dbg_dir, exist_ok=True)
                        cv.imwrite(os.path.join(_dbg_dir,
                            "require_text_{}_{}_{},{}.png".format(_tag, _ts, _pt[0], _pt[1])), _crop)
                        cv.imwrite(os.path.join(_dbg_dir,
                            "require_text_edges_{}_{}_{},{}.png".format(_tag, _ts, _pt[0], _pt[1])), _edges)
                    if _ok:
                        _passed.append(_pt)
                if _rtr_top_k:
                    _all_sorted = sorted(points, key=lambda p: _rtr_density_map.get(tuple(p), 0), reverse=True)
                    _passed = _all_sorted[:_rtr_top_k]
                if not runner._rtr_page_logged:
                    log.info("[match_click] {} -> require_text_in_region ({} trucks, thresh={:.3f}): {}".format(
                        runner._step_label(step), len(points), _rtr_density,
                        " | ".join(_rtr_log_parts)))
                runner._rtr_cache[_rtr_key] = (_passed, _rtr_density_map)
                points = _passed
            runner._rtr_density_map_last = _rtr_density_map

        # ── require_bright_region ─────────────────────────────────────────────
        _rbr = step.get("require_bright_region")
        if _rbr and points:
            _rbr_x   = _rbr.get("x", -0.5)
            _rbr_y   = _rbr.get("y", -2.0)
            _rbr_w   = _rbr.get("w", 1.0)
            _rbr_h   = _rbr.get("h", 1.5)
            _rbr_min = _rbr.get("min_mean", 160)
            _passed  = []
            for _pt in points:
                cx, cy = _pt[0], _pt[1]
                mw, mh = (_pt[2], _pt[3]) if len(_pt) >= 4 else (vision.needle_w, vision.needle_h)
                _crop = _crop_rel(screenshot, cx, cy, mw, mh, _rbr_x, _rbr_y, _rbr_w, _rbr_h)
                if _crop is None:
                    continue
                _gray = cv.cvtColor(_crop, cv.COLOR_BGR2GRAY) if len(_crop.shape) == 3 else _crop
                if float(_gray.mean()) >= _rbr_min:
                    _passed.append(_pt)
                else:
                    log.info("[match_click] {} ({},{}) -> require_bright_region SKIP mean={:.1f} < {:.1f}".format(
                        runner._step_label(step), _pt[0], _pt[1], float(_gray.mean()), _rbr_min))
            points = _passed

        # ── exclude_template ──────────────────────────────────────────────────
        _excl_raw = step.get("exclude_template")
        if _excl_raw and points:
            _excl_tpls     = _excl_raw if isinstance(_excl_raw, list) else [_excl_raw]
            _excl_visions  = [v for v in (runner._get_vision(t) for t in _excl_tpls) if v is not None]
            _excl_threshold = float(step.get("exclude_threshold", threshold))
            _excl_ax        = float(_tpl_overrides.get("exclude_area_x", step.get("exclude_area_x", 1.0)))
            _excl_ay        = float(_tpl_overrides.get("exclude_area_y", step.get("exclude_area_y", 1.0)))
            _excl_match_color = bool(step.get("exclude_match_color", False))
            _excl_color_tol   = step.get("exclude_color_tolerance")
            _excl_debug_save  = step.get("debug_save", False)
            if _excl_visions:
                _passed = []
                for _pt in points:
                    cx, cy = _pt[0], _pt[1]
                    _sh, _sw = screenshot.shape[:2]
                    mw = _pt[2] if len(_pt) >= 3 else vision.needle_w
                    mh = _pt[3] if len(_pt) >= 4 else vision.needle_h
                    _ecx = cx + int(match_center_x * _sw)
                    _ecy = cy + int(match_center_y * _sh)
                    _half_x = int(mw / 2 * _excl_ax)
                    _half_y = int(mh / 2 * _excl_ay)
                    _rx  = max(0, _ecx - _half_x)
                    _ry  = max(0, _ecy - _half_y)
                    _rx2 = min(_sw, _ecx + _half_x)
                    _ry2 = min(_sh, _ecy + _half_y)
                    _crop = screenshot[_ry:_ry2, _rx:_rx2]
                    _excluded = False
                    if debug_log:
                        log.info("[match_click] {} exclude check ({},{}): crop={}×{}px".format(
                            runner._step_label(step), _ecx, _ecy, _rx2 - _rx, _ry2 - _ry))
                    for _ev in _excl_visions:
                        if _excl_match_color:
                            _norm_c, _ = _ev._norm_haystack(_crop)
                            if len(_norm_c.shape) == 2:
                                _norm_c = cv.cvtColor(_norm_c, cv.COLOR_GRAY2BGR)
                            _needle_c = _ev.needle_img if len(_ev.needle_img.shape) == 3 else \
                                cv.cvtColor(_ev.needle_img, cv.COLOR_GRAY2BGR)
                            if _ev.needle_w <= _norm_c.shape[1] and _ev.needle_h <= _norm_c.shape[0]:
                                _res_c = cv.matchTemplate(_norm_c, _needle_c, _ev.method)
                                _, _best_score, _, _ = cv.minMaxLoc(_res_c)
                                _best_score = float(_best_score)
                            else:
                                _best_score = 0.0
                        else:
                            _best_score = _ev.match_score(_crop)
                        if _best_score >= _excl_threshold:
                            log.info("[match_click] {} ({},{}) -> exclude [{}] score={:.3f} -> skip".format(
                                runner._step_label(step), _ecx, _ecy,
                                _ev.needle_name, _best_score))
                            _excluded = True
                            break
                        elif debug_log:
                            log.info("[match_click] {} ({},{}) -> exclude [{}] score={:.3f} < {:.2f} -> keep".format(
                                runner._step_label(step), _ecx, _ecy,
                                _ev.needle_name, _best_score, _excl_threshold))
                    if _excl_debug_save:
                        try:
                            _ts  = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
                            _tag = "excluded" if _excluded else "kept"
                            _dbg_dir = "debug_exclude"
                            os.makedirs(_dbg_dir, exist_ok=True)
                            _fname = os.path.join(_dbg_dir,
                                "exclude_{}_{}_{},{}.png".format(_tag, _ts, _ecx, _ecy))
                            if _crop is not None and _crop.size > 0:
                                cv.imwrite(_fname, _crop)
                        except Exception as _de:
                            log.warning("[match_click] exclude debug_save failed: {}".format(_de))
                    if not _excluded:
                        _passed.append(_pt)
                points = _passed

        # ── dedup_enable ──────────────────────────────────────────────────────
        _dedup = step.get("dedup_enable", False)
        if _dedup and points:
            _dedup_tol_x = int(step.get("dedup_tolerance_x", 30))
            _dedup_tol_y = int(step.get("dedup_tolerance_y", 30))
            _seen = runner._step_dedup_positions.get(runner.step_index, [])
            if _seen:
                _before = len(points)
                points = [pt for pt in points if not any(
                    abs(pt[0] - sx) <= _dedup_tol_x and abs(pt[1] - sy) <= _dedup_tol_y
                    for sx, sy in _seen)]
                if len(points) < _before and debug_log:
                    log.info("[match_click] {} dedup_enable: filtered {}/{}".format(
                        runner._step_label(step), _before - len(points), _before))

        # ── track_tried ───────────────────────────────────────────────────────
        track_tried = step.get("track_tried", False)
        _rtr_density_map = getattr(runner, "_rtr_density_map_last", {})
        if track_tried:
            _tol    = step.get("tried_tolerance_px", 30)
            untried = [pt for pt in points if not any(
                abs(pt[0] - tp[0]) < _tol and abs(pt[1] - tp[1]) < _tol
                for tp in runner._tried_positions)]
            if not runner._rtr_page_logged:
                if meta_list is not None and len(meta_list) == len(points):
                    parts = []
                    for p, m in zip(points, meta_list):
                        s = m.get("score")
                        extras = []
                        _pname = _point_names.get(tuple(p), "")
                        if _pname:
                            extras.append("name={}".format(_pname))
                        for _mk, _mfmt in [("region_bgr", None), ("dominant_color", None),
                                           ("color_dist", "{:.0f}"), ("mean_sat", "{:.0f}"),
                                           ("median_hue", "{:.0f}"), ("in_range_frac", "{:.0f}%")]:
                            if m.get(_mk) is not None:
                                if _mk == "region_bgr":
                                    b, g, r = m["region_bgr"]
                                    extras.append("B={:.0f} G={:.0f} R={:.0f}".format(b, g, r))
                                elif _mk == "in_range_frac":
                                    extras.append("frac={:.0f}%".format(m[_mk] * 100))
                                else:
                                    extras.append("{}={}".format(_mk.replace("_", ""), _mfmt.format(m[_mk])))
                        parts.append("({},{}) score={:.3f} [{}]".format(p[0], p[1], s, " ".join(extras)))
                    _pts_str = " | ".join(parts)
                else:
                    parts = []
                    for p in points:
                        _pname = _point_names.get(tuple(p), "")
                        _d = _rtr_density_map.get(tuple(p))
                        parts.append("({},{}){}{}".format(
                            p[0], p[1],
                            " density={:.3f}".format(_d) if _d is not None else "",
                            " name={}".format(_pname) if _pname else ""))
                    _pts_str = ", ".join(parts)
                log.info("[match_click] {} -> found {} truck(s): [{}] | tried={} untried={}".format(
                    runner._step_label(step), len(points), _pts_str,
                    len(runner._tried_positions), len(untried)))
                runner._rtr_page_logged = True
            if not untried:
                _refresh_tpl = step.get("refresh_template")
                _v_ref = runner.vision_cache.get(_refresh_tpl) if _refresh_tpl else None
                if _v_ref:
                    _ref_pts = _v_ref.find(screenshot, threshold=threshold)
                    if _ref_pts:
                        _rsx, _rsy = wincap.get_screen_position(tuple(_ref_pts[0]))
                        runner._safe_click(_rsx, _rsy, wincap, "match_click refresh")
                        log.info("[match_click] {} -> all {} tried -> clicked refresh".format(
                            runner._step_label(step), len(runner._tried_positions)))
                        time.sleep(refresh_sleep_sec)
                        runner._tried_positions = []
                        runner._rtr_cache = {}
                        runner._rtr_page_logged = False
                        runner.step_start_time = time.time()
                        return "running"
                log.info("[match_click] {} -> all tried, no refresh -> abort".format(runner._step_label(step)))
                runner._fail_step(step, "(all tried, no refresh)")
                return "running"
            points = untried

        # All candidates filtered out
        if not points:
            if _tpls and (now - runner.step_start_time >= timeout_sec):
                tpl_idx = int(getattr(runner, "_tpl_array_idx", 0) or 0)
                _cur_tpl = _tpls[tpl_idx] if 0 <= tpl_idx < len(_tpls) else None
                _cur_tpl_path = (_cur_tpl.get("template", "") if isinstance(_cur_tpl, dict) else _cur_tpl) or ""
                _tpl_name = os.path.splitext(os.path.basename(_cur_tpl_path))[0] if _cur_tpl_path else str(tpl_idx)
                log.info("[match_click] {} [{}] -> all filtered in {}s, next template".format(
                    runner._step_label(step), _tpl_name, timeout_sec))
                runner._tpl_array_idx = tpl_idx + 1
                runner._tpl_array_start_t = None
                runner._tpl_array_last_tpl = None
                runner._step_pos_cache = None
            return "running"

        # ── Compute click position ─────────────────────────────────────────────
        cx, cy, mw, mh = (
            (points[0][0], points[0][1], points[0][2], points[0][3])
            if len(points[0]) >= 4
            else (points[0][0], points[0][1], vision.needle_w, vision.needle_h)
        )
        center = [cx, cy]
        center[0] += int(click_offset_x * mw)
        center[1] += int(click_offset_y * mh)
        if match_center_x or match_center_y:
            _msh, _msw = screenshot.shape[:2]
            center[0] += int(match_center_x * _msw)
            center[1] += int(match_center_y * _msh)
        if click_random_offset_x is not None:
            rx = int(click_random_offset_x * mw)
            if rx > 0:
                center[0] += random.randint(-rx, rx)
        elif click_random_offset > 0:
            center[0] += random.randint(-click_random_offset, click_random_offset)
        if click_random_offset_y is not None:
            ry = int(click_random_offset_y * mh)
            if ry > 0:
                center[1] += random.randint(-ry, ry)
        elif click_random_offset > 0:
            center[1] += random.randint(-click_random_offset, click_random_offset)
        sx, sy    = wincap.get_screen_position(tuple(center))
        raw_center = (cx, cy)

        if debug_log:
            _mc_str = " match_center=({},{})".format(match_center_x, match_center_y) if (match_center_x or match_center_y) else ""
            log.info("[match_click] {} | raw=({},{}) needle=({}x{}) matched=({}x{}) offset=({},{}) final=({},{}) screen=({},{}){}".format(
                runner._step_label(step), raw_center[0], raw_center[1],
                vision.needle_w, vision.needle_h, mw, mh,
                click_offset_x, click_offset_y,
                center[0], center[1], sx, sy, _mc_str))

        if debug_click:
            _is_yellow = "YellowTruckSmall" in (template or "")
            if _is_yellow or not getattr(runner, "_debug_click_saved", False):
                if not _is_yellow:
                    runner._debug_click_saved = True
                _crop_path = _get_save_debug_image()(screenshot, raw_center, tuple(center),
                                                     mw, mh, "match_click", template)
                if _is_yellow and _crop_path:
                    runner._last_truck_crop_path = _crop_path

        # ── click_storm via FastClicker ────────────────────────────────────────
        if click_storm_sec > 0 and not one_shot:
            if not runner._fast_clicker.is_running:
                if click_storm_block_input:
                    try:
                        import ctypes
                        if ctypes.windll.user32.BlockInput(True):
                            log.info("[match_click] {} -> storm: BlockInput(True)".format(runner._step_label(step)))
                    except Exception as e:
                        log.warning("[match_click] storm BlockInput(True) failed: {}".format(e))
                runner._storm_start_t            = time.time()
                runner._storm_position_refresh_t = time.time()
                runner._fast_clicker.start(sx, sy,
                    rate=click_storm_max_rate or 0,
                    offset_x=click_storm_offset_x,
                    offset_y=click_storm_offset_y,
                    corner_pos=click_storm_corner,
                    corner_every=click_storm_corner_every)
                log.info("[match_click] {} -> storm started at ({},{}) rate={}".format(
                    runner._step_label(step), sx, sy, click_storm_max_rate or "unlimited"))
            else:
                if position_refresh_interval > 0 and points:
                    _refresh_elapsed = time.time() - getattr(runner, "_storm_position_refresh_t", 0)
                    if _refresh_elapsed >= position_refresh_interval:
                        runner._fast_clicker.stop()
                        runner._fast_clicker.start(sx, sy,
                            rate=click_storm_max_rate or 0,
                            offset_x=click_storm_offset_x,
                            offset_y=click_storm_offset_y,
                            corner_pos=click_storm_corner,
                            corner_every=click_storm_corner_every)
                        runner._storm_position_refresh_t = time.time()
                        log.info("[match_click] {} -> position refresh at ({},{})".format(
                            runner._step_label(step), sx, sy))
                _n       = runner._fast_clicker.click_count
                _elapsed = max(0.001, time.time() - getattr(runner, "_storm_start_t", time.time()))
                if _n >= max_clicks or _elapsed >= click_storm_sec:
                    runner._fast_clicker.stop()
                    if click_storm_block_input:
                        try:
                            import ctypes
                            ctypes.windll.user32.BlockInput(False)
                        except Exception as e:
                            log.warning("[match_click] storm BlockInput(False) failed: {}".format(e))
                    runner.step_click_count += _n
                    _rate = round(_n / _elapsed, 0)
                    log.info("[match_click] {} | storm={} clicks in {:.1f}s (~{:.0f}/s)".format(
                        runner._step_label(step), _n, _elapsed, _rate))
                    log.info("[match_click] {} -> true (clicked {})".format(
                        runner._step_label(step), runner.step_click_count))
                    runner._advance_step(True, step=step)
            return "running"

        # ── no_click: match only ───────────────────────────────────────────────
        if no_click:
            log.info("[match_click] {} -> match found ({},{}) (no_click)".format(
                runner._step_label(step), points[0][0], points[0][1]))
            runner._advance_step(True, step=step)
            return "running"

        # ── Standard click path ────────────────────────────────────────────────
        try:
            if not runner._safe_move(sx, sy, wincap, "match_click"):
                return "running"
            time.sleep(0.05)
        except Exception as e:
            log.warning("[match_click] Failed to set mouse position: {}".format(e))
            return "running"

        if debug_log:
            actual = _mouse_ctrl.position
            log.info("[match_click] {} | intended=({},{}) actual=({},{}) diff=({},{})".format(
                runner._step_label(step), sx, sy, int(actual[0]), int(actual[1]),
                int(actual[0] - sx), int(actual[1] - sy)))

        if hasattr(wincap, "focus_window"):
            wincap.focus_window()

        burst_start = time.time()
        if not one_shot and click_interval_sec == 0:
            burst_end = burst_start + 0.05
            while time.time() < burst_end and runner.step_click_count < max_clicks:
                _mouse_ctrl.press(Button.left)
                time.sleep(0.03)
                _mouse_ctrl.release(Button.left)
                runner.step_click_count += 1
        else:
            _mouse_ctrl.press(Button.left)
            time.sleep(0.1)
            _mouse_ctrl.release(Button.left)
            runner.step_click_count += 1

        if not one_shot:
            click_t = time.time()
            last_t  = getattr(runner, "_step_last_click_t", None)
            if debug_log and last_t is not None:
                frame_interval = click_t - last_t
                actual_rate    = round(runner.step_click_count / max(0.001, click_t - runner.step_start_time), 1)
                log.info("[match_click] {} | frame={:.3f}s total={} rate={:.1f}/s".format(
                    runner._step_label(step), frame_interval, runner.step_click_count, actual_rate))
            runner._step_last_click_t = click_t
            if click_interval_sec > 0 and last_t is not None:
                elapsed   = click_t - last_t
                remaining = click_interval_sec - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        if step.get("track_tried", False):
            runner._tried_positions.append(tuple(points[0]))
        if step.get("dedup_enable", False):
            _sidx_click = runner.step_index
            runner._step_dedup_positions.setdefault(_sidx_click, []).append(
                (points[0][0], points[0][1]))

        if one_shot:
            runner._last_click_pos = tuple(center)
            _click_truck_pt = tuple(points[0])
            _click_density  = getattr(runner, "_rtr_density_map_last", {}).get(_click_truck_pt)
            _density_str    = " density={:.3f}".format(_click_density) if _click_density is not None else ""
            log.info("[match_click] {} -> true (clicked ({},{}) {})".format(
                runner._step_label(step), _click_truck_pt[0], _click_truck_pt[1], _density_str))
            on_success_goto = step.get("on_success_goto")
            if on_success_goto is not None:
                log.info("[match_click] {} -> success, goto {}".format(
                    runner._step_label(step), on_success_goto))
                runner._goto_step(runner._resolve_goto(on_success_goto))
            else:
                runner._advance_step(True, step=step)
            return "running"

        log.info("[match_click] {} clicking... (count {})".format(
            runner._step_label(step), runner.step_click_count))
        if runner.step_click_count >= max_clicks:
            log.info("[match_click] {} -> true (clicked {})".format(
                runner._step_label(step), runner.step_click_count))
            on_success_goto = step.get("on_success_goto")
            if on_success_goto is not None:
                log.info("[match_click] {} -> success, goto {}".format(
                    runner._step_label(step), on_success_goto))
                runner._goto_step(runner._resolve_goto(on_success_goto))
            else:
                runner._advance_step(True, step=step)
            return "running"

    if now - runner.step_start_time >= timeout_sec:
        log.info("[match_click] {} -> false (not found in {}s)".format(
            runner._step_label(step), timeout_sec))
        runner._fail_step(step, "(timeout)")
    return "running"
