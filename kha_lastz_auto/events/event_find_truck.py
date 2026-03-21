"""
event_find_truck.py
-------------------
Handler for the ``find_truck`` event type.

Flow per tick
-------------
1. Detect top max_truck yellow trucks by match score (find_multi_with_scores).
2. Filter already-tried positions (within this refresh cycle).
3. Click the best untried truck.
4. Synchronously: check OrangeHeroFragment count, then run OCR assertions.
5. All pass → advance to next step (e.g. TruckLootButton).
   Any fail  → mark truck as tried, return "running" to try next on next tick.
6. When all top-N are tried → click refresh, reset tried list, retry.

YAML keys
---------
templates / template : str or list
    Yellow truck template(s) to scan for.
threshold : float
    Template match threshold (default 0.6).
max_truck : int
    Maximum trucks to evaluate per refresh cycle (default 5).
match_color : bool
    Use colour-aware template matching (default True).
color_match_tolerance : int
    Max mean-BGR distance for colour match (optional).
dedup_tolerance : float
    Fraction of screen size — positions closer than this are the same truck (default 0.04).
dedup_tolerance_x / dedup_tolerance_y : float
    Per-axis overrides for dedup_tolerance.
sleep_after_click : float
    Seconds to wait for banner to render after clicking a truck (default 0.8).
refresh_template : str
    Template for the "Refresh" button.
refresh_sleep_sec : float
    Seconds to wait after clicking refresh (default 1.5).
max_refreshes : int
    Abort after this many refresh cycles (default 20).
timeout_sec : float
    Abort after this many seconds total (default 120).
debug_log : bool
    Extra logging of batch positions and scores.
debug_save : bool
    Save OCR crops to debug_ocr/ for inspection.

fragment_template : str
    Template for a single fragment badge (legacy; use fragment_filters for multi).
fragment_count : int
    Minimum count for the single legacy fragment_template (default 2).
fragment_filters : list of dict
    List of ``{template, count}`` pairs. Evaluated together using fragment_filter_mode.
    Takes priority over fragment_template/fragment_count when present.
    Can also be overridden at runtime via fn_settings["fragment_filters"].
fragment_filter_mode : str
    ``"AND"`` (default) — all filters must pass.  ``"OR"`` — any one filter passes.
fragment_threshold : float
    Template match threshold for fragment detection (default 0.90).
ocr_on_fragment_fail : bool
    When True, run OCR even if fragment check fails (for debug logging).

ocr_anchor_template : str
    Template for the truck banner — used as the coordinate anchor for OCR regions.
ocr_threshold : float
    Match threshold for the OCR anchor (default 0.70).
ocr_timeout_sec : float
    Max seconds to poll for the OCR anchor after clicking (default 3).
ocr_regions : list of dict
    Each entry defines one OCR region relative to the anchor match:
      name         : str   — used in log messages
      x, y, w, h   : float — ratios of the anchor template size (can be negative)
      digits_only  : bool  — strip non-numeric characters from OCR result
      pattern      : str   — regex to extract one group from the result
      assert_in    : list  — result must match one of these strings
      assert_max   : int   — numeric result must be < this value
      assert_min   : int   — numeric result must be >= this value
      assert_equals: str   — result must equal this string

OCR engine
----------
Uses OpenOCR (openocr-python, ONNX mobile) for accurate, fast text recognition
without any image preprocessing. The ONNX model (~24 MB) is downloaded once to
``~/.cache/openocr/`` on first use.
"""

import os
import re
import time
import logging
import datetime
import tempfile

from vision import Vision

import cv2 as cv

log = logging.getLogger("kha_lastz")

_DEBUG_DIR = "debug_ocr"

# ── OpenOCR singleton ──────────────────────────────────────────────────────────

_openocr_rec = None


def _ensure_openocr():
    """Lazy-initialize the OpenOCR recognizer (ONNX mobile, downloads once)."""
    global _openocr_rec
    if _openocr_rec is None:
        from openocr import OpenOCR
        _openocr_rec = OpenOCR(task="rec", mode="mobile")
    return _openocr_rec


# ── Internal helpers ───────────────────────────────────────────────────────────

def _save_debug_crop(crop, label: str) -> None:
    """Save *crop* to ``debug_ocr/<label>_<timestamp>_raw.png``."""
    ts = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    path = os.path.join(_DEBUG_DIR, "{}_{}_raw.png".format(label, ts))
    cv.imwrite(path, crop)
    log.info("[find_truck] debug saved -> {}".format(path))


def _crop_relative(screenshot, cx: int, cy: int, needle_w: int, needle_h: int,
                   x: float, y: float, w: float, h: float):
    """Crop a region relative to a template match centre.

    x, y, w, h are multiples of the template's width/height and can be negative
    (e.g. x=-2.9 means 2.9 template-widths to the left of the template's left edge).
    Returns the cropped numpy array or ``None`` if the region is degenerate.
    """
    tl_x = cx - needle_w // 2
    tl_y = cy - needle_h // 2

    rx = max(0, tl_x + int(x * needle_w))
    ry = max(0, tl_y + int(y * needle_h))
    rw = max(1, int(w * needle_w))
    rh = max(1, int(h * needle_h))

    img_h, img_w = screenshot.shape[:2]
    rw = min(rw, img_w - rx)
    rh = min(rh, img_h - ry)
    if rw <= 0 or rh <= 0:
        return None

    crop = screenshot[ry:ry + rh, rx:rx + rw]
    return crop if crop.size > 0 else None


def _openocr_read(crop, digits_only: bool = False, pattern: str = None,
                  debug_label: str = None) -> str:
    """OCR *crop* with OpenOCR and return the extracted string.

    Parameters
    ----------
    digits_only : bool
        When True, strips all non-numeric characters (keeps digits, commas,
        periods) from the raw OCR result — equivalent to EasyOCR's digits_only
        mode but without special image preprocessing (OpenOCR handles it).
    pattern : str or None
        Regex pattern; the first capturing group of the first match is returned.
    debug_label : str or None
        When set, saves the raw crop image to debug_ocr/ for inspection.

    Returns
    -------
    str
        Extracted text, or "" on failure.
    """
    if debug_label:
        _save_debug_crop(crop, debug_label)

    rec = _ensure_openocr()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv.imwrite(tmp_path, crop)
        results = rec(image_path=tmp_path)
        if not results:
            return ""
        texts = [r.get("text", "") or r.get("rec_text", "") for r in results if r]
        raw = " ".join(t for t in texts if t).strip()
    except Exception as exc:
        log.debug("[find_truck] OpenOCR error: {}".format(exc))
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if digits_only:
        raw = re.sub(r"[^\d,.]", "", raw)

    if pattern:
        m = re.search(pattern, raw)
        return m.group(1) if m else ""

    return raw


def _iter_templates(step: dict, *keys):
    """Yield template path(s) from one or more YAML keys (string or list)."""
    for key in keys:
        val = step.get(key)
        if not val:
            continue
        if isinstance(val, list):
            yield from val
        else:
            yield val


# ── Public entry point ─────────────────────────────────────────────────────────

def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute one tick of the ``find_truck`` event.

    Parameters
    ----------
    step : dict
        Parsed YAML step (the ``find_truck`` block).
    screenshot : numpy.ndarray
        Current game screenshot (BGR).
    wincap : WindowCapture
        Provides ``.w``, ``.h``, ``.get_screen_position()``, ``.get_screenshot()``.
    runner : FunctionRunner
        Provides ``vision_cache``, ``step_index``, ``step_start_time``,
        ``_safe_click()``, ``_advance_step()``, ``_fn_setting()``.

    Returns
    -------
    str
        Always ``"running"``.
    """
    import time as _time

    _tpl_paths            = list(_iter_templates(step, "template", "templates"))
    threshold             = float(step.get("threshold", 0.6))
    max_truck             = int(step.get("max_truck", 5))
    match_color           = step.get("match_color", True)
    color_match_tolerance = step.get("color_match_tolerance")
    sleep_after_click     = float(step.get("sleep_after_click", 0.8))
    refresh_template      = step.get("refresh_template")
    refresh_sleep_sec     = float(step.get("refresh_sleep_sec", 1.5))
    max_refreshes         = int(step.get("max_refreshes", 20))
    timeout_sec           = float(step.get("timeout_sec", 120))
    _dedup_base           = float(step.get("dedup_tolerance", 0.04))
    dedup_tol_x           = float(step.get("dedup_tolerance_x", _dedup_base))
    dedup_tol_y           = float(step.get("dedup_tolerance_y", _dedup_base))
    debug_log             = bool(step.get("debug_log", False))
    debug_save            = bool(step.get("debug_save", False))

    fragment_threshold    = float(step.get("fragment_threshold", 0.90))

    # ── Resolve fragment filter list ───────────────────────────────────────────
    # Priority: fn_settings["fragment_filters"] > YAML fragment_filters list
    #           > YAML single fragment_template/fragment_count (backward compat)
    _ff_ov = runner._fn_setting("fragment_filters")
    if _ff_ov and isinstance(_ff_ov, dict) and _ff_ov.get("filters"):
        fragment_filters      = _ff_ov["filters"]
        fragment_filter_mode  = _ff_ov.get("mode", "AND")
    elif step.get("fragment_filters") and isinstance(step.get("fragment_filters"), list):
        fragment_filters      = step["fragment_filters"]
        fragment_filter_mode  = step.get("fragment_filter_mode", "AND")
    elif step.get("fragment_template"):
        fragment_filters      = [{"template": step["fragment_template"],
                                   "count": int(step.get("fragment_count", 2))}]
        fragment_filter_mode  = "AND"
    else:
        fragment_filters      = []
        fragment_filter_mode  = "AND"

    ocr_anchor_template   = step.get("ocr_anchor_template")
    ocr_threshold         = float(step.get("ocr_threshold", 0.70))
    ocr_timeout_sec       = float(step.get("ocr_timeout_sec", 3))
    ocr_regions           = step.get("ocr_regions") or []
    ocr_on_fragment_fail  = bool(step.get("ocr_on_fragment_fail", False))

    # Per-step state keys scoped by step index so they reset between function runs
    _state_start_key = "_ft_start_{}".format(runner.step_index)
    _tried_key       = "_ft_tried_{}".format(runner.step_index)
    _refresh_key     = "_ft_refresh_{}".format(runner.step_index)
    _batch_key       = "_ft_batch_{}".format(runner.step_index)

    # Reset state on fresh entry
    if getattr(runner, _state_start_key, None) != runner.step_start_time:
        setattr(runner, _state_start_key, runner.step_start_time)
        setattr(runner, _tried_key, [])
        setattr(runner, _refresh_key, 0)
        setattr(runner, _batch_key, None)

    tried_positions = getattr(runner, _tried_key, [])
    refresh_count   = getattr(runner, _refresh_key, 0)
    batch           = getattr(runner, _batch_key, None)
    now             = _time.time()

    # Global timeout / max-refresh guard
    if now - runner.step_start_time >= timeout_sec:
        log.info("[find_truck] timeout {}s after {} refreshes -> abort".format(
            timeout_sec, refresh_count))
        runner._advance_step(False)
        return "running"
    if refresh_count > max_refreshes:
        log.info("[find_truck] max_refreshes {} reached -> abort".format(max_refreshes))
        runner._advance_step(False)
        return "running"

    # Load all vision objects; abort only if none are available
    _visions = [runner.vision_cache[p] for p in _tpl_paths if p in runner.vision_cache]
    if not _visions:
        log.info("[find_truck] no templates loaded ({}), abort".format(_tpl_paths))
        runner._advance_step(False)
        return "running"

    # Convert dedup tolerance ratios → pixels
    _scr_h, _scr_w = screenshot.shape[:2]
    _tol_x = int(_scr_w * dedup_tol_x)
    _tol_y = int(_scr_h * dedup_tol_y)

    # Establish a fixed batch at the start of each refresh cycle
    if batch is None:
        _all_matches: list = []
        for _v in _visions:
            _all_matches.extend(_v.find_multi_with_scores(
                screenshot,
                threshold=threshold,
                is_color=match_color,
                color_tolerance=color_match_tolerance,
            ))
        _deduped: list = []
        for (cx, cy, mw, mh, sc) in sorted(_all_matches, key=lambda t: t[4], reverse=True):
            if not any(abs(cx - ex[0]) < _tol_x and abs(cy - ex[1]) < _tol_y
                       for ex in _deduped):
                _deduped.append((cx, cy, mw, mh, sc))
        batch = _deduped[:max_truck]
        setattr(runner, _batch_key, batch)
        log.info("[find_truck] new batch of {} truck(s) | refresh={}/{} (dedup x={}px y={}px)".format(
            len(batch), refresh_count, max_refreshes, _tol_x, _tol_y))
        if debug_log and batch:
            parts = ["({},{}) {:.3f}".format(cx, cy, sc) for (cx, cy, _, _, sc) in batch]
            log.info("[find_truck] batch: {}".format(" | ".join(parts)))

    untried = [
        (cx, cy, mw, mh, sc) for (cx, cy, mw, mh, sc) in batch
        if not any(abs(cx - tp[0]) < _tol_x and abs(cy - tp[1]) < _tol_y for tp in tried_positions)
    ]

    log.info("[find_truck] batch={} tried={} untried={} | refresh={}/{}".format(
        len(batch), len(tried_positions), len(untried), refresh_count, max_refreshes))

    if not untried:
        # All trucks in this cycle tried → click refresh and start a new cycle
        refresh_vision = runner.vision_cache.get(refresh_template) if refresh_template else None
        if refresh_vision:
            ref_scr = wincap.get_screenshot()
            ref_pts = refresh_vision.find(ref_scr, threshold=0.6)
            if ref_pts:
                rsx, rsy = wincap.get_screen_position(tuple(ref_pts[0][:2]))
                runner._safe_click(rsx, rsy, wincap, "find_truck refresh")
                log.info("[find_truck] refreshed #{} (all {} in batch tried)".format(
                    refresh_count + 1, len(batch)))
                _time.sleep(refresh_sleep_sec)
                setattr(runner, _tried_key, [])
                setattr(runner, _refresh_key, refresh_count + 1)
                setattr(runner, _batch_key, None)
                return "running"
            log.info("[find_truck] refresh button not found -> abort")
        else:
            log.info("[find_truck] all tried, no refresh_template -> abort")
        runner._advance_step(False)
        return "running"

    # Click the best untried truck from the fixed batch
    cx, cy, mw, mh, score = untried[0]
    tried_positions.append((cx, cy))
    setattr(runner, _tried_key, tried_positions)

    sx, sy = wincap.get_screen_position((cx, cy))
    runner._safe_click(sx, sy, wincap, "find_truck")
    log.info("[find_truck] clicked truck ({},{}) score={:.3f}".format(cx, cy, score))
    _time.sleep(sleep_after_click)

    # Fresh screenshot after click for fragment + OCR checks
    scr2 = wincap.get_screenshot()

    # ── Fragment check ─────────────────────────────────────────────────────────
    _fragment_failed = False
    if fragment_filters:
        _frag_results = []
        for _filt in fragment_filters:
            _tpl = _filt.get("template")
            _req = max(1, int(_filt.get("count", 1)))
            if not _tpl:
                continue
            v_frag = runner.vision_cache.get(_tpl)
            if v_frag is None:
                # Template not pre-loaded (e.g. added via UI fn_settings) — lazy-load and cache
                try:
                    v_frag = Vision(_tpl)
                    runner.vision_cache[_tpl] = v_frag
                    log.info("[find_truck] lazy-loaded fragment template: {}".format(_tpl))
                except Exception as _le:
                    log.warning("[find_truck] fragment template load failed [{}]: {}".format(_tpl, _le))
            if v_frag:
                frag_pts    = v_frag.find(scr2, threshold=fragment_threshold, multi=True)
                found_frags = len(frag_pts) if frag_pts else 0
                _passed     = found_frags >= _req
                log.info("[find_truck] fragment [{}] {}/{} -> {}".format(
                    os.path.basename(_tpl), found_frags, _req,
                    "OK" if _passed else "skip"))
                _frag_results.append(_passed)
            else:
                log.warning("[find_truck] fragment template not available: {}".format(_tpl))
                _frag_results.append(False)

        if _frag_results:
            _overall = (any(_frag_results) if fragment_filter_mode.upper() == "OR"
                        else all(_frag_results))
            if not _overall:
                if not ocr_on_fragment_fail:
                    return "running"
                _fragment_failed = True

    # ── OCR assertion check ────────────────────────────────────────────────────
    if ocr_regions and ocr_anchor_template:
        v_anchor = runner.vision_cache.get(ocr_anchor_template)
        if not v_anchor:
            log.info("[find_truck] ocr_anchor_template not loaded -> skip OCR")
        else:
            # Poll for banner to appear
            _anchor_pts = None
            _scr_ocr = scr2
            _ocr_wait_start = _time.time()
            while _time.time() - _ocr_wait_start < ocr_timeout_sec:
                _anchor_pts = v_anchor.find(_scr_ocr, threshold=ocr_threshold)
                if _anchor_pts:
                    break
                _time.sleep(0.2)
                _scr_ocr = wincap.get_screenshot()

            if not _anchor_pts:
                log.info("[find_truck] OCR anchor not found in {}s -> skip truck".format(
                    ocr_timeout_sec))
                return "running"

            _pt  = _anchor_pts[0]
            _acx = _pt[0]
            _acy = _pt[1]
            _amw = _pt[2] if len(_pt) >= 3 else v_anchor.needle_w
            _amh = _pt[3] if len(_pt) >= 4 else v_anchor.needle_h
            _tpl_name = os.path.splitext(os.path.basename(ocr_anchor_template))[0]

            _ocr_fail_reason = None
            for region in ocr_regions:
                rname      = region.get("name", "ocr")
                dbg_lbl    = "{}_{}".format(_tpl_name, rname) if debug_save else None
                crop       = _crop_relative(
                    _scr_ocr, _acx, _acy, _amw, _amh,
                    x=region.get("x", 0.0),
                    y=region.get("y", 0.0),
                    w=region.get("w", 1.0),
                    h=region.get("h", 1.0),
                )
                if crop is None:
                    text = ""
                else:
                    text = _openocr_read(
                        crop,
                        digits_only=region.get("digits_only", False),
                        pattern=region.get("pattern"),
                        debug_label=dbg_lbl,
                    )
                log.info("[find_truck] OCR [{}]: {}".format(rname, text or "(no text)"))

                # Assertion params (with fn_settings overrides for server/power)
                assert_in  = region.get("assert_in")
                assert_max = region.get("assert_max")
                assert_min = region.get("assert_min")
                assert_eq  = region.get("assert_equals")

                if rname == "server":
                    _srv_ov = runner._fn_setting("servers")
                    log.info("[find_truck] OCR server: fn_settings[servers]={!r}".format(_srv_ov))
                    if _srv_ov is not None and str(_srv_ov).strip():
                        _srv_str = str(_srv_ov).strip()
                        assert_in = None if _srv_str == "*" else \
                            [s.strip() for s in _srv_str.split(",") if s.strip()]
                        log.info("[find_truck] OCR server: assert_in overridden -> {}".format(assert_in))
                if rname == "power":
                    _mp_ov = runner._fn_setting("max_power")
                    if _mp_ov is not None:
                        try:
                            assert_max = int(_mp_ov)
                        except (ValueError, TypeError):
                            pass

                _fail = None
                if assert_in is not None:
                    allowed = [str(v) for v in (assert_in if isinstance(assert_in, list) else [assert_in])]
                    if text not in allowed:
                        _fail = "assert_in FAIL ({!r} not in {})".format(text, allowed)
                elif assert_eq is not None and text != str(assert_eq):
                    _fail = "assert_equals FAIL ({!r} != {!r})".format(text, str(assert_eq))
                if _fail is None and (assert_max is not None or assert_min is not None):
                    try:
                        num = int(text.replace(",", "").replace(".", ""))
                        if assert_max is not None and num >= int(assert_max):
                            _fail = "assert_max FAIL ({} >= {})".format(num, assert_max)
                        elif assert_min is not None and num < int(assert_min):
                            _fail = "assert_min FAIL ({} < {})".format(num, assert_min)
                    except (ValueError, AttributeError):
                        _fail = "assert numeric FAIL (cannot parse {!r})".format(text)

                if _fail:
                    _ocr_fail_reason = _fail
                    log.info("[find_truck] OCR [{}]: {} -> skip truck".format(rname, _fail))
                    break

            if _ocr_fail_reason:
                return "running"

    # Fragment check failed (but OCR ran for debug) → skip truck now
    if _fragment_failed:
        return "running"

    # All checks passed → proceed to next step (e.g. TruckLootButton)
    log.info("[find_truck] truck ({},{}) passed all checks -> proceed".format(cx, cy))
    runner._advance_step(True)
    return "running"


def collect_templates(step: dict) -> list:
    """Return all template paths used by this step for pre-loading.

    Called by ``bot_engine.collect_templates`` for every ``find_truck`` step.
    """
    templates = []
    for tpl in _iter_templates(step, "template", "templates"):
        templates.append(tpl)
    for key in ("refresh_template", "fragment_template", "ocr_anchor_template"):
        tpl = step.get(key)
        if tpl:
            templates.append(tpl)
    # New: collect templates from fragment_filters list (YAML format)
    for _filt in (step.get("fragment_filters") or []):
        tpl = _filt.get("template")
        if tpl:
            templates.append(tpl)
    return templates
