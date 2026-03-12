import os
import sys
import ctypes
import signal
import time
import queue
import threading
import logging
from datetime import datetime


os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

def _load_dotenv(path=".env"):
    """Load .env file into os.environ. Returns list of keys found."""
    if not os.path.isfile(path):
        return []
    load_dotenv(dotenv_path=path, override=False)
    keys = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                keys.append(line.split("=", 1)[0].strip())
    return keys

LOG_NAME = "kha_lastz"

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, "kha_lastz_{}.log".format(ts))
    fmt = "%(asctime)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("=== Start ===")
    return logger

log = setup_logging()

# Heavy imports in background thread so main thread can react to Ctrl+C (wait with 1s timeout)
log.info("Loading modules...")
sys.stdout.flush()
sys.stderr.flush()

_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002
_imports_ready = threading.Event()
_imports_error = []

def _do_heavy_imports():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
        )
        import cv2 as cv
        import pyautogui
        from croniter import croniter as croniter_class
        from pynput import keyboard
        from windowcapture import WindowCapture
        from bot_engine import (
            load_functions,
            load_config,
            collect_templates,
            build_vision_cache,
            FunctionRunner,
        )
        import vision as vision_module
        from attack_detector import AttackDetector
        from treasure_detector import TreasureDetector
        from ui import BotUI
        import config_manager
        from logout_detector import LogoutDetector
        from alliance_attack_detector import AllianceAttackDetector
        main_mod = sys.modules.get("__main__")
        if main_mod:
            main_mod.cv = cv
            main_mod.pyautogui = pyautogui
            main_mod.keyboard = keyboard
            main_mod.croniter = croniter_class
            main_mod.WindowCapture = WindowCapture
            main_mod.load_functions = load_functions
            main_mod.load_config = load_config
            main_mod.collect_templates = collect_templates
            main_mod.build_vision_cache = build_vision_cache
            main_mod.FunctionRunner = FunctionRunner
            main_mod.vision_module = vision_module
            main_mod.AttackDetector = AttackDetector
            main_mod.TreasureDetector = TreasureDetector
            main_mod.BotUI = BotUI
            main_mod.config_manager = config_manager
            main_mod.LogoutDetector = LogoutDetector
            main_mod.AllianceAttackDetector = AllianceAttackDetector
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = True
    except Exception as e:
        _imports_error.append(e)
    finally:
        _imports_ready.set()

threading.Thread(target=_do_heavy_imports, daemon=True).start()
while True:
    try:
        if _imports_ready.wait(timeout=1.0):
            break
    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")
        sys.exit(130)
if _imports_error:
    log.error("Failed to load modules: {}".format(_imports_error[0]))
    sys.exit(1)

_dotenv_keys = _load_dotenv(".env")

# EasyOCR preload in background so main thread does not freeze (Ctrl+C stays responsive)
def _do_easyocr_preload():
    try:
        from ocr_easyocr import preload as _preload_easyocr
        _preload_easyocr()
    except Exception as e:
        log.warning("[EasyOCR] Preload failed: {}".format(e))
threading.Thread(target=_do_easyocr_preload, daemon=True).start()

if _dotenv_keys:
    log.info("[Env] Loaded from .env: {}".format(", ".join(_dotenv_keys)))
else:
    log.info("[Env] .env not found or empty — set secrets as environment variables")

# ── Load config ──────────────────────────────────────────────────────────────
config = load_config("config.yaml")
fn_configs = config.get("functions") or []
config_manager.apply_overrides(fn_configs)  # .env_config overrides config.yaml

key_bindings      = {}   # key_char -> fn_name
fn_enabled        = {}   # fn_name  -> bool
schedules         = []   # [{ "function": ..., "cron": ... }]
attacked_triggers          = []   # fn_names triggered when attack starts
treasure_detected_triggers = []   # fn_names triggered when treasure is detected
logged_out_triggers        = []   # fn_names triggered when logged out
alliance_attacked_triggers = []   # fn_names triggered when alliance is attacked

for fc in fn_configs:
    name    = fc.get("name")
    key     = fc.get("key")
    cron    = fc.get("cron")
    trigger = fc.get("trigger")
    enabled = fc.get("enabled", True)
    if not name:
        continue
    fn_enabled[name]  = enabled
    if key and enabled:
        key_bindings[key] = name
    if cron and enabled:
        schedules.append({"function": name, "cron": cron})
    if trigger == "attacked" and enabled:
        attacked_triggers.append(name)
    if trigger == "treasure_detected" and enabled:
        treasure_detected_triggers.append(name)
    if trigger == "logged_out" and enabled:
        logged_out_triggers.append(name)
    if trigger == "alliance_attacked" and enabled:
        alliance_attacked_triggers.append(name)

functions    = load_functions("functions")
templates    = collect_templates(functions)
vision_cache = build_vision_cache(templates)

# Create WindowCapture in thread with timeout so FindWindow cannot freeze main (Ctrl+C works)
_wincap_result = []
_wincap_done = threading.Event()
def _create_wincap():
    try:
        _wincap_result.append(WindowCapture("LastZ"))
    except Exception as e:
        log.error("WindowCapture failed: {}".format(e))
    finally:
        _wincap_done.set()
log.info("Connecting to game window 'LastZ'...")
sys.stdout.flush()
threading.Thread(target=_create_wincap, daemon=True).start()
try:
    if not _wincap_done.wait(timeout=15):
        log.error("Window 'LastZ' not found or timeout (15s). Open the game window and restart.")
        sys.exit(1)
except KeyboardInterrupt:
    log.info("Interrupted by user.")
    sys.exit(130)
if not _wincap_result:
    log.error("WindowCapture failed. Check that 'LastZ' window exists.")
    sys.exit(1)
wincap = _wincap_result[0]

_ref_w = config.get("reference_width")   # template capture resolution → vision scale
_ref_h = config.get("reference_height")
_win_w = config.get("window_width")       # game window resize target (can differ from reference)
_win_h = config.get("window_height")
_show_preview = config.get("show_preview", False)

# Resize window to desired size on startup.
# focus_loop will keep enforcing this size throughout the session.
if _win_w and _win_h:
    wincap.resize_to_client(_win_w, _win_h)
    log.info("Window resized to {}x{} (target)".format(wincap.w, wincap.h))
else:
    log.info("window_width/height not set — keeping current window size {}x{}".format(wincap.w, wincap.h))

def update_vision_scale():
    current_w = wincap.w 
    # Dùng luôn biến _ref_w đã load ở trên
    if _ref_w and current_w > 0:
        new_scale = current_w / _ref_w
        vision_module.set_global_scale(new_scale)
        return new_scale
    return 1.0

# vision_module.set_global_scale(1.0)

actual_scale = update_vision_scale()


fn_settings = config_manager.load_fn_settings()
runner = FunctionRunner(vision_cache, fn_settings=fn_settings)
runner.load(functions)

# ── Ctrl+C / SIGINT ──────────────────────────────────────────────────────────
exit_requested = False

def _on_sigint(signum, frame):
    global exit_requested
    exit_requested = True

signal.signal(signal.SIGINT, _on_sigint)

# ── Global hotkey listener ────────────────────────────────────────────────────
key_queue_msg = queue.Queue()
pressed_keys  = set()

def on_press(key):
    try:
        if hasattr(key, "char") and key.char:
            pressed_keys.add(key.char)
            if key.char in key_bindings:
                key_queue_msg.put(("hotkey", key.char))
        else:
            pressed_keys.add(key)
            if key == keyboard.Key.esc and (keyboard.Key.ctrl_l in pressed_keys or keyboard.Key.ctrl_r in pressed_keys):
                key_queue_msg.put("quit")
            elif key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r) and keyboard.Key.esc in pressed_keys:
                key_queue_msg.put("quit")
    except Exception:
        pass

def on_release(key):
    try:
        if hasattr(key, "char") and key.char:
            pressed_keys.discard(key.char)
        else:
            pressed_keys.discard(key)
    except Exception:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True
listener.start()

# ── Cron schedule ─────────────────────────────────────────────────────────────
next_run_at = {}
for item in schedules:
    fn_name   = item.get("function")
    cron_expr = item.get("cron")
    if not fn_name or not cron_expr or fn_name not in functions:
        continue
    it = croniter(cron_expr, datetime.now().astimezone())
    next_run_at[fn_name] = it.get_next(float)

# ── FIFO queue (functions waiting to run) ─────────────────────────────────────
# Each item: (fn_name, trigger_event, trigger_active_cb) so dequeued trigger-based functions keep repeat (e.g. send_zalo).
pending_queue = []   # list of (fn_name, trigger_event, trigger_active_cb)

def _queue_add(fn_name, reason="cron", trigger_event=None, trigger_active_cb=None):
    """Add fn_name to pending_queue if not already queued. Store trigger_* so send_zalo repeat works when dequeued."""
    if any(item[0] == fn_name for item in pending_queue):
        log.info("[Scheduler] {} already in queue, skip duplicate ({})".format(fn_name, reason))
        return
    pending_queue.append((fn_name, trigger_event, trigger_active_cb))
    log.info("[Scheduler] {} queued (reason={})".format(fn_name, reason))

def _try_start(fn_name, trigger="hotkey", trigger_event=None, trigger_active_cb=None):
    """Start fn_name immediately if idle, otherwise queue it.
    When started from a detector, pass trigger_event and trigger_active_cb so steps like send_zalo
    can repeat while the trigger is still active (e.g. alliance icon still visible).
    """
    if not fn_enabled.get(fn_name, True):
        log.info("[Scheduler] {} is disabled, skipping ({})".format(fn_name, trigger))
        return
    if fn_name not in functions:
        log.info("[Scheduler] {} not found in functions".format(fn_name))
        return

    if runner.state == "running":
        cur_name = runner.function_name
        if cur_name == fn_name:
            # Same function: toggle stop
            runner.stop()
            log.info("[Scheduler] {} stopped (same key toggle)".format(fn_name))
        else:
            log.info("[Scheduler] {} blocked by {} [{}] -> queued".format(fn_name, cur_name, trigger))
            _queue_add(fn_name, reason=trigger, trigger_event=trigger_event, trigger_active_cb=trigger_active_cb)
    else:
        runner.start(fn_name, wincap, trigger_event=trigger_event, trigger_active_cb=trigger_active_cb)

def _process_queue():
    """Pop next FIFO item from queue and start it. Pass trigger_* so detector-triggered send_zalo can repeat."""
    while pending_queue:
        item = pending_queue.pop(0)
        fn_name = item[0]
        trigger_event = item[1] if len(item) > 1 else None
        trigger_active_cb = item[2] if len(item) > 2 else None
        if not fn_enabled.get(fn_name, True):
            log.info("[Scheduler] {} dequeued but disabled, skipping".format(fn_name))
            continue
        if fn_name not in functions:
            continue
        log.info("[Scheduler] {} dequeued and starting".format(fn_name))
        runner.start(fn_name, wincap, trigger_event=trigger_event, trigger_active_cb=trigger_active_cb)
        return

def _start_urgent(fn_name, trigger="logged_out"):
    """Stop whatever is running (re-queue it at the front), then start fn_name immediately.

    Used for high-urgency events like logged_out that must run right now.
    The interrupted function is inserted at the head of the queue so it
    resumes (from the beginning) as soon as fn_name finishes.
    """
    if not fn_enabled.get(fn_name, True):
        log.info("[Scheduler] {} is disabled, skipping ({})".format(fn_name, trigger))
        return
    if fn_name not in functions:
        log.info("[Scheduler] {} not found in functions".format(fn_name))
        return

    if runner.state == "running":
        interrupted = runner.function_name
        runner.stop()
        # Re-insert interrupted function at the front of the queue (skip duplicates)
        if interrupted and interrupted != fn_name and not any(x[0] == interrupted for x in pending_queue):
            pending_queue.insert(0, (interrupted, None, None))
            log.info("[Scheduler] {} interrupted by {} [{}], re-queued at front".format(
                interrupted, fn_name, trigger))

    log.info("[Scheduler] {} starting urgently [{}]".format(fn_name, trigger))
    runner.start(fn_name, wincap)

# ── Startup log ───────────────────────────────────────────────────────────────
log.info("Global hotkey: {} | Press same key to stop | Ctrl+Esc = quit".format(key_bindings))
log.info("Function enabled: {}".format({n: fn_enabled[n] for n in fn_enabled}))
if schedules:
    log.info("Auto cron: {}".format([(s["function"], s["cron"]) for s in schedules]))
if attacked_triggers:
    log.info("Attack triggers: {}".format(attacked_triggers))
if logged_out_triggers:
    log.info("Logout triggers: {}".format(logged_out_triggers))
if treasure_detected_triggers:
    log.info("Treasure detected triggers: {}".format(treasure_detected_triggers))
if alliance_attacked_triggers:
    log.info("Alliance attack triggers: {}".format(alliance_attacked_triggers))
    for fn_name, ts in sorted(next_run_at.items(), key=lambda x: x[1]):
        log.info("Cron next run: {} at {}".format(
            fn_name, datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")))
if pending_queue:
    log.info("Pending queue: {}".format([x[0] for x in pending_queue]))

# ── UI ────────────────────────────────────────────────────────────────────────
config_manager.init_if_missing(fn_configs, fn_enabled)
bot_paused = {"paused": False}


def _rebuild_trigger_lists():
    """Rebuild detector trigger lists from fn_configs + fn_enabled so enable/disable in UI takes effect without restart."""
    attacked_triggers.clear()
    treasure_detected_triggers.clear()
    logged_out_triggers.clear()
    alliance_attacked_triggers.clear()
    for fc in fn_configs:
        name = fc.get("name")
        trigger = fc.get("trigger")
        enabled = fn_enabled.get(name, True)
        if not name or not enabled:
            continue
        if trigger == "attacked":
            attacked_triggers.append(name)
        elif trigger == "treasure_detected":
            treasure_detected_triggers.append(name)
        elif trigger == "logged_out":
            logged_out_triggers.append(name)
        elif trigger == "alliance_attacked":
            alliance_attacked_triggers.append(name)


def _rebuild_schedules():
    """Rebuild schedules and next_run_at from fn_configs + fn_enabled (mirror of .env_config).
    Preserves existing next_run_at timing for already-scheduled functions.
    Called after any enabled/cron change so cron state is always consistent with .env_config.
    """
    newly_scheduled = set()
    for fc in fn_configs:
        name = fc.get("name")
        cron = fc.get("cron")
        if name and cron and fn_enabled.get(name, True):
            newly_scheduled.add(name)

    # Remove entries no longer active
    for fn_name in list(next_run_at.keys()):
        if fn_name not in newly_scheduled:
            next_run_at.pop(fn_name)
            log.info("Cron removed: {}".format(fn_name))

    # Rebuild schedules list
    schedules.clear()
    for fc in fn_configs:
        name = fc.get("name")
        cron = fc.get("cron")
        if name and cron and fn_enabled.get(name, True):
            schedules.append({"function": name, "cron": cron})
            if name not in next_run_at:
                it = croniter(cron, datetime.now().astimezone())
                next_run_at[name] = it.get_next(float)
                log.info("Cron scheduled: {} = {} (next: {})".format(
                    name, cron,
                    datetime.fromtimestamp(next_run_at[name]).strftime("%Y-%m-%d %H:%M:%S")))

    # Rebuild key_bindings from fn_configs
    key_bindings.clear()
    for fc in fn_configs:
        name = fc.get("name")
        key  = fc.get("key")
        if name and key and fn_enabled.get(name, True):
            key_bindings[key] = name


def _on_cron_change(fn_name, cron_expr):
    """Called from UI when user saves or clears a schedule for a function."""
    for fc in fn_configs:
        if fc.get("name") == fn_name:
            if cron_expr:
                fc["cron"] = cron_expr
            else:
                fc.pop("cron", None)
            break

    _rebuild_schedules()
    config_manager.save(fn_configs, fn_enabled)
    if cron_expr:
        ts = next_run_at.get(fn_name)
        log.info("Cron updated: {} = {} (next: {})".format(
            fn_name, cron_expr,
            datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "N/A"))
    else:
        log.info("Cron cleared: {}".format(fn_name))


def _on_enabled_change(fn_name, enabled):
    """Called from UI when user toggles a function enabled/disabled."""
    _rebuild_trigger_lists()
    _rebuild_schedules()


# UI will run on main thread via .run_main() after game loop thread is started (reduces startup hang)
_ui = BotUI(fn_enabled, fn_configs, runner, next_run_at,
            key_bindings=key_bindings,
            save_callback=lambda: config_manager.save(fn_configs, fn_enabled),
            bot_paused=bot_paused,
            cron_callback=_on_cron_change,
            fn_settings=fn_settings,
            settings_save_callback=config_manager.save_fn_settings,
            run_callback=lambda fn_name: _try_start(fn_name, trigger="ui_play"),
            enabled_callback=_on_enabled_change,
            quit_check=lambda: exit_requested)

# ── Focus thread ──────────────────────────────────────────────────────────────
running = True

_highgui_warned = False

def _safe_waitkey(ms=1):
    """Yield time / pump HighGUI events if available. Works on OpenCV builds with GUI: NONE."""
    global _highgui_warned
    try:
        return cv.waitKey(ms)
    except Exception as e:
        if not _highgui_warned:
            log.warning("[OpenCV] HighGUI not available (waitKey failed): {}".format(e))
            _highgui_warned = True
        time.sleep(ms / 1000.0)
        return -1

def focus_loop():
    while running and not exit_requested:
        if not bot_paused["paused"]:
            wincap.focus_window()
            if _win_w and _win_h and (wincap.w != _win_w or wincap.h != _win_h):
                wincap.resize_to_client(_win_w, _win_h)
                log.info("[focus_loop] Window resized back to {}x{}".format(wincap.w, wincap.h))
        time.sleep(0.2)

focus_thread = threading.Thread(target=focus_loop, daemon=True)
focus_thread.start()

# Initial focus with timeout so SetForegroundWindow cannot hang startup (Windows can block here)
def _focus_window_with_timeout(timeout_sec=2.0):
    done = threading.Event()
    def _do():
        try:
            wincap.focus_window()
        except Exception as e:
            log.warning("[Startup] focus_window: {}".format(e))
        finally:
            done.set()
    t = threading.Thread(target=_do, daemon=True)
    t.start()
    if not done.wait(timeout=timeout_sec):
        log.warning("[Startup] focus_window did not finish in {}s, continuing anyway".format(timeout_sec))
_focus_window_with_timeout(2.0)
time.sleep(0.2)

attack_detector = AttackDetector(
    warning_template_path="buttons_template/BeingAttackedWarning.png",
    clear_sec=10.0,
)

logout_detector = LogoutDetector(
    template_path="buttons_template/PasswordSlot.png",
    threshold=0.75,
    confirm_sec=1.0,
    clear_sec=5.0,
)

alliance_attack_detector = AllianceAttackDetector(
    warning_template_path="buttons_template/AllianceBeingAttackedWarning.png",
    clear_sec=10.0,
)

treasure_detector = TreasureDetector(
    treasure_template_path="buttons_template/Treasure1.png",
    threshold=0.6,
    clear_sec=10.0,
)

# Callbacks for trigger-based functions: send_zalo with repeat_interval_sec uses these to repeat while detector still active.
# When game window is closed, detector can't run so _attacked/_treasure_visible never clear — we must return False so Zalo repeat stops.
def _is_game_window_valid():
    try:
        import win32gui as _wg
        return wincap.hwnd and _wg.IsWindow(wincap.hwnd)
    except Exception:
        return False

trigger_active_callbacks = {
    "attacked": lambda: _is_game_window_valid() and attack_detector._attacked,
    "alliance_attacked": lambda: _is_game_window_valid() and alliance_attack_detector._attacked,
    "treasure_detected": lambda: _is_game_window_valid() and treasure_detector._treasure_visible,
}

from exit_banner_detector import ExitBannerDetector
exit_banner_detector = ExitBannerDetector(
    template_path="buttons_template/ExitGameBanner.png",
    threshold=0.85,
    check_every=5,   # 5 × 2s = 10s
)

# ── Detector background thread ─────────────────────────────────────────────────
# Detectors run matchTemplate (~0.3s each × 3 = ~0.9s total). Running them on the
# main thread blocks clicking. Instead, run them in a background thread every 2s
# and post events to a queue for the main loop to handle.
_detector_event_queue = queue.Queue()
_DETECTOR_INTERVAL = 5.0  # background detectors check every 2s

def _detector_loop():
    import mss as _mss_lib
    import win32gui as _win32gui
    import numpy as _np
    _mss_inst = _mss_lib.mss()
    log.info("[Detector] Background thread started (interval={}s)".format(_DETECTOR_INTERVAL))
    _tick = 0
    while running and not exit_requested:
        time.sleep(_DETECTOR_INTERVAL)
        if not running or exit_requested:
            break
        _tick += 1
        try:
            w, h = wincap.w, wincap.h
            hwnd = wincap.hwnd
            if w <= 0 or h <= 0 or not hwnd:
                log.warning("[Detector] #{} skip — invalid window size {}x{}".format(_tick, w, h))
                continue
            (left, top) = _win32gui.ClientToScreen(hwnd, (0, 0))
            monitor = {'left': left, 'top': top, 'width': w, 'height': h}
            raw = _mss_inst.grab(monitor)
            img = _np.array(raw)[..., :3]
            img = _np.ascontiguousarray(img)
        except Exception as e:
            log.warning("[Detector] #{} screenshot failed: {}".format(_tick, e))
            continue

        # ── Attack detector ─────────────────────────────
        try:
            attack_event = attack_detector.update(img, log)
            if attack_event == "started":
                log.info("[Detector] #{} → attacked, triggers={}".format(_tick, attacked_triggers))
                _detector_event_queue.put(("attacked", list(attacked_triggers)))
        except Exception as e:
            log.error("[Detector] #{} attack_detector crashed: {}".format(_tick, e))
        
        # ── Logout detector ─────────────────────────────
        try:
            logout_event = logout_detector.update(img, log)
            if logout_event == "started":
                log.info("[Detector] #{} → logged_out, triggers={}".format(_tick, logged_out_triggers))
                _detector_event_queue.put(("logged_out", list(logged_out_triggers)))
        except Exception as e:
            log.error("[Detector] #{} logout_detector crashed: {}".format(_tick, e))

        # ── Alliance attack detector ────────────────────
        try:
            alliance_attack_event = alliance_attack_detector.update(img, log)
            if alliance_attack_event == "started":
                log.info("[Detector] #{} → alliance_attacked, triggers={}".format(_tick, alliance_attacked_triggers))
                _detector_event_queue.put(("alliance_attacked", list(alliance_attacked_triggers)))
        except Exception as e:
            log.error("[Detector] #{} alliance_attack_detector crashed: {}".format(_tick, e))


        # ── Treasure detector ───────────────────────────
        try:
            treasure_event = treasure_detector.update(img, log)
            if treasure_event == "started":
                log.info("[Detector] #{} → treasure_detected, triggers={}".format(_tick, treasure_detected_triggers))
                _detector_event_queue.put(("treasure_detected", list(treasure_detected_triggers)))
        except Exception as e:
            log.error("[Detector] #{} treasure_detector crashed: {}".format(_tick, e))

        # ── Exit banner detector ────────────────────────
        try:
            # Check ExitGameBanner every N ticks — click corner to dismiss
            if exit_banner_detector.update(img, wincap, log):
                sx, sy = exit_banner_detector.corner_screen_pos(wincap)
                pyautogui.click(sx, sy)
                log.info("[Detector] #{} → ExitGameBanner detected, clicked corner ({}, {})".format(_tick, sx, sy))
        except Exception as e:
            log.error("[Detector] #{} exit_banner_detector crashed: {}".format(_tick, e))
    log.info("[Detector] Background thread exited")

_detector_thread = threading.Thread(target=_detector_loop, daemon=True)
_detector_thread.start()

# ── Game loop (background thread; Tkinter runs on main thread to avoid startup hang) ──
_CAPTURE_INTERVAL = 0.1   # 10 FPS cap
_last_capture_time = 0.0
last_stopped_key = None
detector_lock = threading.Lock()
_last_detector_restart = 0

def _game_loop():
    global _last_detector_restart, _detector_thread, running, _last_capture_time, last_stopped_key, _show_preview
    now = time.time()
    while running and not exit_requested:
        if exit_requested:
            break
        with detector_lock:
            if not _detector_thread.is_alive() and now - _last_detector_restart > 5:
                log.error("[Watchdog] Detector thread died! Restarting...")
                _detector_thread = threading.Thread(target=_detector_loop, daemon=True)
                _detector_thread.start()
                _last_detector_restart = now

        try:
            while True:
                msg = key_queue_msg.get_nowait()
                if msg == "quit":
                    running = False
                    break
                if bot_paused["paused"]:
                    continue
                if isinstance(msg, tuple) and msg[0] == "hotkey":
                    key_char = msg[1]
                    fn_name  = key_bindings.get(key_char)
                    if not fn_name:
                        continue
                    if runner.state == "running" and runner.function_name == fn_name:
                        runner.stop()
                        last_stopped_key = key_char
                        log.info("[Runner] Stopped function: {} (hotkey toggle)".format(fn_name))
                    else:
                        if key_char != last_stopped_key:
                            _try_start(fn_name, trigger="hotkey")
                        last_stopped_key = None
        except queue.Empty:
            pass
        if not running:
            break

        if bot_paused["paused"]:
            _safe_waitkey(1)
            continue

        now = time.time()
        for fn_name, next_ts in list(next_run_at.items()):
            if now >= next_ts:
                cron_expr = next((s["cron"] for s in schedules if s["function"] == fn_name), None)
                if cron_expr:
                    it = croniter(cron_expr, datetime.fromtimestamp(now).astimezone())
                    next_run_at[fn_name] = it.get_next(float)
                    log.info("Cron next run: {} at {}".format(
                        fn_name, datetime.fromtimestamp(next_run_at[fn_name]).strftime("%Y-%m-%d %H:%M:%S")))
                    _try_start(fn_name, trigger="cron")

        # Throttle to 10 FPS — hotkey/cron above still run every iteration
        _now = time.time()
        if _now - _last_capture_time < _CAPTURE_INTERVAL:
            _safe_waitkey(1)
            continue
        _last_capture_time = _now
        update_vision_scale()

        # Get screenshot and update runner
        screenshot = wincap.get_screenshot()
        if screenshot is None:
            continue

        # Drain detector events from background thread (zero matchTemplate cost on main thread)
        try:
            while True:
                event_type, triggers = _detector_event_queue.get_nowait()
                if event_type == "logged_out":
                    for fn_name in triggers:
                        _start_urgent(fn_name, trigger="logged_out")
                elif event_type == "attacked":
                    for fn_name in triggers:
                        _try_start(fn_name, trigger="attacked", trigger_event="attacked",
                                   trigger_active_cb=trigger_active_callbacks.get("attacked"))
                elif event_type == "alliance_attacked":
                    for fn_name in triggers:
                        _try_start(fn_name, trigger="alliance_attacked", trigger_event="alliance_attacked",
                                   trigger_active_cb=trigger_active_callbacks.get("alliance_attacked"))
                elif event_type == "treasure_detected":
                    for fn_name in triggers:
                        _try_start(fn_name, trigger="treasure_detected", trigger_event="treasure_detected",
                                   trigger_active_cb=trigger_active_callbacks.get("treasure_detected"))
        except queue.Empty:
            pass

        was_running = runner.state == "running"
        runner.update(screenshot, wincap)

        # After function finishes, process pending queue
        if was_running and runner.state == "idle":
            _process_queue()

        if _show_preview:
            cv.imshow("LastZ Capture", screenshot)
            if _safe_waitkey(1) == ord("q"):
                break
            if _highgui_warned:
                _show_preview = False
        else:
            _safe_waitkey(1)

# Start game loop in background, then run UI on main thread (blocks until window closed)
game_thread = threading.Thread(target=_game_loop, daemon=True)
game_thread.start()
try:
    _ui.run_main()
except KeyboardInterrupt:
    log.info("Ctrl+C received, shutting down...")
    sys.exit(0)

listener.stop()
try:
    cv.destroyAllWindows()
except Exception:
    pass
# Reset ve trang thai mac dinh khi thoat
ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
log.info("=== End ===")
log.info("Done.")
