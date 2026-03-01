import cv2 as cv
import os
import ctypes
import signal
import time
import queue
import threading
import logging
from datetime import datetime

# Ngan Windows sleep / lock man hinh trong khi bot chay
_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002
ctypes.windll.kernel32.SetThreadExecutionState(
    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
)
import pyautogui
pyautogui.PAUSE = 0        # remove default 0.1s pause after every pyautogui call
pyautogui.FAILSAFE = True  # keep failsafe (move mouse to corner to abort)
from pynput import keyboard
from croniter import croniter
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
from logout_detector import LogoutDetector
from alliance_attack_detector import AllianceAttackDetector

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

_dotenv_keys = _load_dotenv(".env")

# Logging: timestamp on every line, write to console + per-run file
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

if _dotenv_keys:
    log.info("[Env] Loaded from .env: {}".format(", ".join(_dotenv_keys)))
else:
    log.info("[Env] .env not found or empty — set secrets as environment variables")

# ── Load config ──────────────────────────────────────────────────────────────
config = load_config("config.yaml")
fn_configs = config.get("functions") or []

key_bindings      = {}   # key_char -> fn_name
fn_enabled        = {}   # fn_name  -> bool
schedules         = []   # [{ "function": ..., "cron": ... }]
attacked_triggers         = []   # fn_names triggered when attack starts
logged_out_triggers       = []   # fn_names triggered when logged out
alliance_attacked_triggers = []  # fn_names triggered when alliance is attacked

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
    if trigger == "logged_out" and enabled:
        logged_out_triggers.append(name)
    if trigger == "alliance_attacked" and enabled:
        alliance_attacked_triggers.append(name)

functions    = load_functions("functions")
templates    = collect_templates(functions)
vision_cache = build_vision_cache(templates)

wincap = WindowCapture("LastZ")

_ref_w = config.get("reference_width")
_ref_h = config.get("reference_height")
_show_preview = config.get("show_preview", False)

# Resize window ve dung kich thuoc ngay khi khoi dong.
# focus_loop se tiep tuc giu kich thuoc nay trong suot qua trinh chay.
if _ref_w and _ref_h:
    wincap.resize_to_client(_ref_w, _ref_h)
    log.info("Vision scale: 1.0 (window {}x{})".format(wincap.w, wincap.h))
else:
    log.info("Vision scale: 1.0 (reference_width/height not set — using current window size)")

vision_module.set_global_scale(1.0)

runner = FunctionRunner(vision_cache)
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
pending_queue = []   # list of fn_name, FIFO order

def _queue_add(fn_name, reason="cron"):
    """Add fn_name to pending_queue if not already queued."""
    if fn_name in pending_queue:
        log.info("[Scheduler] {} already in queue, skip duplicate ({})".format(fn_name, reason))
        return
    pending_queue.append(fn_name)
    log.info("[Scheduler] {} queued (reason={})".format(fn_name, reason))

def _try_start(fn_name, trigger="hotkey"):
    """Start fn_name immediately if idle, otherwise queue it."""
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
            _queue_add(fn_name, reason=trigger)
    else:
        runner.start(fn_name, wincap)

def _process_queue():
    """Pop next FIFO item from queue and start it."""
    while pending_queue:
        fn_name = pending_queue.pop(0)
        if not fn_enabled.get(fn_name, True):
            log.info("[Scheduler] {} dequeued but disabled, skipping".format(fn_name))
            continue
        if fn_name not in functions:
            continue
        log.info("[Scheduler] {} dequeued and starting".format(fn_name))
        runner.start(fn_name, wincap)
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
        if interrupted and interrupted != fn_name and interrupted not in pending_queue:
            pending_queue.insert(0, interrupted)
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
if alliance_attacked_triggers:
    log.info("Alliance attack triggers: {}".format(alliance_attacked_triggers))
    for fn_name, ts in sorted(next_run_at.items(), key=lambda x: x[1]):
        log.info("Cron next run: {} at {}".format(
            fn_name, datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")))
if pending_queue:
    log.info("Pending queue: {}".format(pending_queue))

# ── Focus thread ──────────────────────────────────────────────────────────────
running = True

def focus_loop():
    while running and not exit_requested:
        wincap.focus_window()
        if _ref_w and _ref_h and (wincap.w != _ref_w or wincap.h != _ref_h):
            wincap.resize_to_client(_ref_w, _ref_h)
            log.info("[focus_loop] Window resized back to {}x{}".format(wincap.w, wincap.h))
        time.sleep(0.2)

focus_thread = threading.Thread(target=focus_loop, daemon=True)
focus_thread.start()

wincap.focus_window()
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

# ── Detector background thread ─────────────────────────────────────────────────
# Detectors run matchTemplate (~0.3s each × 3 = ~0.9s total). Running them on the
# main thread blocks clicking. Instead, run them in a background thread every 2s
# and post events to a queue for the main loop to handle.
_detector_event_queue = queue.Queue()
_DETECTOR_INTERVAL = 2.0  # background detectors check every 2s

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

        attack_event = attack_detector.update(img, log)
        if attack_event == "started":
            log.info("[Detector] #{} → attacked, triggers={}".format(_tick, attacked_triggers))
            _detector_event_queue.put(("attacked", list(attacked_triggers)))

        logout_event = logout_detector.update(img, log)
        if logout_event == "started":
            log.info("[Detector] #{} → logged_out, triggers={}".format(_tick, logged_out_triggers))
            _detector_event_queue.put(("logged_out", list(logged_out_triggers)))

        alliance_attack_event = alliance_attack_detector.update(img, log)
        if alliance_attack_event == "started":
            log.info("[Detector] #{} → alliance_attacked, triggers={}".format(_tick, alliance_attacked_triggers))
            _detector_event_queue.put(("alliance_attacked", list(alliance_attacked_triggers)))

    log.info("[Detector] Background thread exited")

_detector_thread = threading.Thread(target=_detector_loop, daemon=True)
_detector_thread.start()

# ── Main loop ─────────────────────────────────────────────────────────────────
_CAPTURE_INTERVAL = 0.1   # 10 FPS cap
_last_capture_time = 0.0
last_stopped_key = None

while running and not exit_requested:
    if exit_requested:
        break

    # Process hotkey messages
    try:
        while True:
            msg = key_queue_msg.get_nowait()
            if msg == "quit":
                running = False
                break
            if isinstance(msg, tuple) and msg[0] == "hotkey":
                key_char = msg[1]
                fn_name  = key_bindings.get(key_char)
                if not fn_name:
                    continue
                # Toggle: press same key while running same function -> stop
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

    # Process cron triggers
    now = time.time()
    for fn_name, next_ts in list(next_run_at.items()):
        if now >= next_ts:
            # Advance next cron time regardless of whether we run now
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
        cv.waitKey(1)
        continue
    _last_capture_time = _now

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
                    _try_start(fn_name, trigger="attacked")
            elif event_type == "alliance_attacked":
                for fn_name in triggers:
                    _try_start(fn_name, trigger="alliance_attacked")
    except queue.Empty:
        pass

    was_running = runner.state == "running"
    runner.update(screenshot, wincap)

    # After function finishes, process pending queue
    if was_running and runner.state == "idle":
        _process_queue()

    if _show_preview:
        cv.imshow("LastZ Capture", screenshot)
        if cv.waitKey(1) == ord("q"):
            break
    else:
        cv.waitKey(1)

listener.stop()
cv.destroyAllWindows()
# Reset ve trang thai mac dinh khi thoat
ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
log.info("=== End ===")
log.info("Done.")
