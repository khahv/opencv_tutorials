import cv2 as cv
import os
import signal
import time
import queue
import threading
import logging
from datetime import datetime
import pyautogui
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
fn_priority       = {}   # fn_name  -> int  (lower = higher priority)
fn_enabled        = {}   # fn_name  -> bool
schedules         = []   # [{ "function": ..., "cron": ... }]
attacked_triggers = []   # fn_names triggered when attack starts

for fc in fn_configs:
    name    = fc.get("name")
    key     = fc.get("key")
    cron    = fc.get("cron")
    trigger = fc.get("trigger")
    prio    = fc.get("priority", 99)
    enabled = fc.get("enabled", True)
    if not name:
        continue
    fn_priority[name] = prio
    fn_enabled[name]  = enabled
    if key and enabled:
        key_bindings[key] = name
    if cron and enabled:
        schedules.append({"function": name, "cron": cron})
    if trigger == "attacked" and enabled:
        attacked_triggers.append(name)

functions    = load_functions("functions")
templates    = collect_templates(functions)
vision_cache = build_vision_cache(templates)

wincap = WindowCapture("LastZ")

_ref_w = config.get("reference_width")
_ref_h = config.get("reference_height")

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

# ── Priority queue (cron-triggered functions waiting to run) ──────────────────
pending_queue = []   # list of fn_name, sorted by priority then arrival order

def _queue_add(fn_name, reason="cron"):
    """Add fn_name to pending_queue if not already queued."""
    if fn_name in pending_queue:
        log.info("[Scheduler] {} already in queue, skip duplicate ({})".format(fn_name, reason))
        return
    pending_queue.append(fn_name)
    # sort by priority (stable sort preserves FIFO for equal priority)
    pending_queue.sort(key=lambda n: fn_priority.get(n, 99))
    log.info("[Scheduler] {} queued (priority={}, reason={})".format(
        fn_name, fn_priority.get(fn_name, 99), reason))

def _try_start(fn_name, trigger="hotkey"):
    """Try to start fn_name. Preempt lower priority. Queue if lower priority than running."""
    if not fn_enabled.get(fn_name, True):
        log.info("[Scheduler] {} is disabled, skipping ({})".format(fn_name, trigger))
        return
    if fn_name not in functions:
        log.info("[Scheduler] {} not found in functions".format(fn_name))
        return

    new_prio = fn_priority.get(fn_name, 99)

    if runner.state == "running":
        cur_name = runner.function_name
        cur_prio = fn_priority.get(cur_name, 99)

        if new_prio < cur_prio:
            # Higher priority preempts running function
            log.info("[Scheduler] {} (priority={}) preempts {} (priority={}) [{}]".format(
                fn_name, new_prio, cur_name, cur_prio, trigger))
            runner.stop()
            runner.start(fn_name, wincap)
        elif new_prio > cur_prio:
            # Lower priority: always queue (hotkey or cron)
            log.info("[Scheduler] {} (priority={}) blocked by {} (priority={}) [{}] -> queued".format(
                fn_name, new_prio, cur_name, cur_prio, trigger))
            _queue_add(fn_name, reason=trigger)
        else:
            # Same priority: toggle or restart
            if cur_name == fn_name:
                runner.stop()
                log.info("[Scheduler] {} stopped (same key toggle)".format(fn_name))
            else:
                runner.stop()
                runner.start(fn_name, wincap)
    else:
        runner.start(fn_name, wincap)

def _process_queue():
    """Pop highest priority item from queue and start it."""
    while pending_queue:
        fn_name = pending_queue.pop(0)
        if not fn_enabled.get(fn_name, True):
            log.info("[Scheduler] {} dequeued but disabled, skipping".format(fn_name))
            continue
        if fn_name not in functions:
            continue
        log.info("[Scheduler] {} dequeued and starting (priority={})".format(
            fn_name, fn_priority.get(fn_name, 99)))
        runner.start(fn_name, wincap)
        return

# ── Startup log ───────────────────────────────────────────────────────────────
log.info("Global hotkey: {} | Press same key to stop | Ctrl+Esc = quit".format(key_bindings))
log.info("Function priorities: {}".format({n: fn_priority[n] for n in fn_priority}))
log.info("Function enabled: {}".format({n: fn_enabled[n] for n in fn_enabled}))
if schedules:
    log.info("Auto cron: {}".format([(s["function"], s["cron"]) for s in schedules]))
if attacked_triggers:
    log.info("Attack triggers: {}".format(attacked_triggers))
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

    # Being-attacked detection
    attack_event = attack_detector.update(screenshot, log)
    if attack_event == "started":
        for fn_name in attacked_triggers:
            _try_start(fn_name, trigger="attacked")

    was_running = runner.state == "running"
    runner.update(screenshot, wincap)

    # After function finishes, process pending queue
    if was_running and runner.state == "idle":
        _process_queue()

    cv.imshow("LastZ Capture", screenshot)
    if cv.waitKey(1) == ord("q"):
        break

listener.stop()
cv.destroyAllWindows()
log.info("=== End ===")
log.info("Done.")
