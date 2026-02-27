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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Log: ngay gio phut giay, vua ra console vua ra file, moi lan chay 1 file rieng
LOG_NAME = "kha_lastz"
_log_file_path = None

def setup_logging():
    global _log_file_path
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _log_file_path = os.path.join(log_dir, "kha_lastz_{}.log".format(ts))
    fmt = "%(asctime)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(_log_file_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("=== Start ===")
    return logger

log = setup_logging()

# Load config va functions
config = load_config("config.yaml")
key_bindings = config.get("key_bindings") or {}
# schedules: list of { function: "CheckMail", cron: "*/5 * * * *" } (dinh dang cron Linux)
schedules = config.get("schedules") or []
functions = load_functions("functions")
templates = collect_templates(functions)
vision_cache = build_vision_cache(templates)

# Log danh sach cua so
# print("=== Cac cua so dang chay (handle, ten) ===")
# WindowCapture.list_window_names()
# print("==========================================")

wincap = WindowCapture("LastZ")
runner = FunctionRunner(vision_cache)
runner.load(functions)

# Ctrl+C trong terminal -> SIGINT; pynput co the khong nhan duoc. Dang ky signal de thoat.
exit_requested = False

def _on_sigint(signum, frame):
    global exit_requested
    exit_requested = True

signal.signal(signal.SIGINT, _on_sigint)

# Global hotkey: listen m/t/... va Ctrl+Esc de thoat
key_queue = queue.Queue()
pressed_keys = set()

def on_press(key):
    try:
        if hasattr(key, "char") and key.char:
            pressed_keys.add(key.char)
            if key.char in key_bindings:
                key_queue.put(("hotkey", key.char))
        else:
            pressed_keys.add(key)
            # Ctrl+Esc = thoat (bat ky cua so nao)
            if key == keyboard.Key.esc and (keyboard.Key.ctrl_l in pressed_keys or keyboard.Key.ctrl_r in pressed_keys):
                key_queue.put("quit")
            elif key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r) and keyboard.Key.esc in pressed_keys:
                key_queue.put("quit")
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

# Tinh lan chay tiep theo cho moi schedule (cron) - dung gio local (tranh croniter naive = UTC)
next_run_at = {}
for item in schedules:
    fn_name = item.get("function")
    cron_expr = item.get("cron")
    if not fn_name or not cron_expr or fn_name not in functions:
        continue
    local_now = datetime.now().astimezone()
    it = croniter(cron_expr, local_now)
    next_run_at[fn_name] = it.get_next(float)

# Main loop: global hotkey (m/t), Ctrl+Esc = thoat, tu dong theo cron
log.info("Global hotkey: {} = run function. Press same key again (e.g. t) to stop. Ctrl+Esc = quit".format(key_bindings))
if schedules:
    log.info("Auto cron: {}".format([(s.get("function"), s.get("cron")) for s in schedules]))
    for fn_name, ts in next_run_at.items():
        log.info("Cron next run: {} at {}".format(fn_name, datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")))

running = True

def focus_loop():
    """Thread phu: lien tuc dua LastZ len truoc moi ~200ms (de khi user click cua so khac van bi keo lai)."""
    while running and not exit_requested:
        wincap.focus_window()
        time.sleep(0.2)

focus_thread = threading.Thread(target=focus_loop, daemon=True)
focus_thread.start()

# Focus ngay khi vao main loop
wincap.focus_window()
time.sleep(0.2)

while running and not exit_requested:
    # Xu ly thoat + hotkey + cron truoc get_screenshot (chay moi vong lap, tranh cron bi bo qua khi screenshot = None)
    if exit_requested:
        break
    try:
        last_stopped_key = None  # tranh key repeat: vua stop bang "t" thi khong start lai ngay
        while True:
            msg = key_queue.get_nowait()
            if msg == "quit":
                running = False
                break
            if isinstance(msg, tuple) and msg[0] == "hotkey":
                key_char = msg[1]
                fn_name = key_bindings.get(key_char)
                if not fn_name:
                    continue
                if runner.state == "running" and runner.function_name == fn_name:
                    runner.stop()
                    last_stopped_key = key_char
                    log.info("[Runner] Stopped function: {} (press same key to stop)".format(fn_name))
                else:
                    if key_char != last_stopped_key:
                        runner.start(fn_name, wincap)
                    last_stopped_key = None
    except queue.Empty:
        pass
    if not running:
        break

    now = time.time()
    if runner.state == "idle" and next_run_at:
        for fn_name, next_ts in list(next_run_at.items()):
            if now >= next_ts:
                runner.start(fn_name, wincap)
                cron_expr = next((s.get("cron") for s in schedules if s.get("function") == fn_name), None)
                if cron_expr:
                    local_now = datetime.fromtimestamp(now).astimezone()
                    it = croniter(cron_expr, local_now)
                    next_run_at[fn_name] = it.get_next(float)
                log.info("[Auto] Running function: {} (cron)".format(fn_name))
                break

    screenshot = wincap.get_screenshot()
    if screenshot is None:
        continue

    runner.update(screenshot, wincap)

    cv.imshow("LastZ Capture", screenshot)
    if cv.waitKey(1) == ord("q"):
        break

listener.stop()
cv.destroyAllWindows()
log.info("=== End ===")
log.info("Done.")
