import os
import sys
import ctypes
import signal
import subprocess
import time
import queue
import threading
import logging
from datetime import datetime


os.chdir(os.path.dirname(os.path.abspath(__file__)))

from ui_locale import normalize_language_code as _norm_lang


def _require_admin_win():
    """If not running as admin on Windows, trigger UAC and re-launch elevated; then exit."""
    if sys.platform != "win32":
        return
    try:
        if ctypes.windll.shell32.IsUserAnAdmin() != 0:
            return
    except Exception:
        pass
    # Not admin: run self with "runas" to show UAC prompt
    script = os.path.abspath(sys.argv[0])
    quoted = '"' + script + '"' if " " in script else script
    params = quoted + (" " + " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "")
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, params, os.getcwd(), 1  # 1 = SW_SHOWNORMAL
    )
    if ret > 32:  # success
        sys.exit(0)
    # UAC cancelled or error: exit anyway so user sees no duplicate window
    sys.exit(1)


_require_admin_win()


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
        from screenshot_provider import create_screenshot_provider as _csp
        import adb_input as _adb_input_mod
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
        from connection_detector import ConnectionDetector
        main_mod = sys.modules.get("__main__")
        if main_mod:
            main_mod.cv = cv
            main_mod.pyautogui = pyautogui
            main_mod.keyboard = keyboard
            main_mod.croniter = croniter_class
            main_mod.WindowCapture = WindowCapture
            main_mod.create_screenshot_provider = _csp
            main_mod.adb_input = _adb_input_mod
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
            main_mod.ConnectionDetector = ConnectionDetector
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

# OpenOCR preload in background so main thread does not freeze (Ctrl+C stays responsive)
def _do_openocr_preload():
    try:
        from ocr_openocr import preload as _preload_openocr
        _preload_openocr()
    except Exception as exc:
        log.warning("[OpenOCR] Preload failed: {}".format(exc))
threading.Thread(target=_do_openocr_preload, daemon=True).start()

if _dotenv_keys:
    log.info("[Env] Loaded from .env: {}".format(", ".join(_dotenv_keys)))
else:
    log.info("[Env] .env not found or empty — set secrets as environment variables")

# ── Load config ──────────────────────────────────────────────────────────────
config = load_config("config.yaml")
fn_configs = config.get("functions") or []
general_settings_ov = config_manager.apply_overrides(fn_configs) or {}  # .env_config overrides config.yaml

# Auto Focus setting
auto_focus = config.get("auto_focus", False)
if "auto_focus" in general_settings_ov:
    auto_focus = bool(general_settings_ov["auto_focus"])

# Window size: prefer .env_config (general_settings), else config.yaml
_win_w = general_settings_ov.get("window_width") or config.get("window_width")
_win_h = general_settings_ov.get("window_height") or config.get("window_height")

# LastZ executable path (startup auto-launch and connection_detector restart)
LASTZ_EXE_PATH = (
    general_settings_ov.get("lastz_exe_path")
    or config.get("lastz_exe_path")
    or r"C:\Users\hongkhavo\AppData\Local\Last Z\Last Z.exe"
)

# Auto start LastZ when window not found (PC mode only)
_auto_start_lastz = bool(
    general_settings_ov.get("auto_start_lastz", config.get("auto_start_lastz", False))
)

# Emulator selection: "pc" (default, win32 window) or "ldplayer" (LDPlayer emulator, ADB screenshots)
_emulator = (general_settings_ov.get("emulator") or config.get("emulator") or "pc").lower()
_GAME_WINDOW_NAME = "LDPlayer" if _emulator == "ldplayer" else "LastZ"
_LDPLAYER_ADB_PATH = config.get("ldplayer_adb_path") or None
log.info("Emulator mode: %s (window='%s')", _emulator, _GAME_WINDOW_NAME)

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

# Per-function cooldown (seconds) for detector triggers; only re-trigger after cooldown has passed.
trigger_cooldown_sec = {}
for fc in fn_configs:
    if fc.get("name") and "cooldown" in fc:
        try:
            trigger_cooldown_sec[fc["name"]] = float(fc["cooldown"])
        except (TypeError, ValueError):
            pass

functions    = load_functions("functions")
templates    = collect_templates(functions)
vision_cache = build_vision_cache(templates)

from game_surface import (
    LdplayerAdbSurface,
    apply_post_connect_window_prefs,
    initial_pc_connect_or_none,
    is_game_surface_valid,
    should_skip_focus_loop,
    sync_lastz_pid_from_wincap,
    PcWin32Surface,
)

wincap = None
screenshot_provider = None

if _emulator == "ldplayer":
    wincap, screenshot_provider = LdplayerAdbSurface.connect(
        adb_path=_LDPLAYER_ADB_PATH,
        device_serial=config.get("ldplayer_device_serial") or None,
        logger=log,
    )
else:
    try:
        wincap, screenshot_provider = initial_pc_connect_or_none(
            window_name=_GAME_WINDOW_NAME,
            adb_path=_LDPLAYER_ADB_PATH,
            logger=log,
            timeout_sec=90.0,
        )
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(130)

lastz_pid = {"pid": None}
sync_lastz_pid_from_wincap(wincap, lastz_pid, log)

_ref_w = config.get("reference_width")   # template capture resolution → vision scale
_ref_h = config.get("reference_height")
_show_preview = config.get("show_preview", False)
# _win_w, _win_h already set from general_settings_ov / config above

apply_post_connect_window_prefs(
    wincap,
    auto_focus=auto_focus,
    win_w=_win_w,
    win_h=_win_h,
    logger=log,
)

def update_vision_scale():
    if wincap is None:
        return 1.0
    current_w = wincap.w
    if _ref_w and current_w > 0:
        new_scale = current_w / _ref_w
        vision_module.set_global_scale(new_scale)
        return new_scale
    return 1.0

actual_scale = update_vision_scale()


fn_settings = config_manager.load_fn_settings()
bot_paused = {"paused": True}
runner = FunctionRunner(vision_cache, fn_settings=fn_settings, bot_paused=bot_paused)
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
last_triggered_at = {}   # fn_name -> time.time() when last triggered by a detector (for cooldown)

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
            if trigger in ("hotkey", "ui_play"):
                # Hotkey or UI play button on same running function: toggle stop
                runner.stop()
                log.info("[Scheduler] {} stopped ({} toggle)".format(fn_name, trigger))
            else:
                # Detector re-fired while already running: ignore to avoid interrupting
                log.debug("[Scheduler] {} already running, ignoring re-trigger ({})".format(fn_name, trigger))
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

def _trigger_cooldown_ok(fn_name):
    """Return True if fn_name is not in cooldown (or has no cooldown), so we may trigger it."""
    cooldown = trigger_cooldown_sec.get(fn_name, 0)
    if cooldown <= 0:
        return True
    last = last_triggered_at.get(fn_name)
    if last is None:
        return True
    return (time.time() - last) >= cooldown

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
    last_triggered_at[fn_name] = time.time()
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


# Mutable dict for general_settings; written to .env_config on save.
_general_settings = {
    "auto_focus": auto_focus,
    "window_width": _win_w,
    "window_height": _win_h,
    "language": _norm_lang(general_settings_ov.get("language", "en")),
    "emulator": _emulator,
    "lastz_exe_path": LASTZ_EXE_PATH or "",
    "auto_start_lastz": _auto_start_lastz,
}


def _on_general_setting_change(key, value):
    global auto_focus, _win_w, _win_h, _emulator, _GAME_WINDOW_NAME, wincap, screenshot_provider
    global LASTZ_EXE_PATH, _auto_start_lastz
    if key == "auto_focus":
        auto_focus = value
        if wincap is not None:
            wincap.auto_focus = value
        _general_settings["auto_focus"] = value
    elif key == "resolution":
        # value is "1080x1920" or "540x960"
        try:
            a, b = value.strip().lower().split("x")
            _win_w, _win_h = int(a.strip()), int(b.strip())
        except (ValueError, AttributeError):
            log.warning("[Settings] Invalid resolution '{}', ignored.".format(value))
            return
        _general_settings["window_width"] = _win_w
        _general_settings["window_height"] = _win_h
        config_manager.save(fn_configs, fn_enabled, general_settings=_general_settings)
        if wincap is not None:
            wincap.resize_to_client(_win_w, _win_h)
            update_vision_scale()
        log.info("[Settings] Resolution → {}x{}, saved to .env_config, window resized.".format(_win_w, _win_h))
        return
    elif key == "language":
        _general_settings["language"] = _norm_lang(value)
        config_manager.save(fn_configs, fn_enabled, general_settings=_general_settings)
        log.info("[Settings] UI language → {}".format(_general_settings["language"]))
        return
    elif key == "emulator":
        _emulator = value.lower()
        _GAME_WINDOW_NAME = "LDPlayer" if _emulator == "ldplayer" else "LastZ"
        wincap = None
        screenshot_provider = None
        if _emulator == "pc":
            adb_input.set_adb_input(None)
            log.info("[Settings] Emulator → pc — ADB input cleared, waiting for LastZ window.")
        else:
            log.info("[Settings] Emulator → ldplayer — waiting for ADB / screencap.")
        _general_settings["emulator"] = _emulator
        config_manager.save(fn_configs, fn_enabled, general_settings=_general_settings)
        _ui.notify_disconnected()
        return
    elif key == "lastz_exe_path":
        LASTZ_EXE_PATH = value or ""
        _general_settings["lastz_exe_path"] = LASTZ_EXE_PATH
        config_manager.save(fn_configs, fn_enabled, general_settings=_general_settings)
        if hasattr(connection_detector, "lastz_exe_path"):
            connection_detector.lastz_exe_path = LASTZ_EXE_PATH
        log.info("[Settings] LastZ exe path → %s", LASTZ_EXE_PATH)
        return
    elif key == "auto_start_lastz":
        _auto_start_lastz = bool(value)
        _general_settings["auto_start_lastz"] = _auto_start_lastz
        config_manager.save(fn_configs, fn_enabled, general_settings=_general_settings)
        log.info("[Settings] Auto start LastZ → %s", _auto_start_lastz)
        return
    _general_settings["auto_focus"] = auto_focus
    config_manager.save(fn_configs, fn_enabled, general_settings=_general_settings)

# UI will run on main thread via .run_main() after game loop thread is started (reduces startup hang)
def _launch_lastz():
    if not LASTZ_EXE_PATH:
        log.warning("[UI] Start LastZ: exe path not configured.")
        return
    if not os.path.isfile(LASTZ_EXE_PATH):
        log.warning("[UI] Start LastZ: file not found: %s", LASTZ_EXE_PATH)
        return
    log.info("[UI] Starting LastZ: %s", LASTZ_EXE_PATH)
    subprocess.Popen(
        [LASTZ_EXE_PATH],
        cwd=os.path.dirname(LASTZ_EXE_PATH),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

_ui = BotUI(fn_enabled, fn_configs, runner, next_run_at,
            key_bindings=key_bindings,
            save_callback=lambda: config_manager.save(fn_configs, fn_enabled, general_settings=_general_settings),
            bot_paused=bot_paused,
            cron_callback=_on_cron_change,
            fn_settings=fn_settings,
            settings_save_callback=config_manager.save_fn_settings,
            run_callback=lambda fn_name: _try_start(fn_name, trigger="ui_play"),
            enabled_callback=_on_enabled_change,
            quit_check=lambda: exit_requested,
            general_settings=_general_settings,
            general_settings_callback=_on_general_setting_change,
            connection_status=lambda: wincap is not None and _is_game_window_valid(),
            start_lastz_callback=_launch_lastz)

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
        if should_skip_focus_loop(wincap):
            time.sleep(0.2)
            continue
        if not bot_paused["paused"]:
            wincap.focus_window()
            if _win_w and _win_h and (wincap.w != _win_w or wincap.h != _win_h):
                wincap.resize_to_client(_win_w, _win_h)
                log.info("[focus_loop] Window resized back to {}x{}".format(wincap.w, wincap.h))
        time.sleep(0.2)

focus_thread = threading.Thread(target=focus_loop, daemon=True)
focus_thread.start()

# ── Window auto-connect thread (logic in ``game_surface``) ───────────────────
_pc_autostart_watch_state = {"t": 0.0}


def _window_connect_loop():
    global wincap, screenshot_provider
    while running and not exit_requested:
        time.sleep(3)
        if not running or exit_requested:
            break

        if _emulator == "ldplayer":
            wincap, screenshot_provider = LdplayerAdbSurface.watcher_step(
                wincap,
                screenshot_provider,
                adb_path=_LDPLAYER_ADB_PATH,
                device_serial=config.get("ldplayer_device_serial") or None,
                logger=log,
                update_vision_scale=update_vision_scale,
            )
        else:
            wincap, screenshot_provider = PcWin32Surface.watcher_step(
                wincap,
                screenshot_provider,
                window_name=_GAME_WINDOW_NAME,
                adb_path=_LDPLAYER_ADB_PATH,
                auto_focus=auto_focus,
                win_w=_win_w,
                win_h=_win_h,
                lastz_pid=lastz_pid,
                lastz_exe_path=LASTZ_EXE_PATH,
                auto_start_lastz=_auto_start_lastz,
                bot_paused=bot_paused,
                autostart_state=_pc_autostart_watch_state,
                logger=log,
                update_vision_scale=update_vision_scale,
            )

_window_watcher_thread = threading.Thread(target=_window_connect_loop, daemon=True)
_window_watcher_thread.start()

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
if wincap is not None and getattr(wincap, "hwnd", None):
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
    treasure_template_path="buttons_template/Treasure1-3.png",
    threshold=0.70,
    clear_sec=10.0,
    re_trigger_interval_sec=60.0,  # emit "started" again while visible; per-function cooldown in config throttles actual runs
    roi_center_x=0.74,
    roi_center_y=0.89,
    roi_padding=2,
)

# Callbacks for trigger-based functions: send_zalo with repeat_interval_sec uses these to repeat while detector still active.
# When game window is closed, detector can't run so _attacked/_treasure_visible never clear — we must return False so Zalo repeat stops.
def _is_game_window_valid():
    return is_game_surface_valid(wincap)

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

connection_detector = ConnectionDetector(
    buff_icon_template_path="buttons_template/BuffIcon.png",
    world_zoomout_template_path="buttons_template/HeadquartersButton.png",
    world_zoomout_button_path="buttons_template/WorldButton.png",
    lastz_exe_path=LASTZ_EXE_PATH,
    lastz_pid_ref=lastz_pid,
    lastz_window_name=_GAME_WINDOW_NAME,
    interval_sec=300.0,   # 5 minutes
    buff_icon_threshold=0.75,
)

# ── Detector background thread ─────────────────────────────────────────────────
# Detectors run matchTemplate (~0.3s each × 3 = ~0.9s total). Running them on the
# main thread blocks clicking. Instead, run them in a background thread every 2s
# and post events to a queue for the main loop to handle.
_detector_event_queue = queue.Queue()
_DETECTOR_INTERVAL = 5.0  # background detectors check every 2s
_last_detector_screenshot_fail_log = 0.0  # throttle "screenshot failed" log

def _detector_loop():
    global _last_detector_screenshot_fail_log
    log.info("[Detector] Background thread started (interval={}s)".format(_DETECTOR_INTERVAL))
    _tick = 0
    while running and not exit_requested:
        time.sleep(_DETECTOR_INTERVAL)
        if not running or exit_requested:
            break
        # When Is Running (UI) = false, pause all detectors — no screenshot, no matchTemplate
        if bot_paused["paused"]:
            continue
        # Skip if window not connected yet
        if wincap is None or screenshot_provider is None:
            continue
        _tick += 1
        try:
            img = screenshot_provider.get_screenshot()
            if img is None:
                _now = time.time()
                if _now - _last_detector_screenshot_fail_log >= 15.0:
                    log.warning(
                        "[Detector] #{} screenshot returned None (window invalid or ADB error)".format(_tick)
                    )
                    _last_detector_screenshot_fail_log = _now
                try:
                    connection_detector.update(wincap, vision_cache, log, current_screenshot=None,
                                               is_busy=lambda: runner.state == "running")
                except Exception as conn_e:
                    log.error("[Detector] connection_detector on screenshot fail: {}".format(conn_e))
                continue
        except Exception as e:
            _now = time.time()
            if _now - _last_detector_screenshot_fail_log >= 15.0:
                log.warning("[Detector] #{} screenshot failed: {} (run connection_detector to recover)".format(_tick, e))
                _last_detector_screenshot_fail_log = _now
            # Still run connection_detector so it can check process and restart game if needed
            try:
                connection_detector.update(wincap, vision_cache, log, current_screenshot=None,
                                           is_busy=lambda: runner.state == "running")
            except Exception as conn_e:
                log.error("[Detector] connection_detector on screenshot fail: {}".format(conn_e))
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
        # Do not trigger treasure when on login screen (PasswordSlot visible) — avoids false positives.
        try:
            treasure_event = treasure_detector.update(img, log)
            if treasure_event == "started":
                _password_visible = False
                try:
                    _v = vision_cache.get("buttons_template/PasswordSlot.png")
                    if _v:
                        _password_visible = _v.exists(img, threshold=0.75)
                except Exception:
                    pass
                if _password_visible:
                    log.info("[Detector] #{} → treasure_detected suppressed (on login screen)".format(_tick))
                else:
                    log.info("[Detector] #{} → treasure_detected, triggers={}".format(_tick, treasure_detected_triggers))
                    _detector_event_queue.put(("treasure_detected", list(treasure_detected_triggers)))
        except Exception as e:
            log.error("[Detector] #{} treasure_detector crashed: {}".format(_tick, e))

        # ── Exit banner detector ────────────────────────
        try:
            # Check ExitGameBanner every N ticks — click corner to dismiss
            if exit_banner_detector.update(img, wincap, log):
                sx, sy = exit_banner_detector.corner_screen_pos(wincap)
                import adb_input as _adb_banner
                _adb_b = _adb_banner.get_adb_input()
                if _adb_b is not None:
                    _adb_b.tap(int(sx), int(sy))
                else:
                    pyautogui.click(sx, sy)
                log.info("[Detector] #{} → ExitGameBanner detected, tapped corner ({}, {})".format(_tick, sx, sy))
        except Exception as e:
            log.error("[Detector] #{} exit_banner_detector crashed: {}".format(_tick, e))

        # ── Connection detector (every 5 min when Is Running: world_zoomout → click middle → BuffIcon; kill+restart if disconnected) ──
        try:
            connection_detector.update(wincap, vision_cache, log, current_screenshot=img,
                                       is_busy=lambda: runner.state == "running")
        except Exception as e:
            log.error("[Detector] #{} connection_detector crashed: {}".format(_tick, e))
    log.info("[Detector] Background thread exited")

_detector_thread = threading.Thread(target=_detector_loop, daemon=True)
_detector_thread.start()

# ── Game loop (background thread; Tkinter runs on main thread to avoid startup hang) ──
_CAPTURE_INTERVAL = 0.1   # 10 FPS cap
_last_capture_time = 0.0
_last_invalid_handle_log = 0.0   # throttle "Invalid window handle" log to at most once per 15s
last_stopped_key = None
detector_lock = threading.Lock()
_last_detector_restart = 0
_was_paused_prev = True   # so on first run we don't reset; reset logout_detector when resuming from pause

def _game_loop():
    global _last_detector_restart, _detector_thread, running, _last_capture_time, _last_invalid_handle_log, last_stopped_key, _show_preview, _was_paused_prev
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
            _was_paused_prev = True
            # Drain detector queue so we don't process stale events on resume
            try:
                while True:
                    _detector_event_queue.get_nowait()
            except queue.Empty:
                pass
            _safe_waitkey(1)
            continue

        # Just resumed from pause: reset all detectors so they re-evaluate and can trigger again
        if _was_paused_prev:
            _was_paused_prev = False
            logout_detector.reset()
            attack_detector.reset()
            alliance_attack_detector.reset()
            treasure_detector.reset()
            exit_banner_detector.reset()
            connection_detector.reset()
            log.info("[Detector] Resumed: all detectors reset, will re-check (logged_out, attacked, treasure, etc.).")

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
        # Skip screenshot loop if window not connected yet
        if wincap is None or screenshot_provider is None:
            _safe_waitkey(100)
            continue
        try:
            update_vision_scale()
        except Exception as _e:
            if getattr(_e, "winerror", None) == 1400:
                _t = time.time()
                if _t - _last_invalid_handle_log >= 15.0:
                    log.warning("[GameLoop] Invalid window handle, skipping (window may be restarting)")
                    _last_invalid_handle_log = _t
                _safe_waitkey(100)
                continue
            raise

        # Get screenshot and update runner (invalid handle possible if connection_detector just killed LastZ)
        try:
            screenshot = screenshot_provider.get_screenshot()
        except Exception as _e:
            if getattr(_e, "winerror", None) == 1400:
                _t = time.time()
                if _t - _last_invalid_handle_log >= 15.0:
                    log.warning("[GameLoop] Invalid window handle, skipping (window may be restarting)")
                    _last_invalid_handle_log = _t
                _safe_waitkey(100)
                continue
            raise
        if screenshot is None:
            continue

        # Drain detector events from background thread (zero matchTemplate cost on main thread)
        try:
            while True:
                event_type, triggers = _detector_event_queue.get_nowait()
                _now = time.time()
                if event_type == "logged_out":
                    for fn_name in triggers:
                        if not _trigger_cooldown_ok(fn_name):
                            log.info("[Scheduler] {} skipped (cooldown) [logged_out]".format(fn_name))
                            continue
                        _start_urgent(fn_name, trigger="logged_out")
                elif event_type == "attacked":
                    for fn_name in triggers:
                        if not _trigger_cooldown_ok(fn_name):
                            log.info("[Scheduler] {} skipped (cooldown) [attacked]".format(fn_name))
                            continue
                        last_triggered_at[fn_name] = _now
                        _try_start(fn_name, trigger="attacked", trigger_event="attacked",
                                   trigger_active_cb=trigger_active_callbacks.get("attacked"))
                elif event_type == "alliance_attacked":
                    for fn_name in triggers:
                        if not _trigger_cooldown_ok(fn_name):
                            log.info("[Scheduler] {} skipped (cooldown) [alliance_attacked]".format(fn_name))
                            continue
                        last_triggered_at[fn_name] = _now
                        _try_start(fn_name, trigger="alliance_attacked", trigger_event="alliance_attacked",
                                   trigger_active_cb=trigger_active_callbacks.get("alliance_attacked"))
                elif event_type == "treasure_detected":
                    for fn_name in triggers:
                        if not _trigger_cooldown_ok(fn_name):
                            log.info("[Scheduler] {} skipped (cooldown) [treasure_detected]".format(fn_name))
                            continue
                        last_triggered_at[fn_name] = _now
                        _try_start(fn_name, trigger="treasure_detected", trigger_event="treasure_detected",
                                   trigger_active_cb=trigger_active_callbacks.get("treasure_detected"))
        except queue.Empty:
            pass

        was_running = runner.state == "running"
        try:
            runner.update(screenshot, wincap)
        except Exception as e:
            log.error("[Runner] CRASH in update: {}".format(e), exc_info=True)
            runner.state = "idle"

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
