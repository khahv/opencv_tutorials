import logging
import os
import threading
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as filedialog
from datetime import datetime

from croniter import croniter
from fn_settings_schema import SCHEMA as _FN_SETTINGS_SCHEMA, COMMON_FIELDS as _FN_COMMON_FIELDS
from ui_locale import get_messages, normalize_language_code

import config_manager
import ocr_openocr
_log = logging.getLogger("kha_lastz")

BG     = "#1e1e2e"
BG2    = "#313244"
BG3    = "#181825"
FG     = "#cdd6f4"
ACCENT = "#89b4fa"
GREEN  = "#a6e3a1"
YELLOW = "#f9e2af"
RED    = "#f38ba8"
GRAY   = "#6c7086"
GRAY2  = "#45475a"
# Secondary line under each function name (on BG2); brighter than GRAY for contrast
FG_MUTED = "#a6adc8"


class BotUI:
    """
    Tkinter UI running in a daemon thread.
    - Checkbox to enable/disable each function (persisted to .env_config)
    - [KEY] badge is clickable to rebind hotkey (only when Is Running = off)
    - Status bar shows current running function / next cron
    - Stop button above the status bar aborts only the current function (Is Running stays on)
    - Turning Is Running OFF clears the scheduler FIFO queue (see clear_pending_queue_callback)
    """

    def __init__(self, fn_enabled: dict, fn_configs: list, runner_ref,
                 next_run_at: dict = None,
                 key_bindings: dict = None,
                 save_callback=None,
                 bot_paused: dict = None,
                 cron_callback=None,
                 fn_settings: dict = None,
                 settings_save_callback=None,
                 run_callback=None,
                 enabled_callback=None,
                 quit_check=None,
                 general_settings: dict = None,
                 general_settings_callback=None,
                 connection_status=None,
                 start_lastz_callback=None,
                 clear_pending_queue_callback=None):
        self._fn_enabled    = fn_enabled
        self._fn_configs    = [fc for fc in fn_configs if fc.get("name")]
        self._runner        = runner_ref
        self._next_run_at   = next_run_at  if next_run_at  is not None else {}
        self._key_bindings  = key_bindings if key_bindings is not None else {}
        self._save_callback = save_callback
        self._bot_paused    = bot_paused   if bot_paused   is not None else {"paused": False}
        self._cron_callback = cron_callback  # fn(fn_name, cron_expr_or_empty)
        self._fn_settings   = fn_settings if fn_settings is not None else {}
        self._settings_save_callback = settings_save_callback  # fn(fn_settings)
        self._run_callback  = run_callback   # fn(fn_name) — trigger function immediately
        self._enabled_callback = enabled_callback  # fn(fn_name, enabled_bool)
        self._quit_check   = quit_check   # callable() -> bool, if True close UI (for Ctrl+C)
        self._general_settings = general_settings if general_settings is not None else {}
        self._general_settings_callback = general_settings_callback
        self._connection_status = connection_status  # callable() -> bool; None = always connected
        self._was_disconnected = False  # tracks last known disconnected state for transition detection
        self._exe_path_var = None    # StringVar for LastZ exe path entry
        self._auto_start_var = None  # BooleanVar for auto-start checkbox
        self._pc_section_toggle = None  # callable() to show/hide PC-only settings rows
        self._start_lastz_callback = start_lastz_callback  # callable() to launch LastZ.exe
        self._clear_pending_queue_callback = clear_pending_queue_callback  # callable() when Is Running → OFF
        self._btn_start_lastz = None  # header button, visible in PC mode only
        self._capture_fps_var = None  # IntVar: max screenshot FPS (1–50) in App settings dialog
        self._show_preview_var = None  # BooleanVar: OpenCV live capture preview window

        self._vars       = {}   # fn_name → BooleanVar
        self._row_frames = {}   # fn_name → (row_frame, name_label, toggle_lbl)
        self._meta_lbls  = {}   # fn_name → subtitle meta tk.Label (hotkey/cron/trigger)
        self._badge_lbls = {}   # fn_name → badge tk.Label
        self._sched_lbls = {}   # fn_name → schedule "S" tk.Label
        self._gear_lbls  = {}   # fn_name → gear tk.Label (or None if no settings)
        self._play_lbls  = {}   # fn_name → play ▶ tk.Label
        self._rebinding  = None
        self._rebind_lbl = None
        self._ocr_ready  = False
        self._root       = None
        self._running_var = None
        self._running_cb  = None
        self._status_lbl  = None
        self._btn_stop_fn = None
        self._fn_objs     = {}
        self._kv_objs     = {}
        self._play_btns   = {}
        self._play_icons  = {}
        self._focus_var   = None
        # Widget refs for language refresh (set in _build)
        self._lbl_app_header = None
        self._focus_cb       = None  # only inside app-settings dialog (if open)
        self._lbl_resolution = None
        self._lbl_language   = None
        self._btn_app_settings = None
        self._app_settings_win = None  # Toplevel or None
        self._lbl_functions  = None
        self._lbl_rebind_hint = None
        self._btn_enable_all  = None
        self._btn_disable_all = None

    # Map YAML trigger id → ui_locale message key
    _TRIGGER_I18N = {
        "logged_out":        "trigger_logged_out",
        "attacked":          "trigger_attacked",
        "alliance_attacked": "trigger_alliance_attacked",
        "treasure_detected": "trigger_treasure",
    }

    def _current_lang(self) -> str:
        return normalize_language_code(self._general_settings.get("language", "en"))

    def _t(self, msg_key: str, **kwargs) -> str:
        """Translate *msg_key* using the current UI language."""
        msgs = get_messages(self._current_lang())
        s = msgs.get(msg_key, msg_key)
        if kwargs:
            try:
                return s.format(**kwargs)
            except (KeyError, ValueError):
                return s
        return s

    def _lang_menu_labels(self):
        """Fixed menu captions: English name / Vietnamese name (not swapped by UI lang)."""
        en = get_messages("en")
        vi = get_messages("vi")
        return en["lang_en"], vi["lang_vi"]

    def _lang_code_from_display(self, display_label: str) -> str:
        en_l, vi_l = self._lang_menu_labels()
        return "vi" if display_label == vi_l else "en"

    def _apply_language_change(self, *_args) -> None:
        """Persist UI language from ``self._lang_var`` and refresh visible strings."""
        code = self._lang_code_from_display(self._lang_var.get())
        self._general_settings["language"] = code
        if self._general_settings_callback:
            self._general_settings_callback("language", code)
        self._refresh_static_ui_texts()
        if self._status_lbl and self._bot_paused.get("paused") and not self._rebinding:
            self._status_lbl.config(text=self._t("status_paused_all"), fg=YELLOW)

    def _sync_general_settings_from_disk(self) -> None:
        """Merge general_settings from .env_config into the live dict.

        Disk supplies hand-edited keys (e.g. fast_user_mouse_*); in-memory values win on
        duplicate keys so runtime stays authoritative for managed settings.
        """
        disk_gs = config_manager.load_general_settings()
        if not disk_gs:
            return
        merged = dict(disk_gs)
        merged.update(self._general_settings)
        self._general_settings.clear()
        self._general_settings.update(merged)

    def _show_app_settings(self) -> None:
        """Modal window: resolution, language, auto-focus (same behavior as former header)."""
        if self._running_var is not None and self._running_var.get():
            self._show_toast(self._t("toast_pause_for_settings"))
            return
        if self._app_settings_win is not None and self._app_settings_win.winfo_exists():
            self._app_settings_win.lift()
            self._app_settings_win.focus_force()
            return

        self._sync_general_settings_from_disk()

        dlg = tk.Toplevel(self._root)
        self._app_settings_win = dlg
        dlg.title(self._t("dlg_app_settings_title"))
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.transient(self._root)
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        self._app_settings_loading = True

        _emulator_at_open = self._general_settings.get("emulator", "pc")

        tk.Label(dlg, text=self._t("dlg_app_settings_heading"),
                 font=("Segoe UI", 12, "bold"), bg=BG, fg=ACCENT
                 ).pack(fill="x", padx=16, pady=(14, 8))
        tk.Frame(dlg, bg=GRAY2, height=1).pack(fill="x", padx=16, pady=(0, 10))

        RESOLUTIONS = ["1080x1920", "540x960"]
        ww = self._general_settings.get("window_width")
        wh = self._general_settings.get("window_height")
        self._resolution_var.set(
            "1080x1920" if (ww == 1080 and wh == 1920) else "540x960")
        self._focus_var.set(bool(self._general_settings.get("auto_focus", False)))
        en_lbl, vi_lbl = self._lang_menu_labels()
        self._lang_var.set(vi_lbl if self._current_lang() == "vi" else en_lbl)
        self._emulator_var.set(self._general_settings.get("emulator", "pc"))

        def _setting_row(label_key: str):
            fr = tk.Frame(dlg, bg=BG)
            fr.pack(fill="x", padx=16, pady=(0, 12))
            tk.Label(fr, text=self._t(label_key), font=("Segoe UI", 10),
                     bg=BG, fg=FG, width=20, anchor="w").pack(side="left")
            return fr

        rf_res = _setting_row("resolution")
        om_res = tk.OptionMenu(rf_res, self._resolution_var, *RESOLUTIONS,
                               command=self._on_resolution_change)
        om_res.config(font=("Segoe UI", 10), bg=BG2, fg=FG, activebackground=GRAY2,
                      activeforeground=FG, highlightthickness=0, relief="flat")
        om_res.pack(side="left", padx=(8, 0))

        rf_lang = _setting_row("language")
        om_lang = tk.OptionMenu(rf_lang, self._lang_var, en_lbl, vi_lbl,
                                command=lambda _v: self._apply_language_change())
        om_lang.config(font=("Segoe UI", 10), bg=BG2, fg=FG, activebackground=GRAY2,
                       activeforeground=FG, highlightthickness=0, relief="flat")
        om_lang.pack(side="left", padx=(8, 0))

        EMULATORS = ["pc", "ldplayer"]
        rf_emu = _setting_row("emulator_platform")
        om_emu = tk.OptionMenu(rf_emu, self._emulator_var, *EMULATORS,
                               command=self._on_emulator_change)
        om_emu.config(font=("Segoe UI", 10), bg=BG2, fg=FG, activebackground=GRAY2,
                      activeforeground=FG, highlightthickness=0, relief="flat")
        om_emu.pack(side="left", padx=(8, 0))

        # ── PC-only settings (shown only when emulator = pc) ──────────────────
        _cur_exe = self._general_settings.get("lastz_exe_path", "")
        _cur_autostart = bool(self._general_settings.get("auto_start_lastz", False))
        if self._exe_path_var is None:
            self._exe_path_var = tk.StringVar(value=_cur_exe)
        else:
            self._exe_path_var.set(_cur_exe)
        if self._auto_start_var is None:
            self._auto_start_var = tk.BooleanVar(value=_cur_autostart)
        else:
            self._auto_start_var.set(_cur_autostart)

        pc_frame = tk.Frame(dlg, bg=BG)

        # LastZ exe path row
        rf_exe = tk.Frame(pc_frame, bg=BG)
        rf_exe.pack(fill="x", padx=0, pady=(0, 8))
        tk.Label(rf_exe, text=self._t("lastz_exe_path"), font=("Segoe UI", 10),
                 bg=BG, fg=FG, width=20, anchor="w").pack(side="left")
        exe_entry = tk.Entry(rf_exe, textvariable=self._exe_path_var,
                             font=("Segoe UI", 9), bg=BG2, fg=FG,
                             insertbackground=FG, relief="flat", width=28)
        exe_entry.pack(side="left", padx=(8, 4))

        def _browse_exe():
            path = filedialog.askopenfilename(
                title="Select LastZ exe",
                filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
                initialfile=self._exe_path_var.get() or "",
            )
            if path:
                self._exe_path_var.set(path)
                _on_exe_path_change()

        def _on_exe_path_change(*_):
            if self._general_settings_callback:
                self._general_settings_callback("lastz_exe_path", self._exe_path_var.get())

        exe_entry.bind("<FocusOut>", _on_exe_path_change)
        exe_entry.bind("<Return>", _on_exe_path_change)
        tk.Button(rf_exe, text=self._t("btn_browse"), command=_browse_exe,
                  font=("Segoe UI", 9), bg=BG2, fg=ACCENT, relief="flat",
                  padx=8, pady=2, cursor="hand2").pack(side="left")

        # Auto-start checkbox row
        rf_as = tk.Frame(pc_frame, bg=BG)
        rf_as.pack(fill="x", padx=0, pady=(0, 4))
        tk.Label(rf_as, bg=BG, width=20).pack(side="left")  # indent to align with label column
        tk.Checkbutton(
            rf_as, text=self._t("auto_start_lastz"), variable=self._auto_start_var,
            command=lambda: (self._general_settings_callback("auto_start_lastz", self._auto_start_var.get())
                             if self._general_settings_callback else None),
            font=("Segoe UI", 10), bg=BG, activebackground=BG,
            selectcolor=GRAY2, fg=FG, activeforeground=FG,
            relief="flat", bd=0, highlightthickness=0,
        ).pack(anchor="w")

        def _refresh_pc_section():
            if self._emulator_var.get() == "pc":
                pc_frame.pack(fill="x", padx=16, pady=(0, 4))
            else:
                pc_frame.pack_forget()

        self._pc_section_toggle = _refresh_pc_section
        _refresh_pc_section()  # apply initial visibility

        rf_af = tk.Frame(dlg, bg=BG)
        rf_af.pack(fill="x", padx=16, pady=(0, 14))
        tk.Checkbutton(
            rf_af, text=self._t("auto_focus_window"), variable=self._focus_var,
            command=self._on_focus_toggle,
            font=("Segoe UI", 10), bg=BG, activebackground=BG,
            selectcolor=GRAY2, fg=FG, activeforeground=FG,
            relief="flat", bd=0, highlightthickness=0,
        ).pack(anchor="w")

        if self._show_preview_var is None:
            self._show_preview_var = tk.BooleanVar(
                value=bool(self._general_settings.get("show_preview", False)))
        else:
            self._show_preview_var.set(bool(self._general_settings.get("show_preview", False)))

        rf_pv = tk.Frame(dlg, bg=BG)
        rf_pv.pack(fill="x", padx=16, pady=(0, 12))
        tk.Checkbutton(
            rf_pv,
            text=self._t("show_preview_window"),
            variable=self._show_preview_var,
            command=lambda: (
                self._general_settings_callback("show_preview", self._show_preview_var.get())
                if self._general_settings_callback else None
            ),
            font=("Segoe UI", 10),
            bg=BG,
            activebackground=BG,
            selectcolor=GRAY2,
            fg=FG,
            activeforeground=FG,
            relief="flat",
            bd=0,
            highlightthickness=0,
        ).pack(anchor="w")

        # Screenshot rate as FPS (1–50); stored as capture_interval_sec = 1/fps in config
        _iv = float(self._general_settings.get("capture_interval_sec", 0.1) or 0.1)
        _fps0 = max(1, min(50, int(round(1.0 / _iv)))) if _iv > 0 else 10
        if self._capture_fps_var is None:
            self._capture_fps_var = tk.IntVar(value=_fps0)
        else:
            self._capture_fps_var.set(_fps0)

        rf_cap = _setting_row("capture_interval")

        def _apply_capture_fps(*_):
            # Prefer widget text so closing the dialog without FocusOut still saves typed value.
            if getattr(self, "_app_settings_loading", False):
                return
            try:
                fps = int(str(sp_cap.get()).strip())
            except (ValueError, TypeError, tk.TclError):
                try:
                    fps = int(self._capture_fps_var.get())
                except (tk.TclError, ValueError, TypeError):
                    return
            fps = max(1, min(50, fps))
            self._capture_fps_var.set(fps)
            interval = 1.0 / float(fps)
            prev = self._general_settings.get("capture_interval_sec")
            if prev is not None and abs(float(interval) - float(prev)) < 1e-5:
                return
            self._general_settings["capture_interval_sec"] = interval
            if self._general_settings_callback:
                self._general_settings_callback("capture_interval_sec", interval)

        sp_cap = tk.Spinbox(
            rf_cap,
            textvariable=self._capture_fps_var,
            from_=1,
            to=50,
            increment=1,
            width=5,
            command=_apply_capture_fps,
            font=("Segoe UI", 10),
            bg=BG2,
            fg=FG,
            insertbackground=FG,
            buttonbackground=GRAY2,
            highlightthickness=0,
            relief="flat",
        )
        sp_cap.pack(side="left", padx=(8, 0))
        sp_cap.bind("<FocusOut>", lambda _e: _apply_capture_fps())
        sp_cap.bind("<Return>", lambda _e: _apply_capture_fps())
        tk.Label(rf_cap, text="FPS", font=("Segoe UI", 10, "bold"), bg=BG, fg=FG).pack(
            side="left", padx=(2, 0)
        )
        tk.Label(
            rf_cap,
            text=self._t("capture_interval_hint"),
            font=("Segoe UI", 8),
            bg=BG,
            fg=GRAY,
            wraplength=280,
            justify="left",
        ).pack(side="left", padx=(8, 0))

        def _close():
            # Always flush FPS from spinbox before destroy (Close / X may skip FocusOut).
            self._app_settings_loading = False  # allow flush if dialog closed before after_idle
            _apply_capture_fps()
            _new_emu = self._emulator_var.get()
            if _new_emu != _emulator_at_open and self._general_settings_callback:
                self._general_settings_callback("emulator", _new_emu)
            self._pc_section_toggle = None
            dlg.grab_release()
            dlg.destroy()
            self._app_settings_win = None

        dlg.protocol("WM_DELETE_WINDOW", _close)

        btn_cfg = dict(font=("Segoe UI", 9, "bold"), relief="flat",
                       padx=16, pady=6, cursor="hand2")
        tk.Button(dlg, text=self._t("btn_close"), command=_close,
                  bg=GRAY2, fg=FG, **btn_cfg).pack(pady=(4, 14))

        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        cx = self._root.winfo_x() + self._root.winfo_width()  // 2
        cy = self._root.winfo_y() + self._root.winfo_height() // 2
        dlg.geometry("+{}+{}".format(cx - w // 2, cy - h // 2))

        def _end_app_settings_loading() -> None:
            self._app_settings_loading = False

        self._root.after_idle(_end_app_settings_loading)

    def _format_row_meta(self, fc: dict) -> str:
        """One-line subtitle under each function name (locale-aware)."""
        trigger = fc.get("trigger", "")
        cron    = fc.get("cron", "")
        key     = fc.get("key", "")
        if trigger:
            return self._t("meta_trigger", trigger=trigger)
        if cron:
            return self._t("meta_cron", cron=cron)
        if key:
            return self._t("meta_hotkey")
        return self._t("meta_manual")

    def _refresh_all_row_meta(self) -> None:
        for fc in self._fn_configs:
            name = fc.get("name")
            if not name or name not in self._meta_lbls:
                continue
            self._meta_lbls[name].config(text=self._format_row_meta(fc))

    def _refresh_static_ui_texts(self) -> None:
        """Update all main-window strings after a language change."""
        if not self._root:
            return
        self._root.title(self._t("app_title"))
        if self._lbl_app_header is not None:
            self._lbl_app_header.config(text=self._t("app_title"))
        if self._running_cb is not None:
            self._running_cb.config(text=self._t("is_running"))
        if self._focus_cb is not None:
            self._focus_cb.config(text=self._t("auto_focus_window"))
        if self._lbl_resolution is not None:
            self._lbl_resolution.config(text=self._t("resolution"))
        if self._lbl_language is not None:
            self._lbl_language.config(text=self._t("language"))
        if self._btn_app_settings is not None:
            self._btn_app_settings.config(text=self._t("btn_app_settings"))
            self._update_app_settings_button_state()
        if self._btn_start_lastz is not None:
            self._btn_start_lastz.config(text=self._t("btn_start_lastz"))
            self._update_start_lastz_button_visibility()
        if self._lbl_functions is not None:
            self._lbl_functions.config(text=self._t("functions_header"))
        if self._lbl_rebind_hint is not None:
            self._lbl_rebind_hint.config(text=self._t("rebind_hint"))
        if self._btn_enable_all is not None:
            self._btn_enable_all.config(text=self._t("enable_all"))
        if self._btn_disable_all is not None:
            self._btn_disable_all.config(text=self._t("disable_all"))
        if self._btn_stop_fn is not None:
            self._btn_stop_fn.config(text=self._t("btn_stop_function"))
        self._refresh_all_row_meta()

    def _preset_definitions(self):
        """(message_key, cron_expr) for schedule presets; first entry is empty placeholder."""
        return [
            ("preset_select",           ""),
            ("preset_every_minute",     "* * * * *"),
            ("preset_every_5_min",      "*/5 * * * *"),
            ("preset_every_10_min",     "*/10 * * * *"),
            ("preset_every_15_min",     "*/15 * * * *"),
            ("preset_every_30_min",     "*/30 * * * *"),
            ("preset_every_hour",       "0 * * * *"),
            ("preset_every_2h",         "0 */2 * * *"),
            ("preset_every_4h",         "0 */4 * * *"),
            ("preset_every_6h",         "0 */6 * * *"),
            ("preset_every_12h",        "0 */12 * * *"),
            ("preset_daily_midnight",   "0 0 * * *"),
            ("preset_daily_8am",        "0 8 * * *"),
            ("preset_daily_noon",       "0 12 * * *"),
            ("preset_weekly_mon_8",     "0 8 * * 1"),
        ]

    def _preset_pairs(self):
        return [(self._t(k), c) for k, c in self._preset_definitions()]

    def start(self):
        """Start UI in a background thread (legacy). Prefer run_main() on main thread to avoid hangs."""
        threading.Thread(target=self._run, daemon=True).start()

    def run_main(self):
        """Run UI mainloop on the current (main) thread. Use this to avoid startup hangs on Windows."""
        self._run()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _run(self):
        self._root = tk.Tk()
        self._root.title(self._t("app_title"))
        self._root.configure(bg=BG)
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        # Force window size (tuned for 4K: larger so window is visible)
        self._root.minsize(1200, 2000)
        self._root.geometry("1200x2000")
        self._root.after(500, self._tick)
        self._root.mainloop()

    def _build(self):
        r = self._root

        # Header
        hf = tk.Frame(r, bg=BG3, padx=16, pady=12)
        hf.pack(fill="x")
        self._lbl_app_header = tk.Label(hf, text=self._t("app_title"),
                                        font=("Segoe UI", 14, "bold"),
                                        bg=BG3, fg=ACCENT)
        self._lbl_app_header.pack(side="left")

        # Is Running toggle — starts OFF, enabled once OpenOCR preloads (or if it fails)
        self._running_var = tk.BooleanVar(value=False)
        self._running_cb  = tk.Checkbutton(
            hf, text=self._t("is_running"),
            variable=self._running_var,
            command=self._on_running_toggle,
            font=("Segoe UI", 10, "bold"),
            bg=BG3, activebackground=BG3,
            fg=RED, activeforeground=RED,
            selectcolor=GRAY2, cursor="arrow",
            state="disabled",
            relief="flat", bd=0, highlightthickness=0)
        self._running_cb.pack(side="right")

        # Resolution / language / auto-focus live in the app-settings dialog (⚙ button).
        self._focus_var = tk.BooleanVar(value=self._general_settings.get("auto_focus", False))
        _ww = self._general_settings.get("window_width")
        _wh = self._general_settings.get("window_height")
        _cur_res = "1080x1920" if (_ww == 1080 and _wh == 1920) else "540x960"
        self._resolution_var = tk.StringVar(value=_cur_res)
        _en0, _vi0 = self._lang_menu_labels()
        self._lang_var = tk.StringVar(value=_vi0 if self._current_lang() == "vi" else _en0)
        _emulator0 = self._general_settings.get("emulator", "pc")
        self._emulator_var = tk.StringVar(value=_emulator0)

        self._btn_app_settings = tk.Button(
            hf, text=self._t("btn_app_settings"),
            command=self._show_app_settings,
            font=("Segoe UI", 10, "bold"), bg=GRAY2, fg=ACCENT,
            relief="flat", padx=12, pady=4, cursor="hand2",
            activebackground=GRAY2, activeforeground=ACCENT)
        self._btn_app_settings.pack(side="right", padx=(0, 16))
        self._update_app_settings_button_state()

        self._btn_start_lastz = tk.Button(
            hf, text=self._t("btn_start_lastz"),
            command=self._on_start_lastz,
            font=("Segoe UI", 10, "bold"), bg=GRAY2, fg=GREEN,
            relief="flat", padx=12, pady=4, cursor="hand2",
            activebackground=GRAY2, activeforeground=GREEN)
        self._update_start_lastz_button_visibility()

        # Stop current function only (does not pause Is Running)
        # Frame() does not accept pady=(top, bottom) — Tcl errors with "bad screen distance".
        stop_row = tk.Frame(r, bg=BG2, padx=16, pady=4)
        stop_row.pack(fill="x", pady=(1, 0))
        self._btn_stop_fn = tk.Button(
            stop_row,
            text=self._t("btn_stop_function"),
            command=self._on_stop_function,
            font=("Segoe UI", 9, "bold"),
            bg=GRAY2,
            fg=RED,
            activebackground=GRAY2,
            activeforeground=RED,
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            state="disabled",
        )
        self._btn_stop_fn.pack(side="right")

        # Status bar
        sf = tk.Frame(r, bg=BG2, padx=16, pady=8)
        sf.pack(fill="x", pady=(0, 0))

        init_text = self._t("status_idle")
        init_fg   = FG
        if not ocr_openocr.OPENOCR_OK:
            init_text = self._t("status_init_ocr")
            init_fg   = ACCENT

        self._status_lbl = tk.Label(sf, text=init_text,
                                    font=("Segoe UI", 10), bg=BG2, fg=init_fg, anchor="w")
        self._status_lbl.pack(fill="x")

        # Section header
        lf = tk.Frame(r, bg=BG, padx=16)
        lf.pack(fill="x", pady=(10, 4))
        self._lbl_functions = tk.Label(lf, text=self._t("functions_header"),
                                       font=("Segoe UI", 8, "bold"),
                                       bg=BG, fg=GRAY)
        self._lbl_functions.pack(side="left")
        self._lbl_rebind_hint = tk.Label(lf, text=self._t("rebind_hint"),
                                         font=("Segoe UI", 8), bg=BG, fg=GRAY)
        self._lbl_rebind_hint.pack(side="right")
        _btn_cfg = dict(font=("Segoe UI", 8, "bold"), relief="flat",
                        padx=8, pady=1, cursor="hand2")
        self._btn_disable_all = tk.Button(lf, text=self._t("disable_all"),
                                          bg=GRAY2, fg=GRAY,
                                          command=lambda: self._toggle_all_enabled(False),
                                          **_btn_cfg)
        self._btn_disable_all.pack(side="right", padx=(0, 4))
        self._btn_enable_all = tk.Button(lf, text=self._t("enable_all"),
                                         bg=GRAY2, fg=GREEN,
                                         command=lambda: self._toggle_all_enabled(True),
                                         **_btn_cfg)
        self._btn_enable_all.pack(side="right", padx=(0, 4))

        # Function list: fixed height + scroll (theo tài liệu Tk: scrollregion + yscrollcommand + command)
        LIST_HEIGHT = 620
        list_container = tk.Frame(r, bg=BG)
        list_container.pack(fill="both", expand=True, pady=(0, 4))

        canvas = tk.Canvas(list_container, bg=BG, highlightthickness=0, height=LIST_HEIGHT)
        scrollbar = tk.Scrollbar(list_container, orient="vertical", command=canvas.yview, bg=BG2)
        # Hai chiều (tkinter docs): canvas báo view cho scrollbar; scrollbar gọi canvas.yview khi kéo
        canvas.configure(yscrollcommand=scrollbar.set)

        ff = tk.Frame(canvas, bg=BG, padx=10)
        canvas_window = canvas.create_window((0, 0), window=ff, anchor="nw")

        def _update_scroll_region():
            """Cập nhật scrollregion theo bbox nội dung (để scroll hoạt động đủ; thumb theo yview)."""
            try:
                canvas.update_idletasks()
                bbox = canvas.bbox("all")
                if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    canvas.configure(scrollregion=bbox)
                w = max(1, canvas.winfo_width())
                canvas.itemconfig(canvas_window, width=w)
            except tk.TclError:
                pass

        def _on_frame_configure(e):
            _update_scroll_region()

        def _on_canvas_configure(e):
            w = max(1, e.width)
            canvas.itemconfig(canvas_window, width=w)
            canvas.after_idle(_update_scroll_region)

        ff.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        # Bind globally while the mouse is anywhere inside list_container.
        # Using the outermost container avoids spurious Leave events that fire
        # when the mouse moves between inner child widgets (buttons, labels, etc.).
        def _on_list_enter(e):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _on_list_leave(e):
            canvas.unbind_all("<MouseWheel>")

        list_container.bind("<Enter>", _on_list_enter)
        list_container.bind("<Leave>", _on_list_leave)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        for fc in self._fn_configs:
            self._build_row(ff, fc)

        tk.Frame(r, bg=BG, height=8).pack(fill="x")

    def _on_stop_function(self):
        """Abort the running YAML function; keep Is Running ON so cron/detectors continue."""
        if self._bot_paused.get("paused"):
            return
        r = self._runner
        if getattr(r, "state", "idle") != "running":
            return
        fn = getattr(r, "function_name", None)
        if not fn:
            return
        if hasattr(r, "abort_current_function"):
            r.abort_current_function()
        else:
            r.stop()
        _log.info("[UI] Stop button — aborted function: {}".format(fn))

    def _sync_stop_function_button(self):
        """Enable Stop only when a function is active and the bot is not globally paused."""
        btn = self._btn_stop_fn
        if btn is None:
            return
        try:
            if not btn.winfo_exists():
                return
        except tk.TclError:
            return
        if self._connection_status is not None and not self._connection_status():
            btn.config(state="disabled", cursor="arrow")
            return
        if not self._ocr_ready:
            btn.config(state="disabled", cursor="arrow")
            return
        if self._bot_paused.get("paused"):
            btn.config(state="disabled", cursor="arrow")
            return
        if getattr(self._runner, "state", "idle") == "running" and getattr(
            self._runner, "function_name", None
        ):
            btn.config(state="normal", cursor="hand2")
        else:
            btn.config(state="disabled", cursor="arrow")

    def _on_running_toggle(self):
        paused = not self._running_var.get()
        self._bot_paused["paused"] = paused
        if paused:
            cancelled = None
            if getattr(self._runner, "state", "idle") == "running":
                cancelled = getattr(self._runner, "function_name", None)
                self._runner.stop()
            if self._clear_pending_queue_callback:
                try:
                    self._clear_pending_queue_callback()
                except Exception as e:
                    _log.warning("[UI] clear_pending_queue_callback failed: %s", e)
            _log.info("[UI] Is Running → OFF (paused)")
            if cancelled:
                _log.info("[UI] Cancelled running function: {}".format(cancelled))
            self._running_cb.config(fg=RED, activeforeground=RED)
            self._status_lbl.config(text=self._t("status_paused_all"), fg=YELLOW)
        else:
            _log.info("[UI] Is Running → ON (resumed)")
            self._running_cb.config(fg=GREEN, activeforeground=GREEN)
            self._status_lbl.config(text=self._t("status_ready"), fg=FG)
        self._update_badge_states()
        self._update_app_settings_button_state()

    def _update_start_lastz_button_visibility(self) -> None:
        """Show the Start LastZ button only in PC emulator mode."""
        btn = self._btn_start_lastz
        if btn is None:
            return
        if self._general_settings.get("emulator", "pc") == "pc" and self._start_lastz_callback:
            btn.pack(side="right", padx=(0, 8))
        else:
            btn.pack_forget()

    def _on_start_lastz(self) -> None:
        if self._start_lastz_callback:
            try:
                self._start_lastz_callback()
            except Exception as e:
                _log.warning("[UI] Start LastZ failed: %s", e)

    def notify_disconnected(self):
        """Called (thread-safe) when the game window is lost or emulator changes.
        Forces Is Running = OFF and disables the toggle until connection is restored."""
        if self._root:
            self._root.after(0, self._apply_disconnected_state)

    def _apply_disconnected_state(self):
        """Runs on Tkinter main thread: turns off Is Running and disables the toggle."""
        if self._running_var and self._running_var.get():
            self._running_var.set(False)
            self._on_running_toggle()
        if self._running_cb:
            self._running_cb.config(state="disabled", cursor="arrow")
        self._was_disconnected = True

    def _update_app_settings_button_state(self) -> None:
        """Enable ⚙ Settings only when Is Running is off (same rule as schedule/gear)."""
        btn = self._btn_app_settings
        if btn is None or self._running_var is None:
            return
        if self._running_var.get():
            btn.config(state="disabled", fg=GRAY, cursor="arrow", takefocus=0)
        else:
            btn.config(state="normal", fg=ACCENT, cursor="hand2")

    def _on_focus_toggle(self):
        if getattr(self, "_app_settings_loading", False):
            return
        if self._general_settings_callback:
            self._general_settings_callback("auto_focus", self._focus_var.get())
        self._update_badge_states()

    def _on_resolution_change(self, value):
        if getattr(self, "_app_settings_loading", False):
            return
        if self._general_settings_callback and value:
            self._general_settings_callback("resolution", value)

    def _on_emulator_change(self, value):
        # Do NOT fire the callback here — emulator change is deferred to when settings dialog closes.
        # Only update the PC-settings section visibility immediately.
        if self._pc_section_toggle:
            self._pc_section_toggle()

    def _update_badge_states(self):
        """Grey out / restore all key badges, schedule and gear buttons based on Is Running state."""
        paused = self._bot_paused["paused"]
        for name, badge_lbl in self._badge_lbls.items():
            if badge_lbl is None:
                continue
            key = next((fc.get("key", "") for fc in self._fn_configs
                        if fc.get("name") == name), "")
            if paused:
                badge_lbl.config(fg=ACCENT if key else GRAY, cursor="hand2")
            else:
                badge_lbl.config(fg=GRAY, cursor="arrow")

        for name, sched_lbl in self._sched_lbls.items():
            if sched_lbl is None:
                continue
            has_cron = bool(next((fc.get("cron", "") for fc in self._fn_configs
                                  if fc.get("name") == name), ""))
            if paused:
                sched_lbl.config(fg=ACCENT if has_cron else FG, cursor="hand2")
            else:
                sched_lbl.config(fg=GRAY, cursor="arrow")

        for name, gear_lbl in self._gear_lbls.items():
            if gear_lbl is None:
                continue
            if paused:
                gear_lbl.config(fg=ACCENT, cursor="hand2")
            else:
                gear_lbl.config(fg=GRAY, cursor="arrow")

        for name, play_lbl in self._play_lbls.items():
            if play_lbl is None:
                continue
            if paused:
                play_lbl.config(fg=GRAY, cursor="arrow")
            else:
                play_lbl.config(fg=GREEN, cursor="hand2")

    def _on_close(self):
        os._exit(0)

    def _build_row(self, parent, fc):
        name    = fc.get("name")
        key     = fc.get("key", "")
        meta    = self._format_row_meta(fc)

        enabled = self._fn_enabled.get(name, True)
        var = tk.BooleanVar(value=enabled)
        self._vars[name] = var

        row = tk.Frame(parent, bg=BG2, padx=10, pady=8)
        row.pack(fill="x", pady=2)

        # Custom toggle label (✓ green / ✗ gray) — clearer than tk.Checkbutton on dark bg
        toggle_lbl = tk.Label(row,
                              text="✓" if enabled else "✗",
                              font=("Segoe UI", 12, "bold"),
                              bg=BG2, fg=GREEN if enabled else GRAY,
                              cursor="hand2", width=2)
        toggle_lbl.pack(side="left")

        def _on_toggle(n=name, v=var, lbl=toggle_lbl):
            new_val = not v.get()
            v.set(new_val)
            lbl.config(text="✓" if new_val else "✗",
                       fg=GREEN if new_val else GRAY)
            self._fn_enabled[n] = new_val
            self._refresh_row(n)
            if self._enabled_callback:
                self._enabled_callback(n, new_val)
            if self._save_callback:
                self._save_callback()

        toggle_lbl.bind("<Button-1>", lambda e: _on_toggle())

        inner = tk.Frame(row, bg=BG2)
        inner.pack(side="left", fill="x", expand=True, padx=(6, 0))

        name_lbl = tk.Label(inner, text=name,
                            font=("Segoe UI", 10, "bold"),
                            bg=BG2, fg=FG if enabled else GRAY, anchor="w")
        name_lbl.pack(fill="x")
        _meta_lbl = tk.Label(inner, text=meta, font=("Segoe UI", 8),
                             bg=BG2, fg=FG_MUTED, anchor="w")
        _meta_lbl.pack(fill="x")
        self._meta_lbls[name] = _meta_lbl

        # Key badge — greyed out when Is Running; clickable only when paused
        badge_text = " {} ".format(key.upper()) if key else " + "
        badge_fg   = GRAY  # greyed out by default (Is Running = true on start)
        badge_lbl  = tk.Label(row, text=badge_text,
                              font=("Consolas", 9, "bold"),
                              bg=GRAY2, fg=badge_fg,
                              padx=5, pady=2, relief="flat", cursor="arrow")
        badge_lbl.pack(side="right")
        badge_lbl.bind("<Button-1>",
                       lambda e, n=name, lbl=badge_lbl: self._start_rebind(n, lbl))

        # Schedule "S" button — greyed out when Is Running (same as key badge)
        sched_lbl = tk.Label(row, text=" S ",
                             font=("Consolas", 9, "bold"),
                             bg=GRAY2, fg=GRAY,
                             padx=5, pady=2, relief="flat", cursor="arrow")
        sched_lbl.pack(side="right", padx=(0, 4))
        sched_lbl.bind("<Button-1>",
                       lambda e, n=name, lbl=sched_lbl: self._show_schedule(n, lbl))

        # Gear "⚙" button — only for functions that have settings schema
        has_settings = bool(_FN_COMMON_FIELDS or _FN_SETTINGS_SCHEMA.get(name))
        if has_settings:
            gear_lbl = tk.Label(row, text=" ⚙ ",
                                font=("Segoe UI", 10),
                                bg=GRAY2, fg=GRAY,
                                padx=4, pady=2, relief="flat", cursor="arrow")
            gear_lbl.pack(side="right", padx=(0, 4))
            gear_lbl.bind("<Button-1>",
                          lambda e, n=name: self._show_fn_settings(n))
        else:
            gear_lbl = None

        # Play "▶" button — active when Is Running = ON (paused=False), greyed when paused
        is_paused = not self._running_var.get()
        play_lbl = tk.Label(row, text=" ▶ ",
                            font=("Segoe UI", 10),
                            bg=GRAY2, fg=GRAY if is_paused else GREEN,
                            padx=4, pady=2, relief="flat", 
                            cursor="arrow" if is_paused else "hand2")
        play_lbl.pack(side="right", padx=(0, 4))
        play_lbl.bind("<Button-1>", lambda e, n=name: self._run_fn(n))

        self._row_frames[name] = (row, name_lbl, toggle_lbl)
        self._badge_lbls[name] = badge_lbl
        self._sched_lbls[name] = sched_lbl
        self._gear_lbls[name]  = gear_lbl
        self._play_lbls[name]  = play_lbl

        # Tooltip on hover — show next run time / trigger / hotkey
        for w in (row, inner, name_lbl):
            self._attach_tooltip(w, name)

    # ── Schedule dialog ───────────────────────────────────────────────────────

    def _show_schedule(self, name, sched_lbl):
        if not self._bot_paused["paused"]:
            self._show_toast(self._t("toast_pause_for_schedule"))
            return

        fc = next((f for f in self._fn_configs if f.get("name") == name), None)
        if not fc:
            return

        current_cron = fc.get("cron", "")
        # Split into 5 fields, pad with * if needed
        parts = current_cron.strip().split() if current_cron.strip() else []
        while len(parts) < 5:
            parts.append("*")
        parts = parts[:5]

        dlg = tk.Toplevel(self._root)
        dlg.title(self._t("dlg_schedule_title", name=name))
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        # ── Title ──────────────────────────────────────────────────────────────
        tk.Label(dlg, text=self._t("dlg_schedule_heading", name=name),
                 font=("Segoe UI", 11, "bold"), bg=BG, fg=ACCENT
                 ).pack(fill="x", padx=14, pady=(10, 6))

        sep = tk.Frame(dlg, bg=GRAY2, height=1)
        sep.pack(fill="x", padx=14, pady=(0, 8))

        # ── Preset dropdown ────────────────────────────────────────────────────
        preset_frame = tk.Frame(dlg, bg=BG)
        preset_frame.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(preset_frame, text=self._t("preset_label"), font=("Segoe UI", 9),
                 bg=BG, fg=GRAY, width=9, anchor="w").pack(side="left")

        preset_pairs  = self._preset_pairs()
        preset_labels = [p[0] for p in preset_pairs]
        preset_var    = tk.StringVar(value=preset_labels[0])
        om = tk.OptionMenu(preset_frame, preset_var, *preset_labels)
        om.config(bg=BG2, fg=FG, activebackground=GRAY2, activeforeground=FG,
                  highlightthickness=0, relief="flat", font=("Segoe UI", 9),
                  width=22)
        om["menu"].config(bg=BG2, fg=FG, activebackground=GRAY2,
                          font=("Segoe UI", 9))
        om.pack(side="left")

        # ── 5 cron field entries ───────────────────────────────────────────────
        fields_frame = tk.Frame(dlg, bg=BG)
        fields_frame.pack(fill="x", padx=14, pady=(2, 0))

        FIELD_DEFS = [
            (self._t("cron_field_min"),     self._t("cron_hint_min"),     parts[0]),
            (self._t("cron_field_hour"),    self._t("cron_hint_hour"),    parts[1]),
            (self._t("cron_field_day"),     self._t("cron_hint_day"),     parts[2]),
            (self._t("cron_field_month"),   self._t("cron_hint_month"),   parts[3]),
            (self._t("cron_field_weekday"), self._t("cron_hint_weekday"), parts[4]),
        ]
        field_vars = []
        for col, (label, hint, default) in enumerate(FIELD_DEFS):
            cell = tk.Frame(fields_frame, bg=BG)
            cell.grid(row=0, column=col, padx=(0, 8))

            tk.Label(cell, text=label, font=("Segoe UI", 8, "bold"),
                     bg=BG, fg=FG).pack()
            var = tk.StringVar(value=default)
            field_vars.append(var)
            tk.Entry(cell, textvariable=var, width=7,
                     bg=BG2, fg=YELLOW, insertbackground=FG,
                     relief="flat", font=("Consolas", 10),
                     justify="center").pack()
            tk.Label(cell, text=hint, font=("Segoe UI", 7),
                     bg=BG, fg=GRAY, justify="center").pack()

        # ── Cron preview + validation ──────────────────────────────────────────
        cron_frame = tk.Frame(dlg, bg=BG)
        cron_frame.pack(fill="x", padx=14, pady=(10, 0))

        tk.Label(cron_frame, text=self._t("cron_label"), font=("Segoe UI", 9),
                 bg=BG, fg=GRAY, width=9, anchor="w").pack(side="left")
        cron_preview_var = tk.StringVar()
        tk.Label(cron_frame, textvariable=cron_preview_var,
                 font=("Consolas", 10, "bold"), bg=BG, fg=YELLOW,
                 width=20, anchor="w").pack(side="left")
        valid_var = tk.StringVar()
        valid_lbl = tk.Label(cron_frame, textvariable=valid_var,
                             font=("Segoe UI", 9, "bold"), bg=BG, fg=GREEN)
        valid_lbl.pack(side="left", padx=(4, 0))

        # ── Next runs ──────────────────────────────────────────────────────────
        next_frame = tk.Frame(dlg, bg=BG)
        next_frame.pack(fill="x", padx=14, pady=(4, 0))
        tk.Label(next_frame, text=self._t("next_label"), font=("Segoe UI", 9),
                 bg=BG, fg=GRAY, width=9, anchor="w").pack(side="left")
        next_var = tk.StringVar()
        tk.Label(next_frame, textvariable=next_var,
                 font=("Segoe UI", 9), bg=BG, fg=FG, anchor="w").pack(side="left")

        # ── Live update logic ──────────────────────────────────────────────────
        save_btn_ref = []   # filled after button creation

        def _update(*_):
            cron = " ".join(v.get().strip() or "*" for v in field_vars)
            cron_preview_var.set(cron)
            if croniter.is_valid(cron):
                valid_var.set(self._t("valid_ok"))
                valid_lbl.config(fg=GREEN)
                it = croniter(cron, datetime.now().astimezone())
                runs = [datetime.fromtimestamp(it.get_next(float)).strftime("%m/%d %H:%M")
                        for _ in range(3)]
                next_var.set("  ›  ".join(runs))
                if save_btn_ref:
                    save_btn_ref[0].config(state="normal", bg=ACCENT, fg=BG3)
            else:
                valid_var.set(self._t("valid_bad"))
                valid_lbl.config(fg=RED)
                next_var.set("")
                if save_btn_ref:
                    save_btn_ref[0].config(state="disabled", bg=GRAY2, fg=GRAY)

        for v in field_vars:
            v.trace_add("write", _update)

        def _apply_preset(*_):
            label = preset_var.get()
            cron  = next((p[1] for p in preset_pairs if p[0] == label), "")
            if not cron:
                return
            pparts = cron.split()
            for i, fv in enumerate(field_vars):
                fv.set(pparts[i] if i < len(pparts) else "*")

        preset_var.trace_add("write", _apply_preset)

        # initial render
        _update()

        # ── Separator + buttons ────────────────────────────────────────────────
        tk.Frame(dlg, bg=GRAY2, height=1).pack(fill="x", padx=14, pady=(10, 0))

        btn_frame = tk.Frame(dlg, bg=BG)
        btn_frame.pack(fill="x", padx=14, pady=(6, 10))

        btn_cfg = dict(font=("Segoe UI", 9, "bold"), relief="flat",
                       padx=10, pady=4, cursor="hand2")

        def _save():
            cron = cron_preview_var.get().strip()
            if not croniter.is_valid(cron):
                return
            fc["cron"] = cron
            sched_lbl.config(fg=ACCENT)
            if self._cron_callback:
                self._cron_callback(name, cron)
            dlg.destroy()

        def _clear():
            fc.pop("cron", None)
            sched_lbl.config(fg=GRAY)
            if self._cron_callback:
                self._cron_callback(name, "")
            dlg.destroy()

        tk.Button(btn_frame, text=self._t("btn_clear_schedule"), bg=GRAY2, fg=GRAY,
                  command=_clear, **btn_cfg).pack(side="left")
        tk.Button(btn_frame, text=self._t("btn_cancel"), bg=GRAY2, fg=FG,
                  command=dlg.destroy, **btn_cfg).pack(side="left", padx=(6, 0))
        save_btn = tk.Button(btn_frame, text=self._t("btn_save"), bg=ACCENT, fg=BG3,
                             command=_save, **btn_cfg)
        save_btn.pack(side="right")
        save_btn_ref.append(save_btn)

        # re-run update now that save_btn exists
        _update()

        # Center over root
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        cx = self._root.winfo_x() + self._root.winfo_width()  // 2
        cy = self._root.winfo_y() + self._root.winfo_height() // 2
        dlg.geometry("+{}+{}".format(cx - w // 2, cy - h // 2))

    # ── Function settings dialog ──────────────────────────────────────────────

    def _fn_field_label(self, field: dict) -> str:
        """Translated label from schema (label_key → ui_locale), else English fallback."""
        lk = field.get("label_key")
        if lk:
            return self._t(lk)
        return str(field.get("label", ""))

    def _fn_field_description(self, field: dict) -> str:
        """Translated hint from schema (description_key → ui_locale), else fallback."""
        dk = field.get("description_key")
        if dk:
            return self._t(dk)
        return str(field.get("description", "") or "")

    def _show_fn_settings(self, name):
        if not self._bot_paused["paused"]:
            self._show_toast(self._t("toast_pause_for_settings"))
            return
        schema = _FN_COMMON_FIELDS + (_FN_SETTINGS_SCHEMA.get(name) or [])
        if not schema:
            return

        current = self._fn_settings.get(name, {})

        dlg = tk.Toplevel(self._root)
        dlg.title(self._t("dlg_settings_title", name=name))
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        tk.Label(dlg, text=self._t("dlg_settings_heading", name=name),
                 font=("Segoe UI", 11, "bold"), bg=BG, fg=ACCENT
                 ).pack(fill="x", padx=14, pady=(12, 6))
        tk.Frame(dlg, bg=GRAY2, height=1).pack(fill="x", padx=14, pady=(0, 10))

        field_vars  = {}  # key → tk var
        _extra_state = {}  # key → state dict for complex types (e.g. fragment_filters)

        for field in schema:
            key    = field["key"]
            label  = self._fn_field_label(field)
            ftype  = field.get("type", "str")
            desc   = self._fn_field_description(field)
            fmin   = field.get("min")
            fmax   = field.get("max")
            fdef   = field.get("default")
            stored = current.get(key, fdef)

            # ── fragment_filters: full-width dynamic list widget ───────────────
            if ftype == "fragment_filters":
                choices = list(field.get("choices") or [])
                # Load choices from the function YAML if choices_yaml_key is set.
                # If choices_yaml_step_type is also set, scan that event_type step for the key;
                # otherwise fall back to the top-level YAML key.
                _yaml_key       = field.get("choices_yaml_key")
                _yaml_step_type = field.get("choices_yaml_step_type")
                if _yaml_key:
                    try:
                        import yaml as _yaml
                        _fn_yaml = os.path.join("functions", "{}.yaml".format(name))
                        with open(_fn_yaml, encoding="utf-8") as _fy:
                            _fn_data = _yaml.safe_load(_fy)
                        _yaml_choices = []
                        if _yaml_step_type:
                            for _step in (_fn_data.get("steps") or []):
                                if _step.get("event_type") == _yaml_step_type:
                                    _yaml_choices = _step.get(_yaml_key) or []
                                    break
                        if not _yaml_choices:
                            _yaml_choices = _fn_data.get(_yaml_key) or []
                        if _yaml_choices:
                            choices = list(_yaml_choices)
                    except Exception as _ye:
                        _log.debug("[ui] fragment_filters: could not load choices from YAML: %s", _ye)
                _val_to_disp = {c: os.path.splitext(os.path.basename(c))[0] for c in choices}
                _disp_to_val = {v: k for k, v in _val_to_disp.items()}
                _disp_list   = [_val_to_disp.get(c, c) for c in choices]
                _sv          = stored if isinstance(stored, dict) else {}
                _mode_var    = tk.StringVar(value=_sv.get("mode", "AND"))
                _rows_data   = []  # list of [tpl_var, count_var]
                _extra_state[key] = {
                    "mode_var":   _mode_var,
                    "rows_data":  _rows_data,
                    "disp_to_val": _disp_to_val,
                }
                field_vars[key] = tk.StringVar(value="__ff__")

                _ff = tk.Frame(dlg, bg=BG)
                _ff.pack(fill="x", padx=14, pady=(0, 8))

                # Header: label on left, "mode:" label + AND/OR radios on right
                _hdr = tk.Frame(_ff, bg=BG)
                _hdr.pack(fill="x")
                tk.Label(_hdr, text=label, font=("Segoe UI", 9, "bold"),
                         bg=BG, fg=FG, anchor="w").pack(side="left")
                tk.Label(_hdr, text=self._t("fragment_mode"), font=("Segoe UI", 8),
                         bg=BG, fg=GRAY).pack(side="right", padx=(8, 2))
                for _m in ["OR", "AND"]:
                    tk.Radiobutton(_hdr, text=_m, variable=_mode_var, value=_m,
                                   bg=BG, activebackground=BG, selectcolor=BG2,
                                   fg=ACCENT, activeforeground=ACCENT,
                                   font=("Segoe UI", 9, "bold")).pack(side="right", padx=1)

                _rows_frame = tk.Frame(_ff, bg=BG)
                _rows_frame.pack(fill="x", pady=(2, 0))

                def _add_ff_row(tpl_val=None, cnt_val=1,
                                _rc=_rows_frame, _rd=_rows_data,
                                _dl=_disp_list,  _vtd=_val_to_disp):
                    _rf = tk.Frame(_rc, bg=BG2)
                    _rf.pack(fill="x", pady=1)
                    _init = _vtd.get(tpl_val, _dl[0] if _dl else "")
                    _tv   = tk.StringVar(value=_init)
                    _cv   = tk.IntVar(value=max(1, int(cnt_val or 1)))
                    _e    = [_tv, _cv]
                    _rd.append(_e)
                    ttk.Combobox(_rf, textvariable=_tv, values=_dl,
                                 state="readonly", width=22,
                                 font=("Consolas", 10)).pack(side="left", padx=(4, 2), pady=2)
                    tk.Spinbox(_rf, from_=1, to=99, textvariable=_cv, width=4,
                               bg=BG2, fg=YELLOW, buttonbackground=GRAY2,
                               insertbackground=FG, relief="flat",
                               font=("Consolas", 11, "bold"),
                               justify="center").pack(side="left", padx=2)
                    def _rm(f=_rf, e=_e, rd=_rd):
                        f.pack_forget()
                        f.destroy()
                        try:
                            rd.remove(e)
                        except ValueError:
                            pass
                    tk.Button(_rf, text="✕", command=_rm, bg=GRAY2, fg=RED,
                              relief="flat", font=("Segoe UI", 9),
                              cursor="hand2", padx=4).pack(side="left", padx=2)
                    dlg.update_idletasks()

                for _filt in _sv.get("filters", []):
                    _add_ff_row(_filt.get("template"), _filt.get("count", 1))

                tk.Button(_ff, text=self._t("add_filter"), command=_add_ff_row,
                          bg=GRAY2, fg=ACCENT, relief="flat",
                          font=("Segoe UI", 9, "bold"), cursor="hand2",
                          padx=8, pady=2).pack(anchor="w", pady=(4, 0))
                if desc:
                    tk.Label(_ff, text=desc, font=("Segoe UI", 8),
                             bg=BG, fg=GRAY).pack(anchor="w")
                continue
            # ── end fragment_filters ───────────────────────────────────────────

            row = tk.Frame(dlg, bg=BG)
            row.pack(fill="x", padx=14, pady=(0, 8))

            tk.Label(row, text=label, font=("Segoe UI", 9, "bold"),
                     bg=BG, fg=FG, width=22, anchor="w").pack(side="left")

            if ftype == "int":
                var = tk.IntVar(value=int(stored) if stored is not None else (fmin or 0))
                field_vars[key] = var
                spin = tk.Spinbox(
                    row, from_=fmin if fmin is not None else 0,
                    to=fmax if fmax is not None else 9999,
                    textvariable=var, width=8,
                    bg=BG2, fg=YELLOW, buttonbackground=GRAY2,
                    insertbackground=FG, relief="flat",
                    font=("Consolas", 11, "bold"), justify="center")
                spin.pack(side="left")
                if desc:
                    tk.Label(row, text=desc, font=("Segoe UI", 8),
                             bg=BG, fg=GRAY, padx=8).pack(side="left")

            elif ftype == "float":
                fstep = field.get("step", 0.1)
                init_val = float(stored) if stored is not None else float(fmin or 0)
                var = tk.StringVar(value="{:.3g}".format(init_val))
                field_vars[key] = var
                fmin_f = float(fmin) if fmin is not None else 0.0
                fmax_f = float(fmax) if fmax is not None else 9999.0
                spin = tk.Spinbox(
                    row, from_=fmin_f, to=fmax_f, increment=fstep,
                    textvariable=var, width=8,
                    bg=BG2, fg=YELLOW, buttonbackground=GRAY2,
                    insertbackground=FG, relief="flat",
                    font=("Consolas", 11, "bold"), justify="center")
                spin.pack(side="left")
                if desc:
                    tk.Label(row, text=desc, font=("Segoe UI", 8),
                             bg=BG, fg=GRAY, padx=8).pack(side="left")

            elif ftype == "bool":
                var = tk.BooleanVar(value=bool(stored))
                field_vars[key] = var
                tk.Checkbutton(row, variable=var, bg=BG, activebackground=BG,
                               selectcolor=GRAY2, fg=FG, activeforeground=FG,
                               relief="flat", bd=0, highlightthickness=0).pack(side="left")
                if desc:
                    tk.Label(row, text=desc, font=("Segoe UI", 8),
                             bg=BG, fg=GRAY, padx=4).pack(side="left")

            elif ftype == "password":
                var = tk.StringVar(value=str(stored) if stored else "")
                field_vars[key] = var
                pw_frame = tk.Frame(row, bg=BG)
                pw_frame.pack(side="left")
                pw_entry = tk.Entry(pw_frame, textvariable=var, width=18,
                                    show="*", bg=BG2, fg=YELLOW,
                                    insertbackground=FG, relief="flat",
                                    font=("Consolas", 10))
                pw_entry.pack(side="left")
                _show_pw = {"v": False}
                def _toggle_pw(e=pw_entry, s=_show_pw):
                    s["v"] = not s["v"]
                    e.config(show="" if s["v"] else "*")
                eye_btn = tk.Label(pw_frame, text="👁", font=("Segoe UI", 9),
                                   bg=BG, fg=GRAY, cursor="hand2", padx=4)
                eye_btn.pack(side="left")
                eye_btn.bind("<Button-1>", lambda e: _toggle_pw())
                if desc:
                    tk.Label(row, text=desc, font=("Segoe UI", 8),
                             bg=BG, fg=GRAY, padx=8).pack(side="left")

            else:  # str
                var = tk.StringVar(value=str(stored) if stored is not None else "")
                field_vars[key] = var
                tk.Entry(row, textvariable=var, width=22,
                         bg=BG2, fg=YELLOW, insertbackground=FG,
                         relief="flat", font=("Consolas", 10)).pack(side="left")
                if desc:
                    tk.Label(row, text=desc, font=("Segoe UI", 8),
                             bg=BG, fg=GRAY, padx=8).pack(side="left")

        # ── Buttons ────────────────────────────────────────────────────────────
        tk.Frame(dlg, bg=GRAY2, height=1).pack(fill="x", padx=14, pady=(4, 0))
        btn_frame = tk.Frame(dlg, bg=BG)
        btn_frame.pack(fill="x", padx=14, pady=(6, 12))
        btn_cfg = dict(font=("Segoe UI", 9, "bold"), relief="flat",
                       padx=10, pady=4, cursor="hand2")

        def _save():
            saved = self._fn_settings.setdefault(name, {})
            for field in schema:
                k     = field["key"]
                ftype = field.get("type", "str")
                if ftype == "fragment_filters":
                    state = _extra_state.get(k)
                    if state is None:
                        continue
                    _mode    = state["mode_var"].get()
                    _filters = []
                    for _tv, _cv in state["rows_data"]:
                        _disp = _tv.get()
                        _tpl  = state["disp_to_val"].get(_disp, _disp)
                        try:
                            _cnt = max(1, int(_cv.get()))
                        except (ValueError, TypeError):
                            _cnt = 1
                        if _tpl:
                            _filters.append({"template": _tpl, "count": _cnt})
                    v = {"mode": _mode, "filters": _filters}
                else:
                    raw = field_vars[k].get()
                    try:
                        if ftype == "int":
                            v = int(raw)
                        elif ftype == "float":
                            v = float(raw)
                        elif ftype == "bool":
                            v = bool(raw)
                        else:
                            v = str(raw)
                    except (ValueError, TypeError):
                        v = raw
                saved[k] = v
                # Also push to runner immediately so bot picks it up without restart
                if hasattr(self._runner, "fn_settings"):
                    self._runner.fn_settings.setdefault(name, {})[k] = v
            if self._settings_save_callback:
                self._settings_save_callback(self._fn_settings)
            dlg.destroy()

        def _reset():
            for field in schema:
                k     = field["key"]
                ftype = field.get("type", "str")
                v     = field.get("default")
                if ftype == "fragment_filters":
                    state = _extra_state.get(k)
                    if state and isinstance(v, dict):
                        state["mode_var"].set(v.get("mode", "AND"))
                elif k in field_vars and v is not None:
                    field_vars[k].set(v)

        tk.Button(btn_frame, text=self._t("reset_defaults"), bg=GRAY2, fg=GRAY,
                  command=_reset, **btn_cfg).pack(side="left")
        tk.Button(btn_frame, text=self._t("btn_cancel"), bg=GRAY2, fg=FG,
                  command=dlg.destroy, **btn_cfg).pack(side="left", padx=(6, 0))
        tk.Button(btn_frame, text=self._t("btn_save"), bg=ACCENT, fg=BG3,
                  command=_save, **btn_cfg).pack(side="right")

        # Center over root
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        cx = self._root.winfo_x() + self._root.winfo_width()  // 2
        cy = self._root.winfo_y() + self._root.winfo_height() // 2
        dlg.geometry("+{}+{}".format(cx - w // 2, cy - h // 2))

    # ── Manual run ────────────────────────────────────────────────────────────

    def _run_fn(self, name):
        if self._bot_paused["paused"]:
            self._show_toast(self._t("toast_enable_running_first"))
            return
        if not self._fn_enabled.get(name, True):
            self._show_toast(self._t("toast_fn_disabled", name=name))
            return
        if self._run_callback:
            self._run_callback(name)

    # ── Row tooltip ───────────────────────────────────────────────────────────

    def _row_tooltip_text(self, name):
        fc = next((f for f in self._fn_configs if f.get("name") == name), None)
        if not fc:
            return ""

        fn_disabled = not self._fn_enabled.get(name, True)
        bot_paused  = self._bot_paused["paused"]   # True = bot is NOT running

        lines = []

        trigger = fc.get("trigger", "")
        if trigger:
            i18n_k = self._TRIGGER_I18N.get(trigger)
            label  = self._t(i18n_k) if i18n_k else trigger
            if fn_disabled:
                lines.append(self._t("tt_trigger_disabled", label=label))
            elif bot_paused:
                lines.append(self._t("tt_trigger_paused", label=label))
            else:
                lines.append(self._t("tt_trigger", label=label))

        cron = fc.get("cron", "")
        if cron and croniter.is_valid(cron):
            it     = croniter(cron, datetime.now().astimezone())
            nxt    = datetime.fromtimestamp(it.get_next(float))
            diff_s = (nxt - datetime.now()).total_seconds()
            mins   = int(diff_s / 60)
            eta    = (self._t("tt_eta_hm", h=mins // 60, m=mins % 60) if mins >= 60
                      else self._t("tt_eta_m", m=mins))
            tstr   = nxt.strftime("%H:%M")
            if fn_disabled:
                lines.append(self._t("tt_next_disabled"))
            elif bot_paused:
                lines.append(self._t("tt_next_paused", time=tstr, eta=eta))
            else:
                lines.append(self._t("tt_next", time=tstr, eta=eta))
            lines.append(self._t("tt_cron", cron=cron))

        key = fc.get("key", "")
        if key:
            ku = key.upper()
            if fn_disabled:
                lines.append(self._t("tt_hotkey_disabled", hotkey=ku))
            else:
                lines.append(self._t("tt_hotkey", hotkey=ku))

        return "\n".join(lines)

    def _attach_tooltip(self, widget, name):
        tip = {"win": None}

        def _enter(_e):
            text = self._row_tooltip_text(name)
            if not text:
                return
            if tip["win"]:
                tip["win"].destroy()
            x = widget.winfo_rootx() + widget.winfo_width() + 6
            y = widget.winfo_rooty()
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.attributes("-topmost", True)
            tk.Label(win, text=text,
                     font=("Segoe UI", 9), bg=BG2, fg=FG,
                     padx=10, pady=6, justify="left",
                     relief="flat").pack()
            win.update_idletasks()
            # keep inside screen horizontally
            sw = win.winfo_screenwidth()
            wx = x if x + win.winfo_width() < sw else widget.winfo_rootx() - win.winfo_width() - 6
            win.geometry("+{}+{}".format(wx, y))
            tip["win"] = win

        def _leave(_e):
            if tip["win"]:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", _enter, add="+")
        widget.bind("<Leave>", _leave, add="+")

    def _refresh_row(self, name):
        if name not in self._row_frames:
            return
        _, name_lbl, toggle_lbl = self._row_frames[name]
        enabled = self._fn_enabled.get(name, True)
        name_lbl.config(fg=FG if enabled else GRAY)
        toggle_lbl.config(text="✓" if enabled else "✗",
                          fg=GREEN if enabled else GRAY)

    def _toggle_all_enabled(self, enable: bool) -> None:
        """Enable or disable all functions at once and persist the change."""
        for name in list(self._vars):
            self._vars[name].set(enable)
            self._fn_enabled[name] = enable
            self._refresh_row(name)
            if self._enabled_callback:
                self._enabled_callback(name, enable)
        if self._save_callback:
            self._save_callback()

    # ── Key rebinding ─────────────────────────────────────────────────────────

    def _show_toast(self, message, duration_ms=2500):
        """Small auto-closing popup centered over the main window."""
        toast = tk.Toplevel(self._root)
        toast.overrideredirect(True)
        toast.configure(bg=GRAY2)
        toast.attributes("-topmost", True)
        tk.Label(toast, text=message,
                 font=("Segoe UI", 10), bg=GRAY2, fg=YELLOW,
                 padx=16, pady=10).pack()
        self._root.update_idletasks()
        rx = self._root.winfo_x() + self._root.winfo_width()  // 2
        ry = self._root.winfo_y() + self._root.winfo_height() // 2
        toast.update_idletasks()
        toast.geometry("+{}+{}".format(rx - toast.winfo_width()  // 2,
                                       ry - toast.winfo_height() // 2))
        toast.after(duration_ms, toast.destroy)

    def _start_rebind(self, name, badge_lbl):
        if not self._bot_paused["paused"]:
            self._show_toast(self._t("toast_pause_rebind"))
            return
        if self._rebinding:
            self._cancel_rebind()
            return
        self._rebinding  = name
        self._rebind_lbl = badge_lbl
        badge_lbl.config(text=" ▶ ... ", bg=RED, fg=BG3)
        self._status_lbl.config(
            text=self._t("status_rebind_press", name=name),
            fg=YELLOW)
        self._root.focus_force()
        self._root.bind("<Key>",    self._capture_key)
        self._root.bind("<Escape>", lambda e: self._cancel_rebind())

    def _capture_key(self, event):
        if not self._rebinding:
            return
        char = event.char.lower() if event.char else ""
        if not char or not char.isalnum() or len(char) != 1:
            return

        name = self._rebinding

        # Check duplicate — another function already owns this key
        conflict = self._key_bindings.get(char)
        if conflict and conflict != name:
            # Pause key capture while asking
            self._root.unbind("<Key>")
            self._root.unbind("<Escape>")
            self._confirm_overwrite_key(name, char, conflict)
            return

        self._apply_rebind(name, char)

    def _confirm_overwrite_key(self, name, char, conflict):
        """Modal dialog asking whether to overwrite a conflicting key binding."""
        dlg = tk.Toplevel(self._root)
        dlg.title(self._t("dlg_key_conflict"))
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        tk.Label(dlg,
                 text=self._t("dlg_key_conflict_msg", char=char.upper(), conflict=conflict),
                 font=("Segoe UI", 10), bg=BG, fg=FG,
                 padx=20, pady=14, justify="center").pack()

        btn_frame = tk.Frame(dlg, bg=BG)
        btn_frame.pack(pady=(0, 12))

        btn_cfg = dict(font=("Segoe UI", 9, "bold"), relief="flat",
                       padx=14, pady=4, cursor="hand2")

        def _yes():
            dlg.destroy()
            self._apply_rebind(name, char)

        def _no():
            dlg.destroy()
            # Resume key capture so the user can pick a different key
            self._status_lbl.config(
                text=self._t("status_rebind_press", name=name),
                fg=YELLOW)
            self._root.bind("<Key>",    self._capture_key)
            self._root.bind("<Escape>", lambda e: self._cancel_rebind())

        tk.Button(btn_frame, text=self._t("btn_overwrite"), bg=RED, fg=BG3,
                  command=_yes, **btn_cfg).pack(side="left", padx=(0, 8))
        tk.Button(btn_frame, text=self._t("btn_cancel"), bg=GRAY2, fg=FG,
                  command=_no, **btn_cfg).pack(side="left")

        # Center over root
        dlg.update_idletasks()
        w, h = dlg.winfo_width(), dlg.winfo_height()
        cx = self._root.winfo_x() + self._root.winfo_width()  // 2
        cy = self._root.winfo_y() + self._root.winfo_height() // 2
        dlg.geometry("+{}+{}".format(cx - w // 2, cy - h // 2))

    def _apply_rebind(self, name, char):
        """Commit the key binding change and persist."""
        # Remove old key for this function
        old_key = ""
        for fc in self._fn_configs:
            if fc.get("name") == name:
                old_key = fc.get("key", "")
                fc["key"] = char
                break
        if old_key and self._key_bindings.get(old_key) == name:
            del self._key_bindings[old_key]

        # Evict the conflicting function (if any) — user already confirmed
        if char in self._key_bindings and self._key_bindings[char] != name:
            evicted = self._key_bindings[char]
            for fc in self._fn_configs:
                if fc.get("name") == evicted:
                    fc["key"] = ""
                    break
            if evicted in self._badge_lbls:
                self._badge_lbls[evicted].config(text=" + ", fg=ACCENT)

        # Register new binding
        if self._fn_enabled.get(name, True):
            self._key_bindings[char] = name

        if self._rebind_lbl:
            self._rebind_lbl.config(text=" {} ".format(char.upper()),
                                    bg=GRAY2, fg=ACCENT)
        if self._save_callback:
            self._save_callback()
        self._finish_rebind()

    def _cancel_rebind(self):
        if not self._rebinding:
            return
        name = self._rebinding
        key  = next((fc.get("key", "") for fc in self._fn_configs
                     if fc.get("name") == name), "")
        if self._rebind_lbl:
            self._rebind_lbl.config(
                text=" {} ".format(key.upper()) if key else " + ",
                bg=GRAY2, fg=ACCENT if key else GRAY)
        self._finish_rebind()

    def _finish_rebind(self):
        self._rebinding  = None
        self._rebind_lbl = None
        self._root.unbind("<Key>")
        self._root.unbind("<Escape>")
        self._status_lbl.config(text=self._t("status_paused_all"), fg=YELLOW)

    # ── Status tick ───────────────────────────────────────────────────────────

    def _tick(self):
        if not self._root:
            return
        if self._quit_check and self._quit_check():
            self._root.quit()
            return
        if self._rebinding:
            self._sync_stop_function_button()
            self._root.after(500, self._tick)
            return

        # Keep Start LastZ button visibility in sync with current emulator setting
        self._update_start_lastz_button_visibility()

        # ── Window connection check ───────────────────────────────────────────
        if self._connection_status is not None:
            connected = self._connection_status()
            if not connected:
                self._running_cb.config(state="disabled", cursor="arrow")
                emulator = self._general_settings.get("emulator", "pc")
                window_name = "LDPlayer" if emulator == "ldplayer" else "LastZ"
                self._status_lbl.config(
                    text=self._t("status_waiting_window").format(window=window_name),
                    fg=YELLOW,
                )
                self._was_disconnected = True
                self._sync_stop_function_button()
                self._root.after(500, self._tick)
                return
            elif self._was_disconnected:
                # Window just became available: re-enable toggle if OCR already done
                self._was_disconnected = False
                if self._ocr_ready:
                    self._running_cb.config(state="normal", cursor="hand2", fg=GREEN, activeforeground=GREEN)
                    self._running_var.set(True)
                    self._on_running_toggle()
                    _log.info("[UI] Window connected. Bot resumed.")

        # ── OpenOCR dynamic state check ───────────────────────────────────────
        if not self._ocr_ready:
            if ocr_openocr.OPENOCR_OK:
                # Success: first time ready
                self._ocr_ready = True
                self._running_cb.config(state="normal", cursor="hand2", fg=GREEN, activeforeground=GREEN)
                self._running_var.set(True)
                self._on_running_toggle()  # resumes bot
                _log.info("[UI] OpenOCR loaded. Bot resumed.")
            elif ocr_openocr._loading or not ocr_openocr._tried:
                # Still loading or preload thread hasn't started yet
                self._status_lbl.config(text=self._t("status_init_ocr"), fg=ACCENT)
                self._sync_stop_function_button()
                self._root.after(500, self._tick)
                return
            else:
                # _tried=True, OPENOCR_OK=False → load failed, but app can run without OCR
                self._ocr_ready = True
                self._running_cb.config(state="normal", cursor="hand2", fg=GREEN, activeforeground=GREEN)
                self._running_var.set(True)
                self._on_running_toggle()
                self._status_lbl.config(text=self._t("status_ocr_failed"), fg=ACCENT)
                _log.warning("[UI] OpenOCR not available. Bot running without OCR support.")

        if self._bot_paused["paused"]:
            self._sync_stop_function_button()
            self._root.after(500, self._tick)
            return
        try:
            state = getattr(self._runner, "state", "idle")
            fn    = getattr(self._runner, "function_name", None)
            if state == "running" and fn:
                self._status_lbl.config(text=self._t("status_running", fn=fn), fg=GREEN)
            else:
                now = datetime.now().timestamp()
                upcoming = [(ts, n) for n, ts in self._next_run_at.items()
                            if self._fn_enabled.get(n, True)]
                if upcoming:
                    ts, n = min(upcoming)
                    dt = datetime.fromtimestamp(ts).strftime("%H:%M")
                    self._status_lbl.config(
                        text=self._t("status_idle_next", name=n, time=dt), fg=FG)
                else:
                    self._status_lbl.config(text=self._t("status_idle"), fg=FG)
        except Exception:
            pass
        self._sync_stop_function_button()
        self._root.after(500, self._tick)
