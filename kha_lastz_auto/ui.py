import logging
import os
import threading
import tkinter as tk
from datetime import datetime

from croniter import croniter
from fn_settings_schema import SCHEMA as _FN_SETTINGS_SCHEMA

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


class BotUI:
    """
    Tkinter UI running in a daemon thread.
    - Checkbox to enable/disable each function (persisted to .env_config)
    - [KEY] badge is clickable to rebind hotkey (only when Is Running = off)
    - Status bar shows current running function / next cron
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
                 enabled_callback=None):
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

        self._vars       = {}   # fn_name → BooleanVar
        self._row_frames = {}   # fn_name → (row_frame, name_label)
        self._badge_lbls = {}   # fn_name → badge tk.Label
        self._sched_lbls = {}   # fn_name → schedule "S" tk.Label
        self._gear_lbls  = {}   # fn_name → gear tk.Label (or None if no settings)
        self._play_lbls  = {}   # fn_name → play ▶ tk.Label
        self._rebinding  = None
        self._rebind_lbl = None
        self._root       = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    # ── Build ─────────────────────────────────────────────────────────────────

    def _run(self):
        self._root = tk.Tk()
        self._root.title("KhaLastZ Bot")
        self._root.configure(bg=BG)
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self._root.after(500, self._tick)
        self._root.mainloop()

    def _build(self):
        r = self._root

        # Header
        hf = tk.Frame(r, bg=BG3, padx=16, pady=12)
        hf.pack(fill="x")
        tk.Label(hf, text="KhaLastZ Bot", font=("Segoe UI", 14, "bold"),
                 bg=BG3, fg=ACCENT).pack(side="left")

        # Is Running toggle
        self._running_var = tk.BooleanVar(value=True)
        self._running_cb  = tk.Checkbutton(
            hf, text="Is Running",
            variable=self._running_var,
            command=self._on_running_toggle,
            font=("Segoe UI", 10, "bold"),
            bg=BG3, activebackground=BG3,
            fg=GREEN, activeforeground=GREEN,
            selectcolor=GRAY2, cursor="hand2",
            relief="flat", bd=0, highlightthickness=0)
        self._running_cb.pack(side="right")

        # Status bar
        sf = tk.Frame(r, bg=BG2, padx=16, pady=8)
        sf.pack(fill="x", pady=(1, 0))
        self._status_lbl = tk.Label(sf, text="◼  Idle",
                                    font=("Segoe UI", 10), bg=BG2, fg=FG, anchor="w")
        self._status_lbl.pack(fill="x")

        # Section header
        lf = tk.Frame(r, bg=BG, padx=16)
        lf.pack(fill="x", pady=(10, 4))
        tk.Label(lf, text="FUNCTIONS", font=("Segoe UI", 8, "bold"),
                 bg=BG, fg=GRAY).pack(side="left")
        tk.Label(lf, text="click [key] to rebind  (pause first)",
                 font=("Segoe UI", 8), bg=BG, fg=GRAY).pack(side="right")

        # Function rows
        ff = tk.Frame(r, bg=BG, padx=10)
        ff.pack(fill="both", expand=True, pady=(0, 4))
        for fc in self._fn_configs:
            self._build_row(ff, fc)

        tk.Frame(r, bg=BG, height=8).pack(fill="x")

    def _on_running_toggle(self):
        paused = not self._running_var.get()
        self._bot_paused["paused"] = paused
        if paused:
            cancelled = None
            if getattr(self._runner, "state", "idle") == "running":
                cancelled = getattr(self._runner, "function_name", None)
                self._runner.stop()
            _log.info("[UI] Is Running → OFF (paused)")
            if cancelled:
                _log.info("[UI] Cancelled running function: {}".format(cancelled))
            self._running_cb.config(fg=RED, activeforeground=RED)
            self._status_lbl.config(text="⏸  Paused — all functions suspended", fg=YELLOW)
        else:
            _log.info("[UI] Is Running → ON (resumed)")
            self._running_cb.config(fg=GREEN, activeforeground=GREEN)
        self._update_badge_states()

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
        cron    = fc.get("cron", "")
        trigger = fc.get("trigger", "")

        if trigger:
            meta = "trigger: {}".format(trigger)
        elif cron:
            meta = "cron: {}".format(cron)
        elif key:
            meta = "hotkey"
        else:
            meta = "manual"

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
        tk.Label(inner, text=meta, font=("Segoe UI", 8),
                 bg=BG2, fg=GRAY, anchor="w").pack(fill="x")

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
        has_settings = bool(_FN_SETTINGS_SCHEMA.get(name))
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
        play_lbl = tk.Label(row, text=" ▶ ",
                            font=("Segoe UI", 10),
                            bg=GRAY2, fg=GREEN,
                            padx=4, pady=2, relief="flat", cursor="hand2")
        play_lbl.pack(side="right", padx=(0, 4))
        play_lbl.bind("<Button-1>", lambda e, n=name: self._run_fn(n))

        self._row_frames[name] = (row, name_lbl)
        self._badge_lbls[name] = badge_lbl
        self._sched_lbls[name] = sched_lbl
        self._gear_lbls[name]  = gear_lbl
        self._play_lbls[name]  = play_lbl

        # Tooltip on hover — show next run time / trigger / hotkey
        for w in (row, inner, name_lbl):
            self._attach_tooltip(w, name)

    # ── Schedule dialog ───────────────────────────────────────────────────────

    # Preset label → cron string
    _PRESETS = [
        ("— select preset —",  ""),
        ("Every minute",        "* * * * *"),
        ("Every 5 minutes",     "*/5 * * * *"),
        ("Every 10 minutes",    "*/10 * * * *"),
        ("Every 15 minutes",    "*/15 * * * *"),
        ("Every 30 minutes",    "*/30 * * * *"),
        ("Every hour",          "0 * * * *"),
        ("Every 2 hours",       "0 */2 * * *"),
        ("Every 4 hours",       "0 */4 * * *"),
        ("Every 6 hours",       "0 */6 * * *"),
        ("Every 12 hours",      "0 */12 * * *"),
        ("Daily at midnight",   "0 0 * * *"),
        ("Daily at 8 AM",       "0 8 * * *"),
        ("Daily at noon",       "0 12 * * *"),
        ("Weekly (Mon 8 AM)",   "0 8 * * 1"),
    ]

    def _show_schedule(self, name, sched_lbl):
        if not self._bot_paused["paused"]:
            self._show_toast("Bot is running — pause it first (uncheck Is Running) to edit schedules.")
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
        dlg.title("Schedule — {}".format(name))
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        # ── Title ──────────────────────────────────────────────────────────────
        tk.Label(dlg, text="Schedule  —  {}".format(name),
                 font=("Segoe UI", 11, "bold"), bg=BG, fg=ACCENT
                 ).pack(fill="x", padx=14, pady=(10, 6))

        sep = tk.Frame(dlg, bg=GRAY2, height=1)
        sep.pack(fill="x", padx=14, pady=(0, 8))

        # ── Preset dropdown ────────────────────────────────────────────────────
        preset_frame = tk.Frame(dlg, bg=BG)
        preset_frame.pack(fill="x", padx=14, pady=(0, 6))
        tk.Label(preset_frame, text="Preset:", font=("Segoe UI", 9),
                 bg=BG, fg=GRAY, width=9, anchor="w").pack(side="left")

        preset_labels = [p[0] for p in self._PRESETS]
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
            ("Min",     "0-59\n*/N\n0,30", parts[0]),
            ("Hour",    "0-23\n*/N",       parts[1]),
            ("Day",     "1-31\n*/N",       parts[2]),
            ("Month",   "1-12\nJAN-DEC",   parts[3]),
            ("Weekday", "0-6\nSUN-SAT",    parts[4]),
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

        tk.Label(cron_frame, text="Cron:", font=("Segoe UI", 9),
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
        tk.Label(next_frame, text="Next:", font=("Segoe UI", 9),
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
                valid_var.set("✓ valid")
                valid_lbl.config(fg=GREEN)
                it = croniter(cron, datetime.now().astimezone())
                runs = [datetime.fromtimestamp(it.get_next(float)).strftime("%m/%d %H:%M")
                        for _ in range(3)]
                next_var.set("  ›  ".join(runs))
                if save_btn_ref:
                    save_btn_ref[0].config(state="normal", bg=ACCENT, fg=BG3)
            else:
                valid_var.set("✗ invalid")
                valid_lbl.config(fg=RED)
                next_var.set("")
                if save_btn_ref:
                    save_btn_ref[0].config(state="disabled", bg=GRAY2, fg=GRAY)

        for v in field_vars:
            v.trace_add("write", _update)

        def _apply_preset(*_):
            label = preset_var.get()
            cron  = next((p[1] for p in self._PRESETS if p[0] == label), "")
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

        tk.Button(btn_frame, text="Clear schedule", bg=GRAY2, fg=GRAY,
                  command=_clear, **btn_cfg).pack(side="left")
        tk.Button(btn_frame, text="Cancel", bg=GRAY2, fg=FG,
                  command=dlg.destroy, **btn_cfg).pack(side="left", padx=(6, 0))
        save_btn = tk.Button(btn_frame, text="Save", bg=ACCENT, fg=BG3,
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

    def _show_fn_settings(self, name):
        if not self._bot_paused["paused"]:
            self._show_toast("Bot is running — pause it first (uncheck Is Running) to edit settings.")
            return
        schema = _FN_SETTINGS_SCHEMA.get(name)
        if not schema:
            return

        current = self._fn_settings.get(name, {})

        dlg = tk.Toplevel(self._root)
        dlg.title("Settings — {}".format(name))
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        tk.Label(dlg, text="Settings  —  {}".format(name),
                 font=("Segoe UI", 11, "bold"), bg=BG, fg=ACCENT
                 ).pack(fill="x", padx=14, pady=(12, 6))
        tk.Frame(dlg, bg=GRAY2, height=1).pack(fill="x", padx=14, pady=(0, 10))

        field_vars = {}  # key → tk var

        for field in schema:
            key   = field["key"]
            label = field["label"]
            ftype = field.get("type", "str")
            desc  = field.get("description", "")
            fmin  = field.get("min")
            fmax  = field.get("max")
            fdef  = field.get("default")
            stored = current.get(key, fdef)

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
                    font=("Consolas", 11, "bold"), justify="center", format="%.3g")
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
                k = field["key"]
                ftype = field.get("type", "str")
                raw = field_vars[k].get()
                # Coerce to correct Python type
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
                k = field["key"]
                v = field.get("default")
                if k in field_vars and v is not None:
                    field_vars[k].set(v)

        tk.Button(btn_frame, text="Reset defaults", bg=GRAY2, fg=GRAY,
                  command=_reset, **btn_cfg).pack(side="left")
        tk.Button(btn_frame, text="Cancel", bg=GRAY2, fg=FG,
                  command=dlg.destroy, **btn_cfg).pack(side="left", padx=(6, 0))
        tk.Button(btn_frame, text="Save", bg=ACCENT, fg=BG3,
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
            self._show_toast("Bot is paused — enable Is Running first.")
            return
        if not self._fn_enabled.get(name, True):
            self._show_toast("{} is disabled — enable it first.".format(name))
            return
        if self._run_callback:
            self._run_callback(name)

    # ── Row tooltip ───────────────────────────────────────────────────────────

    _TRIGGER_LABELS = {
        "logged_out":        "Logout detected",
        "attacked":          "Under attack detected",
        "alliance_attacked": "Alliance attack detected",
        "treasure_detected": "Treasure detected",
    }

    def _row_tooltip_text(self, name):
        fc = next((f for f in self._fn_configs if f.get("name") == name), None)
        if not fc:
            return ""

        fn_disabled = not self._fn_enabled.get(name, True)
        bot_paused  = self._bot_paused["paused"]   # True = bot is NOT running

        lines = []

        trigger = fc.get("trigger", "")
        if trigger:
            label = self._TRIGGER_LABELS.get(trigger, trigger)
            if fn_disabled:
                lines.append("Trigger: {}  (disabled)".format(label))
            elif bot_paused:
                lines.append("Trigger: {}  (bot paused)".format(label))
            else:
                lines.append("Trigger: {}".format(label))

        cron = fc.get("cron", "")
        if cron and croniter.is_valid(cron):
            it     = croniter(cron, datetime.now().astimezone())
            nxt    = datetime.fromtimestamp(it.get_next(float))
            diff_s = (nxt - datetime.now()).total_seconds()
            mins   = int(diff_s / 60)
            eta    = "in {}h {}m".format(mins // 60, mins % 60) if mins >= 60 \
                     else "in {}m".format(mins)
            if fn_disabled:
                lines.append("Next run: —  (function disabled)")
            elif bot_paused:
                lines.append("Next run: {}  ({}, bot paused)".format(
                    nxt.strftime("%H:%M"), eta))
            else:
                lines.append("Next run: {}  ({})".format(nxt.strftime("%H:%M"), eta))
            lines.append("Cron: {}".format(cron))

        key = fc.get("key", "")
        if key:
            if fn_disabled:
                lines.append("Hotkey: {}  (disabled)".format(key.upper()))
            else:
                lines.append("Hotkey: {}".format(key.upper()))

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
        _, name_lbl = self._row_frames[name]
        enabled = self._fn_enabled.get(name, True)
        name_lbl.config(fg=FG if enabled else GRAY)

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
            self._show_toast("⚠  Pause 'Is Running' before rebinding a key")
            return
        if self._rebinding:
            self._cancel_rebind()
            return
        self._rebinding  = name
        self._rebind_lbl = badge_lbl
        badge_lbl.config(text=" ▶ ... ", bg=RED, fg=BG3)
        self._status_lbl.config(
            text="⌨  Press new key for [{}]  —  Esc to cancel".format(name),
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
        dlg.title("Key conflict")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.attributes("-topmost", True)
        dlg.grab_set()

        tk.Label(dlg,
                 text="Key  [ {} ]  is already used by\n\"{}\"\n\nOverwrite?".format(
                     char.upper(), conflict),
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
                text="⌨  Press new key for [{}]  —  Esc to cancel".format(name),
                fg=YELLOW)
            self._root.bind("<Key>",    self._capture_key)
            self._root.bind("<Escape>", lambda e: self._cancel_rebind())

        tk.Button(btn_frame, text="Overwrite", bg=RED, fg=BG3,
                  command=_yes, **btn_cfg).pack(side="left", padx=(0, 8))
        tk.Button(btn_frame, text="Cancel", bg=GRAY2, fg=FG,
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
        self._status_lbl.config(text="⏸  Paused — all functions suspended", fg=YELLOW)

    # ── Status tick ───────────────────────────────────────────────────────────

    def _tick(self):
        if not self._root:
            return
        if self._rebinding or self._bot_paused["paused"]:
            self._root.after(500, self._tick)
            return
        try:
            state = getattr(self._runner, "state", "idle")
            fn    = getattr(self._runner, "function_name", None)
            if state == "running" and fn:
                self._status_lbl.config(text="▶  Running: {}".format(fn), fg=GREEN)
            else:
                now = datetime.now().timestamp()
                upcoming = [(ts, n) for n, ts in self._next_run_at.items()
                            if self._fn_enabled.get(n, True)]
                if upcoming:
                    ts, n = min(upcoming)
                    dt = datetime.fromtimestamp(ts).strftime("%H:%M")
                    self._status_lbl.config(
                        text="◼  Idle  —  next: {} @ {}".format(n, dt), fg=FG)
                else:
                    self._status_lbl.config(text="◼  Idle", fg=FG)
        except Exception:
            pass
        self._root.after(500, self._tick)
