"""
config_manager.py
-----------------
Manages .env_config (YAML) — user overrides for config.yaml.

Load priority:  config.yaml  →  .env_config (override)
Saves:          key bindings, enabled states, cron overrides.

Format:
    key_bindings:
      PinLoggin: l
      FightBoomer: ""       # empty = no hotkey
    cron_overrides:
      HelpAlliance: "*/5 * * * *"
    fn_enabled:
      PinLoggin: true
      FightBoomer: false
"""

import logging
import os
import re
import sys

import yaml

_log = logging.getLogger("kha_lastz")

ENV_CONFIG_PATH = ".env_config"

# More than one top-level `general_settings:` is a config error: PyYAML keeps only the last
# mapping, so earlier keys are dropped silently.
_TOP_LEVEL_GENERAL_SETTINGS = re.compile(r"^general_settings:\s*$", re.MULTILINE)


def _env_config_abort(message: str) -> None:
    """Log a fatal .env_config error and exit the process."""
    _log.critical("[config_manager] .env_config — %s Fix the file and restart.", message)
    sys.exit(1)


def _check_duplicate_top_level_general_settings(text: str, abs_path: str) -> None:
    matches = list(_TOP_LEVEL_GENERAL_SETTINGS.finditer(text))
    if len(matches) <= 1:
        return
    lines = [text[: m.start()].count("\n") + 1 for m in matches]
    _env_config_abort(
        "invalid structure: multiple top-level 'general_settings:' keys at lines {} "
        "(PyYAML only keeps the last block; merge into one section). Path: {}".format(
            ", ".join(str(n) for n in lines),
            abs_path,
        )
    )


def load_env_config() -> dict:
    """Load .env_config as a dict. Exits the process on YAML syntax errors or invalid structure."""
    if not os.path.isfile(ENV_CONFIG_PATH):
        return {}
    abs_path = os.path.abspath(ENV_CONFIG_PATH)
    try:
        with open(ENV_CONFIG_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        _env_config_abort("cannot read file: {} ({})".format(abs_path, exc))

    _check_duplicate_top_level_general_settings(text, abs_path)

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        _log.critical(
            "[config_manager] .env_config YAML syntax error in %s —\n%s\nFix the file and restart.",
            abs_path,
            exc,
        )
        sys.exit(1)

    if data is None:
        return {}
    if not isinstance(data, dict):
        _env_config_abort(
            "expected a YAML mapping (key: value) at the root, got {!r}. Path: {}".format(
                type(data).__name__,
                abs_path,
            )
        )
    return data


def load_general_settings() -> dict:
    """Return the general_settings mapping from .env_config, or {} if missing/unreadable."""
    gs = load_env_config().get("general_settings")
    return dict(gs) if isinstance(gs, dict) else {}


def apply_overrides(fn_configs: list) -> None:
    """Load .env_config and apply key_bindings, cron_overrides, fn_enabled to fn_configs in-place.
    Must be called BEFORE the loop that builds key_bindings / fn_enabled in main.py.
    """
    abs_path = os.path.abspath(ENV_CONFIG_PATH)
    if not os.path.isfile(ENV_CONFIG_PATH):
        _log.warning("[config_manager] .env_config not found at %s — using defaults", abs_path)
        return {}
    env = load_env_config()

    kb_ov   = env.get("key_bindings",   {})
    cr_ov   = env.get("cron_overrides", {})
    en_ov   = env.get("fn_enabled",     {})
    gs_ov   = env.get("general_settings") or {}
    _log.info(
        "[config_manager] apply_overrides: loaded general_settings from %s → %s",
        abs_path, gs_ov,
    )

    for fc in fn_configs:
        name = fc.get("name")
        if not name:
            continue
        if name in kb_ov:
            fc["key"] = kb_ov[name] or ""
        if name in cr_ov:
            if cr_ov[name]:
                fc["cron"] = cr_ov[name]
            else:
                fc.pop("cron", None)
        if name in en_ov:
            fc["enabled"] = bool(en_ov[name])
    
    return gs_ov


def save(fn_configs: list, fn_enabled: dict, general_settings: dict = None) -> None:
    """Persist key bindings, cron overrides, and enabled states to .env_config.
    Preserves all other sections (fn_settings, etc.) already in the file."""
    # Read existing file to preserve sections we don't manage here (e.g. fn_settings)
    if os.path.isfile(ENV_CONFIG_PATH):
        data = load_env_config()
    else:
        data = {}

    kb = {}
    cr = {}
    for fc in fn_configs:
        name = fc.get("name")
        if not name:
            continue
        kb[name] = fc.get("key") or ""
        cron = fc.get("cron", "")
        if cron:
            cr[name] = cron

    data["key_bindings"]   = kb
    data["cron_overrides"] = cr
    data["fn_enabled"]     = {k: bool(v) for k, v in fn_enabled.items()}
    if general_settings is not None:
        # Merge rather than replace: preserve any keys already in the file that
        # are not managed by the app (e.g. fast_user_mouse_* tuning values added
        # manually).  Managed keys in general_settings always win.
        existing_gs = data.get("general_settings") or {}
        merged_gs = dict(existing_gs)
        merged_gs.update(general_settings)
        data["general_settings"] = merged_gs
        _log.info(
            "[config_manager] save: writing general_settings → %s",
            {k: v for k, v in merged_gs.items() if k in (
                "window_width", "window_height", "language", "auto_start_lastz",
                "capture_interval_sec", "show_preview", "auto_focus",
            )},
        )

    with open(ENV_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_fn_settings() -> dict:
    """Return fn_settings dict {fn_name: {key: value}} from .env_config."""
    if not os.path.isfile(ENV_CONFIG_PATH):
        return {}
    data = load_env_config()
    return data.get("fn_settings") or {}


def save_fn_settings(fn_settings: dict) -> None:
    """Persist fn_settings into the fn_settings section of .env_config."""
    if os.path.isfile(ENV_CONFIG_PATH):
        data = load_env_config()
    else:
        data = {}
    data["fn_settings"] = fn_settings
    with open(ENV_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def init_if_missing(fn_configs: list, fn_enabled: dict) -> None:
    """Create .env_config if missing, or add any absent cron_overrides from fn_configs."""
    if not os.path.isfile(ENV_CONFIG_PATH):
        save(fn_configs, fn_enabled)
        return

    # File exists — backfill cron_overrides for functions that have cron in
    # fn_configs but are not yet recorded in the file (e.g. file was created
    # before cron support was added).
    data = load_env_config()

    cr = data.get("cron_overrides") or {}
    changed = False
    for fc in fn_configs:
        name = fc.get("name")
        if name and fc.get("cron") and name not in cr:
            cr[name] = fc["cron"]
            changed = True

    if changed:
        data["cron_overrides"] = cr
        with open(ENV_CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
