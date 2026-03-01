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

import os
import yaml

ENV_CONFIG_PATH = ".env_config"


def apply_overrides(fn_configs: list) -> None:
    """Load .env_config and apply key_bindings, cron_overrides, fn_enabled to fn_configs in-place.
    Must be called BEFORE the loop that builds key_bindings / fn_enabled in main.py.
    """
    if not os.path.isfile(ENV_CONFIG_PATH):
        return
    with open(ENV_CONFIG_PATH, "r", encoding="utf-8") as f:
        env = yaml.safe_load(f) or {}

    kb_ov   = env.get("key_bindings",   {})
    cr_ov   = env.get("cron_overrides", {})
    en_ov   = env.get("fn_enabled",     {})

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


def save(fn_configs: list, fn_enabled: dict) -> None:
    """Persist key bindings, cron overrides, and enabled states to .env_config."""
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

    data = {
        "key_bindings":   kb,
        "cron_overrides": cr,
        "fn_enabled":     {k: bool(v) for k, v in fn_enabled.items()},
    }
    with open(ENV_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_fn_settings() -> dict:
    """Return fn_settings dict {fn_name: {key: value}} from .env_config."""
    if not os.path.isfile(ENV_CONFIG_PATH):
        return {}
    with open(ENV_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("fn_settings") or {}


def save_fn_settings(fn_settings: dict) -> None:
    """Persist fn_settings into the fn_settings section of .env_config."""
    if os.path.isfile(ENV_CONFIG_PATH):
        with open(ENV_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
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
    with open(ENV_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

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
