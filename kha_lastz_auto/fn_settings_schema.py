"""
fn_settings_schema.py
---------------------
Per-function UI settings schema.
Each entry is a list of field definitions that the Settings dialog renders.

Field definition keys:
    key         str   – storage key in fn_settings[fn_name]
    label       str   – fallback display label (English) if no label_key
    label_key   str   – optional ui_locale message key for translated label
    description str   – fallback hint text if no description_key
    description_key str – optional ui_locale message key for translated description
    type        str   – "int" | "float" | "str" | "bool" | "password" | "fragment_filters"
    min/max     num   – for int/float: bounds
    default     any   – default value if not stored

COMMON_FIELDS: fields that appear in the settings dialog for ALL functions.
SCHEMA: per-function additional fields (merged after COMMON_FIELDS in the dialog).

To add settings for a new function, add an entry to SCHEMA.
bot_engine.py reads overrides from fn_settings at runtime.
"""

COMMON_FIELDS: list = [
    {
        "key":             "run_count",
        "label":           "Repeat count",
        "label_key":       "fn_run_count_label",
        "description":     "Number of runs for this function (1 = once)",
        "description_key": "fn_run_count_desc",
        "type":            "int",
        "min":             1,
        "max":             20,
        "default":         1,
    },
]

SCHEMA: dict = {
    "PinLoggin": [
        {
            "key":             "password",
            "label":           "PIN password",
            "label_key":       "fn_password_label",
            "description":     "Login PIN (stored in .env_config, not .env)",
            "description_key": "fn_password_desc",
            "type":            "password",
            "default":         "",
        },
    ],

    "TruckPlunder": [
        {
            "key":             "servers",
            "label":           "Server filter",
            "label_key":       "fn_truck_servers_label",
            "description":     "Comma-separated server numbers to plunder (e.g. 600, 601)",
            "description_key": "fn_truck_servers_desc",
            "type":            "str",
            "default":         "600, 601",
        },
        {
            "key":             "max_power",
            "label":           "Max power",
            "label_key":       "fn_truck_max_power_label",
            "description":     "Skip trucks with power ≥ this value",
            "description_key": "fn_truck_max_power_desc",
            "type":            "int",
            "min":             0,
            "max":             99999999,
            "default":         10000000,
        },
        {
            "key":                   "fragment_filters",
            "label":                 "Fragment filters",
            "label_key":             "fn_fragment_filters_label",
            "description":           "Badge conditions — AND: all must pass  |  OR: any one passes",
            "description_key":       "fn_fragment_filters_desc",
            "type":                  "fragment_filters",
            "choices":               [],   # populated at dialog open from YAML step
            "choices_yaml_key":      "fragment_template_choices",
            "choices_yaml_step_type": "find_truck",   # scan this event_type step for the key
            "default":               {"mode": "AND", "filters": []},
        },
    ],

    "FightBoomer": [
        {
            "key":             "target_level",
            "label":           "Target level (Lv.)",
            "label_key":       "fn_target_level_label",
            "description":     "Boomer fight level to set before searching (1–10)",
            "description_key": "fn_target_level_desc",
            "type":            "int",
            "min":             1,
            "max":             10,
            "default":         10,
        },
    ],

    "BeingAttacked": [
        {
            "key":             "send_zalo_message",
            "label":           "Zalo message",
            "label_key":       "fn_zalo_message_label",
            "description":     "Message when under attack (@All = whole group)",
            "description_key": "fn_zalo_message_desc_beingattacked",
            "type":            "str",
            "default":         "@All Ối dời ơi!, Kem Chua đang bị tấn công",
        },
        {
            "key":             "send_zalo_receiver_name",
            "label":           "Zalo receiver",
            "label_key":       "fn_zalo_receiver_label",
            "description":     "Chat/group display name (e.g. Nhóm HLSE)",
            "description_key": "fn_zalo_receiver_desc_beingattacked",
            "type":            "str",
            "default":         "Nhóm HLSE",
        },
        {
            "key":             "send_zalo_repeat_interval_sec",
            "label":           "Repeat interval (sec)",
            "label_key":       "fn_zalo_repeat_interval_label",
            "description":     "Resend every N seconds while attack icon visible (0 = once)",
            "description_key": "fn_zalo_repeat_desc_beingattacked",
            "type":            "int",
            "min":             0,
            "max":             3600,
            "default":         300,
        },
    ],

    "TestAllianceAttack": [
        {
            "key":             "send_zalo_message",
            "label":           "Zalo message",
            "label_key":       "fn_zalo_message_label",
            "description":     "Message when alliance is under attack",
            "description_key": "fn_zalo_message_desc_testalliance",
            "type":            "str",
            "default":         " @All Bớ làng nước ơi!, liên minh đang bị tấn công",
        },
        {
            "key":             "send_zalo_receiver_name",
            "label":           "Zalo receiver",
            "label_key":       "fn_zalo_receiver_label",
            "description":     "Chat/group display name",
            "description_key": "fn_zalo_receiver_desc_testalliance",
            "type":            "str",
            "default":         "Nhóm HLSE",
        },
        {
            "key":             "send_zalo_repeat_interval_sec",
            "label":           "Repeat interval (sec)",
            "label_key":       "fn_zalo_repeat_interval_label",
            "description":     "Resend every N seconds while icon visible (0 = once)",
            "description_key": "fn_zalo_repeat_desc_testalliance",
            "type":            "int",
            "min":             0,
            "max":             3600,
            "default":         300,
        },
    ],

    "TurnOnShield": [
        {
            "key":             "shield_duration",
            "label":           "Shield type",
            "label_key":       "fn_shield_duration_label",
            "description":     "8h | 24h | 3d — fixed click target per option",
            "description_key": "fn_shield_duration_desc",
            "type":            "str",
            "default":         "8h",
        },
    ],


    "ClickTreasure": [
        {
            "key":             "send_zalo_message",
            "label":           "Zalo message (treasure)",
            "label_key":       "fn_clicktreasure_zalo_msg_label",
            "description":     "Message text when treasure is detected",
            "description_key": "fn_clicktreasure_zalo_msg_desc",
            "type":            "str",
            "default":         "@All Có kho báo!",
        },
        {
            "key":             "send_zalo_receiver_name",
            "label":           "Zalo receiver",
            "label_key":       "fn_clicktreasure_zalo_recv_label",
            "description":     "Chat/group display name",
            "description_key": "fn_clicktreasure_zalo_recv_desc",
            "type":            "str",
            "default":         "Nhóm HLSE",
        },
        {
            "key":             "send_zalo_repeat_interval_sec",
            "label":           "Zalo repeat (sec)",
            "label_key":       "fn_clicktreasure_zalo_repeat_label",
            "description":     "Resend every N seconds while treasure visible (0 = once)",
            "description_key": "fn_clicktreasure_zalo_repeat_desc",
            "type":            "int",
            "min":             0,
            "max":             3600,
            "default":         60,
        },
        {
            "key":             "max_clicks",
            "label":           "Max clicks",
            "label_key":       "fn_max_clicks_label",
            "description":     "Maximum clicks per session",
            "description_key": "fn_max_clicks_desc",
            "type":            "int",
            "min":             1,
            "max":             99999,
            "default":         2000,
        },
        {
            "key":             "click_interval_sec",
            "label":           "Click interval (sec)",
            "label_key":       "fn_click_interval_label",
            "description":     "Seconds between clicks (0 = fastest)",
            "description_key": "fn_click_interval_desc",
            "type":            "float",
            "min":             0.0,
            "max":             5.0,
            "step":            0.01,
            "default":         0.01,
        },
    ],
}
