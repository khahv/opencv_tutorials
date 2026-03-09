"""
fn_settings_schema.py
---------------------
Per-function UI settings schema.
Each entry is a list of field definitions that the Settings dialog renders.

Field definition keys:
    key         str   – storage key in fn_settings[fn_name]
    label       str   – display label in dialog
    description str   – hint/tooltip text (optional)
    type        str   – "int" | "float" | "str" | "bool" | "password"
    min/max     num   – for int/float: bounds
    default     any   – default value if not stored

To add settings for a new function, add an entry here.
bot_engine.py reads overrides from fn_settings at runtime.
"""

SCHEMA: dict = {
    "PinLoggin": [
        {
            "key":         "password",
            "label":       "PIN Password",
            "description": "Login PIN code (stored in .env_config, not .env)",
            "type":        "password",
            "default":     "",
        },
    ],

    "ClickTreasure": [
        {
            "key":         "max_clicks",
            "label":       "Max Clicks",
            "description": "Maximum clicks per session",
            "type":        "int",
            "min":         1,
            "max":         99999,
            "default":     2000,
        },
        {
            "key":         "click_interval_sec",
            "label":       "Click Interval (sec)",
            "description": "Seconds between clicks (0 = max speed)",
            "type":        "float",
            "min":         0.0,
            "max":         5.0,
            "step":        0.01,
            "default":     0.01,
        },
    ],

    "TruckPlunder": [
        {
            "key":         "servers",
            "label":       "Server Filter",
            "description": "Comma-separated server numbers to plunder (e.g. 600, 601)",
            "type":        "str",
            "default":     "600, 601",
        },
        {
            "key":         "max_power",
            "label":       "Max Power",
            "description": "Skip trucks with power ≥ this value",
            "type":        "int",
            "min":         0,
            "max":         99999999,
            "default":     10000000,
        },
    ],

    "FightBoomer": [
        {
            "key":         "target_level",
            "label":       "Target Level (Lv.)",
            "description": "Boomer fight level to set before searching (1–10)",
            "type":        "int",
            "min":         1,
            "max":         10,
            "default":     10,
        },
    ],

    "BeingAttacked": [
        {
            "key":         "send_zalo_message",
            "label":       "Zalo message",
            "description": "Nội dung tin nhắn gửi khi bị tấn công (@All = báo cả nhóm)",
            "type":        "str",
            "default":     "@All Ối dời ơi!, Kem Chua đang bị tấn công",
        },
        {
            "key":         "send_zalo_repeat_interval_sec",
            "label":       "Repeat interval (sec)",
            "description": "Gửi lặp mỗi N giây khi icon attack còn hiện (0 = chỉ gửi 1 lần)",
            "type":        "int",
            "min":         0,
            "max":         3600,
            "default":     300,
        },
    ],

    "TestAllianceAttack": [
        {
            "key":         "send_zalo_message",
            "label":       "Zalo message",
            "description": "Nội dung tin nhắn gửi khi liên minh bị tấn công",
            "type":        "str",
            "default":     " @All Bớ làng nước ơi!, liên minh đang bị tấn công",
        },
        {
            "key":         "send_zalo_repeat_interval_sec",
            "label":       "Repeat interval (sec)",
            "description": "Gửi lặp mỗi N giây khi icon còn hiện (0 = chỉ gửi 1 lần)",
            "type":        "int",
            "min":         0,
            "max":         3600,
            "default":     300,
        },
    ],

    "ClickTreasure": [
        {
            "key":         "send_zalo_message",
            "label":       "Zalo message (treasure)",
            "description": "Nội dung tin nhắn khi có kho báo",
            "type":        "str",
            "default":     "@All Có kho báo!",
        },
        {
            "key":         "send_zalo_repeat_interval_sec",
            "label":       "Zalo repeat (sec)",
            "description": "Gửi lặp mỗi N giây khi treasure còn hiện (0 = chỉ gửi 1 lần)",
            "type":        "int",
            "min":         0,
            "max":         3600,
            "default":     60,
        },
        {
            "key":         "max_clicks",
            "label":       "Max Clicks",
            "description": "Maximum clicks per session",
            "type":        "int",
            "min":         1,
            "max":         99999,
            "default":     2000,
        },
        {
            "key":         "click_interval_sec",
            "label":       "Click Interval (sec)",
            "description": "Seconds between clicks (0 = max speed)",
            "type":        "float",
            "min":         0.0,
            "max":         5.0,
            "step":        0.01,
            "default":     0.01,
        },
    ],
}
