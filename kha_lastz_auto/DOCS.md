# kha_lastz_auto — Documentation

An automation bot for the game LastZ. Uses OpenCV template matching to detect and interact with UI elements.

---

## Project Structure

```
kha_lastz_auto/
├── main.py               # Entry point — main loop, hotkey listener, cron scheduler
├── bot_engine.py         # Loads and executes functions, handles each step type
├── attack_detector.py    # Detects being-attacked state via icon template matching
├── vision.py             # Template matching wrapper (OpenCV)
├── windowcapture.py      # Captures screenshots from the game window
├── config.yaml           # Main config: window size, function list
├── functions/            # Each .yaml file defines one automation function
├── buttons_template/     # Template images used for screen matching
└── .env                  # Secrets (e.g. PIN_PASSWORD=1234)
```

---

## config.yaml

```yaml
reference_width: 2400
reference_height: 1600

functions:
  - name: FightBoomer
    key: b
    cron: "*/2 * * * *"
    priority: 2
    enabled: true
```

| Field | Description |
|---|---|
| `reference_width/height` | Client area size of the game window when templates were captured. The bot resizes the window to this size on startup so scale = 1.0 and matching is fast. |
| `name` | Function name — must match the filename in `functions/` (without `.yaml`). |
| `key` | Hotkey to toggle the function on/off. Press the same key while running to stop. |
| `cron` | Cron expression for automatic scheduling. See [crontab.guru](https://crontab.guru/). |
| `priority` | Lower number = higher priority. A higher-priority function can preempt a running one. |
| `trigger` | Special event trigger. Currently supports `attacked` — fires the function when attack is detected. Cannot be combined with `cron`. |
| `enabled` | `true/false` — disables the function from both cron and hotkey when `false`. |

### Priority & Scheduler

When a function is triggered (hotkey or cron) while another is already running:

- **Higher priority** → preempts immediately; the current function is stopped.
- **Lower priority** → added to the queue; runs after the current function finishes.
- **Same priority** → stops the current function and starts the new one.

After any function finishes, the queue is drained in priority order.

### Controls

| Input | Action |
|---|---|
| Hotkey (e.g. `b`) | Start function / press again to stop |
| `Ctrl + Esc` | Quit the bot cleanly |
| `q` on preview window | Quit the bot |

### Trigger: attacked

A function with `trigger: attacked` starts automatically the moment an attack is detected — no hotkey or cron needed.

```yaml
  - name: TurnOnShield
    trigger: attacked
    priority: 1        # high priority so it preempts other running functions
    enabled: true
```

Multiple functions can use `trigger: attacked`. They are all queued/started using the same priority rules as cron and hotkey triggers.

---

## Functions

Each function is a YAML file in `functions/` containing a `description` and a list of `steps`.

Steps run **sequentially**. If a step returns `false` (not found within timeout), all remaining steps are **skipped** (function aborted), unless a step has `run_always: true`.

### Available Functions

| Function | Description |
|---|---|
| `PinLoggin` | Auto-login when the PIN entry screen appears |
| `FightBoomer` | Open Magnifier → select Boomer → set level → Search → Team Up → March |
| `ClickTreasure` | Continuously click the Treasure Helicopter when visible |
| `CollectExplorationReward` | Zoom out → HQ → ExplorationReward → Claim → Collect |
| `SoliderTrain` | Zoom out → HQ → train Shooter / Assault / Rider soldiers |
| `HelpAlliance` | Click the Help Alliance button when visible |
| `CheckMail` | Open Mail → Alliance tab → Claim All → System tab → Claim All → close |
| `DonateAllianceTech` | Open Alliance → Techs → Like → Donate x20 |

---

## Event Types

### `match_click`
Find a template on screen and click it.

```yaml
- event_type: match_click
  template: buttons_template/MyButton.png
  threshold: 0.75          # match confidence (0.0–1.0), default 0.75
  one_shot: true           # true  = click once then advance
                           # false = keep clicking until max_clicks or timeout
  timeout_sec: 10          # seconds to wait before giving up
  max_clicks: 20           # (one_shot: false) maximum number of clicks
  click_interval_sec: 0.15 # (one_shot: false) pause between clicks
  click_offset_x: 0.5     # shift click right by N × template_width  (e.g. 0.5 = half width)
  click_offset_y: 0.0     # shift click down  by N × template_height (negative = up)
  click_random_offset: 20  # randomize click position ±N pixels (anti-bot detection)
  run_always: false        # true = run even if a previous step returned false
```

Returns `true` on successful click, `false` when `timeout_sec` expires without a match.

#### `click_offset_x` / `click_offset_y`

Shift the click position relative to the center of the matched template, expressed as a **ratio of the template size**.

```yaml
click_offset_x: 0.5    # shift right  by 50% of template width
click_offset_y: -0.5   # shift up     by 50% of template height
```

| Value | Effect |
|---|---|
| `0.0` (default) | click the template center |
| `0.5` | shift right / down by half the template dimension |
| `-0.5` | shift left / up by half the template dimension |

Applied before `click_random_offset`.

---

### `match_move`
Find a template on screen and move the mouse to it — no click.

```yaml
- event_type: match_move
  template: buttons_template/MyTarget.png
  threshold: 0.75
  timeout_sec: 10
  click_offset_x: 0.0   # same offset rules as match_click
  click_offset_y: 0.0
```

Returns `true` when the mouse is moved successfully, `false` on timeout.

---

### `match_multi_click`
Find **all** visible instances of a template and click each one.

```yaml
- event_type: match_multi_click
  template: buttons_template/MyButton.png
  threshold: 0.75
  timeout_sec: 10
  click_interval_sec: 0.15
```

Returns `true` after clicking all matches. If none found within `timeout_sec`, still returns `true` (does not abort).

---

### `match_count`
Check how many times a template appears on screen. Does not click.

```yaml
- event_type: match_count
  template: buttons_template/PasswordSlot.png
  count: 6          # require at least N instances
  threshold: 0.75
  timeout_sec: 8
  debug_save: false # true = save screenshot to debug/ on failure
```

Returns `true` when at least `count` instances are found. Returns `false` on timeout.

---

### `wait_until_match`
Wait until a template appears on screen. Does not click.

```yaml
- event_type: wait_until_match
  template: buttons_template/LoadingDone.png
  threshold: 0.75
  timeout_sec: 30
```

Returns `true` when found, `false` on timeout.

---

### `click_unless_visible`
If `visible_template` is **already on screen** → skip (already on the right screen).
If **not found** → click `click_template` to navigate there.

```yaml
- event_type: click_unless_visible
  visible_template: buttons_template/MagnifierButton.png
  click_template: buttons_template/WorldButton.png
  threshold: 0.75
  timeout_sec: 3
```

Always returns `true`.

---

### `sleep`
Wait for a fixed duration.

```yaml
- event_type: sleep
  duration_sec: 1.5
  run_always: false
```

---

### `click_position`
Click a fixed position defined as a ratio of the window size.

```yaml
- event_type: click_position
  offset_x: 0.15   # 15% from the left edge
  offset_y: 0.15   # 15% from the top edge
  run_always: false
```

Commonly used to close popups or dismiss banners by clicking an empty corner.

---

### `key_press`
Press a keyboard key.

```yaml
- event_type: key_press
  key: escape
```

---

### `type_text`
Type a string. Supports `.env` variable substitution via `${VAR_NAME}`.

```yaml
- event_type: type_text
  text: "${PIN_PASSWORD}"
  interval_sec: 0.1   # delay between keystrokes in seconds
```

---

### `set_level`
Use OCR to read the current level, then click Plus/Minus until `target_level` is reached.

```yaml
- event_type: set_level
  target_level: 10
  level_anchor_template: buttons_template/Slider.png   # template used as position anchor
  level_anchor_offset: [-45, -65, 110, 35]             # [dx, dy, w, h] from anchor center
  plus_template: buttons_template/PlusButton.png
  minus_template: buttons_template/MinusButton.png
  threshold: 0.75
  timeout_sec: 20
  debug_save_roi: true   # save the first OCR crop to a file for debugging
```

Requires Tesseract OCR to be installed.

---

### `base_zoomout`
Click the Headquarters button (if visible) then scroll to zoom out to the world map.

```yaml
- event_type: base_zoomout
  template: buttons_template/HeadquartersButton.png
  threshold: 0.75
  scroll_times: 5
  scroll_interval_sec: 0.1
  timeout_sec: 5
```

Always returns `true`. Use as the first step in functions that require the world map view.

---

## Attack Detector

`attack_detector.py` detects when the player's base is under attack by matching the `BeingAttackedWarning` icon on screen.

- **Attack starts**: icon found in any single frame → logs `[Alert] House is being attacked!`
- **Attack ends**: icon absent continuously for `clear_sec` seconds → logs `[Alert] Attack has ended.`

Configured in `main.py`:

```python
attack_detector = AttackDetector(
    warning_template_path="buttons_template/BeingAttackedWarning.png",
    clear_sec=10.0,   # seconds of no icon before declaring attack over
)
```

---

## Adding a New Function

1. Create `functions/MyFunction.yaml`:

```yaml
description: "Short description of what this function does"
steps:
  - event_type: match_click
    template: buttons_template/SomeButton.png
    threshold: 0.75
    one_shot: true
    timeout_sec: 10
```

2. Register it in `config.yaml`:

```yaml
  - name: MyFunction
    key: x
    cron: "*/5 * * * *"
    priority: 50
    enabled: true
```

3. Add template images to `buttons_template/` — crop them from the game window at exactly `reference_width x reference_height` resolution.

---

## .env

Stores secrets. Never commit this file.

```env
PIN_PASSWORD=1234
```

Reference in YAML steps with `${PIN_PASSWORD}`.
