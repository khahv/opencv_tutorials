"""
event_send_zalo.py
------------------
Handler for the ``send_zalo`` event type.

Sends a Zalo message via ``zalo_web_clicker``.

Behaviour
---------
- One-shot: message sent in a background thread, step advances immediately.
- Repeat mode: when ``repeat_interval_sec > 0`` **and** a ``trigger_active_cb``
  is attached to the runner, a daemon thread keeps resending every N seconds
  for as long as the trigger callback returns truthy.  Pause-aware: the timer
  resets after the bot resumes so the first message is sent immediately.

All parameters can be overridden at runtime via ``fn_settings``:

  ``send_zalo_message``          — overrides ``message``
  ``send_zalo_receiver_name``    — overrides ``receiver_name``
  ``send_zalo_repeat_interval_sec`` — overrides ``repeat_interval_sec``
"""

import time
import threading
import logging

log = logging.getLogger("kha_lastz")

try:
    import zalo_web_clicker as _zalo_web_clicker
except ImportError:
    _zalo_web_clicker = None


def run(step: dict, screenshot, wincap, runner) -> str:
    """Execute the ``send_zalo`` event (fires once then advances)."""
    # ── Resolve message / receiver / interval (fn_settings override first) ───
    _msg_ov = runner._fn_setting("send_zalo_message")
    message = (_msg_ov if _msg_ov is not None and str(_msg_ov).strip() else "") \
              or (step.get("message") or "")

    _rcv_ov = runner._fn_setting("send_zalo_receiver_name")
    receiver_name = (_rcv_ov if _rcv_ov is not None and str(_rcv_ov).strip() else None) \
                    or step.get("receiver_name")

    _int_ov = runner._fn_setting("send_zalo_repeat_interval_sec")
    if _int_ov is not None:
        try:
            repeat_interval_sec = int(_int_ov)
        except (ValueError, TypeError):
            repeat_interval_sec = step.get("repeat_interval_sec") or 0
    else:
        repeat_interval_sec = step.get("repeat_interval_sec") or 0

    trigger_cb = getattr(runner, "trigger_active_cb", None)

    def _do_send():
        if _zalo_web_clicker:
            _zalo_web_clicker.send_zalo_message(message, receiver_name=receiver_name, logger=log)
        else:
            log.warning("[send_zalo] zalo_web_clicker not available, skip")

    if repeat_interval_sec > 0 and trigger_cb and callable(trigger_cb):
        def _repeat_loop():
            _do_send()
            next_send_at = time.time() + repeat_interval_sec
            paused_ref   = getattr(runner, "bot_paused", None)
            was_paused   = bool(paused_ref and paused_ref.get("paused", False))
            while trigger_cb():
                if paused_ref and paused_ref.get("paused", False):
                    was_paused = True
                    time.sleep(1)
                    continue
                if was_paused:
                    # Reset timer after resume so the next message sends immediately.
                    next_send_at = time.time()
                    was_paused   = False
                    log.info("[send_zalo] repeat counter reset (resumed), will send next tick")
                if time.time() >= next_send_at:
                    _do_send()
                    next_send_at = time.time() + repeat_interval_sec
                    log.info("[send_zalo] repeat (interval={}s)".format(repeat_interval_sec))
                time.sleep(1)

        threading.Thread(target=_repeat_loop, daemon=True).start()
        log.info("[send_zalo] {} -> started (repeat every {}s while trigger active)".format(
            runner._step_label(step), repeat_interval_sec))
    else:
        threading.Thread(target=_do_send, daemon=True).start()
        log.info("[send_zalo] {} -> sent once".format(runner._step_label(step)))

    runner._advance_step(True)
    return "running"
