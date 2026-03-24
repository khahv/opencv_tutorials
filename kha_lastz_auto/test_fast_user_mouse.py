"""
Standalone test for zoom-shake user-mouse abort (Windows).

Detects macOS-style horizontal shake: move left, then right, within a short window.
Default: GetCursorPos polling. Optional ``--hook`` uses WH_MOUSE_LL.

Usage (from kha_lastz_auto directory):
    python test_fast_user_mouse.py --synthetic-tests   # logic in user_mouse_shake_detect.py only
    python test_fast_user_mouse.py
    python test_fast_user_mouse.py --debug
    python test_fast_user_mouse.py --hook --left-leg 60 --right-leg 60

Ctrl+C to exit (live test).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test zoom-shake mouse abort (Windows).")
    parser.add_argument(
        "--synthetic-tests",
        action="store_true",
        help="Run built-in tests in user_mouse_shake_detect.py and exit (no Windows needed)",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose stats every ~1s")
    parser.add_argument(
        "--count-all",
        action="store_true",
        help="Treat injected moves as physical (debug drivers)",
    )
    parser.add_argument("--left-leg", type=float, default=75.0, help="Min px moved left (smaller x) before trough")
    parser.add_argument("--right-leg", type=float, default=75.0, help="Min px moved right from trough")
    parser.add_argument("--window-sec", type=float, default=0.42, help="Sliding window (seconds)")
    parser.add_argument(
        "--max-down",
        type=float,
        default=0.22,
        help="Max seconds from local high to trough (0 = disable cap)",
    )
    parser.add_argument(
        "--max-up",
        type=float,
        default=0.22,
        help="Max seconds from trough to recovery high (0 = disable cap)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.4,
        help="Min min(L,R)/max(L,R) so both legs are similar (see user_mouse_shake_detect.py)",
    )
    parser.add_argument(
        "--max-leg",
        type=float,
        default=400.0,
        help="Reject if either leg exceeds this (px); 0 = no cap",
    )
    parser.add_argument("--cooldown", type=float, default=0.45, help="Seconds after trip before re-arming")
    parser.add_argument(
        "--hook",
        action="store_true",
        help="Use WH_MOUSE_LL instead of polling",
    )
    parser.add_argument("--poll-interval", type=float, default=0.008, help="Poll interval when not --hook")
    args = parser.parse_args()

    if args.synthetic_tests:
        from user_mouse_shake_detect import run_synthetic_self_tests

        run_synthetic_self_tests()
        print("user_mouse_shake_detect.run_synthetic_self_tests() OK")
        return 0

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if sys.platform != "win32":
        logging.error("This test only runs on Windows.")
        return 1

    import user_mouse_abort as uma

    uma.configure(
        enabled=True,
        gesture_left_leg_px=args.left_leg,
        gesture_right_leg_px=args.right_leg,
        gesture_window_sec=args.window_sec,
        gesture_max_downstroke_sec=args.max_down,
        gesture_max_upstroke_sec=args.max_up,
        gesture_min_leg_balance_ratio=args.balance,
        gesture_max_single_leg_px=args.max_leg,
        cooldown_after_abort_sec=args.cooldown,
        count_all_moves_as_physical=args.count_all,
        use_low_level_hook=args.hook,
        poll_interval_sec=args.poll_interval,
    )
    uma.start()

    mode = "WH_MOUSE_LL" if args.hook else "GetCursorPos poll"
    logging.info(
        "%s running. Shake mouse LEFT then RIGHT quickly (no buttons held). Ctrl+C to stop. "
        "left>=%.0f right>=%.0f bal>=%.2f max_leg<=%.0f window=%.2fs",
        mode,
        args.left_leg,
        args.right_leg,
        args.balance,
        args.max_leg,
        args.window_sec,
    )

    trips = 0
    try:
        while True:
            if uma.consume_abort_request():
                trips += 1
                logging.warning(">>> TRIP #%d — zoom-shake detected", trips)
            time.sleep(0.02)
    except KeyboardInterrupt:
        logging.info("Stopped by user (trips=%d).", trips)
    finally:
        uma.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
