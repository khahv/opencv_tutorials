"""
Standalone test for fast physical mouse movement detection (Windows).

Uses the same WH_MOUSE_LL hook as user_mouse_abort.py. Run while moving the
mouse quickly; prints when the abort latch trips.

Usage (from kha_lastz_auto directory):
    python test_fast_user_mouse.py
    python test_fast_user_mouse.py --debug
    python test_fast_user_mouse.py --count-all --min-speed 4000

Ctrl+C to exit.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Ensure this directory is on path when run as: python test_fast_user_mouse.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test fast user mouse detection (Windows).")
    parser.add_argument("--debug", action="store_true", help="Verbose hook stats every ~1s")
    parser.add_argument(
        "--count-all",
        action="store_true",
        help="Treat injected moves as physical (debug drivers; bot would also trip)",
    )
    parser.add_argument("--min-speed", type=float, default=7500.0, help="Instant speed threshold px/s")
    parser.add_argument("--window-sec", type=float, default=0.12, help="Path sliding window (seconds)")
    parser.add_argument("--min-path", type=float, default=380.0, help="Min path length in window (px)")
    parser.add_argument("--cooldown", type=float, default=0.45, help="Seconds after trip before re-arming")
    args = parser.parse_args()

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
        min_instant_speed_px_s=args.min_speed,
        window_sec=args.window_sec,
        window_min_path_px=args.min_path,
        cooldown_after_abort_sec=args.cooldown,
        count_all_moves_as_physical=args.count_all,
    )
    uma.start()

    logging.info(
        "Hook running. Move the physical mouse quickly to trip detection. Ctrl+C to stop. "
        "count_all=%s min_speed=%.0f min_path=%.0f",
        args.count_all,
        args.min_speed,
        args.min_path,
    )

    trips = 0
    try:
        while True:
            if uma.consume_abort_request():
                trips += 1
                logging.warning(">>> TRIP #%d — fast movement detected (same latch as bot would use)", trips)
            time.sleep(0.02)
    except KeyboardInterrupt:
        logging.info("Stopped by user (trips=%d).", trips)
    finally:
        uma.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
