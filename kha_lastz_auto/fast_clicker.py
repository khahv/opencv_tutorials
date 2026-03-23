"""
FastClicker — dedicated background thread for rapid mouse clicking.

Uses pynput.mouse to simulate clicks — pynput generates native OS-level input
events that are indistinguishable from real hardware input.

Cursor is moved to the target position each iteration, then LEFT press + release.
Optional ``corner:`` is handled by the runner: the thread exits, the runner sends
one isolated corner click, then calls ``start`` again (see ``corner_pause``).

Usage:
    clicker = FastClicker()
    clicker.start(sx=1200, sy=800, rate=300)
    time.sleep(1.0)
    clicker.stop()
    print(clicker.click_count)
"""

import logging
import random
import threading
import time
from pynput.mouse import Button, Controller

log = logging.getLogger("kha_lastz")

_mouse = Controller()


class FastClicker:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.click_count = 0
        self.offset_changed = False  # True when thread exited because offset_change_time was reached
        self.corner_pause = False  # True when thread exited so the runner can corner-click, then restart
        self.last_target_xy: tuple[int, int] | None = None  # last storm click pixel (screen coords)

    def start(self, sx: int, sy: int, rate: int = 0,
              offset_x: int = 0, offset_y: int = 0,
              corner_pos: tuple | None = None, corner_every: int = 1000,
              win_bounds: tuple | None = None,
              offset_change_time: float = 1.0,
              initial_click_count: int | None = None,
              offset_epoch_mono: float | None = None,
              fixed_target_xy: tuple[int, int] | None = None,
              click_interval_sec: float | None = None):
        """Start clicking around (sx, sy).
        offset_x/offset_y    : max random deviation in pixels.
        corner_pos            : (cx, cy) screen coords to click every `corner_every` clicks.
        corner_every          : click corner once per this many normal clicks (default 1000).
        win_bounds            : (left, top, right, bottom) screen coords of the game window.
                                Each click target is checked against these bounds before firing;
                                clicks landing outside are skipped and logged.
        offset_change_time    : seconds between offset re-randomizations (default 1.0).
                                Measured from offset_epoch_mono (runner sets this once per phase).
        offset_epoch_mono     : time.monotonic() when the current random pixel was chosen; pass the
                                same value across corner restarts so offset_change_time is not reset.
        fixed_target_xy       : exact screen (x, y) to click; skips re-randomizing (corner resume).
        initial_click_count   : resume storm with this counter (after an isolated corner click).
        click_interval_sec    : sleep after each click (default 0.1). If None and rate > 0, uses 1/rate.
        rate                  : clicks per second when click_interval_sec is None (default 0 = use 0.1s).
        """
        _sleep = 0.1
        if click_interval_sec is not None:
            _sleep = max(0.001, min(10.0, float(click_interval_sec)))
        else:
            try:
                _rf = float(rate)
            except (TypeError, ValueError):
                _rf = 0.0
            if _rf > 0:
                _sleep = max(0.001, min(10.0, 1.0 / _rf))

        self.stop()
        self._stop_event.clear()
        self.click_count = initial_click_count if initial_click_count is not None else 0
        self.offset_changed = False
        self.corner_pause = False
        self._thread = threading.Thread(
            target=self._loop,
            args=(sx, sy, offset_x, offset_y, corner_pos, corner_every,
                  win_bounds, offset_change_time, offset_epoch_mono, fixed_target_xy, _sleep),
            daemon=True, name="FastClicker"
        )
        self._thread.start()

    def stop(self):
        """Stop and wait for thread (max 1 s)."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self, sx: int, sy: int, offset_x: int, offset_y: int,
              corner_pos: tuple | None, corner_every: int,
              win_bounds: tuple | None, offset_change_time: float,
              offset_epoch_mono: float | None, fixed_target_xy: tuple[int, int] | None,
              click_sleep_sec: float):
        # One "phase" = same pixel until offset_change_time elapses (wall in monotonic time from
        # offset_epoch_mono). Corner restarts pass fixed_target_xy + same epoch so the phase is not reset.
        _epoch = offset_epoch_mono if offset_epoch_mono is not None else time.monotonic()
        if fixed_target_xy is not None:
            _cur_x, _cur_y = int(fixed_target_xy[0]), int(fixed_target_xy[1])
        else:
            _cur_x = sx + (random.randint(-offset_x, offset_x) if offset_x > 0 else 0)
            _cur_y = sy + (random.randint(-offset_y, offset_y) if offset_y > 0 else 0)
        self.last_target_xy = (_cur_x, _cur_y)

        while not self._stop_event.is_set():
            now_mono = time.monotonic()

            # Offset_change_time reached → runner restarts with a new random pixel and new epoch.
            if now_mono - _epoch >= offset_change_time:
                log.debug("[FastClicker] offset_change_time reached — exiting thread for restart")
                self.offset_changed = True
                break

            # Corner focus: exit thread so the runner can stop the storm, click corner alone, then restart.
            if corner_pos and self.click_count > 0 and self.click_count % corner_every == 0:
                log.debug(
                    "[FastClicker] corner_pause at count={} (runner will corner-click then restart)".format(
                        self.click_count,
                    )
                )
                self.corner_pause = True
                break

            if win_bounds is not None:
                _l, _t, _r, _b = win_bounds
                if not (_l <= _cur_x < _r and _t <= _cur_y < _b):
                    log.warning(
                        "[FastClicker] skip click ({},{}) — outside game window ({},{})→({},{})".format(
                            _cur_x, _cur_y, _l, _t, _r, _b,
                        )
                    )
                    time.sleep(click_sleep_sec)
                    continue

            _mouse.position = (_cur_x, _cur_y)
            self.last_target_xy = (_cur_x, _cur_y)
            _mouse.click(Button.left)
            self.click_count += 1
            time.sleep(click_sleep_sec)
