"""
FastClicker — dedicated background thread for rapid mouse clicking.

Uses pynput.mouse to simulate clicks — pynput generates native OS-level input
events that are indistinguishable from real hardware input.

Cursor is moved to the target position ONCE before the loop via pynput's
position property, then LEFT button press + release is fired at maximum rate.
No cursor movement inside the loop = no drag.

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

    def start(self, sx: int, sy: int, rate: int = 0,
              offset_x: int = 0, offset_y: int = 0,
              corner_pos: tuple | None = None, corner_every: int = 1000):
        """Start clicking around (sx, sy).
        offset_x/offset_y : max random deviation in pixels.
        corner_pos         : (cx, cy) screen coords to click every `corner_every` clicks.
        corner_every       : click corner once per this many normal clicks (default 1000).
        """
        self.stop()
        self._stop_event.clear()
        self.click_count = 0
        self._thread = threading.Thread(
            target=self._loop, args=(sx, sy, offset_x, offset_y, corner_pos, corner_every),
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
              corner_pos: tuple | None, corner_every: int):
        while not self._stop_event.is_set():
            # Every corner_every clicks: send one click to the corner position
            if corner_pos and self.click_count > 0 and self.click_count % corner_every == 0:
                _mouse.position = corner_pos
                _mouse.click(Button.left)
                log.info("[FastClicker] corner click at {} (total={})".format(corner_pos, self.click_count))

            x = sx + (random.randint(-offset_x, offset_x) if offset_x > 0 else 0)
            y = sy + (random.randint(-offset_y, offset_y) if offset_y > 0 else 0)
            _mouse.position = (x, y)
            _mouse.click(Button.left)
            self.click_count += 1
            time.sleep(0.1)
