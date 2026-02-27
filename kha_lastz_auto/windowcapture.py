import ctypes
import time
import logging
import numpy as np
import win32gui, win32ui, win32con
import mss

log = logging.getLogger("kha_lastz")

# Bat DPI awareness de toa do tra ve la pixel vat ly thuc te,
# khong bi Windows scale theo DPI (125%, 150%...).
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-monitor DPI aware (Windows 8.1+)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()        # System DPI aware (Windows Vista+)


class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name=None):
        # find the handle for the window we want to capture.
        # if no window name is given, capture the entire screen
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))

        self._is_desktop = window_name is None
        if self._is_desktop:
            # desktop: dung toan man hinh
            window_rect = win32gui.GetWindowRect(self.hwnd)
            self.w = window_rect[2] - window_rect[0]
            self.h = window_rect[3] - window_rect[1]
            self.cropped_x = 0
            self.cropped_y = 0
            self.offset_x = 0
            self.offset_y = 0
        else:
            # dung GetClientRect + ClientToScreen de lay dung client area (khong can biet border/titlebar)
            client_rect = win32gui.GetClientRect(self.hwnd)
            self.w = client_rect[2] - client_rect[0]
            self.h = client_rect[3] - client_rect[1]
            (client_left, client_top) = win32gui.ClientToScreen(self.hwnd, (0, 0))
            window_rect = win32gui.GetWindowRect(self.hwnd)
            self.cropped_x = client_left - window_rect[0]
            self.cropped_y = client_top  - window_rect[1]
            self.offset_x = client_left
            self.offset_y = client_top

        log.info('WindowCapture: client size = {}x{}, screen offset = ({}, {})'.format(self.w, self.h, self.offset_x, self.offset_y))

    def refresh_geometry(self):
        """Cap nhat lai kich thuoc va offset cua cua so (can thiet sau khi restore tu minimize)."""
        if self._is_desktop or not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return
        try:
            client_rect = win32gui.GetClientRect(self.hwnd)
            self.w = client_rect[2] - client_rect[0]
            self.h = client_rect[3] - client_rect[1]
            (client_left, client_top) = win32gui.ClientToScreen(self.hwnd, (0, 0))
            window_rect = win32gui.GetWindowRect(self.hwnd)
            self.cropped_x = client_left - window_rect[0]
            self.cropped_y = client_top - window_rect[1]
            self.offset_x = client_left
            self.offset_y = client_top
        except Exception:
            pass

    def resize_to_client(self, target_w, target_h):
        """
        Resize cua so sao cho CLIENT AREA (phan game, khong tinh title bar / border)
        dung bang target_w x target_h.
        Goi 1 lan luc khoi dong de dam bao window luon o dung kich thuoc.
        Tra ve True neu thanh cong, False neu khong the resize (fullscreen / bi khoa).
        """
        if self._is_desktop or not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return False
        try:
            # Tinh kich thuoc border tu window_rect - client_rect
            window_rect = win32gui.GetWindowRect(self.hwnd)
            client_rect = win32gui.GetClientRect(self.hwnd)
            border_w = (window_rect[2] - window_rect[0]) - client_rect[2]
            border_h = (window_rect[3] - window_rect[1]) - client_rect[3]

            total_w = target_w + border_w
            total_h = target_h + border_h

            win32gui.SetWindowPos(
                self.hwnd, None,
                window_rect[0], window_rect[1],  # giu nguyen vi tri
                total_w, total_h,
                win32con.SWP_NOZORDER,
            )
            time.sleep(0.1)
            self.refresh_geometry()
            return abs(self.w - target_w) <= 2 and abs(self.h - target_h) <= 2
        except Exception:
            return False

    def focus_window(self):
        """Dua cua so len truoc va phuc hoi neu dang minimize. Goi truoc get_screenshot de ket qua chuan hon."""
        if self._is_desktop or not self.hwnd or not win32gui.IsWindow(self.hwnd):
            return
        try:
            was_minimized = win32gui.IsIconic(self.hwnd)
            if was_minimized:
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            # Windows chi cho SetForegroundWindow khi process da nhan input gan day. Trick: nhan Alt truoc
            # thi Windows "mo khoa" (LockSetForegroundWindow) -> focus hoat dong ngay ca luc chua chay function.
            user32 = ctypes.windll.user32
            KEYEVENTF_KEYUP = 0x0002
            user32.keybd_event(win32con.VK_MENU, 0, 0, 0)  # Alt down
            try:
                win32gui.SetForegroundWindow(self.hwnd)
            finally:
                user32.keybd_event(win32con.VK_MENU, 0, KEYEVENTF_KEYUP, 0)  # Alt up
            # Chi sleep + refresh geometry khi vua restore tu minimize (tranh size 0, offset -32000)
            if was_minimized:
                time.sleep(0.15)
                self.refresh_geometry()
        except Exception:
            pass

    def get_screenshot(self):
        if not self._is_desktop:
            self.refresh_geometry()
        if self.w <= 0 or self.h <= 0:
            return None

        (left, top) = win32gui.ClientToScreen(self.hwnd, (0, 0))

        with mss.mss() as sct:
            monitor = {'left': left, 'top': top, 'width': self.w, 'height': self.h}
            raw = sct.grab(monitor)

        img = np.array(raw)          # BGRA uint8
        img = img[..., :3]
        img = np.ascontiguousarray(img)
        return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                log.info("{} {}".format(hex(hwnd), win32gui.GetWindowText(hwnd)))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        (left, top) = win32gui.ClientToScreen(self.hwnd, (0, 0))
        return (pos[0] + left, pos[1] + top)
