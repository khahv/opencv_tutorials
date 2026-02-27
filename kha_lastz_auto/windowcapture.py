import ctypes
import numpy as np
import win32gui, win32ui, win32con
import mss

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

        if window_name is None:
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

        print(f'WindowCapture: client size = {self.w}x{self.h}, screen offset = ({self.offset_x}, {self.offset_y})')

    def get_screenshot(self):
        # Lay toa do client area hien tai (cap nhat moi frame phong truong hop cua so di chuyen)
        (left, top) = win32gui.ClientToScreen(self.hwnd, (0, 0))

        # mss chup theo toa do man hinh -> lay duoc ca noi dung GPU (Unity, DX, GL)
        with mss.mss() as sct:
            monitor = {'left': left, 'top': top, 'width': self.w, 'height': self.h}
            raw = sct.grab(monitor)

        img = np.array(raw)          # BGRA uint8

        # drop alpha channel
        img = img[..., :3]

        # make C_CONTIGUOUS
        img = np.ascontiguousarray(img)

        return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        (left, top) = win32gui.ClientToScreen(self.hwnd, (0, 0))
        return (pos[0] + left, pos[1] + top)
