"""In ra danh sach ten cac cua so dang hien (visible). Chay: python list_windows.py"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import win32gui

def callback(hwnd, _):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd)
        if title:
            print("{}  {}".format(hex(hwnd), title))

if __name__ == "__main__":
    print("Cac cua so dang hien:\n")
    win32gui.EnumWindows(callback, None)
