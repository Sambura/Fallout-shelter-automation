from ctypes import windll
import numpy as np
import win32gui
import win32ui

def find_window(condition):
    handle = None

    def window_enum_handler(hwnd, ctx):
        nonlocal handle
        if win32gui.IsWindowVisible(hwnd):
            if condition(win32gui.GetWindowText(hwnd)):
                if handle is not None: return
                handle = hwnd
                return

    win32gui.EnumWindows(window_enum_handler, None)
    return handle

target_hwnd = None
width, height = None, None
hwndDC = None
mfcDC = None
saveDC = None
saveBitMap = None

def init_window_capture(window_title):
    global target_hwnd, width, height, hwndDC, mfcDC, saveDC, saveBitMap
    target_hwnd = find_window(lambda x: window_title in x)
    if target_hwnd is None: return False
    left, top, right, bot = win32gui.GetClientRect(target_hwnd)
    width = right - left
    height = bot - top
    hwndDC = win32gui.GetWindowDC(target_hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    return True

def do_capture():
    result = windll.user32.PrintWindow(target_hwnd, saveDC.GetSafeHdc(), 1)
    bmpstr = saveBitMap.GetBitmapBits(True)
    return np.frombuffer(bmpstr, dtype=np.uint8).reshape(height, width, 4)[:,:,:3][:,:,::-1]

def finish_window_capture():
    # presumably some of these fail if the game is closed at this point
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(target_hwnd, hwndDC)
