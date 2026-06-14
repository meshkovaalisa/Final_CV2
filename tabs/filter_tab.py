import cv2

def process_filter(frame, ui):
    method = ui.combo_smooth_type.currentIndex()
    if method == 0:
        return frame.copy()
    win_size = ui.spin_hist_win.value()
    if method == 1:
        return cv2.blur(frame, (win_size, win_size))
    elif method == 2:
        win_size = max(1, win_size) | 1
        return cv2.GaussianBlur(frame, (win_size, win_size), sigmaX=0)
    return frame.copy()
