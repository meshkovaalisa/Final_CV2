import cv2

def process_binarization(frame, ui):
    method = ui.combo_bin_method.currentIndex()
    if method == 0:
        return frame.copy()
    thresh_t = ui.spin_bin_threshold.value()
    win_size = ui.spin_bin_win.value() | 1
    c_val = ui.spin_bin_c_val.value()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if method == 1:
        _, binarized = cv2.threshold(gray, thresh_t, 255, cv2.THRESH_BINARY)
    elif method == 2:
        _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 3:
        binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, win_size, c_val)
    elif method == 4:
        binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, win_size, c_val)
    else:
        return frame.copy()
    return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
