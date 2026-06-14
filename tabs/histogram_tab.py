import cv2
import numpy as np

def process_histogram(frame, ui, template_image):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_f = gray.astype(np.float32)
    win = ui.spin_win.value() | 1
    mG, sigmaG = np.mean(img_f), np.std(img_f)
    mS = cv2.blur(img_f, (win, win))
    mS_sq = cv2.blur(img_f ** 2, (win, win))
    sigmaS = np.sqrt(np.maximum(mS_sq - mS ** 2, 0))
    mask = (mS <= ui.spin_k0.value() * mG) & (sigmaS >= ui.spin_k1.value() * sigmaG) & (sigmaS <= ui.spin_k2.value() * sigmaG)
    res = img_f.copy()
    res[mask] *= ui.spin_E.value()
    result = np.clip(res, 0, 255).astype(np.uint8)
    if template_image is not None:
        hist_src = cv2.calcHist([result], [0], None, [256], [0, 256]).flatten()
        hist_tpl = cv2.calcHist([template_image], [0], None, [256], [0, 256]).flatten()
        cdf_src, cdf_tpl = hist_src.cumsum(), hist_tpl.cumsum()
        if cdf_src[-1] != 0 and cdf_tpl[-1] != 0:
            cdf_src /= cdf_src[-1]
            cdf_tpl /= cdf_tpl[-1]
            lut = np.argmin(np.abs(cdf_src[:, None] - cdf_tpl), axis=1).astype(np.uint8)
            result = cv2.LUT(result, lut)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
