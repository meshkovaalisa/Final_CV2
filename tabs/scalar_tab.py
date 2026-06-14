import cv2
import numpy as np

def process_scalar(frame, ui):
    transform_type = ui.combo_transform_type.currentIndex()
    c = ui.spin_c.value()
    gamma = ui.spin_gamma.value()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_float = gray.astype(np.float32)
    if transform_type == 1:
        processed = 255.0 - img_float
    elif transform_type == 2:
        processed = c * np.log1p(img_float) * (255.0 / np.log1p(255.0))
    elif transform_type == 3:
        processed = c * np.power(img_float / 255.0, gamma) * 255.0
    elif transform_type == 4:
        processed = c * np.log1p(255.0 - img_float) * (255.0 / np.log1p(255.0))
    else:
        processed = img_float
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
