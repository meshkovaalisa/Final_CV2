import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from PyQt6 import uic
import os


class CVApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('main_window.ui', self)

        self._prev_cx = None
        self._prev_cy = None

        self.hist_window = HistogramWindow()
        self.setWindowTitle("OpenCV Processing App (PyQt6)")
        self.setMinimumSize(900, 600)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        from PyQt6.QtWidgets import QSizePolicy
        self.frame_video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_video.setMinimumWidth(600)
        self.sidebar_visible = True
        self.frame_sidebar.setMinimumWidth(300)
        self.frame_sidebar.setMaximumWidth(300)

        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_tab = 0
        self.template_image = None

        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.btn_load_template.clicked.connect(self.load_template_image)
        self.check_show_hist.clicked.connect(self.show_histogram)
        self.combo_bin_method.currentIndexChanged.connect(self.toggle_binarization_widgets)
        self.toggle_binarization_widgets()

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.frame_sidebar.setMaximumWidth(0)
            self.frame_sidebar.setMinimumWidth(0)
            self.btn_toggle_sidebar.setText("Показать панель")
        else:
            self.frame_sidebar.setMinimumWidth(300)
            self.frame_sidebar.setMaximumWidth(300)
            self.btn_toggle_sidebar.setText("Скрыть панель")
        self.sidebar_visible = not self.sidebar_visible

    def on_tab_changed(self, index):
        self.current_tab = index

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        processed_frame = self.process_frame(frame)

        self.display_image(frame, self.label_original)
        self.display_image(processed_frame, self.label_processed)

        current_widget = self.tab_widget.currentWidget()

        if current_widget and current_widget.objectName() == "tab_detectors":
            self.hist_window.update_histograms(frame, frame, main_app=self)
        else:
            self.hist_window.update_histograms(frame, processed_frame, main_app=self)

    def process_frame(self, frame):
        current_widget = self.tab_widget.currentWidget()
        if current_widget is None:
            return frame.copy()

        tab_name = current_widget.objectName()

        if tab_name == "tab_scalar":
            return self.apply_scalar_transform(frame)

        elif tab_name == "tab_histogram":
            return self.apply_histogram_transform(frame)

        elif tab_name == "tab_binarization":
            return self.apply_binarization(frame)
        elif tab_name == "tab_detectors":
            return self.apply_all_detections(frame)
        elif tab_name == "tab_hist_filter":
            return self.apply_hist_filter(frame)
        elif tab_name == "tab_motion":
            return self.apply_motion_detection(frame)

        return frame.copy()

    def apply_scalar_transform(self, frame):
        transform_type = self.combo_transform_type.currentIndex()
        c = self.spin_c.value()
        gamma = self.spin_gamma.value()

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

    def load_template_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите эталонное изображение", "",
            "Images (*.png *.jpg *.bmp)"
        )
        if file_path:
            self.template_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.template_image is not None:
                self.label_template_name.setText(f"Загружен: {file_path.split('/')[-1]}")

    def toggle_binarization_widgets(self):
        if not hasattr(self, 'combo_bin_method'):
            return

        method = self.combo_bin_method.currentIndex()

        self.spin_bin_threshold.setEnabled(method == 1)

        self.spin_bin_win.setEnabled(method in (3, 4))
        self.spin_bin_c_val.setEnabled(method in (3, 4))

    def apply_all_detections(self, frame):
        display_frame = frame.copy()

        corn_method = self.combo_corn_method.currentIndex()
        cont_method = self.combo_cont_method.currentIndex()
        line_method = self.combo_line_method.currentIndex()
        face_method = self.combo_face_method.currentIndex()

        if corn_method == 0 and cont_method == 0 and line_method == 0 and face_method == 0:
            return display_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = self.spin_det_thresh.value()
        param = self.spin_det_param.value()

        if corn_method == 1:
            gray_float = np.float32(gray)
            dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            limit = (thresh / 10.0) / 100.0 * dst.max()
            display_frame[dst > limit] = [0, 0, 255]

        elif corn_method == 2:
            quality_level = max(0.01, thresh / 100.0)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=param, qualityLevel=quality_level, minDistance=10)
            if corners is not None:
                for c in corners:
                    x, y = map(int, c.ravel())
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)

        elif corn_method == 3:
            fast = cv2.FastFeatureDetector_create(threshold=thresh)
            keypoints = fast.detect(gray, None)
            display_frame = cv2.drawKeypoints(display_frame, keypoints, None, color=(255, 0, 255))

        elif corn_method == 4:
            block_size = (param // 2) * 2 + 3
            dst = cv2.cornerMinEigenVal(gray, blockSize=block_size, ksize=3)
            limit = (101 - thresh) / 100.0 * dst.max()
            display_frame[dst > limit] = [0, 255, 255]

        elif corn_method == 5:
            sift = cv2.SIFT_create(nfeatures=param)
            keypoints = sift.detect(gray, None)
            display_frame = cv2.drawKeypoints(
                display_frame, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 255)
            )

        if cont_method == 1:
            lower_thresh = thresh * 2
            edges = cv2.Canny(gray, lower_thresh, lower_thresh * 2)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            display_frame[edges > 0] = [255, 255, 255]

        elif cont_method == 2:
            ksize = (param // 20) * 2 + 1
            ksize = max(1, min(7, ksize))

            sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=ksize)

            abs_sobel_x = cv2.convertScaleAbs(sobel_x)
            abs_sobel_y = cv2.convertScaleAbs(sobel_y)

            magnitude = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

            adjusted_magnitude = cv2.subtract(magnitude, thresh * 2)

            contour_color = np.array([0, 255, 0], dtype=np.uint8)

            mask = adjusted_magnitude > 10
            display_frame[mask] = contour_color

        elif cont_method == 3:
            edges = cv2.Canny(gray, thresh, thresh * 2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [c for c in contours if len(c) > param]
            cv2.drawContours(display_frame, filtered_contours, -1, (255, 150, 0), 2)

        if line_method == 1:
            edges = cv2.Canny(gray, 50, 150)
            hough_threshold = max(10, thresh)
            min_line_length = max(5, param)

            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=hough_threshold,
                                    minLineLength=min_line_length, maxLineGap=15)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        elif line_method == 2:
            lsd = cv2.createLineSegmentDetector()

            lines, _, _, _ = lsd.detect(gray)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = map(int, line.flatten())

                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length > param:
                        cv2.line(display_frame, (x1, y1), (x2, y2), (0, 130, 255), 2)

        if face_method == 1:
            min_neighbors = max(3, int(thresh / 10))
            min_size_val = max(30, param)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=min_neighbors,
                minSize=(min_size_val, min_size_val)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        return display_frame

    def apply_binarization(self, frame):

        method = self.combo_bin_method.currentIndex()
        thresh_t = self.spin_bin_threshold.value()
        win_size = self.spin_bin_win.value() | 1
        c_val = self.spin_bin_c_val.value()

        if method == 0:
            return frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method == 1:
            _, binarized = cv2.threshold(gray, thresh_t, 255, cv2.THRESH_BINARY)

        elif method == 2:
            _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 3:
            binarized = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, win_size, c_val
            )

        elif method == 4:
            binarized = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, win_size, c_val
            )
        else:
            return frame.copy()

        return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    def enhance_image(self, img, k0, k1, k2, E, win):
        img_f = img.astype(np.float32)
        mG = np.mean(img_f)
        sigmaG = np.std(img_f)

        mS = cv2.blur(img_f, (win, win))
        mS_sq = cv2.blur(img_f ** 2, (win, win))
        sigmaS = np.sqrt(np.maximum(mS_sq - mS ** 2, 0))

        mask = (mS <= k0 * mG) & (sigmaS >= k1 * sigmaG) & (sigmaS <= k2 * sigmaG)

        res = img_f.copy()
        res[mask] *= E

        return np.clip(res, 0, 255).astype(np.uint8)

    def match_histograms(self, source, template):
        hist_src = cv2.calcHist([source], [0], None, [256], [0, 256]).flatten()
        hist_tpl = cv2.calcHist([template], [0], None, [256], [0, 256]).flatten()

        cdf_src = hist_src.cumsum()
        cdf_tpl = hist_tpl.cumsum()

        if cdf_src[-1] == 0 or cdf_tpl[-1] == 0:
            return source

        cdf_src /= cdf_src[-1]
        cdf_tpl /= cdf_tpl[-1]

        lut = np.argmin(np.abs(cdf_src[:, None] - cdf_tpl), axis=1).astype(np.uint8)
        return cv2.LUT(source, lut)

    def apply_histogram_transform(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        win = self.spin_win.value() | 1

        result = self.enhance_image(
            gray,
            self.spin_k0.value(),
            self.spin_k1.value(),
            self.spin_k2.value(),
            self.spin_E.value(),
            win
        )

        if self.template_image is not None:
            result = self.match_histograms(result, self.template_image)

        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def display_image(self, cv_img, label_widget: QLabel):
        if cv_img is None:
            return
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_image)
        label_widget.setPixmap(pixmap.scaled(
            label_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def closeEvent(self, event):
        self.timer.stop()
        self.hist_window.close()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

    def show_histogram(self):
        self.hist_window.show()

    def apply_hist_filter(self, frame):
        method = self.combo_smooth_type.currentIndex()

        if method == 0:
            return frame.copy()

        win_size = self.spin_hist_win.value()

        if method == 1:
            win_size = win_size
            return cv2.blur(frame, (win_size, win_size))

        elif method == 2:
            win_size = max(1, win_size)
            win_size = win_size | 1

            return cv2.GaussianBlur(frame, (win_size, win_size), sigmaX=0)

        return frame.copy()

    def apply_motion_detection(self, frame):
        display_frame = frame.copy()

        if self.combo_motion_type.currentIndex() == 0:
            if hasattr(self, '_motion_history_queue'):
                self._motion_history_queue.clear()
            return display_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50))

        if not hasattr(self, '_motion_history_queue'):
            self._motion_history_queue = []

        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                cx = int(x + w // 2)
                cy = int(y + h // 2)

                is_moving = False

                if len(self._motion_history_queue) > 0:
                    old_cx, old_cy = self._motion_history_queue[0]

                    distance = ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5

                    thresh_val = self.spin_motion_thresh.value() / 3.0

                    if distance > thresh_val:
                        is_moving = True

                self._motion_history_queue.append((cx, cy))

                if len(self._motion_history_queue) > 4:
                    self._motion_history_queue.pop(0)

                if is_moving:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "AI MOTION DETECTED", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 165, 255), 1)
                    cv2.putText(display_frame, "AI STILL", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

                break
        else:
            if len(self._motion_history_queue) > 0:
                self._motion_history_queue.pop(0)

        return display_frame


class HistogramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("histogram_window.ui", self)
        self.setWindowTitle("Гистограммы сигналов")

        self._last_src = None
        self._last_res = None

    def calculate_hist_image(self, gray_img, size):
        if gray_img is None or size.width() <= 5 or size.height() <= 5:
            return None

        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()

        max_val = hist.max()
        if max_val == 0:
            max_val = 1

        w, h = size.width(), size.height()

        graph_h = h - 25
        hist_normalized = (hist / max_val) * (graph_h - 10)

        from PyQt6.QtGui import QPainter, QColor, QPen, QFont

        pixmap = QPixmap(w, h)
        pixmap.fill(QColor(30, 30, 30))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bin_w = w / 256.0
        painter.setPen(QPen(QColor(0, 150, 255), 1))

        for i in range(256):
            x = int(i * bin_w)
            y_start = graph_h
            y_end = int(graph_h - hist_normalized[i])

            painter.drawLine(x, y_start, x, y_end)

        painter.setPen(QColor(180, 180, 180))
        font = QFont("Segoe UI", 9)
        painter.setFont(font)

        labels_to_draw = [0, 64, 128, 192, 255]

        for val in labels_to_draw:
            text_x = int(val * bin_w)
            text_y = h - 5

            if val == 0:
                offset = 0
            elif val == 255:
                offset = -20
            else:
                offset = -10

            painter.drawText(text_x + offset, text_y, str(val))

            painter.drawLine(text_x, graph_h, text_x, graph_h + 4)

        painter.end()
        return pixmap

    def update_histograms(self, source_frame, processed_frame, main_app=None):

        self._last_src = source_frame.copy()
        self._last_res = processed_frame.copy()

        if not self.isVisible():
            return

        self._redraw()

    def _redraw(self):
        if self._last_src is None or self._last_res is None:
            return

        gray_src = cv2.cvtColor(self._last_src, cv2.COLOR_BGR2GRAY) if len(
            self._last_src.shape) == 3 else self._last_src
        rect_src = self.label_hist_source.contentsRect()
        pix_src = self.calculate_hist_image(gray_src, rect_src.size())
        if pix_src:
            self.label_hist_source.setPixmap(pix_src)

        gray_res = cv2.cvtColor(self._last_res, cv2.COLOR_BGR2GRAY) if len(
            self._last_res.shape) == 3 else self._last_res
        rect_res = self.label_hist_result.contentsRect()
        pix_res = self.calculate_hist_image(gray_res, rect_res.size())
        if pix_res:
            self.label_hist_result.setPixmap(pix_res)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible():
            self._redraw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CVApp()
    window.show()
    sys.exit(app.exec())
