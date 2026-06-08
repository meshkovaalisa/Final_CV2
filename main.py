import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from PyQt6 import uic


class CVApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('main_window.ui', self)

        self.hist_window = HistogramWindow()
        self.setWindowTitle("OpenCV Processing App (PyQt6)")
        self.setMinimumSize(900, 600)

        from PyQt6.QtWidgets import QSizePolicy
        self.frame_video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_video.setMinimumWidth(600)
        self.sidebar_visible = True
        self.frame_sidebar.setMinimumWidth(300)
        self.frame_sidebar.setMaximumWidth(300)

        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar)
        self.btn_reset_filters.clicked.connect(self.reset_filters)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.current_tab = 0
        self.template_image = None

        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.btn_load_template.clicked.connect(self.load_template_image)
        self.check_show_hist.clicked.connect(self.show_histogram)

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

    def reset_filters(self):
        self.combo_transform_type.setCurrentIndex(0)
        self.spin_c.setValue(1.0)
        self.spin_gamma.setValue(1.0)

        self.spin_k0.setValue(0.4)
        self.spin_k1.setValue(0.02)
        self.spin_k2.setValue(0.4)
        self.spin_E.setValue(4.0)
        self.spin_win.setValue(31)
        self.comboBox.setCurrentIndex(0)
        self.spinBox.setValue(5)
        self.checkBox.setChecked(False)

        self.template_image = None
        self.label_template_name.setText("Файл не выбран")

    def on_tab_changed(self, index):
        self.current_tab = index

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        processed_frame = self.process_frame(frame)

        self.display_image(frame, self.label_original)
        self.display_image(processed_frame, self.label_processed)
        self.hist_window.update_histograms(frame, processed_frame,main_app=self)

    def process_frame(self, frame):
        if self.current_tab == 0:
            return self.apply_scalar_transform(frame)
        elif self.current_tab == 1:
            return self.apply_histogram_transform(frame)
        elif self.current_tab == 2:
            return frame.copy()
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


class HistogramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("histogram_window.ui", self)
        self.setWindowTitle("Гистограммы сигналов")

    def draw_filtered_histogram(self, gray_img, smooth_type, win_size):
        if gray_img is None:
            return None

        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()

        if smooth_type == 1:
            kernel = np.ones(win_size, dtype=np.float32) / win_size
            hist = cv2.filter2D(hist.reshape(-1, 1), -1, kernel).flatten()
        elif smooth_type == 2:
            kernel = cv2.getGaussianKernel(win_size, sigma=-1)
            hist = cv2.filter2D(hist.reshape(-1, 1), -1, kernel).flatten()

        hist_w, hist_h = 256, 150
        hist_image = np.zeros((hist_h, hist_w, 3), dtype=np.uint8) + 20

        cv2.normalize(hist, hist, 0, hist_h - 10, cv2.NORM_MINMAX)

        for i in range(1, 256):
            pt1 = (i - 1, hist_h - int(hist[i - 1]))
            pt2 = (i, hist_h - int(hist[i]))
            cv2.line(hist_image, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

        return hist_image

    def update_histograms(self, source_frame, processed_frame, main_app=None):
        if not self.isVisible():
            return

        smooth_type = 0
        win_size = 5

        if main_app is not None:
            try:
                smooth_type = main_app.comboBox.currentIndex()
                win_size = main_app.spinBox.value() | 1

                if win_size < 3:
                    win_size = 3
            except AttributeError:
                pass

        gray_src = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        gray_res = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

        hist_src_img = self.draw_filtered_histogram(gray_src, smooth_type, win_size)
        hist_res_img = self.draw_filtered_histogram(gray_res, smooth_type, win_size)

        self.display_on_label(hist_src_img, self.label_hist_source)
        self.display_on_label(hist_res_img, self.label_hist_result)

    def display_on_label(self, cv_img, label_widget: QLabel):
        if cv_img is None or label_widget is None:
            return
        h, w, ch = cv_img.shape
        qt_image = QImage(
            cv_img.data, w, h, ch * w, QImage.Format.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(qt_image)
        label_widget.setPixmap(
            pixmap.scaled(
                label_widget.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CVApp()
    window.show()
    sys.exit(app.exec())
