import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from PyQt6 import uic


class CVApp(QMainWindow):
    def __init__(self):
        super().__init__()

        uic.loadUi('main_window.ui', self)
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
        self.brightness = 0
        self.contrast = 1.0
        self.current_tab = 0

        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.slider_brightness.valueChanged.connect(self.spin_brightness.setValue)
        self.spin_brightness.valueChanged.connect(self.slider_brightness.setValue)
        self.slider_contrast.valueChanged.connect(lambda v: self.dspin_contrast.setValue(v / 100.0))
        self.dspin_contrast.valueChanged.connect(lambda v: self.slider_contrast.setValue(int(v * 100)))

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
        self.slider_brightness.setValue(0)
        self.spin_brightness.setValue(0)
        self.slider_contrast.setValue(100)
        self.dspin_contrast.setValue(1.0)
        print("Ползунки сброшены")

    def on_tab_changed(self, index):
        self.current_tab = index

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        processed_frame = self.process_frame(frame)

        self.display_image(frame, self.label_original)
        self.display_image(processed_frame, self.label_processed)

    def process_frame(self, frame):
        if self.current_tab == 0:
            return self.apply_scalar_transform(frame)
        return frame.copy()

    def apply_scalar_transform(self, frame):
        """Применение скалярных преобразований (яркость/контраст)"""
        self.brightness = self.spin_brightness.value()
        self.contrast = self.dspin_contrast.value()

        alpha = self.contrast
        beta = self.brightness
        processed = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return processed

    def display_image(self, cv_img, label_widget: QLabel):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line,
                          QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        label_widget.setPixmap(pixmap.scaled(
            label_widget.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CVApp()
    window.show()
    sys.exit(app.exec())