import sys
import os
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from PyQt6 import uic

import tabs.scalar_tab as scalar_tab
import tabs.histogram_tab as histogram_tab
import tabs.filter_tab as filter_tab
import tabs.binarization_tab as binarization_tab
import tabs.detection_tab as detection_tab
import tabs.motion_tab as motion_tab
from tabs.histogram_window import HistogramWindow


class CVApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('main_window.ui', self)

        self.setWindowTitle("OpenCV Processing App (PyQt6)")
        self.setMinimumSize(900, 600)
        self.frame_video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.frame_video.setMinimumWidth(600)
        self.frame_sidebar.setMinimumWidth(300)
        self.frame_sidebar.setMaximumWidth(300)

        self.motion_history = []

        self.sidebar_visible = True
        self.template_image = None
        self.hist_window = HistogramWindow()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar)
        self.btn_load_template.clicked.connect(self.load_template_image)
        self.check_show_hist.clicked.connect(self.show_histogram)
        self.combo_bin_method.currentIndexChanged.connect(self.toggle_binarization_widgets)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.toggle_binarization_widgets()

    def toggle_sidebar(self):
        w = 300 if not self.sidebar_visible else 0
        self.frame_sidebar.setMinimumWidth(w)
        self.frame_sidebar.setMaximumWidth(w)
        self.btn_toggle_sidebar.setText("Показать панель" if self.sidebar_visible else "Скрыть панель")
        self.sidebar_visible = not self.sidebar_visible

    def load_template_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите эталонное изображение", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.template_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.template_image is not None:
                self.label_template_name.setText(f"Загружен: {path.split('/')[-1]}")

    def toggle_binarization_widgets(self):
        method = self.combo_bin_method.currentIndex()
        self.spin_bin_threshold.setEnabled(method == 1)
        self.spin_bin_win.setEnabled(method in (3, 4))
        self.spin_bin_c_val.setEnabled(method in (3, 4))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        processed_frame = self.process_frame(frame)
        self.display_image(frame, self.label_original)
        self.display_image(processed_frame, self.label_processed)

        widget = self.tab_widget.currentWidget()
        if widget and widget.objectName() == "tab_detectors":
            self.hist_window.update_histograms(frame, frame)
        else:
            self.hist_window.update_histograms(frame, processed_frame)

    def process_frame(self, frame):
        widget = self.tab_widget.currentWidget()
        if widget is None:
            return frame.copy()
        name = widget.objectName()

        if name == "tab_scalar":
            return scalar_tab.process_scalar(frame, self)
        elif name == "tab_histogram":
            return histogram_tab.process_histogram(frame, self, self.template_image)
        elif name == "tab_binarization":
            return binarization_tab.process_binarization(frame, self)
        elif name == "tab_detectors":
            return detection_tab.process_detections(frame, self, self.face_cascade)
        elif name == "tab_hist_filter":
            return filter_tab.process_filter(frame, self)
        elif name == "tab_motion":
            return motion_tab.process_motion(frame, self, self.face_cascade, self.motion_history)
        return frame.copy()

    def display_image(self, cv_img, label_widget: QLabel):
        if cv_img is None:
            return
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        label_widget.setPixmap(
            QPixmap.fromImage(qt_image).scaled(label_widget.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))

    def show_histogram(self):
        self.hist_window.show()

    def closeEvent(self, event):
        self.timer.stop()
        self.hist_window.close()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CVApp()
    window.show()
    sys.exit(app.exec())
