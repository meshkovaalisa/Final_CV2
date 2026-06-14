import cv2
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont
from PyQt6 import uic


class HistogramWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("histogram_window.ui", self)
        self._last_src, self._last_res = None, None

    def calculate_hist_image(self, gray_img, size):
        if gray_img is None or size.width() <= 5 or size.height() <= 5:
            return None

        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
        max_val = hist.max() if hist.max() > 0 else 1
        w, h = size.width(), size.height()
        graph_h = h - 25
        hist_normalized = (hist / max_val) * (graph_h - 10)

        pixmap = QPixmap(w, h)
        pixmap.fill(QColor(30, 30, 30))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bin_w = w / 256.0
        painter.setPen(QPen(QColor(0, 150, 255), 1))
        for i in range(256):
            painter.drawLine(int(i * bin_w), graph_h, int(i * bin_w), int(graph_h - hist_normalized[i]))

        painter.setPen(QColor(180, 180, 180))
        painter.setFont(QFont("Segoe UI", 9))
        labels_to_draw = [0, 64, 128, 192, 255]

        for val in labels_to_draw:
            tx = int(val * bin_w)
            offset = 0 if val == 0 else (-20 if val == 255 else -10)
            painter.drawText(tx + offset, h - 5, str(val))
            painter.drawLine(tx, graph_h, tx, graph_h + 4)

        painter.end()
        return pixmap

    def update_histograms(self, source_frame, processed_frame):
        self._last_src, self._last_res = source_frame.copy(), processed_frame.copy()
        if self.isVisible():
            self._redraw()

    def _redraw(self):
        if self._last_src is None or self._last_res is None:
            return
        for frame, widget in [(self._last_src, self.label_hist_source), (self._last_res, self.label_hist_result)]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            pix = self.calculate_hist_image(gray, widget.contentsRect().size())
            if pix:
                widget.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible():
            self._redraw()
