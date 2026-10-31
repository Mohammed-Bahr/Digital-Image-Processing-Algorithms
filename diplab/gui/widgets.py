"""Custom widgets: image viewer, histogram plot, transfer-curve plot."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import (QColor, QFont, QImage, QPainter, QPen, QPixmap,
                         QPolygonF)
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsView, QSizePolicy,
                             QWidget)


def array_to_qimage(arr) -> QImage:
    """uint8 ndarray (2-D or HxWx3) -> QImage (deep copy)."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    a = np.ascontiguousarray(a)
    if a.ndim == 2:
        h, w = a.shape
        return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, c = a.shape
    if c == 4:
        return QImage(a.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
    return QImage(a.data, w, h, w * 3, QImage.Format_RGB888).copy()


def array_to_pixmap(arr) -> QPixmap:
    return QPixmap.fromImage(array_to_qimage(arr))


class ImageView(QGraphicsView):
    """Zoomable / pannable image canvas with a placeholder message."""

    def __init__(self, placeholder: str = "No image", parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._fit_mode = True

        self.setBackgroundBrush(QColor("#0d1118"))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setRenderHints(QPainter.SmoothPixmapTransform |
                            QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setMinimumSize(160, 120)

        self._placeholder = self._scene.addText(placeholder)
        self._placeholder.setDefaultTextColor(QColor("#5b677a"))
        self._placeholder.setFont(QFont("Sans", 10))
        self._placeholder.setPos(14, 12)
        self._scene.setSceneRect(0, 0, 360, 260)

    # ------------------------------------------------------------------
    def set_image(self, arr):
        pix = array_to_pixmap(arr)
        if self._item is None:
            self._item = self._scene.addPixmap(pix)
            self._item.setTransformationMode(Qt.SmoothTransformation)
        else:
            self._item.setPixmap(pix)
        self._placeholder.setVisible(False)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._fit_mode = True
        self.fit_image()

    def clear_image(self, message: str = "No image"):
        if self._item is not None:
            self._scene.removeItem(self._item)
            self._item = None
        self._placeholder.setPlainText(message)
        self._placeholder.setVisible(True)
        self._scene.setSceneRect(0, 0, 360, 260)
        self.resetTransform()

    def fit_image(self):
        if self._item is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_image()

    def wheelEvent(self, event):
        if self._item is None:
            return
        self._fit_mode = False
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        self._fit_mode = True
        self.fit_image()
        super().mouseDoubleClickEvent(event)


class HistogramWidget(QWidget):
    """Input vs result histogram, 256 bins."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.before = None
        self.after = None
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, before, after=None):
        self.before = None if before is None else np.asarray(before, float)
        self.after = None if after is None else np.asarray(after, float)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#10151e"))

        left, top, right, bottom = 44, 30, 14, 22
        w = self.width() - left - right
        h = self.height() - top - bottom
        p.setPen(QColor("#7d8aa0"))
        p.setFont(QFont("Sans", 8))
        p.drawText(QRectF(left, 6, w, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   "Histogram  (number of pixels per intensity)")
        if w < 20 or h < 20:
            return

        # grid + ticks
        for i in range(5):
            x = left + w * i / 4.0
            p.setPen(QPen(QColor("#1d2534"), 1))
            p.drawLine(QPointF(x, top), QPointF(x, top + h))
            p.setPen(QColor("#5b677a"))
            p.drawText(QRectF(x - 20, top + h + 2, 40, 16), Qt.AlignCenter,
                       str(int(round(255 * i / 4.0))))
        for j in range(4):
            y = top + h * j / 3.0
            p.setPen(QPen(QColor("#1d2534"), 1))
            p.drawLine(QPointF(left, y), QPointF(left + w, y))

        counts = [c for c in (self.before, self.after)
                  if c is not None and c.size]
        if not counts:
            p.setPen(QColor("#5b677a"))
            p.drawText(QRectF(left, top, w, h), Qt.AlignCenter,
                       "No histogram yet")
            return

        maxv = max(float(c.max()) for c in counts)
        if maxv <= 0:
            return

        def draw(counts, line_color, fill_color, width):
            n = len(counts)
            poly = QPolygonF()
            poly.append(QPointF(left, top + h))
            for i, c in enumerate(counts):
                x = left + w * i / max(n - 1, 1)
                y = top + h - h * float(c) / maxv
                poly.append(QPointF(x, y))
            poly.append(QPointF(left + w, top + h))
            p.setBrush(fill_color)
            p.setPen(Qt.NoPen)
            p.drawPolygon(poly)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(line_color, width))
            p.drawPolyline(poly)

        if self.before is not None:
            draw(self.before, QColor(150, 158, 175),
                 QColor(150, 158, 175, 45), 1.2)
        if self.after is not None:
            draw(self.after, QColor(92, 214, 205),
                 QColor(92, 214, 205, 50), 1.8)

        # legend
        x = left + w - 150
        for label, color in (("input", QColor(150, 158, 175)),
                             ("result", QColor(92, 214, 205))):
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            p.drawRect(QRectF(x, 8, 9, 9))
            p.setPen(QColor("#aab5c7"))
            p.drawText(QRectF(x + 13, 4, 52, 16),
                       Qt.AlignLeft | Qt.AlignVCenter, label)
            x += 68


class CurveWidget(QWidget):
    """Transfer function s = T(r) for point operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.x = None
        self.y = None
        self.setMinimumHeight(150)

    def set_curve(self, x, y):
        self.x = None if x is None else np.asarray(x, float)
        self.y = None if y is None else np.asarray(y, float)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#10151e"))
        p.setPen(QColor("#7d8aa0"))
        p.setFont(QFont("Sans", 8))
        p.drawText(QRectF(44, 6, self.width() - 58, 18),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   "Transfer curve  s = T(r)")

        left, top, right, bottom = 44, 28, 18, 38
        w = self.width() - left - right
        h = self.height() - top - bottom
        if w < 20 or h < 20:
            return

        for i in range(5):
            x = left + w * i / 4.0
            p.setPen(QPen(QColor("#1d2534"), 1))
            p.drawLine(QPointF(x, top), QPointF(x, top + h))
            p.setPen(QColor("#5b677a"))
            p.drawText(QRectF(x - 20, top + h + 2, 40, 14), Qt.AlignCenter,
                       str(int(round(255 * i / 4.0))))
            y = top + h - h * i / 4.0
            p.drawText(QRectF(2, y - 8, 38, 16), Qt.AlignRight | Qt.AlignVCenter,
                       str(int(round(255 * i / 4.0))))

        # identity
        pen = QPen(QColor("#39445a"), 1, Qt.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(left, top + h), QPointF(left + w, top))

        if self.x is None or self.y is None:
            p.setPen(QColor("#5b677a"))
            p.drawText(QRectF(left, top, w, h), Qt.AlignCenter,
                       "Not a point operation.\nTry Negative, Log, Gamma or "
                       "Piecewise.")
            return

        poly = QPolygonF()
        for xv, yv in zip(self.x, self.y):
            px = left + w * float(xv) / 255.0
            py = top + h - h * float(yv) / 255.0
            poly.append(QPointF(px, py))
        p.setPen(QPen(QColor(92, 214, 205), 2))
        p.setBrush(Qt.NoBrush)
        p.drawPolyline(poly)

        p.setPen(QColor("#8b97aa"))
        p.drawText(QRectF(left, self.height() - 16, w, 14), Qt.AlignCenter,
                   "input intensity r")
