from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    if arr.ndim == 2:
        h, w = arr.shape
        img = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    elif arr.shape[2] == 4:
        h, w, _ = arr.shape
        img = QImage(arr.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
    else:
        rgb = np.ascontiguousarray(arr[..., :3])
        h, w, _ = rgb.shape
        img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class ZoomPanGraphicsView(QGraphicsView):
    ZOOM_IN = 1.25
    ZOOM_OUT = 1 / 1.25

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._overlay_items: list = []

        self.setRenderHints(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)

    def set_image(self, arr: Optional[np.ndarray]) -> None:
        self.clear_overlays()
        if self._pix_item is not None:
            self._scene.removeItem(self._pix_item)
            self._pix_item = None
        if arr is None:
            return
        pix = ndarray_to_qpixmap(arr)
        self._pix_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(pix.rect().toRectF())
        # Fit immediately AND once more on the next event-loop tick, so that
        # if the viewport hasn't been sized yet (e.g. during a tab switch
        # or the initial show), the deferred call lands after layout.
        self.fit_to_view()
        QTimer.singleShot(0, self.fit_to_view)

    def fit_to_view(self) -> None:
        if self._pix_item is None:
            return
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def add_overlay_item(self, item) -> None:
        self._scene.addItem(item)
        self._overlay_items.append(item)

    def clear_overlays(self) -> None:
        for it in self._overlay_items:
            try:
                self._scene.removeItem(it)
            except RuntimeError:
                pass
        self._overlay_items.clear()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        factor = self.ZOOM_IN if event.angleDelta().y() > 0 else self.ZOOM_OUT
        self.scale(factor, factor)
