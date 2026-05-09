from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPixmap, QWheelEvent
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
    """Zoomable / pannable image view with arrow-key navigation hooks.

    Arrow keys are reserved for navigation (tab switch + image switch) and
    are forwarded as signals rather than consumed by the default scroll
    behaviour. The main window decides what they do.
    """

    ZOOM_IN = 1.25
    ZOOM_OUT = 1 / 1.25

    # Direction signals — emitted on Left/Right/Up/Down key press.
    nav_left = Signal()
    nav_right = Signal()
    nav_up = Signal()
    nav_down = Signal()

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
        # The view must accept focus so it sees arrow-key presses.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

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

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        # Arrow keys are repurposed for navigation. Anything with a modifier
        # falls through to the base class so e.g. Ctrl+Arrow still works for
        # finer-grained scroll if the user wants it.
        if event.modifiers() == Qt.KeyboardModifier.NoModifier:
            key = event.key()
            if key == Qt.Key.Key_Left:
                self.nav_left.emit()
                event.accept()
                return
            if key == Qt.Key.Key_Right:
                self.nav_right.emit()
                event.accept()
                return
            if key == Qt.Key.Key_Up:
                self.nav_up.emit()
                event.accept()
                return
            if key == Qt.Key.Key_Down:
                self.nav_down.emit()
                event.accept()
                return
        super().keyPressEvent(event)
