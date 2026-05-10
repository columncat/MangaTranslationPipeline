"""Per-bbox inpainting mask editor.

Opens a modal dialog showing the bbox crop with a paintable overlay.
The user paints (red translucent) the regions that should be inpainted,
then clicks OK to write the result back as a uint8 mask. The mask lives
in bbox-local coordinates and is consumed by Step 5's
``build_inpaint_mask``.

Tools:
- Left-drag = paint
- Right-drag = erase
- Brush size spinbox + mouse wheel
- Fill all / Clear / Invert buttons
- Reset button restores the original bbox-rectangle behaviour (returns
  ``None`` from the dialog so the caller drops the per-bbox mask).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QSize
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..i18n import tr


def _ndarray_to_qpixmap(arr: np.ndarray) -> QPixmap:
    if arr.ndim == 2:
        h, w = arr.shape
        img = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(arr[..., :3])
        h, w, _ = rgb.shape
        img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class _MaskCanvas(QLabel):
    """Fixed-scale canvas that shows the bbox crop and the paintable mask.

    Coordinates are kept in mask-pixel space (i.e. the bbox's native
    resolution). The widget scales up by ``zoom`` for display only.
    """

    def __init__(
        self,
        crop_rgb: np.ndarray,
        initial_mask: Optional[np.ndarray] = None,
        zoom: int = 4,
        parent=None,
    ):
        super().__init__(parent)
        h, w = crop_rgb.shape[:2]
        self._w = w
        self._h = h
        self._zoom = max(1, int(zoom))
        self._brush_size = max(2, min(w, h) // 16 or 2)
        self._base_pix = _ndarray_to_qpixmap(crop_rgb)

        if initial_mask is not None and initial_mask.shape[:2] == (h, w):
            self._mask = (initial_mask > 0).astype(np.uint8) * 255
        else:
            self._mask = np.zeros((h, w), dtype=np.uint8)

        self.setFixedSize(QSize(w * self._zoom, h * self._zoom))
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._last_pos: Optional[tuple[int, int]] = None

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()

    @property
    def brush_size(self) -> int:
        return self._brush_size

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, min(self._w * 4, int(size)))
        self.update()

    def fill_all(self) -> None:
        self._mask[:] = 255
        self.update()

    def clear_all(self) -> None:
        self._mask[:] = 0
        self.update()

    def invert(self) -> None:
        self._mask = 255 - self._mask
        self.update()

    # ---- input ----

    def _to_mask_xy(self, pos: QPointF) -> tuple[int, int]:
        x = int(pos.x() / self._zoom)
        y = int(pos.y() / self._zoom)
        return x, y

    def _stamp(self, mx: int, my: int, value: int) -> None:
        # Soft brush via a filled circle. Always-on edges so the user has
        # a predictable footprint regardless of zoom.
        r = self._brush_size
        y0, y1 = max(0, my - r), min(self._h, my + r + 1)
        x0, x1 = max(0, mx - r), min(self._w, mx + r + 1)
        if y1 <= y0 or x1 <= x0:
            return
        ys = np.arange(y0, y1) - my
        xs = np.arange(x0, x1) - mx
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        disk = (xx * xx + yy * yy) <= r * r
        sub = self._mask[y0:y1, x0:x1]
        sub[disk] = value

    def _stroke_to(self, mx: int, my: int, value: int) -> None:
        """Stamp along the line from the last position to (mx, my) so we
        don't get gaps when the mouse moves quickly."""
        if self._last_pos is None:
            self._stamp(mx, my, value)
            self._last_pos = (mx, my)
            return
        x0, y0 = self._last_pos
        dx, dy = mx - x0, my - y0
        steps = max(abs(dx), abs(dy), 1)
        for i in range(steps + 1):
            t = i / steps
            self._stamp(int(x0 + dx * t), int(y0 + dy * t), value)
        self._last_pos = (mx, my)

    def mousePressEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        mx, my = self._to_mask_xy(ev.position())
        if ev.button() == Qt.MouseButton.LeftButton:
            self._last_pos = None
            self._stroke_to(mx, my, 255)
            self.update()
        elif ev.button() == Qt.MouseButton.RightButton:
            self._last_pos = None
            self._stroke_to(mx, my, 0)
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        mx, my = self._to_mask_xy(ev.position())
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._stroke_to(mx, my, 255)
            self.update()
        elif ev.buttons() & Qt.MouseButton.RightButton:
            self._stroke_to(mx, my, 0)
            self.update()
        else:
            # Just trigger a repaint so the brush preview follows the cursor.
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:  # type: ignore[override]
        self._last_pos = None

    def leaveEvent(self, ev: QEvent) -> None:  # type: ignore[override]
        self._last_pos = None
        self.update()

    def wheelEvent(self, ev: QWheelEvent) -> None:  # type: ignore[override]
        delta = 1 if ev.angleDelta().y() > 0 else -1
        self.set_brush_size(self._brush_size + delta)

    # ---- paint ----

    def paintEvent(self, ev: QPaintEvent) -> None:  # type: ignore[override]
        p = QPainter(self)
        # 1) base crop, scaled up
        p.drawPixmap(self.rect(), self._base_pix, self._base_pix.rect())

        # 2) translucent red overlay where the mask is set
        if self._mask.any():
            # Build an RGBA image where (r=255, a=120) wherever mask>0.
            rgba = np.zeros((self._h, self._w, 4), dtype=np.uint8)
            sel = self._mask > 0
            rgba[..., 0] = 255
            rgba[..., 3] = sel * 120
            qimg = QImage(
                rgba.tobytes(), self._w, self._h, 4 * self._w,
                QImage.Format.Format_RGBA8888,
            ).copy()
            p.drawImage(self.rect(), qimg)

        # 3) brush preview at the cursor
        cursor_pos = self.mapFromGlobal(self.cursor().pos())
        if self.rect().contains(cursor_pos):
            pen = QPen(QColor(0, 200, 255, 220))
            pen.setWidth(1)
            p.setPen(pen)
            r_view = self._brush_size * self._zoom
            p.drawEllipse(cursor_pos, r_view, r_view)

        p.end()


class MaskEditorDialog(QDialog):
    """Modal mask editor.

    On accept, ``result_mask`` is the uint8 mask in bbox-local coordinates
    (or ``None`` if the user clicked Reset).
    """

    def __init__(
        self,
        crop_rgb: np.ndarray,
        initial_mask: Optional[np.ndarray] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("mask.title"))
        self.setModal(True)
        self._reset = False

        h, w = crop_rgb.shape[:2]
        # Pick a zoom that keeps the dialog around 600 px wide max.
        zoom = max(1, min(8, 600 // max(1, w)))

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr("mask.hint")))

        self.canvas = _MaskCanvas(crop_rgb, initial_mask, zoom=zoom)
        layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)

        # Brush size row
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel(tr("mask.brush")))
        self.brush_box = QSpinBox()
        self.brush_box.setRange(1, max(w, h))
        self.brush_box.setValue(self.canvas.brush_size)
        self.brush_box.valueChanged.connect(self.canvas.set_brush_size)
        size_row.addWidget(self.brush_box)
        size_row.addStretch(1)

        fill_btn = QPushButton(tr("mask.fill"))
        fill_btn.clicked.connect(self.canvas.fill_all)
        size_row.addWidget(fill_btn)
        clear_btn = QPushButton(tr("mask.clear"))
        clear_btn.clicked.connect(self.canvas.clear_all)
        size_row.addWidget(clear_btn)
        invert_btn = QPushButton(tr("mask.invert"))
        invert_btn.clicked.connect(self.canvas.invert)
        size_row.addWidget(invert_btn)
        layout.addLayout(size_row)

        # OK / Reset / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        reset_btn = QPushButton(tr("mask.reset"))
        reset_btn.setToolTip(tr("mask.reset_tip"))
        reset_btn.clicked.connect(self._do_reset)
        buttons.addButton(reset_btn, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _do_reset(self) -> None:
        self._reset = True
        self.accept()

    @property
    def result_mask(self) -> Optional[np.ndarray]:
        if self._reset:
            return None
        return self.canvas.mask
