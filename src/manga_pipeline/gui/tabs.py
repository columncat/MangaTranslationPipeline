from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..models import PageContext
from .image_view import ZoomPanGraphicsView
from .items import ClickableRectItem, DraggableTextItem, EditableBBoxItem


class PhaseTabWidget(QWidget):
    """Base for the Detect / Translate phase tabs.

    Contains a horizontal splitter with two graphics views:
      - ``original_view`` (left, hidden by default) — always shows ``ctx.source``
      - ``view``          (right, always visible)  — phase-specific image

    A ``View Original`` checkbox in the toolbar toggles the left view.
    ``rerun_requested`` is emitted with the phase name when the user clicks
    the local re-run button.
    """

    rerun_requested = Signal(str)

    def __init__(self, phase: str, title: str, parent=None):
        super().__init__(parent)
        self.phase = phase
        self.title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._toolbar_layout = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        self._toolbar_layout.addWidget(self.title_label)
        self._toolbar_layout.addStretch(1)

        self.status_label = QLabel("not run")
        self.status_label.setStyleSheet("color: gray;")
        self._toolbar_layout.addWidget(self.status_label)

        self.view_original_box = QCheckBox("View Original")
        self.view_original_box.setToolTip(
            "Split this tab and show the original source image on the left half"
        )
        self.view_original_box.toggled.connect(self._on_toggle_view_original)
        self._toolbar_layout.addWidget(self.view_original_box)

        self.fit_button = QPushButton("Fit")
        self.fit_button.setMaximumWidth(60)
        self._toolbar_layout.addWidget(self.fit_button)

        self.rerun_button = QPushButton(f"Re-run {phase}")
        self._toolbar_layout.addWidget(self.rerun_button)
        self.rerun_button.clicked.connect(lambda: self.rerun_requested.emit(self.phase))

        layout.addLayout(self._toolbar_layout)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.original_view = ZoomPanGraphicsView(self._splitter)
        self.original_view.setVisible(False)
        self._splitter.addWidget(self.original_view)

        self.view = ZoomPanGraphicsView(self._splitter)
        self._splitter.addWidget(self.view)
        self._splitter.setSizes([0, 1000])
        layout.addWidget(self._splitter, 1)

        self.fit_button.clicked.connect(self._fit_visible_views)

        self._latest_source = None  # cached for the original_view

    # ---- view management ----

    def _on_toggle_view_original(self, checked: bool) -> None:
        self.original_view.setVisible(checked)
        if checked:
            # Give the left half a sensible width and update its image.
            total = max(1, self._splitter.size().width() or self.width())
            self._splitter.setSizes([total // 2, total - total // 2])
            if self._latest_source is not None:
                self.original_view.set_image(self._latest_source)
        else:
            self._splitter.setSizes([0, 1000])
        QTimer.singleShot(0, self._fit_visible_views)

    def _set_original_image(self, source) -> None:
        """Subclasses call this whenever the source image changes."""
        self._latest_source = source
        if self.view_original_box.isChecked():
            self.original_view.set_image(source)

    def _fit_visible_views(self) -> None:
        if self.view_original_box.isChecked():
            self.original_view.fit_to_view()
        self.view.fit_to_view()

    # ---- show / status ----

    def showEvent(self, event):  # type: ignore[override]
        super().showEvent(event)
        # Re-fit every time the user switches to this tab, so the image
        # always lands centered and at the largest size that fits.
        QTimer.singleShot(0, self._fit_visible_views)

    def set_status(self, text: str, ok: bool = True) -> None:
        color = "#2e7d32" if ok else "#c62828"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)

    def update_from_context(self, ctx: PageContext) -> None:
        raise NotImplementedError


class SourceTab(PhaseTabWidget):
    def __init__(self, parent=None):
        super().__init__(phase="source", title="0. Original", parent=parent)
        self.rerun_button.hide()
        self.view_original_box.hide()  # already showing the original

    def update_from_context(self, ctx: PageContext) -> None:
        self.view.set_image(ctx.source)
        self._set_original_image(ctx.source)


class DetectTab(PhaseTabWidget):
    """Detect phase view.

    - Right-click a box to delete it (always available).
    - Toggle ``Edit`` to drag boxes around and resize via corner handles.
    - ``Add Box`` inserts a default-sized rectangle at the image center.
    """

    bbox_delete_requested = Signal(int)
    bbox_geometry_changed = Signal(int, int, int, int, int)  # idx, x, y, w, h
    bbox_add_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(phase="detect", title="1. Detect", parent=parent)
        hint = QLabel("Right-click: delete  |  Edit: drag to move/resize")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        self._toolbar_layout.insertWidget(1, hint)

        self.add_button = QPushButton("Add Box")
        self.add_button.setToolTip("Insert a default-sized box at the image center")
        self.add_button.clicked.connect(self.bbox_add_requested.emit)
        self._toolbar_layout.insertWidget(2, self.add_button)

        self._edit_enabled = False
        self.edit_button = QPushButton("Edit")
        self.edit_button.setCheckable(True)
        self.edit_button.setToolTip("Toggle drag-to-move / corner-resize for boxes")
        self.edit_button.toggled.connect(self._on_toggle_edit)
        self._toolbar_layout.insertWidget(3, self.edit_button)

        self._ctx: Optional[PageContext] = None

    def _on_toggle_edit(self, checked: bool) -> None:
        self._edit_enabled = checked
        # In edit mode left-button drag must move/resize boxes, so disable pan.
        self.view.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if checked
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        if self._ctx is not None:
            self.update_from_context(self._ctx)

    def update_from_context(self, ctx: PageContext) -> None:
        self._ctx = ctx
        self.view.set_image(ctx.source)
        self._set_original_image(ctx.source)
        if not ctx.bboxes:
            return
        pen = QPen(QColor(255, 64, 64))
        pen.setWidth(2)
        for i, bbox in enumerate(ctx.bboxes):
            rect = EditableBBoxItem(
                QRectF(bbox.x, bbox.y, bbox.w, bbox.h),
                idx=i,
                on_right_click=self.bbox_delete_requested.emit,
                on_geometry_changed=self.bbox_geometry_changed.emit,
                edit_enabled=self._edit_enabled,
            )
            rect.setPen(pen)
            rect.setBrush(QBrush(QColor(255, 64, 64, 30)))
            self.view.add_overlay_item(rect)


class TranslateTab(PhaseTabWidget):
    """Combined OCR + Translation + Final view.

    Edit modes:
      - ``Move Text`` shows draggable text overlays on top of the cleaned image.
        Releasing a drag emits ``text_offset_changed(idx, dx, dy)`` so the
        renderer can pick up the new offset on the next Render.
    """

    bbox_delete_requested = Signal(int)
    translation_edit_requested = Signal(int)
    text_offset_changed = Signal(int, int, int)  # idx, offset_x, offset_y

    def __init__(self, parent=None):
        super().__init__(phase="translate", title="2. Translate", parent=parent)
        hint = QLabel("Right-click: delete  |  Double-click: edit text")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        self._toolbar_layout.insertWidget(1, hint)

        self._show_korean = True
        self.toggle_button = QPushButton("Show JA")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setToolTip("Toggle overlay text language")
        self._toolbar_layout.insertWidget(2, self.toggle_button)
        self.toggle_button.toggled.connect(self._on_toggle_lang)

        self._show_cleaned_only = False
        self.hide_text_button = QPushButton("Hide text")
        self.hide_text_button.setCheckable(True)
        self.hide_text_button.setToolTip("Show cleaned image without rendered translations")
        self._toolbar_layout.insertWidget(3, self.hide_text_button)
        self.hide_text_button.toggled.connect(self._on_toggle_hide_text)

        self._move_text = False
        self.move_text_button = QPushButton("Move Text")
        self.move_text_button.setCheckable(True)
        self.move_text_button.setToolTip(
            "Drag dialogue text to reposition without moving its bbox.\n"
            "Press Render after moving to update the final image."
        )
        self.move_text_button.toggled.connect(self._on_toggle_move_text)
        self._toolbar_layout.insertWidget(4, self.move_text_button)

        # Renderer defaults — set by main_window so Move-Text previews match
        # the actual font face and size used at render time.
        self._default_font_path: Optional[str] = None
        self._default_font_pt: int = 36

        self._ctx: Optional[PageContext] = None

    def _on_toggle_lang(self, checked: bool) -> None:
        self._show_korean = not checked
        self.toggle_button.setText("Show KO" if checked else "Show JA")
        if self._ctx:
            self.update_from_context(self._ctx)

    def _on_toggle_hide_text(self, checked: bool) -> None:
        self._show_cleaned_only = checked
        if self._ctx:
            self.update_from_context(self._ctx)

    def _on_toggle_move_text(self, checked: bool) -> None:
        self._move_text = checked
        # Pan must yield to draggable text items in move-text mode.
        self.view.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if checked
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        if self._ctx:
            self.update_from_context(self._ctx)

    def set_render_defaults(self, font_path: Optional[str], default_pt: int) -> None:
        """Tell the tab which font/size the renderer would use by default,
        so Move-Text previews match the actual rendered text."""
        self._default_font_path = font_path or None
        self._default_font_pt = max(6, int(default_pt))
        if self._move_text and self._ctx is not None:
            self.update_from_context(self._ctx)

    def update_from_context(self, ctx: PageContext) -> None:
        self._ctx = ctx

        # Move-Text mode hides the rendered text so the user can place
        # draggable previews on the clean background.
        show_clean = self._show_cleaned_only or self._move_text
        if show_clean and ctx.cleaned is not None:
            base = ctx.cleaned
        elif ctx.final is not None:
            base = ctx.final
        elif ctx.cleaned is not None:
            base = ctx.cleaned
        else:
            base = ctx.source
        self.view.set_image(base)
        self._set_original_image(ctx.source)

        if ctx.translations:
            self._draw_translation_overlays(ctx)
        elif ctx.ocr:
            self._draw_ocr_overlays(ctx)
        elif ctx.bboxes:
            self._draw_bbox_overlays(ctx)

    def _draw_bbox_overlays(self, ctx: PageContext) -> None:
        pen = QPen(QColor(255, 64, 64))
        pen.setWidth(2)
        for i, bbox in enumerate(ctx.bboxes):
            rect = ClickableRectItem(
                QRectF(bbox.x, bbox.y, bbox.w, bbox.h),
                idx=i,
                on_right_click=self.bbox_delete_requested.emit,
            )
            rect.setPen(pen)
            rect.setBrush(QBrush(QColor(255, 64, 64, 30)))
            self.view.add_overlay_item(rect)

    def _draw_ocr_overlays(self, ctx: PageContext) -> None:
        pen = QPen(QColor(64, 128, 255))
        pen.setWidth(2)
        for r in ctx.ocr:
            try:
                idx = next(j for j, b in enumerate(ctx.bboxes) if b is r.bbox)
            except StopIteration:
                idx = -1
            rect = ClickableRectItem(
                QRectF(r.bbox.x, r.bbox.y, r.bbox.w, r.bbox.h),
                idx=idx,
                on_right_click=(
                    self.bbox_delete_requested.emit if idx >= 0 else None
                ),
            )
            rect.setPen(pen)
            rect.setBrush(QBrush(QColor(64, 128, 255, 30)))
            self.view.add_overlay_item(rect)
            self._add_label(r.bbox, r.text_ja)

    def _draw_translation_overlays(self, ctx: PageContext) -> None:
        pen = QPen(QColor(80, 200, 80))
        pen.setWidth(2)
        for i, tr in enumerate(ctx.translations):
            rect = ClickableRectItem(
                QRectF(tr.bbox.x, tr.bbox.y, tr.bbox.w, tr.bbox.h),
                idx=i,
                on_right_click=self.bbox_delete_requested.emit,
                on_double_click=self.translation_edit_requested.emit,
            )
            rect.setPen(pen)
            rect.setBrush(QBrush(QColor(80, 200, 80, 30)))
            self.view.add_overlay_item(rect)

            if self._move_text:
                self._add_draggable_text(i, tr)
            else:
                text = tr.text_ko if self._show_korean else tr.text_ja
                if tr.ignore_boundary:
                    text = "↔ " + text
                self._add_label(tr.bbox, text.replace("\n", " ⏎ "))

    def _add_label(self, bbox, text: str) -> None:
        label = QGraphicsSimpleTextItem(text)
        label.setBrush(QBrush(QColor("white")))
        font = QFont()
        font.setPointSize(10)
        label.setFont(font)
        label.setPos(bbox.x, max(0, bbox.y - 14))
        self.view.add_overlay_item(label)

    def _add_draggable_text(self, idx: int, tr) -> None:
        bbox_center = QPointF(tr.bbox.x + tr.bbox.w / 2.0, tr.bbox.y + tr.bbox.h / 2.0)
        offset = QPointF(int(tr.text_offset_x or 0), int(tr.text_offset_y or 0))
        font_size = (
            int(tr.font_pt)
            if (tr.font_pt and tr.font_pt > 0)
            else int(self._default_font_pt)
        )
        font_path = tr.font_path or self._default_font_path
        item = DraggableTextItem(
            idx=idx,
            bbox_center=bbox_center,
            offset=offset,
            text=tr.text_ko or tr.text_ja or "(empty)",
            on_offset_changed=self.text_offset_changed.emit,
            font_size=max(6, font_size),
            font_path=font_path,
            rotation=int(getattr(tr, "text_rotation", 0) or 0),
        )
        self.view.add_overlay_item(item)


__all__ = [
    "SourceTab",
    "DetectTab",
    "TranslateTab",
    "PhaseTabWidget",
]
