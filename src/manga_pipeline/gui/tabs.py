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

from ..i18n import tr
from ..models import PageContext
from .image_view import ZoomPanGraphicsView
from .items import ClickableRectItem, DraggableTextItem, EditableBBoxItem


class PhaseTabWidget(QWidget):
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

        self.status_label = QLabel(tr("tabs.status_not_run"))
        self.status_label.setStyleSheet("color: gray;")
        self._toolbar_layout.addWidget(self.status_label)

        self.view_original_box = QCheckBox(tr("tabs.view_original"))
        self.view_original_box.setToolTip(tr("tabs.view_original_tip"))
        self.view_original_box.toggled.connect(self._on_toggle_view_original)
        self._toolbar_layout.addWidget(self.view_original_box)

        self.fit_button = QPushButton(tr("tabs.fit"))
        self.fit_button.setMaximumWidth(60)
        self._toolbar_layout.addWidget(self.fit_button)

        # Localised phase label for the re-run button. Falls back to the raw
        # phase name (e.g. "source") for tabs that don't hide the button.
        phase_label = tr(f"tabs.phase.{phase}") if phase in ("detect", "translate") else phase
        self.rerun_button = QPushButton(tr("tabs.rerun", phase=phase_label))
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

        self._latest_source = None

    def _on_toggle_view_original(self, checked: bool) -> None:
        self.original_view.setVisible(checked)
        if checked:
            total = max(1, self._splitter.size().width() or self.width())
            self._splitter.setSizes([total // 2, total - total // 2])
            if self._latest_source is not None:
                self.original_view.set_image(self._latest_source)
        else:
            self._splitter.setSizes([0, 1000])
        QTimer.singleShot(0, self._fit_visible_views)

    def _set_original_image(self, source) -> None:
        self._latest_source = source
        if self.view_original_box.isChecked():
            self.original_view.set_image(source)

    def _fit_visible_views(self) -> None:
        if self.view_original_box.isChecked():
            self.original_view.fit_to_view()
        self.view.fit_to_view()

    def showEvent(self, event):  # type: ignore[override]
        super().showEvent(event)
        QTimer.singleShot(0, self._fit_visible_views)

    def set_status(self, text: str, ok: bool = True) -> None:
        color = "#2e7d32" if ok else "#c62828"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)

    def update_from_context(self, ctx: PageContext) -> None:
        raise NotImplementedError


class SourceTab(PhaseTabWidget):
    def __init__(self, parent=None):
        super().__init__(phase="source", title=tr("tabs.title.original"), parent=parent)
        self.rerun_button.hide()
        self.view_original_box.hide()

    def update_from_context(self, ctx: PageContext) -> None:
        self.view.set_image(ctx.source)
        self._set_original_image(ctx.source)


class DetectTab(PhaseTabWidget):
    bbox_delete_requested = Signal(int)
    bbox_geometry_changed = Signal(int, int, int, int, int)
    bbox_add_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(phase="detect", title=tr("tabs.title.detect"), parent=parent)
        hint = QLabel(tr("tabs.detect.hint"))
        hint.setStyleSheet("color: #666; font-size: 11px;")
        self._toolbar_layout.insertWidget(1, hint)

        self.add_button = QPushButton(tr("tabs.detect.add"))
        self.add_button.setToolTip(tr("tabs.detect.add_tip"))
        self.add_button.clicked.connect(self.bbox_add_requested.emit)
        self._toolbar_layout.insertWidget(2, self.add_button)

        self._edit_enabled = False
        self.edit_button = QPushButton(tr("tabs.detect.edit"))
        self.edit_button.setCheckable(True)
        self.edit_button.setToolTip(tr("tabs.detect.edit_tip"))
        self.edit_button.toggled.connect(self._on_toggle_edit)
        self._toolbar_layout.insertWidget(3, self.edit_button)

        self._ctx: Optional[PageContext] = None

    def _on_toggle_edit(self, checked: bool) -> None:
        self._edit_enabled = checked
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
    bbox_delete_requested = Signal(int)
    translation_edit_requested = Signal(int)
    text_offset_changed = Signal(int, int, int)
    # Fired when the user disables Move-Text mode so the main window can
    # re-run Step 5 to bake in any text positions that were dragged while
    # the mode was active.
    render_requested = Signal()
    # User clicked "Add text" — main window should create a new bbox
    # plus a TranslationResult and open the edit dialog.
    add_text_requested = Signal()
    # Mask edit mode: double-clicking a translation routes to this
    # signal instead of translation_edit_requested so the main window
    # can open the mask editor on the bbox crop.
    mask_edit_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(phase="translate", title=tr("tabs.title.translate"), parent=parent)
        hint = QLabel(tr("tabs.translate.hint"))
        hint.setStyleSheet("color: #666; font-size: 11px;")
        self._toolbar_layout.insertWidget(1, hint)

        self._show_korean = True
        self.toggle_button = QPushButton(tr("tabs.translate.show_ja"))
        self.toggle_button.setCheckable(True)
        self.toggle_button.setToolTip(tr("tabs.translate.lang_tip"))
        self._toolbar_layout.insertWidget(2, self.toggle_button)
        self.toggle_button.toggled.connect(self._on_toggle_lang)

        self._show_cleaned_only = False
        self.hide_text_button = QPushButton(tr("tabs.translate.hide_text"))
        self.hide_text_button.setCheckable(True)
        self.hide_text_button.setToolTip(tr("tabs.translate.hide_tip"))
        self._toolbar_layout.insertWidget(3, self.hide_text_button)
        self.hide_text_button.toggled.connect(self._on_toggle_hide_text)

        self._move_text = False
        self.move_text_button = QPushButton(tr("tabs.translate.move_text"))
        self.move_text_button.setCheckable(True)
        self.move_text_button.setToolTip(tr("tabs.translate.move_tip"))
        self.move_text_button.toggled.connect(self._on_toggle_move_text)
        self._toolbar_layout.insertWidget(4, self.move_text_button)

        # New: insert a free-floating text bubble that wasn't auto-detected.
        self.add_text_button = QPushButton(tr("tabs.translate.add_text"))
        self.add_text_button.setToolTip(tr("tabs.translate.add_text_tip"))
        self.add_text_button.clicked.connect(self.add_text_requested.emit)
        self._toolbar_layout.insertWidget(5, self.add_text_button)

        # Mask-edit toggle: when on, a double-click on a translation opens
        # the per-bbox mask editor instead of the text editor.
        self._mask_edit_mode = False
        self.mask_edit_button = QPushButton(tr("tabs.translate.mask_edit"))
        self.mask_edit_button.setCheckable(True)
        self.mask_edit_button.setToolTip(tr("tabs.translate.mask_edit_tip"))
        self.mask_edit_button.toggled.connect(self._on_toggle_mask_edit)
        self._toolbar_layout.insertWidget(6, self.mask_edit_button)

        self._default_font_path: Optional[str] = None
        self._default_font_pt: int = 36

        self._ctx: Optional[PageContext] = None

    def _on_toggle_lang(self, checked: bool) -> None:
        self._show_korean = not checked
        self.toggle_button.setText(
            tr("tabs.translate.show_ko") if checked else tr("tabs.translate.show_ja")
        )
        if self._ctx:
            self.update_from_context(self._ctx)

    def _on_toggle_hide_text(self, checked: bool) -> None:
        self._show_cleaned_only = checked
        if self._ctx:
            self.update_from_context(self._ctx)

    def _on_toggle_mask_edit(self, checked: bool) -> None:
        self._mask_edit_mode = checked
        # Mask editing happens in a modal dialog so we don't need to
        # change the view drag mode. The double-click router takes care
        # of the rest. Repaint so any visual hint can update.
        if self._ctx:
            self.update_from_context(self._ctx)

    def _on_toggle_move_text(self, checked: bool) -> None:
        was_active = self._move_text
        self._move_text = checked
        self.view.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if checked
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        if self._ctx:
            self.update_from_context(self._ctx)
        # Disabling the mode → kick off a render so the user immediately
        # sees the text in its dragged-to position baked into the final
        # image, instead of having to press Render manually. Skipped if
        # there are no translations (renderer would error anyway) or if
        # the toggle didn't actually change state.
        if was_active and not checked and self._ctx and self._ctx.translations:
            self.render_requested.emit()

    def set_render_defaults(self, font_path: Optional[str], default_pt: int) -> None:
        self._default_font_path = font_path or None
        self._default_font_pt = max(6, int(default_pt))
        if self._move_text and self._ctx is not None:
            self.update_from_context(self._ctx)

    def update_from_context(self, ctx: PageContext) -> None:
        self._ctx = ctx

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
        # In mask-edit mode the overlay turns purple to make the changed
        # double-click target visible at a glance.
        if self._mask_edit_mode:
            pen = QPen(QColor(160, 80, 200))
            brush = QBrush(QColor(160, 80, 200, 60))
            dbl_target = self.mask_edit_requested.emit
        else:
            pen = QPen(QColor(80, 200, 80))
            brush = QBrush(QColor(80, 200, 80, 30))
            dbl_target = self.translation_edit_requested.emit
        pen.setWidth(2)
        for i, tr_item in enumerate(ctx.translations):
            rect = ClickableRectItem(
                QRectF(tr_item.bbox.x, tr_item.bbox.y, tr_item.bbox.w, tr_item.bbox.h),
                idx=i,
                on_right_click=self.bbox_delete_requested.emit,
                on_double_click=dbl_target,
            )
            rect.setPen(pen)
            rect.setBrush(brush)
            self.view.add_overlay_item(rect)

            if self._move_text:
                self._add_draggable_text(i, tr_item)
            else:
                text = tr_item.text_ko if self._show_korean else tr_item.text_ja
                # Translations are always center-on-bbox, so prefix all of them.
                text = "↔ " + text
                self._add_label(tr_item.bbox, text.replace("\n", " ⏎ "))

    def _add_label(self, bbox, text: str) -> None:
        label = QGraphicsSimpleTextItem(text)
        label.setBrush(QBrush(QColor("white")))
        font = QFont()
        font.setPointSize(10)
        label.setFont(font)
        label.setPos(bbox.x, max(0, bbox.y - 14))
        self.view.add_overlay_item(label)

    def _add_draggable_text(self, idx: int, tr_item) -> None:
        bbox_center = QPointF(
            tr_item.bbox.x + tr_item.bbox.w / 2.0,
            tr_item.bbox.y + tr_item.bbox.h / 2.0,
        )
        offset = QPointF(int(tr_item.text_offset_x or 0), int(tr_item.text_offset_y or 0))
        font_size = (
            int(tr_item.font_pt)
            if (tr_item.font_pt and tr_item.font_pt > 0)
            else int(self._default_font_pt)
        )
        font_path = tr_item.font_path or self._default_font_path
        item = DraggableTextItem(
            idx=idx,
            bbox_center=bbox_center,
            offset=offset,
            text=tr_item.text_ko or tr_item.text_ja or "(empty)",
            on_offset_changed=self.text_offset_changed.emit,
            font_size=max(6, font_size),
            font_path=font_path,
            rotation=int(getattr(tr_item, "text_rotation", 0) or 0),
        )
        self.view.add_overlay_item(item)


__all__ = [
    "SourceTab",
    "DetectTab",
    "TranslateTab",
    "PhaseTabWidget",
]
