from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig


class SidePanel(QWidget):
    request_run_phase = Signal(str)  # "detect" | "translate"
    request_run_all = Signal()
    request_save = Signal()
    request_save_as = Signal()
    request_render = Signal()
    api_key_clicked = Signal()
    config_changed = Signal()

    AVAILABLE_MODELS = [
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        layout = QVBoxLayout(inner)
        layout.setSpacing(8)

        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_step1_group())
        layout.addWidget(self._build_step2_group())
        layout.addWidget(self._build_step4_group())
        layout.addWidget(self._build_step5_group())
        layout.addWidget(self._build_fonts_group())
        layout.addStretch(1)

    def _build_run_group(self) -> QGroupBox:
        g = QGroupBox("Pipeline")
        v = QVBoxLayout(g)

        top = QHBoxLayout()
        run_all = QPushButton("Run all (Detect → Translate)")
        run_all.clicked.connect(self.request_run_all.emit)
        top.addWidget(run_all, 1)

        save_btn = QPushButton("Save")
        save_btn.setToolTip(
            "Save final image to <source_dir>/translated/<source_name> "
            "(overwrites if exists)"
        )
        save_btn.clicked.connect(self.request_save.emit)
        top.addWidget(save_btn)

        save_as_btn = QPushButton("Save as")
        save_as_btn.setToolTip("Save final image to a custom file name")
        save_as_btn.clicked.connect(self.request_save_as.emit)
        top.addWidget(save_as_btn)
        v.addLayout(top)

        h = QHBoxLayout()
        detect_btn = QPushButton("Detect")
        detect_btn.setToolTip("Re-run detection (mask + bboxes)")
        detect_btn.clicked.connect(lambda: self.request_run_phase.emit("detect"))
        h.addWidget(detect_btn, 1)

        translate_btn = QPushButton("Translate")
        translate_btn.setToolTip("Re-run translation (OCR + Claude + render)")
        translate_btn.clicked.connect(lambda: self.request_run_phase.emit("translate"))
        h.addWidget(translate_btn, 1)

        render_btn = QPushButton("Render")
        render_btn.setToolTip("Re-run only Step 5 (inpaint + render) using current translations")
        render_btn.clicked.connect(self.request_render.emit)
        h.addWidget(render_btn, 1)
        v.addLayout(h)
        return g

    def _build_step1_group(self) -> QGroupBox:
        g = QGroupBox("Step 1 — Mask")
        f = QFormLayout(g)

        self.s1_threshold = QDoubleSpinBox()
        self.s1_threshold.setRange(0.05, 0.95)
        self.s1_threshold.setSingleStep(0.05)
        self.s1_threshold.setValue(self.config.step1.mask_threshold)
        self.s1_threshold.valueChanged.connect(self._sync)
        f.addRow("Mask threshold", self.s1_threshold)

        self.s1_dilate = QSpinBox()
        self.s1_dilate.setRange(0, 30)
        self.s1_dilate.setValue(self.config.step1.mask_dilate_px)
        self.s1_dilate.valueChanged.connect(self._sync)
        f.addRow("Mask dilate (px)", self.s1_dilate)

        self.s1_inpaint_dilate = QSpinBox()
        self.s1_inpaint_dilate.setRange(0, 30)
        self.s1_inpaint_dilate.setValue(self.config.step1.inpaint_dilate_px)
        self.s1_inpaint_dilate.valueChanged.connect(self._sync)
        f.addRow("Inpaint dilate (px)", self.s1_inpaint_dilate)
        return g

    def _build_step2_group(self) -> QGroupBox:
        g = QGroupBox("Step 2 — Bounding Boxes")
        f = QFormLayout(g)

        self.s2_kw = QSpinBox()
        self.s2_kw.setRange(1, 99)
        self.s2_kw.setValue(self.config.step2.kernel_w)
        self.s2_kw.valueChanged.connect(self._sync)
        f.addRow("Kernel W", self.s2_kw)

        self.s2_kh = QSpinBox()
        self.s2_kh.setRange(1, 99)
        self.s2_kh.setValue(self.config.step2.kernel_h)
        self.s2_kh.valueChanged.connect(self._sync)
        f.addRow("Kernel H", self.s2_kh)

        self.s2_iter = QSpinBox()
        self.s2_iter.setRange(1, 10)
        self.s2_iter.setValue(self.config.step2.iterations)
        self.s2_iter.valueChanged.connect(self._sync)
        f.addRow("Iterations", self.s2_iter)

        self.s2_min = QSpinBox()
        self.s2_min.setRange(0, 100000)
        self.s2_min.setValue(self.config.step2.min_area)
        self.s2_min.valueChanged.connect(self._sync)
        f.addRow("Min area (px)", self.s2_min)

        self.s2_max_ratio = QDoubleSpinBox()
        self.s2_max_ratio.setRange(0.01, 1.0)
        self.s2_max_ratio.setSingleStep(0.05)
        self.s2_max_ratio.setValue(self.config.step2.max_area_ratio)
        self.s2_max_ratio.valueChanged.connect(self._sync)
        f.addRow("Max area ratio", self.s2_max_ratio)
        return g

    def _build_step4_group(self) -> QGroupBox:
        g = QGroupBox("Step 4 — Translation")
        v = QVBoxLayout(g)

        self.s4_skip = QCheckBox(
            "Skip translation (use original Japanese OCR text as the dialogue)"
        )
        self.s4_skip.setChecked(bool(self.config.step4.skip_translation))
        self.s4_skip.toggled.connect(self._sync)
        v.addWidget(self.s4_skip)

        f = QFormLayout()
        self.s4_model = QComboBox()
        self.s4_model.addItems(self.AVAILABLE_MODELS)
        idx = self.s4_model.findText(self.config.step4.model)
        self.s4_model.setCurrentIndex(idx if idx >= 0 else 0)
        self.s4_model.currentTextChanged.connect(self._sync)
        f.addRow("Model", self.s4_model)

        self.s4_max_tokens = QSpinBox()
        self.s4_max_tokens.setRange(256, 16384)
        self.s4_max_tokens.setValue(self.config.step4.max_tokens)
        self.s4_max_tokens.valueChanged.connect(self._sync)
        f.addRow("Max tokens", self.s4_max_tokens)

        self.s4_style = QLineEdit(self.config.step4.style_notes)
        self.s4_style.editingFinished.connect(self._sync)
        f.addRow("Style notes", self.s4_style)
        v.addLayout(f)

        v.addWidget(QLabel("Glossary (사이타마=사이타마 형태)"))
        self.s4_glossary = QPlainTextEdit(self.config.step4.glossary)
        self.s4_glossary.setMaximumHeight(120)
        self.s4_glossary.textChanged.connect(self._sync)
        v.addWidget(self.s4_glossary)

        api_row = QHBoxLayout()
        self.api_status = QLabel("API key: (not set)")
        api_row.addWidget(self.api_status, 1)
        btn = QPushButton("Set…")
        btn.clicked.connect(self.api_key_clicked.emit)
        api_row.addWidget(btn)
        v.addLayout(api_row)
        return g

    def _build_step5_group(self) -> QGroupBox:
        g = QGroupBox("Step 5 — Render")
        v = QVBoxLayout(g)
        f = QFormLayout()

        font_row = QHBoxLayout()
        self.s5_font_path = QLineEdit(self.config.step5.font_path)
        self.s5_font_path.editingFinished.connect(self._sync)
        font_row.addWidget(self.s5_font_path, 1)
        font_btn = QPushButton("…")
        font_btn.setMaximumWidth(28)
        font_btn.clicked.connect(self._pick_font)
        font_row.addWidget(font_btn)
        f.addRow("Font (TTF/OTF)", self._wrap(font_row))

        self.s5_min_pt = QSpinBox()
        self.s5_min_pt.setRange(6, 200)
        self.s5_min_pt.setValue(self.config.step5.min_pt)
        self.s5_min_pt.valueChanged.connect(self._sync)
        f.addRow("Min font pt", self.s5_min_pt)

        self.s5_max_pt = QSpinBox()
        self.s5_max_pt.setRange(6, 200)
        self.s5_max_pt.setValue(self.config.step5.max_pt)
        self.s5_max_pt.valueChanged.connect(self._sync)
        f.addRow("Max font pt", self.s5_max_pt)

        self.s5_outside_pt = QSpinBox()
        self.s5_outside_pt.setRange(6, 200)
        self.s5_outside_pt.setValue(self.config.step5.outside_pt)
        self.s5_outside_pt.valueChanged.connect(self._sync)
        self.s5_outside_pt.setToolTip("Fixed font size used when 'Ignore Boundary' is on")
        f.addRow("Outside-bbox pt", self.s5_outside_pt)

        self.s5_padding = QSpinBox()
        self.s5_padding.setRange(0, 50)
        self.s5_padding.setValue(self.config.step5.padding)
        self.s5_padding.valueChanged.connect(self._sync)
        f.addRow("Padding (px)", self.s5_padding)

        self.s5_stroke = QSpinBox()
        self.s5_stroke.setRange(0, 10)
        self.s5_stroke.setValue(self.config.step5.stroke_px)
        self.s5_stroke.valueChanged.connect(self._sync)
        f.addRow("Stroke (px)", self.s5_stroke)

        self.s5_spacing = QDoubleSpinBox()
        self.s5_spacing.setRange(0.8, 3.0)
        self.s5_spacing.setSingleStep(0.05)
        self.s5_spacing.setValue(self.config.step5.line_spacing)
        self.s5_spacing.valueChanged.connect(self._sync)
        f.addRow("Line spacing", self.s5_spacing)
        v.addLayout(f)
        return g

    def _build_fonts_group(self) -> QGroupBox:
        g = QGroupBox("Fonts library")
        v = QVBoxLayout(g)

        hint = QLabel("Available in the Edit Translation dialog's font dropdown.")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        v.addWidget(hint)

        self.font_list = QListWidget()
        self.font_list.setMaximumHeight(140)
        for p in self.config.fonts:
            self._add_font_item(p)
        v.addWidget(self.font_list)

        h = QHBoxLayout()
        add_btn = QPushButton("Add font…")
        add_btn.clicked.connect(self._on_add_font)
        h.addWidget(add_btn, 1)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._on_remove_font)
        h.addWidget(rm_btn)
        v.addLayout(h)
        return g

    def _add_font_item(self, path: str) -> None:
        from pathlib import Path as _P

        item = QListWidgetItem(_P(path).name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.font_list.addItem(item)

    def _on_add_font(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add font to library", "", "Fonts (*.ttf *.otf *.ttc)"
        )
        if not path:
            return
        if path in self.config.fonts:
            return
        self.config.fonts.append(path)
        self._add_font_item(path)
        self.config_changed.emit()

    def _on_remove_font(self) -> None:
        row = self.font_list.currentRow()
        if row < 0:
            return
        item = self.font_list.takeItem(row)
        path = item.data(Qt.ItemDataRole.UserRole)
        if path in self.config.fonts:
            self.config.fonts.remove(path)
        self.config_changed.emit()

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _pick_font(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Korean font", "", "Fonts (*.ttf *.otf *.ttc)"
        )
        if path:
            self.s5_font_path.setText(path)
            self._sync()

    def set_api_key_status(self, present: bool, source: Optional[str] = None) -> None:
        if present:
            src = f" ({source})" if source else ""
            self.api_status.setText(f"API key: ✓ set{src}")
            self.api_status.setStyleSheet("color: #2e7d32;")
        else:
            self.api_status.setText("API key: (not set)")
            self.api_status.setStyleSheet("color: #c62828;")

    def _sync(self) -> None:
        c = self.config
        c.step1.mask_threshold = float(self.s1_threshold.value())
        c.step1.mask_dilate_px = int(self.s1_dilate.value())
        c.step1.inpaint_dilate_px = int(self.s1_inpaint_dilate.value())
        c.step2.kernel_w = int(self.s2_kw.value())
        c.step2.kernel_h = int(self.s2_kh.value())
        c.step2.iterations = int(self.s2_iter.value())
        c.step2.min_area = int(self.s2_min.value())
        c.step2.max_area_ratio = float(self.s2_max_ratio.value())
        c.step4.model = self.s4_model.currentText()
        c.step4.max_tokens = int(self.s4_max_tokens.value())
        c.step4.style_notes = self.s4_style.text()
        c.step4.glossary = self.s4_glossary.toPlainText()
        c.step4.skip_translation = bool(self.s4_skip.isChecked())
        c.step5.font_path = self.s5_font_path.text()
        c.step5.min_pt = int(self.s5_min_pt.value())
        c.step5.outside_pt = int(self.s5_outside_pt.value())
        c.step5.max_pt = int(self.s5_max_pt.value())
        c.step5.padding = int(self.s5_padding.value())
        c.step5.stroke_px = int(self.s5_stroke.value())
        c.step5.line_spacing = float(self.s5_spacing.value())
        self.config_changed.emit()
