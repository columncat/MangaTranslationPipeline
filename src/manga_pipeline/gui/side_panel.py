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
from ..i18n import tr
from ..utils.fonts import list_bundled_fonts


class SidePanel(QWidget):
    request_run_phase = Signal(str)  # "detect" | "translate"
    request_run_all = Signal()
    request_save = Signal()
    request_save_as = Signal()
    request_render = Signal()
    api_key_clicked = Signal()
    config_changed = Signal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        # Set of currently-known font absolute paths (bundled + external),
        # rebuilt on every refresh.
        self._known_fonts: list[str] = []

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

        # Initial scan so the Step5 font combo is populated.
        self._refresh_fonts(emit=False)

    # ---- public ----

    @property
    def known_fonts(self) -> list[str]:
        """Absolute paths of every font currently in the library."""
        return list(self._known_fonts)

    # ---- groups ----

    def _build_run_group(self) -> QGroupBox:
        g = QGroupBox(tr("side.pipeline"))
        v = QVBoxLayout(g)

        top = QHBoxLayout()
        run_all = QPushButton(tr("side.run_all_btn"))
        run_all.clicked.connect(self.request_run_all.emit)
        top.addWidget(run_all, 1)

        save_btn = QPushButton(tr("side.save_btn"))
        save_btn.setToolTip(tr("side.save_tip"))
        save_btn.clicked.connect(self.request_save.emit)
        top.addWidget(save_btn)

        save_as_btn = QPushButton(tr("side.save_as_btn"))
        save_as_btn.setToolTip(tr("side.save_as_tip"))
        save_as_btn.clicked.connect(self.request_save_as.emit)
        top.addWidget(save_as_btn)
        v.addLayout(top)

        h = QHBoxLayout()
        detect_btn = QPushButton(tr("side.detect_btn"))
        detect_btn.setToolTip(tr("side.detect_tip"))
        detect_btn.clicked.connect(lambda: self.request_run_phase.emit("detect"))
        h.addWidget(detect_btn, 1)

        translate_btn = QPushButton(tr("side.translate_btn"))
        translate_btn.setToolTip(tr("side.translate_tip"))
        translate_btn.clicked.connect(lambda: self.request_run_phase.emit("translate"))
        h.addWidget(translate_btn, 1)

        render_btn = QPushButton(tr("side.render_btn"))
        render_btn.setToolTip(tr("side.render_tip"))
        render_btn.clicked.connect(self.request_render.emit)
        h.addWidget(render_btn, 1)
        v.addLayout(h)
        return g

    def _build_step1_group(self) -> QGroupBox:
        g = QGroupBox(tr("side.step1"))
        f = QFormLayout(g)

        self.s1_threshold = QDoubleSpinBox()
        self.s1_threshold.setRange(0.05, 0.95)
        self.s1_threshold.setSingleStep(0.05)
        self.s1_threshold.setValue(self.config.step1.mask_threshold)
        self.s1_threshold.valueChanged.connect(self._sync)
        f.addRow(tr("side.s1.threshold"), self.s1_threshold)

        self.s1_dilate = QSpinBox()
        self.s1_dilate.setRange(0, 30)
        self.s1_dilate.setValue(self.config.step1.mask_dilate_px)
        self.s1_dilate.valueChanged.connect(self._sync)
        f.addRow(tr("side.s1.dilate"), self.s1_dilate)

        self.s1_inpaint_dilate = QSpinBox()
        self.s1_inpaint_dilate.setRange(0, 30)
        self.s1_inpaint_dilate.setValue(self.config.step1.inpaint_dilate_px)
        self.s1_inpaint_dilate.valueChanged.connect(self._sync)
        f.addRow(tr("side.s1.inpaint_dilate"), self.s1_inpaint_dilate)
        return g

    def _build_step2_group(self) -> QGroupBox:
        g = QGroupBox(tr("side.step2"))
        f = QFormLayout(g)

        self.s2_kw = QSpinBox()
        self.s2_kw.setRange(1, 99)
        self.s2_kw.setValue(self.config.step2.kernel_w)
        self.s2_kw.valueChanged.connect(self._sync)
        f.addRow(tr("side.s2.kw"), self.s2_kw)

        self.s2_kh = QSpinBox()
        self.s2_kh.setRange(1, 99)
        self.s2_kh.setValue(self.config.step2.kernel_h)
        self.s2_kh.valueChanged.connect(self._sync)
        f.addRow(tr("side.s2.kh"), self.s2_kh)

        self.s2_iter = QSpinBox()
        self.s2_iter.setRange(1, 10)
        self.s2_iter.setValue(self.config.step2.iterations)
        self.s2_iter.valueChanged.connect(self._sync)
        f.addRow(tr("side.s2.iter"), self.s2_iter)

        self.s2_min = QSpinBox()
        self.s2_min.setRange(0, 100000)
        self.s2_min.setValue(self.config.step2.min_area)
        self.s2_min.valueChanged.connect(self._sync)
        f.addRow(tr("side.s2.min"), self.s2_min)

        self.s2_max_ratio = QDoubleSpinBox()
        self.s2_max_ratio.setRange(0.01, 1.0)
        self.s2_max_ratio.setSingleStep(0.05)
        self.s2_max_ratio.setValue(self.config.step2.max_area_ratio)
        self.s2_max_ratio.valueChanged.connect(self._sync)
        f.addRow(tr("side.s2.max_ratio"), self.s2_max_ratio)
        return g

    def _build_step4_group(self) -> QGroupBox:
        from ..ai import AIProvider, PROVIDERS

        # Reasonable model presets per provider — the user can still type
        # any model name into the editable combo box.
        self._provider_default_models = {
            AIProvider.ANTHROPIC: [
                "claude-sonnet-4-6",
                "claude-opus-4-7",
                "claude-haiku-4-5-20251001",
            ],
            AIProvider.OPENAI_COMPAT: [
                "gpt-4o-mini",
                "gpt-4o",
            ],
            AIProvider.GEMINI: [
                "gemini-2.0-flash",
                "gemini-2.5-pro",
            ],
            AIProvider.OLLAMA: [
                "qwen3:8b",
                "qwen3:14b",
                "llama3.1:8b",
            ],
            AIProvider.LLAMACPP: [],  # uses model_path instead
        }
        self._all_providers = list(PROVIDERS)

        g = QGroupBox(tr("side.step4"))
        v = QVBoxLayout(g)

        self.s4_skip = QCheckBox(tr("side.s4.skip"))
        self.s4_skip.setChecked(bool(self.config.step4.skip_translation))
        self.s4_skip.toggled.connect(self._sync)
        v.addWidget(self.s4_skip)

        f = QFormLayout()

        # Provider picker
        self.s4_provider = QComboBox()
        for p in self._all_providers:
            self.s4_provider.addItem(tr(f"side.s4.provider.{p}"), p)
        cur_provider = self.config.step4.provider or AIProvider.ANTHROPIC
        idx = self.s4_provider.findData(cur_provider)
        self.s4_provider.setCurrentIndex(idx if idx >= 0 else 0)
        self.s4_provider.currentIndexChanged.connect(self._on_provider_changed)
        f.addRow(tr("side.s4.provider"), self.s4_provider)

        self.s4_model = QComboBox()
        self.s4_model.setEditable(True)
        self.s4_model.editTextChanged.connect(self._sync)
        f.addRow(tr("side.s4.model"), self.s4_model)

        # Backend-specific fields. Always present in the layout but hidden
        # when irrelevant so the user only sees what matters.
        self.s4_base_url = QLineEdit(self.config.step4.base_url or "")
        self.s4_base_url.setPlaceholderText("http://localhost:11434")
        self.s4_base_url.editingFinished.connect(self._sync)
        f.addRow(tr("side.s4.base_url"), self.s4_base_url)

        self.s4_model_path = QLineEdit(self.config.step4.model_path or "")
        self.s4_model_path.setPlaceholderText("/path/to/model.gguf")
        self.s4_model_path.editingFinished.connect(self._sync)
        model_path_row = QHBoxLayout()
        model_path_row.addWidget(self.s4_model_path, 1)
        browse_btn = QPushButton("…")
        browse_btn.setMaximumWidth(28)
        browse_btn.clicked.connect(self._pick_gguf)
        model_path_row.addWidget(browse_btn)
        model_path_wrap = QWidget()
        model_path_wrap.setLayout(model_path_row)
        f.addRow(tr("side.s4.model_path"), model_path_wrap)
        self._s4_model_path_wrap = model_path_wrap

        self.s4_max_tokens = QSpinBox()
        self.s4_max_tokens.setRange(256, 16384)
        self.s4_max_tokens.setValue(self.config.step4.max_tokens)
        self.s4_max_tokens.valueChanged.connect(self._sync)
        f.addRow(tr("side.s4.max_tokens"), self.s4_max_tokens)

        self.s4_style = QLineEdit(self.config.step4.style_notes)
        self.s4_style.editingFinished.connect(self._sync)
        f.addRow(tr("side.s4.style"), self.s4_style)
        v.addLayout(f)
        # Save references to the form rows so we can hide / show them.
        self._s4_form = f

        v.addWidget(QLabel(tr("side.s4.glossary_label")))
        self.s4_glossary = QPlainTextEdit(self.config.step4.glossary)
        self.s4_glossary.setMaximumHeight(120)
        self.s4_glossary.textChanged.connect(self._sync)
        v.addWidget(self.s4_glossary)

        api_row = QHBoxLayout()
        self.api_status = QLabel(tr("apikey.status.unset"))
        api_row.addWidget(self.api_status, 1)
        btn = QPushButton(tr("side.api.set"))
        btn.clicked.connect(self.api_key_clicked.emit)
        api_row.addWidget(btn)
        v.addLayout(api_row)

        # Apply provider-conditional visibility now that all widgets exist.
        self._on_provider_changed(self.s4_provider.currentIndex())
        return g

    def _pick_gguf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("side.s4.pick_gguf"), "", "GGUF (*.gguf)"
        )
        if path:
            self.s4_model_path.setText(path)
            self._sync()

    def _on_provider_changed(self, _idx: int) -> None:
        """Refresh the model dropdown choices and hide irrelevant fields."""
        from ..ai import AIProvider

        provider = self.s4_provider.currentData() or AIProvider.ANTHROPIC

        # Repopulate model presets, preserving any custom value the user
        # may have typed for the previous provider.
        prev_text = self.s4_model.currentText()
        self.s4_model.blockSignals(True)
        self.s4_model.clear()
        for m in self._provider_default_models.get(provider, []):
            self.s4_model.addItem(m)
        # Try to keep the user's currently-saved model for this provider;
        # fall back to the first preset.
        if (
            self.config.step4.provider == provider
            and self.config.step4.model
        ):
            target = self.config.step4.model
        elif prev_text and prev_text in self._provider_default_models.get(provider, []):
            target = prev_text
        elif self._provider_default_models.get(provider):
            target = self._provider_default_models[provider][0]
        else:
            target = ""
        i = self.s4_model.findText(target)
        if i >= 0:
            self.s4_model.setCurrentIndex(i)
        elif target:
            self.s4_model.setCurrentText(target)
        self.s4_model.blockSignals(False)

        # Field visibility per provider:
        #   anthropic     → model only (api key handled by keyring)
        #   openai-compat → model + base_url + api key
        #   gemini        → model + api key
        #   ollama        → model + base_url
        #   llama.cpp     → model_path (model name field hidden)
        wants_base_url = provider in (
            AIProvider.OPENAI_COMPAT,
            AIProvider.OLLAMA,
        )
        wants_model_path = provider == AIProvider.LLAMACPP
        wants_model_name = provider != AIProvider.LLAMACPP

        self._set_form_row_visible(self.s4_model, wants_model_name)
        self._set_form_row_visible(self.s4_base_url, wants_base_url)
        self._set_form_row_visible(self._s4_model_path_wrap, wants_model_path)

        self._sync()

    def _set_form_row_visible(self, widget: QWidget, visible: bool) -> None:
        """Hide both the field and its label in the QFormLayout."""
        widget.setVisible(visible)
        label = self._s4_form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)

    def _build_step5_group(self) -> QGroupBox:
        g = QGroupBox(tr("side.step5"))
        v = QVBoxLayout(g)
        f = QFormLayout()

        self.s5_font_combo = QComboBox()
        self.s5_font_combo.setToolTip(tr("side.s5.font_tip"))
        self.s5_font_combo.currentIndexChanged.connect(self._on_step5_font_changed)
        f.addRow(tr("side.s5.font"), self.s5_font_combo)

        self.s5_outside_pt = QSpinBox()
        self.s5_outside_pt.setRange(6, 200)
        self.s5_outside_pt.setValue(self.config.step5.outside_pt)
        self.s5_outside_pt.valueChanged.connect(self._sync)
        f.addRow(tr("side.s5.outside_pt"), self.s5_outside_pt)

        self.s5_stroke = QSpinBox()
        self.s5_stroke.setRange(0, 10)
        self.s5_stroke.setValue(self.config.step5.stroke_px)
        self.s5_stroke.valueChanged.connect(self._sync)
        f.addRow(tr("side.s5.stroke"), self.s5_stroke)

        self.s5_spacing = QDoubleSpinBox()
        self.s5_spacing.setRange(0.8, 3.0)
        self.s5_spacing.setSingleStep(0.05)
        self.s5_spacing.setValue(self.config.step5.line_spacing)
        self.s5_spacing.valueChanged.connect(self._sync)
        f.addRow(tr("side.s5.spacing"), self.s5_spacing)
        v.addLayout(f)
        return g

    def _build_fonts_group(self) -> QGroupBox:
        g = QGroupBox(tr("side.fonts"))
        v = QVBoxLayout(g)

        hint = QLabel(tr("side.fonts.hint"))
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.font_list = QListWidget()
        self.font_list.setMaximumHeight(160)
        v.addWidget(self.font_list)

        h = QHBoxLayout()
        refresh_btn = QPushButton(tr("side.fonts.refresh"))
        refresh_btn.clicked.connect(lambda: self._refresh_fonts(emit=True))
        h.addWidget(refresh_btn, 1)
        add_btn = QPushButton(tr("side.fonts.add"))
        add_btn.clicked.connect(self._on_add_external_font)
        h.addWidget(add_btn, 1)
        rm_btn = QPushButton(tr("side.fonts.remove"))
        rm_btn.clicked.connect(self._on_remove_external_font)
        h.addWidget(rm_btn)
        v.addLayout(h)
        return g

    # ---- fonts library ----

    def _refresh_fonts(self, *, emit: bool) -> None:
        """Re-scan ``fonts/`` and merge with external entries.

        Auto-discovered entries (inside ``fonts/``) are unremovable from the
        UI; external entries (added via the file picker) carry an asterisk
        marker and can be removed via the ``Remove external`` button.
        """
        bundled = [str(p) for p in list_bundled_fonts()]
        external = list(self.config.external_fonts or [])

        # Drop external entries that no longer exist on disk.
        external = [p for p in external if Path(p).exists()]
        # Drop external entries that have since been moved into fonts/ — they
        # would otherwise show up twice.
        bundled_set = {p.lower() for p in bundled}
        external = [p for p in external if p.lower() not in bundled_set]
        self.config.external_fonts = external

        merged = bundled + external
        self._known_fonts = merged

        self.font_list.clear()
        for p in bundled:
            item = QListWidgetItem(Path(p).name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self.font_list.addItem(item)
        for p in external:
            item = QListWidgetItem(f"{Path(p).name}  *")
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(f"(external) {p}")
            self.font_list.addItem(item)

        # Repopulate the Step5 font combo, preserving the current selection
        # (or the value already in config) if still available.
        prev = self.config.step5.font_path or self.s5_font_combo.currentData() or ""
        self.s5_font_combo.blockSignals(True)
        self.s5_font_combo.clear()
        self.s5_font_combo.addItem(tr("edit.font_default"), "")
        for p in merged:
            self.s5_font_combo.addItem(Path(p).name, p)
        if prev:
            for i in range(self.s5_font_combo.count()):
                if self.s5_font_combo.itemData(i) == prev:
                    self.s5_font_combo.setCurrentIndex(i)
                    break
            else:
                self.s5_font_combo.addItem(
                    tr("edit.font_orphan", name=Path(prev).name), prev
                )
                self.s5_font_combo.setCurrentIndex(self.s5_font_combo.count() - 1)
        self.s5_font_combo.blockSignals(False)

        if emit:
            self._sync()

    def _on_step5_font_changed(self, _idx: int) -> None:
        self._sync()

    def _on_add_external_font(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("side.fonts.add"), "", "Fonts (*.ttf *.otf *.ttc)"
        )
        if not path:
            return
        if path in (self.config.external_fonts or []):
            return
        # Skip if it's already inside fonts/ (would duplicate).
        bundled = {str(p).lower() for p in list_bundled_fonts()}
        if path.lower() in bundled:
            return
        self.config.external_fonts.append(path)
        self._refresh_fonts(emit=True)

    def _on_remove_external_font(self) -> None:
        item = self.font_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path not in (self.config.external_fonts or []):
            # bundled entries cannot be removed via the UI
            return
        self.config.external_fonts.remove(path)
        self._refresh_fonts(emit=True)

    # ---- api status ----

    def set_api_key_status(self, present: bool, source: Optional[str] = None) -> None:
        if present:
            src = f" ({source})" if source else ""
            self.api_status.setText(tr("apikey.status.set", src=src))
            self.api_status.setStyleSheet("color: #2e7d32;")
        else:
            self.api_status.setText(tr("apikey.status.unset"))
            self.api_status.setStyleSheet("color: #c62828;")

    # ---- sync ----

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
        c.step4.provider = self.s4_provider.currentData() or "anthropic"
        c.step4.model = self.s4_model.currentText()
        c.step4.base_url = self.s4_base_url.text().strip() or None
        c.step4.model_path = self.s4_model_path.text().strip() or None
        c.step4.max_tokens = int(self.s4_max_tokens.value())
        c.step4.style_notes = self.s4_style.text()
        c.step4.glossary = self.s4_glossary.toPlainText()
        c.step4.skip_translation = bool(self.s4_skip.isChecked())
        c.step5.font_path = str(self.s5_font_combo.currentData() or "")
        c.step5.outside_pt = int(self.s5_outside_pt.value())
        c.step5.stroke_px = int(self.s5_stroke.value())
        c.step5.line_spacing = float(self.s5_spacing.value())
        self.config_changed.emit()
