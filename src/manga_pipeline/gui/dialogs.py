from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr


class _ColorSwatchButton(QPushButton):
    """Small button that acts as a colour swatch and opens a colour picker.

    The current colour is stored as an ``(r, g, b)`` tuple; ``None`` means
    "use the renderer default" and is rendered as a hatched / faded
    appearance so the user can tell the override is unset.
    """

    def __init__(
        self,
        rgb: Optional[tuple[int, int, int]],
        *,
        default_rgb: tuple[int, int, int] = (0, 0, 0),
        parent=None,
    ):
        super().__init__(parent)
        self.setFixedSize(48, 22)
        self._rgb = tuple(rgb) if rgb is not None else None
        self._default_rgb = default_rgb
        self.clicked.connect(self._pick)
        self._refresh_swatch()

    @property
    def rgb(self) -> Optional[tuple[int, int, int]]:
        return self._rgb

    def set_rgb(self, rgb: Optional[tuple[int, int, int]]) -> None:
        self._rgb = tuple(rgb) if rgb is not None else None
        self._refresh_swatch()

    def reset_to_default(self) -> None:
        self._rgb = None
        self._refresh_swatch()

    def _refresh_swatch(self) -> None:
        if self._rgb is None:
            r, g, b = self._default_rgb
            self.setText("(default)")
            self.setStyleSheet(
                f"QPushButton {{ background-color: rgb({r},{g},{b}); "
                f"color: {'white' if (r + g + b) < 384 else 'black'}; "
                f"border: 1px dashed #888; font-size: 9px; }}"
            )
        else:
            r, g, b = self._rgb
            self.setText("")
            self.setStyleSheet(
                f"QPushButton {{ background-color: rgb({r},{g},{b}); "
                f"border: 1px solid #555; }}"
            )

    def _pick(self) -> None:
        seed = self._rgb if self._rgb is not None else self._default_rgb
        chosen = QColorDialog.getColor(
            QColor(*seed), self, tr("edit.color_pick_title")
        )
        if chosen.isValid():
            self._rgb = (chosen.red(), chosen.green(), chosen.blue())
            self._refresh_swatch()


class TranslationEditDialog(QDialog):
    """Edit the Korean translation of a single bbox.

    Translations are always rendered with ``ignore_boundary`` semantics
    (centered on the bbox, only user-inserted ``\\n`` splits lines).

    Controls:
      - Korean text editor (Ctrl+Enter accepts)
      - Font dropdown — choices come from the side panel's Fonts library
      - Font size spinbox — initial value is the dialogue's current
        effective size (per-item override if set, otherwise Step5 default)
      - Alignment radio (left / center / right)
      - Rotation spinbox in degrees
      - Text fill / stroke colour swatches (per-dialogue overrides)
      - Optional ellipse background fill with its own colour and padding
    """

    def __init__(
        self,
        japanese: str,
        korean: str,
        font_path: Optional[str] = None,
        font_pt: Optional[int] = None,
        text_align: str = "center",
        text_rotation: int = 0,
        fill_rgb: Optional[tuple[int, int, int]] = None,
        stroke_rgb: Optional[tuple[int, int, int]] = None,
        bg_fill_enabled: bool = False,
        bg_fill_rgb: tuple[int, int, int] = (255, 255, 255),
        bg_fill_pad: int = 6,
        available_fonts: Optional[Sequence[str]] = None,
        default_font_pt: int = 36,
        default_fill_rgb: tuple[int, int, int] = (0, 0, 0),
        default_stroke_rgb: tuple[int, int, int] = (255, 255, 255),
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("edit.title"))
        self.setMinimumSize(520, 480)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(tr("edit.ja_label")))
        ja_view = QPlainTextEdit(japanese)
        ja_view.setReadOnly(True)
        ja_view.setMaximumHeight(80)
        ja_view.setStyleSheet("color: #555; background: #f5f5f5;")
        layout.addWidget(ja_view)

        layout.addWidget(QLabel(tr("edit.ko_label")))
        self.editor = QPlainTextEdit(korean)
        self.editor.setMinimumHeight(80)
        layout.addWidget(self.editor, 1)

        # Per-dialogue overrides
        overrides = QFormLayout()

        self.font_combo = QComboBox()
        self.font_combo.addItem(tr("edit.font_default"), None)
        seen: set[str] = set()
        for fp in available_fonts or []:
            if fp in seen:
                continue
            seen.add(fp)
            self.font_combo.addItem(Path(fp).name, fp)
        selected_idx = 0
        if font_path:
            for i in range(1, self.font_combo.count()):
                if self.font_combo.itemData(i) == font_path:
                    selected_idx = i
                    break
            else:
                self.font_combo.addItem(
                    tr("edit.font_orphan", name=Path(font_path).name), font_path
                )
                selected_idx = self.font_combo.count() - 1
        self.font_combo.setCurrentIndex(selected_idx)
        overrides.addRow(tr("edit.font_label"), self.font_combo)

        initial_pt = int(font_pt) if (font_pt and font_pt > 0) else int(default_font_pt)
        self.font_pt_box = QSpinBox()
        self.font_pt_box.setRange(6, 300)
        self.font_pt_box.setValue(max(6, initial_pt))
        self.font_pt_box.setSuffix(" pt")
        self.font_pt_box.setToolTip(tr("edit.font_size_tip"))
        overrides.addRow(tr("edit.font_size"), self.font_pt_box)

        # Alignment radio buttons
        align_row = QHBoxLayout()
        self.align_left_btn = QRadioButton(tr("edit.align_left"))
        self.align_center_btn = QRadioButton(tr("edit.align_center"))
        self.align_right_btn = QRadioButton(tr("edit.align_right"))
        self._align_group = QButtonGroup(self)
        self._align_group.addButton(self.align_left_btn)
        self._align_group.addButton(self.align_center_btn)
        self._align_group.addButton(self.align_right_btn)
        align_row.addWidget(self.align_left_btn)
        align_row.addWidget(self.align_center_btn)
        align_row.addWidget(self.align_right_btn)
        align_row.addStretch(1)
        align_wrap = QWidget()
        align_wrap.setLayout(align_row)
        align_norm = (text_align or "center").lower()
        if align_norm == "left":
            self.align_left_btn.setChecked(True)
        elif align_norm == "right":
            self.align_right_btn.setChecked(True)
        else:
            self.align_center_btn.setChecked(True)
        overrides.addRow(tr("edit.align"), align_wrap)

        # Rotation
        self.rotation_box = QSpinBox()
        self.rotation_box.setRange(-180, 180)
        self.rotation_box.setSingleStep(5)
        self.rotation_box.setSuffix("°")
        self.rotation_box.setValue(int(text_rotation) if text_rotation else 0)
        self.rotation_box.setToolTip(tr("edit.rotation_tip"))
        overrides.addRow(tr("edit.rotation"), self.rotation_box)

        # ----- Colour overrides -----
        # Fill colour (the body of each glyph). Has a Reset button to revert
        # to the renderer's Step5Params default.
        self.fill_swatch = _ColorSwatchButton(fill_rgb, default_rgb=default_fill_rgb)
        fill_row = QHBoxLayout()
        fill_row.addWidget(self.fill_swatch)
        fill_reset = QPushButton(tr("edit.color_reset"))
        fill_reset.setMaximumWidth(72)
        fill_reset.clicked.connect(self.fill_swatch.reset_to_default)
        fill_row.addWidget(fill_reset)
        fill_row.addStretch(1)
        fill_wrap = QWidget()
        fill_wrap.setLayout(fill_row)
        overrides.addRow(tr("edit.text_color"), fill_wrap)

        self.stroke_swatch = _ColorSwatchButton(
            stroke_rgb, default_rgb=default_stroke_rgb
        )
        stroke_row = QHBoxLayout()
        stroke_row.addWidget(self.stroke_swatch)
        stroke_reset = QPushButton(tr("edit.color_reset"))
        stroke_reset.setMaximumWidth(72)
        stroke_reset.clicked.connect(self.stroke_swatch.reset_to_default)
        stroke_row.addWidget(stroke_reset)
        stroke_row.addStretch(1)
        stroke_wrap = QWidget()
        stroke_wrap.setLayout(stroke_row)
        overrides.addRow(tr("edit.stroke_color"), stroke_wrap)

        # ----- Background fill (ellipse) -----
        self.bg_check = QCheckBox(tr("edit.bg_fill"))
        self.bg_check.setChecked(bool(bg_fill_enabled))
        self.bg_check.setToolTip(tr("edit.bg_fill_tip"))
        overrides.addRow("", self.bg_check)

        self.bg_swatch = _ColorSwatchButton(bg_fill_rgb, default_rgb=(255, 255, 255))
        self.bg_pad_box = QSpinBox()
        self.bg_pad_box.setRange(0, 80)
        self.bg_pad_box.setSuffix(" px")
        self.bg_pad_box.setValue(int(bg_fill_pad))
        bg_row = QHBoxLayout()
        bg_row.addWidget(self.bg_swatch)
        bg_row.addSpacing(6)
        bg_row.addWidget(QLabel(tr("edit.bg_fill_pad")))
        bg_row.addWidget(self.bg_pad_box)
        bg_row.addStretch(1)
        bg_wrap = QWidget()
        bg_wrap.setLayout(bg_row)
        overrides.addRow("", bg_wrap)

        # Disable the bg colour/pad widgets when the checkbox is off so it's
        # obvious that they're not in effect.
        def _sync_bg_enabled(state: bool) -> None:
            self.bg_swatch.setEnabled(state)
            self.bg_pad_box.setEnabled(state)

        self.bg_check.toggled.connect(_sync_bg_enabled)
        _sync_bg_enabled(self.bg_check.isChecked())

        layout.addLayout(overrides)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for seq in ("Ctrl+Return", "Ctrl+Enter"):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(self.accept)

        self.editor.setFocus(Qt.FocusReason.OtherFocusReason)
        self.editor.selectAll()

    # ---- result properties ----

    @property
    def korean(self) -> str:
        return self.editor.toPlainText().strip("\n")

    @property
    def font_path(self) -> Optional[str]:
        return self.font_combo.currentData()

    @property
    def font_pt(self) -> int:
        return int(self.font_pt_box.value())

    @property
    def text_align(self) -> str:
        if self.align_left_btn.isChecked():
            return "left"
        if self.align_right_btn.isChecked():
            return "right"
        return "center"

    @property
    def text_rotation(self) -> int:
        return int(self.rotation_box.value())

    @property
    def fill_rgb(self) -> Optional[tuple[int, int, int]]:
        return self.fill_swatch.rgb

    @property
    def stroke_rgb(self) -> Optional[tuple[int, int, int]]:
        return self.stroke_swatch.rgb

    @property
    def bg_fill_enabled(self) -> bool:
        return self.bg_check.isChecked()

    @property
    def bg_fill_rgb(self) -> tuple[int, int, int]:
        # Background swatch is never None — falls back to white if untouched.
        return self.bg_swatch.rgb or (255, 255, 255)

    @property
    def bg_fill_pad(self) -> int:
        return int(self.bg_pad_box.value())
