from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr


class TranslationEditDialog(QDialog):
    """Edit the Korean translation of a single bbox.

    Translations are now always rendered with ``ignore_boundary`` semantics
    (centered on the bbox, only user-inserted ``\\n`` splits lines), so the
    Ignore Boundary checkbox has been removed.

    Controls:
      - Korean text editor (Ctrl+Enter accepts)
      - Font dropdown — choices come from the side panel's Fonts library;
        ``(default)`` falls back to the renderer's default font
      - Font size spinbox — initial value is the dialogue's current effective
        size (per-item override if set, otherwise the Step5 default)
      - Alignment radio (left / center / right)
      - Rotation spinbox in degrees
    """

    def __init__(
        self,
        japanese: str,
        korean: str,
        font_path: Optional[str] = None,
        font_pt: Optional[int] = None,
        text_align: str = "center",
        text_rotation: int = 0,
        available_fonts: Optional[Sequence[str]] = None,
        default_font_pt: int = 36,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("edit.title"))
        self.setMinimumSize(480, 380)

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
        # Pre-select the currently stored font, or insert as orphan if it's
        # not in the library so the user doesn't lose the existing setting.
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
