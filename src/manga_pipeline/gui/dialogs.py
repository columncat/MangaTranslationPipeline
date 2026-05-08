from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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


class TranslationEditDialog(QDialog):
    """Edit the Korean translation of a single bbox.

    Controls:
      - Korean text editor (Ctrl+Enter accepts)
      - Ignore Boundary checkbox (centered, no auto-wrap)
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
        ignore_boundary: bool = True,
        font_path: Optional[str] = None,
        font_pt: Optional[int] = None,
        text_align: str = "center",
        text_rotation: int = 0,
        available_fonts: Optional[Sequence[str]] = None,
        default_font_pt: int = 36,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit translation")
        self.setMinimumSize(480, 420)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Japanese (original):"))
        ja_view = QPlainTextEdit(japanese)
        ja_view.setReadOnly(True)
        ja_view.setMaximumHeight(80)
        ja_view.setStyleSheet("color: #555; background: #f5f5f5;")
        layout.addWidget(ja_view)

        layout.addWidget(QLabel("Korean (Ctrl+Enter to accept):"))
        self.editor = QPlainTextEdit(korean)
        self.editor.setMinimumHeight(80)
        layout.addWidget(self.editor, 1)

        self.ignore_boundary_box = QCheckBox(
            "Ignore Boundary  (center on bbox; only your line breaks split lines)"
        )
        self.ignore_boundary_box.setChecked(bool(ignore_boundary))
        layout.addWidget(self.ignore_boundary_box)

        # Per-dialogue overrides
        overrides = QFormLayout()

        self.font_combo = QComboBox()
        self.font_combo.addItem("(default)", None)
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
                    f"[not in library] {Path(font_path).name}", font_path
                )
                selected_idx = self.font_combo.count() - 1
        self.font_combo.setCurrentIndex(selected_idx)
        self.font_combo.setToolTip(
            "Manage fonts in the side panel's 'Fonts library' group."
        )
        overrides.addRow("Font:", self.font_combo)

        initial_pt = int(font_pt) if (font_pt and font_pt > 0) else int(default_font_pt)
        self.font_pt_box = QSpinBox()
        self.font_pt_box.setRange(6, 300)
        self.font_pt_box.setValue(max(6, initial_pt))
        self.font_pt_box.setSuffix(" pt")
        self.font_pt_box.setToolTip(
            "Defaults to this dialogue's current effective size on open."
        )
        overrides.addRow("Font size:", self.font_pt_box)

        # Alignment radio buttons
        align_row = QHBoxLayout()
        self.align_left_btn = QRadioButton("Left")
        self.align_center_btn = QRadioButton("Center")
        self.align_right_btn = QRadioButton("Right")
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
        overrides.addRow("Align:", align_wrap)

        # Rotation
        self.rotation_box = QSpinBox()
        self.rotation_box.setRange(-180, 180)
        self.rotation_box.setSingleStep(5)
        self.rotation_box.setSuffix("°")
        self.rotation_box.setValue(int(text_rotation) if text_rotation else 0)
        self.rotation_box.setToolTip(
            "Counter-clockwise rotation around the bbox center."
        )
        overrides.addRow("Rotation:", self.rotation_box)

        layout.addLayout(overrides)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Ctrl+Enter (and Ctrl+Return on numpad) → OK from anywhere in the dialog.
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
    def ignore_boundary(self) -> bool:
        return self.ignore_boundary_box.isChecked()

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
