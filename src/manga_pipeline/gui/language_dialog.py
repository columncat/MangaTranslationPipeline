"""First-run language picker.

Displayed when ``AppConfig.ui_language`` is ``None``. The dialog text is
intentionally bilingual so it's understandable regardless of which UI
language is currently loaded (English by default at boot).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)


class LanguageDialog(QDialog):
    def __init__(self, current: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose language / 언어 선택")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Choose the interface language.\n"
                "인터페이스 언어를 선택하세요.\n\n"
                "(You can change it later from the toolbar.\n"
                " 도구 모음에서 나중에 변경할 수 있습니다.)"
            )
        )

        self.ko_btn = QRadioButton("한국어 (Korean)")
        self.en_btn = QRadioButton("English")
        group = QButtonGroup(self)
        group.addButton(self.ko_btn)
        group.addButton(self.en_btn)
        # Default: Korean — the project ships KO by request.
        if current == "en":
            self.en_btn.setChecked(True)
        else:
            self.ko_btn.setChecked(True)

        layout.addWidget(self.ko_btn)
        layout.addWidget(self.en_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        # Hitting Enter accepts.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def selected(self) -> str:
        return "en" if self.en_btn.isChecked() else "ko"
