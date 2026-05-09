from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..i18n import tr
from ..utils.secrets import delete_anthropic_key, get_anthropic_key, set_anthropic_key


class ApiKeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("apikey.title"))
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        body = QLabel(tr("apikey.body"))
        body.setWordWrap(True)
        layout.addWidget(body)

        row = QHBoxLayout()
        self.edit = QLineEdit(get_anthropic_key() or "")
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        row.addWidget(self.edit, 1)

        toggle = QPushButton("👁")
        toggle.setMaximumWidth(36)
        toggle.setCheckable(True)
        toggle.toggled.connect(
            lambda checked: self.edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        row.addWidget(toggle)
        layout.addLayout(row)

        clear_btn = QPushButton(tr("apikey.remove"))
        clear_btn.clicked.connect(self._clear)
        layout.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self) -> None:
        key = self.edit.text().strip()
        if key:
            set_anthropic_key(key)
        self.accept()

    def _clear(self) -> None:
        delete_anthropic_key()
        self.edit.setText("")
