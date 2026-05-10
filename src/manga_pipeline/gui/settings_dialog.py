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

from ..ai import AIProvider
from ..i18n import tr
from ..utils.secrets import delete_api_key, get_api_key, set_api_key


# Providers that need an API key managed via this dialog.
_KEYED_PROVIDERS = (
    AIProvider.ANTHROPIC,
    AIProvider.OPENAI_COMPAT,
    AIProvider.GEMINI,
)


class ApiKeyDialog(QDialog):
    """Edits the cloud-provider API key for a chosen ``provider``.

    Local backends (Ollama, llama.cpp) don't appear here; the dialog
    silently no-ops if asked to manage one of them.
    """

    def __init__(self, provider: str = AIProvider.ANTHROPIC, parent=None):
        super().__init__(parent)
        self.provider = (
            provider if provider in _KEYED_PROVIDERS else AIProvider.ANTHROPIC
        )
        self.setWindowTitle(
            tr("apikey.title.dynamic", provider=tr(f"side.s4.provider.{self.provider}"))
        )
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        body = QLabel(tr("apikey.body"))
        body.setWordWrap(True)
        layout.addWidget(body)

        row = QHBoxLayout()
        self.edit = QLineEdit(get_api_key(self.provider) or "")
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
            set_api_key(key, self.provider)
        self.accept()

    def _clear(self) -> None:
        delete_api_key(self.provider)
        self.edit.setText("")
