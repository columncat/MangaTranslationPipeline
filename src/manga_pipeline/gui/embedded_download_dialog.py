"""One-shot download progress popup for the embedded LLM weights.

Shown the first time the user picks the ``embedded`` translator
backend and runs Translate. The actual download (~5 GB) happens on a
worker QThread so the UI stays responsive; the dialog blocks until it
finishes (or fails / is cancelled).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from ..i18n import tr
from ..ml.weights import EMBEDDED_LLM_URL, ensure_embedded_llm_weights
from ..paths import EMBEDDED_LLM_FILENAME, EMBEDDED_LLM_WEIGHTS


class _Worker(QObject):
    progress = Signal(int, int)        # downloaded_bytes, total_bytes
    finished = Signal(bool, str)       # ok, error_message

    def run(self) -> None:
        try:
            ensure_embedded_llm_weights(progress=self._progress_cb)
        except Exception as e:  # noqa: BLE001 — surface anything to the UI
            self.finished.emit(False, str(e))
            return
        self.finished.emit(True, "")

    def _progress_cb(self, done: int, total: int) -> None:
        self.progress.emit(int(done), int(total))


class EmbeddedDownloadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("embedded.download.title"))
        self.setMinimumWidth(560)
        self.setModal(True)
        # Block window-close while downloading; the cancel button is
        # the only way out so we don't leave a half-finished GGUF
        # behind silently.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self._cancelled = False
        self._success = False

        layout = QVBoxLayout(self)

        intro = QLabel(tr("embedded.download.intro", filename=EMBEDDED_LLM_FILENAME))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        url_lbl = QLabel(EMBEDDED_LLM_URL)
        url_lbl.setStyleSheet("color: #555; font-size: 10px;")
        url_lbl.setWordWrap(True)
        url_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(url_lbl)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # indeterminate until the first chunk arrives
        layout.addWidget(self.bar)

        self.detail = QLabel(tr("embedded.download.starting"))
        self.detail.setStyleSheet("color: #555;")
        layout.addWidget(self.detail)

        self.summary = QLabel("")
        self.summary.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.summary)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.buttons.rejected.connect(self._on_cancel)
        layout.addWidget(self.buttons)

        self._thread = QThread(self)
        self._worker = _Worker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)

    def start(self) -> None:
        self._thread.start()

    @property
    def success(self) -> bool:
        return self._success

    # ---- worker callbacks ----

    def _on_progress(self, done: int, total: int) -> None:
        if self._cancelled:
            return
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(min(done, total))
            mb_done = done // (1024 * 1024)
            mb_total = total // (1024 * 1024)
            pct = int(done * 100 / total)
            self.detail.setText(
                tr(
                    "embedded.download.progress",
                    pct=pct,
                    mb_done=mb_done,
                    mb_total=mb_total,
                )
            )
        else:
            self.detail.setText(
                tr("embedded.download.progress_unknown",
                   mb_done=done // (1024 * 1024))
            )

    def _on_finished(self, ok: bool, err: str) -> None:
        self._thread.quit()
        self._thread.wait()
        if ok:
            self._success = True
            self.summary.setStyleSheet("font-weight: bold; color: #2e7d32;")
            self.summary.setText(tr("embedded.download.done"))
            self.detail.setText(str(EMBEDDED_LLM_WEIGHTS))
            self.buttons.clear()
            ok_btn = self.buttons.addButton(QDialogButtonBox.StandardButton.Ok)
            ok_btn.clicked.connect(self.accept)
        else:
            self.summary.setStyleSheet("font-weight: bold; color: #c62828;")
            self.summary.setText(tr("embedded.download.failed", err=err))
            self.bar.setValue(0)
            self.buttons.clear()
            close_btn = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
            close_btn.clicked.connect(self.reject)

    def _on_cancel(self) -> None:
        # Mark for dismissal; we can't actually interrupt requests
        # mid-stream (download_file uses a synchronous loop). The
        # partial .part file is left on disk and resumed on the next
        # download attempt because download_file restarts cleanly.
        self._cancelled = True
        self.reject()

    def closeEvent(self, event):  # type: ignore[override]
        if self._thread.isRunning():
            event.ignore()
            return
        super().closeEvent(event)
