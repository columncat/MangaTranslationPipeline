"""Modal popup that runs the queue's "save all final images" pass with a
visible progress bar, then auto-closes after a short countdown.

Save IO happens on a worker QThread so the UI stays responsive even when
copying many large PNGs to a slow disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

from .. import persistence
from ..i18n import tr
from ..utils.image_io import save_image

OUT_DIR_NAME = "translated"
AUTOCLOSE_SECONDS = 3


class _SaveAllWorker(QObject):
    item = Signal(int, int, str)            # done, total, name
    finished = Signal(int, int, int)        # saved, skipped, total

    def __init__(self, paths: Sequence[Path]):
        super().__init__()
        self._paths = list(paths)

    def run(self) -> None:
        saved = 0
        skipped = 0
        total = len(self._paths)
        for idx, p in enumerate(self._paths, start=1):
            try:
                ctx = persistence.load_context(p)
            except Exception:  # noqa: BLE001
                ctx = None
            if ctx is None or ctx.final is None:
                skipped += 1
                self.item.emit(idx, total, p.name)
                continue
            target = p.parent / OUT_DIR_NAME / p.name
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                save_image(ctx.final, target)
                saved += 1
            except Exception:  # noqa: BLE001
                skipped += 1
            self.item.emit(idx, total, p.name)
        self.finished.emit(saved, skipped, total)


class SaveAllProgressDialog(QDialog):
    def __init__(self, paths: Sequence[Path], parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("save_all.title"))
        self.setMinimumWidth(480)
        self.setModal(True)
        # Block window close while saving — countdown handles closure.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)

        self.bar = QProgressBar()
        self.bar.setRange(0, max(1, len(paths)))
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.detail = QLabel("")
        self.detail.setStyleSheet("color: #555;")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

        self.summary = QLabel("")
        self.summary.setStyleSheet("font-weight: bold; color: #2e7d32;")
        layout.addWidget(self.summary)

        self.countdown = QLabel("")
        self.countdown.setStyleSheet("color: #555;")
        layout.addWidget(self.countdown)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setText(
            tr("save_all.close")
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setEnabled(False)
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self.accept)
        layout.addWidget(self.buttons)

        self._thread = QThread(self)
        self._worker = _SaveAllWorker(paths)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.item.connect(self._on_item)
        self._worker.finished.connect(self._on_finished)

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)
        self._countdown_remaining = AUTOCLOSE_SECONDS

    def start(self) -> None:
        self._thread.start()

    # ---- worker callbacks ----

    def _on_item(self, done: int, total: int, name: str) -> None:
        self.bar.setValue(done)
        self.detail.setText(tr("save_all.progress", done=done, total=total, name=name))

    def _on_finished(self, saved: int, skipped: int, total: int) -> None:
        self._thread.quit()
        self._thread.wait()
        self.bar.setValue(total)
        self.detail.setText("")
        if skipped == 0:
            self.summary.setText(
                tr("save_all.summary_ok", saved=saved, total=total)
            )
        else:
            self.summary.setText(
                tr(
                    "save_all.summary_skipped",
                    saved=saved,
                    total=total,
                    skipped=skipped,
                )
            )
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setEnabled(True)
        self._countdown_remaining = AUTOCLOSE_SECONDS
        self.countdown.setText(
            tr("save_all.autoclose", sec=self._countdown_remaining)
        )
        self._countdown_timer.start()

    def _on_countdown_tick(self) -> None:
        self._countdown_remaining -= 1
        if self._countdown_remaining <= 0:
            self._countdown_timer.stop()
            self.accept()
            return
        self.countdown.setText(
            tr("save_all.autoclose", sec=self._countdown_remaining)
        )

    def closeEvent(self, event):  # type: ignore[override]
        if self._thread.isRunning():
            event.ignore()
            return
        self._countdown_timer.stop()
        super().closeEvent(event)
