"""First-run model-download progress popup.

Triggered before the main window appears whenever any of the required model
weights is missing. A worker QThread runs the actual downloads (CTD weight
file, LaMa, manga-ocr) and reports progress back to the dialog via signals.

The popup makes the one-time nature of the operation explicit so users
don't think the app is hanging.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr
from ..ml.weights import (
    ensure_ctd_weights,
    ensure_lama_weights,
    ensure_manga_ocr_weights,
)


# Items the dialog walks through. Each entry is a (key, label, fn) tuple
# where ``fn(progress_cb)`` performs the download.
def _build_jobs() -> list[tuple[str, str, callable]]:
    return [
        ("ctd", "comic-text-detector (~200 MB)", ensure_ctd_weights),
        ("lama", "LaMa inpainting (~200 MB)", ensure_lama_weights),
        ("ocr", "manga-ocr (~400 MB)", ensure_manga_ocr_weights),
    ]


class DownloadWorker(QObject):
    item_started = Signal(int, str)            # idx, label
    item_progress = Signal(int, int, int)      # idx, downloaded_bytes, total_bytes
    item_finished = Signal(int, bool, str)     # idx, ok, error_msg
    all_finished = Signal()

    def __init__(self, jobs):
        super().__init__()
        self._jobs = jobs

    def run(self) -> None:
        for idx, (_key, label, fn) in enumerate(self._jobs):
            self.item_started.emit(idx, label)

            def progress(done: int, total: int, _idx=idx) -> None:
                self.item_progress.emit(_idx, int(done), int(total))

            try:
                fn(progress)
                self.item_finished.emit(idx, True, "")
            except Exception as e:  # noqa: BLE001
                self.item_finished.emit(idx, False, str(e))
        self.all_finished.emit()


class _ItemRow(QWidget):
    """One label + progress bar pair, restyled per state."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self._name = name
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self.label = QLabel(tr("download.item_pending", name=name))
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # indeterminate until first progress tick
        self.bar.setVisible(False)
        v.addWidget(self.label)
        v.addWidget(self.bar)

    def set_running(self) -> None:
        self.label.setText(tr("download.item_running", name=self._name, pct=0, mb_done=0, mb_total=0))
        self.label.setStyleSheet("color: #1565c0;")
        self.bar.setRange(0, 0)
        self.bar.setVisible(True)

    def set_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.bar.setRange(0, 0)
            self.label.setText(
                tr("download.item_running", name=self._name, pct=0, mb_done=0, mb_total=0)
            )
            return
        pct = int(done * 100 / total) if total else 0
        mb_done = done // (1024 * 1024)
        mb_total = total // (1024 * 1024)
        self.bar.setRange(0, total)
        self.bar.setValue(min(done, total))
        self.label.setText(
            tr(
                "download.item_running",
                name=self._name,
                pct=pct,
                mb_done=mb_done,
                mb_total=mb_total,
            )
        )

    def set_done(self) -> None:
        self.label.setText(tr("download.item_done", name=self._name))
        self.label.setStyleSheet("color: #2e7d32;")
        self.bar.setRange(0, 1)
        self.bar.setValue(1)
        self.bar.setVisible(False)

    def set_failed(self, err: str) -> None:
        self.label.setText(tr("download.item_failed", name=self._name, err=err))
        self.label.setStyleSheet("color: #c62828;")
        self.bar.setVisible(False)


class DownloadProgressDialog(QDialog):
    """Modal popup that runs ``ensure_*`` jobs on a worker thread.

    Closes itself once all downloads finish (or after the user clicks Close
    on a partial-failure result).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("download.title"))
        self.setMinimumWidth(520)
        self.setModal(True)
        # No close button on the title bar — user must wait or cancel.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)

        intro = QLabel(tr("download.intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("font-weight: bold;")
        layout.addWidget(intro)

        warn = QLabel(tr("download.warning_keep_open"))
        warn.setStyleSheet("color: #c62828;")
        warn.setWordWrap(True)
        layout.addWidget(warn)

        layout.addSpacing(8)

        jobs = _build_jobs()
        self._rows: list[_ItemRow] = []
        for _key, label, _fn in jobs:
            row = _ItemRow(label)
            self._rows.append(row)
            layout.addWidget(row)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #2e7d32; font-weight: bold;")
        layout.addWidget(self.summary)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setEnabled(False)
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self.accept)
        layout.addWidget(self.buttons)

        self._thread = QThread(self)
        self._worker = DownloadWorker(jobs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.item_started.connect(self._on_item_started)
        self._worker.item_progress.connect(self._on_item_progress)
        self._worker.item_finished.connect(self._on_item_finished)
        self._worker.all_finished.connect(self._on_all_finished)
        self._all_ok = True

    def start(self) -> None:
        self._thread.start()

    # ---- worker callbacks ----

    def _on_item_started(self, idx: int, _label: str) -> None:
        self._rows[idx].set_running()

    def _on_item_progress(self, idx: int, done: int, total: int) -> None:
        self._rows[idx].set_progress(done, total)

    def _on_item_finished(self, idx: int, ok: bool, err: str) -> None:
        if ok:
            self._rows[idx].set_done()
        else:
            self._rows[idx].set_failed(err)
            self._all_ok = False

    def _on_all_finished(self) -> None:
        self._thread.quit()
        self._thread.wait()
        self.summary.setText(tr("download.all_done"))
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setEnabled(True)

    # ---- public ----

    @property
    def all_ok(self) -> bool:
        return self._all_ok

    def closeEvent(self, event):  # type: ignore[override]
        if self._thread.isRunning():
            event.ignore()
            return
        super().closeEvent(event)
