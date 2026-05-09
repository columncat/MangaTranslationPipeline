"""Left-side dock: folder browser + work queue.

The folder browser lists images in a directory; single-click opens an image
in the main view (any saved metadata in ``<dir>/metadata/`` is restored).
The queue panel lets the user batch-add files and run any of:

- ``Detect``     — only the Detect phase across the queue
- ``Translate``  — only the Translate phase (use after Detect + manual bbox tweaks)
- ``Run All``    — Detect then Translate

Internally everything runs in ``per-step batch`` mode (each step iterates
over all queued images before advancing) so ML models stay hot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..i18n import tr

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

QUEUE_MODE_SEQUENTIAL = "sequential"
QUEUE_MODE_PER_STEP = "per_step"

STATUS_ICONS = {
    "pending": "⏳",
    "running": "▶",
    "done": "✓",
    "failed": "✗",
}


def _strip_status(text: str) -> str:
    if text and text[0] in STATUS_ICONS.values():
        return text[1:].lstrip()
    return text


class ExplorerPanel(QWidget):
    image_selected = Signal(object)                       # Path
    queue_process_requested = Signal(list, str)           # (paths, "detect" | "translate" | "all")
    queue_save_all_requested = Signal(list)               # paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder: Optional[Path] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(splitter, 1)

        # ---- folder browser ----
        browser = QWidget()
        bl = QVBoxLayout(browser)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)

        h = QHBoxLayout()
        self.folder_label = QLabel(tr("explorer.no_folder"))
        self.folder_label.setStyleSheet("color: #444; font-style: italic;")
        h.addWidget(self.folder_label, 1)
        open_btn = QPushButton(tr("explorer.open_folder"))
        open_btn.setMaximumWidth(72)
        open_btn.clicked.connect(self._on_pick_folder)
        h.addWidget(open_btn)
        bl.addLayout(h)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.file_list.itemClicked.connect(self._on_file_clicked)
        bl.addWidget(self.file_list, 1)

        add_btn = QPushButton(tr("explorer.add_to_queue"))
        add_btn.clicked.connect(self._on_add_to_queue)
        bl.addWidget(add_btn)

        splitter.addWidget(browser)

        # ---- queue ----
        queue_box = QWidget()
        ql = QVBoxLayout(queue_box)
        ql.setContentsMargins(0, 0, 0, 0)
        ql.setSpacing(2)

        ql.addWidget(QLabel(tr("explorer.queue_label")))
        self.queue_list = QListWidget()
        self.queue_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.queue_list.itemClicked.connect(self._on_queue_clicked)
        ql.addWidget(self.queue_list, 1)

        h2 = QHBoxLayout()
        rm = QPushButton(tr("explorer.remove_selected"))
        rm.clicked.connect(self._on_remove_selected)
        h2.addWidget(rm, 1)
        clr = QPushButton(tr("explorer.clear"))
        clr.clicked.connect(self.queue_list.clear)
        h2.addWidget(clr)
        ql.addLayout(h2)

        h3 = QHBoxLayout()
        run_detect_btn = QPushButton(tr("explorer.detect"))
        run_detect_btn.setToolTip(tr("explorer.detect_tip"))
        run_detect_btn.clicked.connect(lambda: self._on_process_queue("detect"))
        h3.addWidget(run_detect_btn, 2)

        run_translate_btn = QPushButton(tr("explorer.translate"))
        run_translate_btn.setToolTip(tr("explorer.translate_tip"))
        run_translate_btn.clicked.connect(lambda: self._on_process_queue("translate"))
        h3.addWidget(run_translate_btn, 2)

        run_all_btn = QPushButton(tr("explorer.run_all"))
        run_all_btn.setToolTip(tr("explorer.run_all_tip"))
        run_all_btn.clicked.connect(lambda: self._on_process_queue("all"))
        h3.addWidget(run_all_btn, 1)
        ql.addLayout(h3)

        save_all_btn = QPushButton(tr("explorer.save_all"))
        save_all_btn.setToolTip(tr("explorer.save_all_tip"))
        save_all_btn.clicked.connect(self._on_save_all)
        ql.addWidget(save_all_btn)

        splitter.addWidget(queue_box)
        splitter.setSizes([400, 320])

    # ---- folder ----

    def _on_pick_folder(self) -> None:
        start = str(self._folder) if self._folder else ""
        folder = QFileDialog.getExistingDirectory(self, "Open folder", start)
        if not folder:
            return
        self.set_folder(Path(folder))

    def set_folder(self, folder: Path) -> None:
        self._folder = folder
        self.folder_label.setText(folder.name or str(folder))
        self.folder_label.setToolTip(str(folder))
        self.folder_label.setStyleSheet("color: #222;")
        self.file_list.clear()
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return
        for p in entries:
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                item = QListWidgetItem(p.name)
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                item.setToolTip(str(p))
                self.file_list.addItem(item)

    def _on_file_clicked(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.ItemDataRole.UserRole)
        if p:
            self.file_list.setCurrentItem(item)
            self.image_selected.emit(Path(p))

    def _file_paths(self) -> list[Path]:
        return [
            Path(self.file_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.file_list.count())
        ]

    def _select_relative_in_list(
        self,
        list_widget: QListWidget,
        delta: int,
        current: Optional[Path],
    ) -> bool:
        """Move the current selection in ``list_widget`` by ``delta`` rows.

        Returns ``True`` if a new image was emitted, ``False`` otherwise
        (empty list, or single-element list with delta off the edge — then
        the caller may want to fall back to a different list).
        """
        items_count = list_widget.count()
        if items_count == 0:
            return False
        idx = -1
        if current is not None:
            target = str(current)
            for i in range(items_count):
                if list_widget.item(i).data(Qt.ItemDataRole.UserRole) == target:
                    idx = i
                    break
        if idx < 0:
            idx = list_widget.currentRow()
        if idx < 0:
            idx = 0 if delta > 0 else items_count - 1
        else:
            idx = max(0, min(items_count - 1, idx + delta))
        item = list_widget.item(idx)
        if item is None:
            return False
        list_widget.setCurrentRow(idx)
        list_widget.scrollToItem(item)
        p = item.data(Qt.ItemDataRole.UserRole)
        if not p:
            return False
        self.image_selected.emit(Path(p))
        return True

    def is_in_queue(self, path: Path) -> bool:
        target = str(path)
        for i in range(self.queue_list.count()):
            if self.queue_list.item(i).data(Qt.ItemDataRole.UserRole) == target:
                return True
        return False

    def select_relative(self, delta: int, current: Optional[Path] = None) -> None:
        """Jump ``delta`` images forward/backward in the folder browser."""
        self._select_relative_in_list(self.file_list, delta, current)

    def select_relative_in_queue(
        self, delta: int, current: Optional[Path] = None
    ) -> bool:
        """Jump ``delta`` images forward/backward inside the work queue.

        Returns ``True`` if navigation happened. Returns ``False`` if the
        queue is empty so callers can fall back to folder navigation.
        """
        if self.queue_list.count() == 0:
            return False
        return self._select_relative_in_list(self.queue_list, delta, current)

    # ---- queue ----

    def _on_add_to_queue(self) -> None:
        existing: set[str] = set()
        for i in range(self.queue_list.count()):
            existing.add(self.queue_list.item(i).data(Qt.ItemDataRole.UserRole))
        for it in self.file_list.selectedItems():
            p = it.data(Qt.ItemDataRole.UserRole)
            if p in existing:
                continue
            existing.add(p)
            qi = QListWidgetItem(f"{STATUS_ICONS['pending']} {Path(p).name}")
            qi.setData(Qt.ItemDataRole.UserRole, p)
            qi.setToolTip(p)
            self.queue_list.addItem(qi)

    def _on_queue_clicked(self, item: QListWidgetItem) -> None:
        p = item.data(Qt.ItemDataRole.UserRole)
        if p:
            self.image_selected.emit(Path(p))

    def _on_remove_selected(self) -> None:
        for it in list(self.queue_list.selectedItems()):
            self.queue_list.takeItem(self.queue_list.row(it))

    def _on_process_queue(self, phases: str) -> None:
        """phases is ``"detect"``, ``"translate"``, or ``"all"``."""
        paths = self._queued_paths()
        if not paths:
            return
        # reset all icons to pending before starting
        for i in range(self.queue_list.count()):
            it = self.queue_list.item(i)
            base = _strip_status(it.text())
            it.setText(f"{STATUS_ICONS['pending']} {base}")
        self.queue_process_requested.emit(paths, phases)

    def _on_save_all(self) -> None:
        paths = self._queued_paths()
        if not paths:
            return
        self.queue_save_all_requested.emit(paths)

    def _queued_paths(self) -> list[Path]:
        out: list[Path] = []
        for i in range(self.queue_list.count()):
            p = self.queue_list.item(i).data(Qt.ItemDataRole.UserRole)
            if p:
                out.append(Path(p))
        return out

    def mark_queue_status(self, source_path: Path, status: str) -> None:
        emoji = STATUS_ICONS.get(status, "?")
        target = str(source_path)
        for i in range(self.queue_list.count()):
            it = self.queue_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == target:
                base = _strip_status(it.text())
                it.setText(f"{emoji} {base}")
                if status == "running":
                    self.queue_list.scrollToItem(it)
                break
