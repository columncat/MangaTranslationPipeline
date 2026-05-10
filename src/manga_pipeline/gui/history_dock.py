"""History side dock: visualises the per-image undo / redo timeline.

Each entry in :class:`manga_pipeline.history.HistoryManager` is rendered
as a row with its label and a relative timestamp. The current state is
highlighted; clicking another row jumps to that snapshot (treats it as
a sequence of undo / redo operations).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..history import HistoryManager
from ..i18n import tr


class HistoryDock(QWidget):
    """Read + click-to-jump view onto a :class:`HistoryManager`."""

    # Emitted when the user clicks an entry; payload is the desired
    # current-index. Main window then performs enough undo/redo calls to
    # land there. Decoupled this way so the dock doesn't touch the live
    # context directly.
    jump_requested = Signal(int)
    undo_requested = Signal()
    redo_requested = Signal()
    clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._manager: Optional[HistoryManager] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Toolbar with Undo / Redo / Clear buttons. Keyboard shortcuts
        # live on the main window's QActions; these are convenience.
        toolbar = QHBoxLayout()
        self.undo_btn = QPushButton(tr("history.undo"))
        self.undo_btn.setToolTip(tr("history.undo_tip"))
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        toolbar.addWidget(self.undo_btn, 1)

        self.redo_btn = QPushButton(tr("history.redo"))
        self.redo_btn.setToolTip(tr("history.redo_tip"))
        self.redo_btn.clicked.connect(self.redo_requested.emit)
        toolbar.addWidget(self.redo_btn, 1)
        layout.addLayout(toolbar)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list, 1)

        self.hint = QLabel(tr("history.hint"))
        self.hint.setStyleSheet("color: #666; font-size: 11px;")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.refresh()

    def set_manager(self, manager: HistoryManager) -> None:
        self._manager = manager
        self.refresh()

    def refresh(self) -> None:
        """Repopulate the list from the current manager state."""
        self.list.clear()
        if self._manager is None:
            self.undo_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
            return

        entries = self._manager.entries()
        current = self._manager.current_index
        bold = QFont()
        bold.setBold(True)
        for i, e in enumerate(entries):
            label = e.label
            ts = self._format_timestamp(e.timestamp)
            item = QListWidgetItem(f"{label}    [{ts}]")
            if i == current:
                item.setFont(bold)
                item.setBackground(QBrush(QColor(80, 200, 80, 60)))
            else:
                # Entries past the current index represent redo'able
                # future states (none in this layout because we render
                # only the undo stack); future-proofing for when we
                # decide to interleave redo entries with a separator.
                pass
            self.list.addItem(item)
        # Keep the current row visible.
        if 0 <= current < self.list.count():
            self.list.setCurrentRow(current)
            self.list.scrollToItem(self.list.item(current))

        self.undo_btn.setEnabled(self._manager.can_undo())
        self.redo_btn.setEnabled(self._manager.can_redo())

    @staticmethod
    def _format_timestamp(ts: float) -> str:
        # Show only HH:MM:SS — full ISO is too noisy for a sidebar.
        import time as _time

        return _time.strftime("%H:%M:%S", _time.localtime(ts))

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        row = self.list.row(item)
        if row >= 0:
            self.jump_requested.emit(row)
