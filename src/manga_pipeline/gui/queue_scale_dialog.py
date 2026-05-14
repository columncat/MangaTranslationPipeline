"""Pick a uniform scale factor that the queue worker should apply to
every queued image before running the pipeline.

The dialog reads each queued image's source dimensions (lazy — it does
not load the pixel data) and shows the original-vs-scaled comparison
so the user knows exactly what the chosen factor will produce.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..i18n import tr


def _read_dims(path: Path) -> Optional[tuple[int, int]]:
    """Return (w, h) from the image header without decoding pixel data."""
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:  # noqa: BLE001
        return None


def _summarise_dims(paths: Sequence[Path]) -> dict:
    """Aggregate per-image dims into the stats the dialog wants to show."""
    dims: list[tuple[int, int]] = []
    for p in paths:
        d = _read_dims(p)
        if d is not None:
            dims.append(d)
    if not dims:
        return {
            "count": 0,
            "min": (0, 0),
            "max": (0, 0),
            "avg": (0, 0),
            "total_pixels": 0,
        }
    ws = [d[0] for d in dims]
    hs = [d[1] for d in dims]
    return {
        "count": len(dims),
        "min": (min(ws), min(hs)),
        "max": (max(ws), max(hs)),
        "avg": (sum(ws) // len(ws), sum(hs) // len(hs)),
        "total_pixels": sum(w * h for w, h in dims),
    }


class QueueScaleDialog(QDialog):
    """Modal dialog returning a queue-wide scale factor."""

    def __init__(
        self,
        paths: Sequence[Path],
        current_scale: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("queue_scale.title"))
        self.setMinimumWidth(540)
        self._paths = list(paths)
        self._stats = _summarise_dims(self._paths)
        self._chosen: float = float(current_scale)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_intro_label())

        # Original-stats grid
        orig_box = QFormLayout()
        orig_box.addRow(QLabel(f"<b>{tr('queue_scale.original')}</b>"))
        orig_box.addRow(
            tr("queue_scale.image_count"),
            QLabel(str(self._stats["count"])),
        )
        orig_box.addRow(
            tr("queue_scale.min_size"),
            QLabel(self._fmt_size(self._stats["min"])),
        )
        orig_box.addRow(
            tr("queue_scale.max_size"),
            QLabel(self._fmt_size(self._stats["max"])),
        )
        orig_box.addRow(
            tr("queue_scale.avg_size"),
            QLabel(self._fmt_size(self._stats["avg"])),
        )
        orig_box.addRow(
            tr("queue_scale.total_pixels"),
            QLabel(self._fmt_pixels(self._stats["total_pixels"])),
        )
        layout.addLayout(orig_box)

        # Scale factor input
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel(tr("queue_scale.factor")))
        self.scale_box = QDoubleSpinBox()
        self.scale_box.setRange(0.1, 4.0)
        self.scale_box.setSingleStep(0.05)
        self.scale_box.setDecimals(2)
        self.scale_box.setValue(float(current_scale))
        self.scale_box.setToolTip(tr("queue_scale.factor_tip"))
        self.scale_box.valueChanged.connect(self._refresh_scaled_labels)
        scale_row.addWidget(self.scale_box)
        # Quick presets
        for label, val in (("0.5×", 0.5), ("0.75×", 0.75), ("1.0×", 1.0), ("1.5×", 1.5)):
            btn = QPushButton(label)
            btn.setMaximumWidth(48)
            btn.clicked.connect(lambda _checked, v=val: self.scale_box.setValue(v))
            scale_row.addWidget(btn)
        scale_row.addStretch(1)
        layout.addLayout(scale_row)

        # Scaled-stats (live-updates as the spin box changes)
        scaled_box = QFormLayout()
        scaled_box.addRow(QLabel(f"<b>{tr('queue_scale.scaled')}</b>"))
        self._scaled_min = QLabel("")
        self._scaled_max = QLabel("")
        self._scaled_avg = QLabel("")
        self._scaled_total = QLabel("")
        scaled_box.addRow(tr("queue_scale.min_size"), self._scaled_min)
        scaled_box.addRow(tr("queue_scale.max_size"), self._scaled_max)
        scaled_box.addRow(tr("queue_scale.avg_size"), self._scaled_avg)
        scaled_box.addRow(tr("queue_scale.total_pixels"), self._scaled_total)
        layout.addLayout(scaled_box)

        self._refresh_scaled_labels(self.scale_box.value())

        # Notice + buttons
        notice = QLabel(tr("queue_scale.notice"))
        notice.setStyleSheet("color: #555; font-size: 11px;")
        notice.setWordWrap(True)
        layout.addWidget(notice)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---- helpers ----

    def _build_intro_label(self) -> QLabel:
        intro = QLabel(tr("queue_scale.intro"))
        intro.setWordWrap(True)
        return intro

    @staticmethod
    def _fmt_size(wh: tuple[int, int]) -> str:
        w, h = wh
        return f"{w} × {h}  ({w * h:,} px)"

    @staticmethod
    def _fmt_pixels(total: int) -> str:
        if total >= 1_000_000:
            return f"{total / 1_000_000:.2f} MP  ({total:,} px)"
        return f"{total:,} px"

    def _refresh_scaled_labels(self, value: float) -> None:
        s = float(value)
        if self._stats["count"] == 0:
            return
        sw_min = max(1, int(round(self._stats["min"][0] * s)))
        sh_min = max(1, int(round(self._stats["min"][1] * s)))
        sw_max = max(1, int(round(self._stats["max"][0] * s)))
        sh_max = max(1, int(round(self._stats["max"][1] * s)))
        sw_avg = max(1, int(round(self._stats["avg"][0] * s)))
        sh_avg = max(1, int(round(self._stats["avg"][1] * s)))
        scaled_total = int(round(self._stats["total_pixels"] * s * s))
        self._scaled_min.setText(self._fmt_size((sw_min, sh_min)))
        self._scaled_max.setText(self._fmt_size((sw_max, sh_max)))
        self._scaled_avg.setText(self._fmt_size((sw_avg, sh_avg)))
        self._scaled_total.setText(self._fmt_pixels(scaled_total))

    def _accept(self) -> None:
        self._chosen = float(self.scale_box.value())
        self.accept()

    @property
    def scale(self) -> float:
        return self._chosen
