from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QToolBar,
)

from .. import persistence
from ..config import AppConfig
from ..device import auto_device
from ..models import BBox, PageContext
from ..utils.fonts import find_default_font
from ..utils.image_io import load_rgb, save_image
from ..utils.secrets import get_anthropic_key
from .dialogs import TranslationEditDialog
from .explorer import ExplorerPanel
from .settings_dialog import ApiKeyDialog
from .side_panel import SidePanel
from .tabs import DetectTab, SourceTab, TranslateTab
from .workers import (
    PHASE_DETECT,
    PHASE_TRANSLATE,
    PipelineThread,
    PipelineWorker,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Translation Pipeline")
        self.resize(1400, 900)

        self.config = AppConfig.load()
        self.ctx: Optional[PageContext] = None
        self._source_path: Optional[Path] = None
        self.worker = PipelineWorker()
        self._thread: Optional[QThread] = None

        self._build_tabs()
        self._build_explorer()
        self._build_side_panel()
        self._build_toolbar()
        self._build_status_bar()
        self._wire_signals()

        self.side_panel.set_api_key_status(get_anthropic_key() is not None)
        self._push_render_defaults()
        self.status_bar.showMessage(f"Device: {auto_device()} — open an image to begin")

    # ------------------------------------------------------------------ build

    def _build_tabs(self) -> None:
        self.tabs = QTabWidget(self)
        self.source_tab = SourceTab()
        self.detect_tab = DetectTab()
        self.translate_tab = TranslateTab()
        self.tabs.addTab(self.source_tab, "0. Original")
        self.tabs.addTab(self.detect_tab, "1. Detect")
        self.tabs.addTab(self.translate_tab, "2. Translate")
        self.setCentralWidget(self.tabs)

        self.detect_tab.rerun_requested.connect(self._on_run_phase)
        self.translate_tab.rerun_requested.connect(self._on_run_phase)

        self.detect_tab.bbox_delete_requested.connect(self._on_bbox_delete)
        self.detect_tab.bbox_geometry_changed.connect(self._on_bbox_geometry_changed)
        self.detect_tab.bbox_add_requested.connect(self._on_bbox_add)
        self.translate_tab.bbox_delete_requested.connect(self._on_bbox_delete)
        self.translate_tab.translation_edit_requested.connect(self._on_translation_edit)
        self.translate_tab.text_offset_changed.connect(self._on_text_offset_changed)

    def _build_explorer(self) -> None:
        self.explorer = ExplorerPanel()
        dock = QDockWidget("Files & Queue", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(self.explorer)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        self.explorer.image_selected.connect(self._on_explorer_image_selected)
        self.explorer.queue_process_requested.connect(self._on_queue_process)
        self.explorer.queue_save_all_requested.connect(self._on_queue_save_all)

    def _build_side_panel(self) -> None:
        self.side_panel = SidePanel(self.config)
        dock = QDockWidget("Settings", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(self.side_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.side_panel.request_run_phase.connect(self._on_run_phase)
        self.side_panel.request_run_all.connect(self._on_run_all)
        self.side_panel.request_save.connect(self._on_save_overwrite)
        self.side_panel.request_save_as.connect(self._on_save_as)
        self.side_panel.request_render.connect(self._on_render)
        self.side_panel.api_key_clicked.connect(self._on_set_api_key)
        self.side_panel.config_changed.connect(self._save_config)
        self.side_panel.config_changed.connect(self._push_render_defaults)

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        open_act = QAction("Open…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open)
        tb.addAction(open_act)

        save_act = QAction("Save final…", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._on_save)
        tb.addAction(save_act)

        tb.addSeparator()

        run_all_act = QAction("Run all", self)
        run_all_act.setShortcut("Ctrl+R")
        run_all_act.triggered.connect(self._on_run_all)
        tb.addAction(run_all_act)

        cancel_act = QAction("Cancel", self)
        cancel_act.triggered.connect(lambda: self.worker.cancel())
        tb.addAction(cancel_act)

        tb.addSeparator()

        api_act = QAction("API key…", self)
        api_act.triggered.connect(self._on_set_api_key)
        tb.addAction(api_act)

    def _build_status_bar(self) -> None:
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(280)
        self.progress.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress)

    def _wire_signals(self) -> None:
        self.worker.started.connect(self._on_step_started)
        self.worker.progress.connect(self._on_step_progress)
        self.worker.finished.connect(self._on_step_finished)
        self.worker.failed.connect(self._on_step_failed)
        self.worker.page_updated.connect(self._refresh_tabs)
        self.worker.phase_finished.connect(self._on_phase_finished)
        self.worker.queue_started.connect(self._on_queue_started)
        self.worker.queue_item_started.connect(self._on_queue_item_started)
        self.worker.queue_item_finished.connect(self._on_queue_item_finished)
        self.worker.queue_finished.connect(self._on_queue_finished)

    # ----------------------------------------------------------------- file IO

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open manga page", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        self._load_image_path(Path(path))

    def _on_explorer_image_selected(self, path: Path) -> None:
        self._load_image_path(path)

    def _load_image_path(self, path: Path) -> None:
        """Open ``path``, restoring saved metadata; auto-save the previous one."""
        # 1. Auto-save the currently-open image before switching.
        if (
            self.ctx is not None
            and self._source_path is not None
            and self._source_path != path
        ):
            try:
                persistence.save_context(self.ctx, self._source_path)
            except Exception:  # noqa: BLE001
                pass

        # 2. Load the new image (with metadata if present).
        try:
            ctx = persistence.load_context(path)
            if ctx is None:
                ctx = PageContext(source=load_rgb(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.ctx = ctx
        self._source_path = path
        self._refresh_tabs(ctx)

        # 3. Pick the most-advanced tab that has results.
        if ctx.final is not None or ctx.translations:
            target = 2  # Translate
        elif ctx.bboxes or ctx.mask is not None:
            target = 1  # Detect
        else:
            target = 0  # Original
        self.tabs.setCurrentIndex(target)

        h, w = ctx.source.shape[:2]
        restored = ""
        if ctx.bboxes or ctx.translations or ctx.mask is not None:
            restored = " [restored from metadata]"
        self.status_bar.showMessage(f"Loaded: {path.name} ({w}x{h}){restored}")

    def _on_save(self) -> None:
        """Toolbar 'Save final…' — file dialog (any location)."""
        if self.ctx is None or self.ctx.final is None:
            QMessageBox.information(self, "Nothing to save", "Run Translate first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save final image", "translated.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        save_image(self.ctx.final, path)
        self.status_bar.showMessage(f"Saved: {path}", 5000)

    def _on_save_overwrite(self) -> None:
        """Side-panel 'Save' — overwrite ``<src_dir>/translated/<src_name>`` silently."""
        target = self._default_save_target()
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            save_image(self.ctx.final, target)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self.status_bar.showMessage(f"Saved: {target}", 5000)

    def _on_save_as(self) -> None:
        """Side-panel 'Save as' — always prompt for a file name."""
        target = self._default_save_target()
        if target is None:
            return
        out_dir = target.parent
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Cannot create folder", f"{out_dir}\n{e}")
            return

        new_name, ok = QInputDialog.getText(
            self,
            "Save as",
            f"Save into:\n{out_dir}\n\nEnter file name (with extension):",
            text=target.name,
        )
        if not ok or not new_name.strip():
            self.status_bar.showMessage("Save cancelled", 4000)
            return
        new_target = out_dir / new_name.strip()
        try:
            save_image(self.ctx.final, new_target)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self.status_bar.showMessage(f"Saved: {new_target}", 5000)

    def _default_save_target(self) -> Optional[Path]:
        """Compute ``<src_dir>/translated/<src_name>`` and validate prerequisites."""
        if self.ctx is None or self.ctx.final is None:
            QMessageBox.information(self, "Nothing to save", "Run Translate first.")
            return None
        if self._source_path is None:
            QMessageBox.warning(self, "No source", "Open an image before saving.")
            return None
        return self._source_path.parent / "translated" / self._source_path.name

    def _on_set_api_key(self) -> None:
        dlg = ApiKeyDialog(self)
        if dlg.exec():
            self.side_panel.set_api_key_status(get_anthropic_key() is not None)

    # ----------------------------------------------------------------- pipeline

    def _on_run_phase(self, phase: str) -> None:
        if self.ctx is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        if phase == PHASE_TRANSLATE and not get_anthropic_key():
            self._on_set_api_key()
            if not get_anthropic_key():
                return
        self._launch_thread(phase=phase)

    def _on_run_all(self) -> None:
        if self.ctx is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        if not get_anthropic_key():
            self._on_set_api_key()
            if not get_anthropic_key():
                return
        self._launch_thread(all_phases=True)

    def _on_render(self) -> None:
        if self.ctx is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        if not self.ctx.translations:
            QMessageBox.information(
                self,
                "Nothing to render",
                "Translate first (or skip-translate) before rendering.",
            )
            return
        self._launch_thread(step=5)

    def _on_queue_process(self, paths: list, phases_label: str) -> None:
        if not paths:
            return
        # Anything that triggers Step 4 needs an API key (unless skip-translation is on).
        needs_translate = (
            phases_label in ("translate", "all")
            and not self.config.step4.skip_translation
        )
        if needs_translate and not get_anthropic_key():
            self._on_set_api_key()
            if not get_anthropic_key():
                return

        # Auto-save the currently-open image before the queue clobbers state.
        if self.ctx is not None and self._source_path is not None:
            try:
                persistence.save_context(self.ctx, self._source_path)
            except Exception:  # noqa: BLE001
                pass

        # If anything in the queue already has results that this run would
        # invalidate, ask before proceeding.
        if not self._confirm_queue_overwrite(paths, phases_label):
            return

        if phases_label == "detect":
            phases: tuple[str, ...] = (PHASE_DETECT,)
        elif phases_label == "translate":
            phases = (PHASE_TRANSLATE,)
        else:
            phases = (PHASE_DETECT, PHASE_TRANSLATE)

        self._launch_thread(
            queue=[Path(p) for p in paths],
            queue_phases=phases,
        )

    def _confirm_queue_overwrite(self, paths: list, phases_label: str) -> bool:
        """Ask the user before overwriting existing intermediate data.

        - ``detect``    → warn if any queued image already has bboxes
        - ``translate`` → warn if any queued image already has translations
        - ``all``       → warn if any has either bboxes or translations
        """
        check_bboxes = phases_label in ("detect", "all")
        check_translations = phases_label in ("translate", "all")

        affected: list[str] = []
        for raw in paths:
            p = Path(str(raw))
            try:
                ctx = persistence.load_context(p)
            except Exception:  # noqa: BLE001
                continue
            if ctx is None:
                continue
            has_bboxes = bool(ctx.bboxes)
            has_translations = bool(ctx.translations)
            if (check_bboxes and has_bboxes) or (check_translations and has_translations):
                affected.append(p.name)

        if not affected:
            return True

        sample = "\n  • ".join(affected[:8])
        if len(affected) > 8:
            sample += f"\n  …and {len(affected) - 8} more"
        action = {
            "detect": "Detect (this will overwrite existing bboxes)",
            "translate": "Translate (this will overwrite existing translations)",
            "all": "Run All (this will overwrite both bboxes and translations)",
        }.get(phases_label, phases_label)
        reply = QMessageBox.question(
            self,
            "Existing results will be overwritten",
            f"The following {len(affected)} queued image(s) already have results:\n\n"
            f"  • {sample}\n\n"
            f"Proceed with {action}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_queue_save_all(self, paths: list) -> None:
        """Save the rendered final of every queued image (overwrite). Skip those without final."""
        if not paths:
            return
        out_dir_root = "translated"
        saved = 0
        skipped = 0
        for raw in paths:
            p = Path(str(raw))
            try:
                ctx = persistence.load_context(p)
            except Exception:  # noqa: BLE001
                ctx = None
            if ctx is None or ctx.final is None:
                skipped += 1
                continue
            target = p.parent / out_dir_root / p.name
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                save_image(ctx.final, target)
                saved += 1
            except Exception:  # noqa: BLE001
                skipped += 1
        self.status_bar.showMessage(
            f"Save all done: {saved} saved, {skipped} skipped (no final image)",
            8000,
        )

    def _launch_thread(
        self,
        *,
        step: Optional[int] = None,
        phase: Optional[str] = None,
        all_phases: bool = False,
        queue: Optional[list] = None,
        queue_phases: tuple = (PHASE_DETECT, PHASE_TRANSLATE),
    ) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "Pipeline already running.")
            return

        if queue is not None:
            self._thread = PipelineThread(
                self.worker,
                None,
                self.config,
                queue=queue,
                queue_phases=queue_phases,
                load_fn=self._queue_load_fn,
                save_fn=self._queue_save_fn,
            )
        else:
            assert self.ctx is not None
            self._thread = PipelineThread(
                self.worker,
                self.ctx,
                self.config,
                step=step,
                phase=phase,
                all_phases=all_phases,
            )
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _queue_load_fn(self, path: Path) -> Optional[PageContext]:
        """Load a PageContext for queue runs, preferring saved metadata.

        Without this, ``Translate Only`` would start each image with no
        bboxes and immediately fail at the OCR step. The phase-specific
        cascade-clear inside ``_run_queue_per_step`` then trims the right
        downstream state (e.g. it does *not* clear bboxes when only the
        Translate phase is requested).
        """
        try:
            ctx = persistence.load_context(path)
            if ctx is not None:
                return ctx
            return PageContext(source=load_rgb(path))
        except Exception:  # noqa: BLE001
            return None

    def _queue_save_fn(self, ctx: PageContext, path: Path) -> None:
        try:
            persistence.save_context(ctx, path)
        except Exception:  # noqa: BLE001
            pass

    def _on_thread_done(self) -> None:
        self.progress.setVisible(False)
        self._thread = None

    def _on_step_started(self, idx: int, name: str) -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status_bar.showMessage(f"Running {name}…")

    def _on_step_progress(self, idx: int, cur: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(cur)
        self.status_bar.showMessage(f"[Step {idx}] {msg} ({cur}/{total})")

    def _on_step_finished(self, idx: int, ok: bool, msg: str) -> None:
        if not ok:
            self.status_bar.showMessage(f"[Step {idx}] FAILED: {msg}", 10000)

    def _on_step_failed(self, idx: int, msg: str) -> None:
        QMessageBox.critical(self, f"Step {idx} failed", msg)

    def _on_phase_finished(self, phase: str, ok: bool, msg: str) -> None:
        tab = self.detect_tab if phase == PHASE_DETECT else self.translate_tab
        tab.set_status(msg or ("done" if ok else "failed"), ok=ok)
        if ok:
            target = 1 if phase == PHASE_DETECT else 2
            self.tabs.setCurrentIndex(target)
        # Auto-save metadata so the user can later resume from a single click.
        if ok and self.ctx is not None and self._source_path is not None:
            try:
                persistence.save_context(self.ctx, self._source_path)
            except Exception:  # noqa: BLE001
                pass

    # ---- queue ----

    def _on_queue_started(self, total: int) -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        self._queue_total = total
        self._queue_done = 0
        self.status_bar.showMessage(f"Processing queue (0 / {total})…")

    def _on_queue_item_started(self, path: object, idx: int, total: int) -> None:
        p = Path(str(path))
        self.explorer.mark_queue_status(p, "running")
        # Display the image being processed in the main view as the queue runs.
        self._load_image_path(p)

    def _on_queue_item_finished(self, path: object, ok: bool, msg: str) -> None:
        p = Path(str(path))
        self.explorer.mark_queue_status(p, "done" if ok else "failed")
        self._queue_done = getattr(self, "_queue_done", 0) + 1
        total = getattr(self, "_queue_total", 1)
        self.progress.setValue(self._queue_done)
        self.status_bar.showMessage(
            f"Queue: {self._queue_done} / {total}  —  last: {p.name} ({'OK' if ok else 'FAIL'}: {msg})"
        )

    def _on_queue_finished(self) -> None:
        self.progress.setVisible(False)
        total = getattr(self, "_queue_total", 0)
        self.status_bar.showMessage(f"Queue done — {total} item(s) processed", 8000)

    # ----------------------------------------------------------------- editing

    def _on_bbox_geometry_changed(
        self, idx: int, x: int, y: int, w: int, h: int
    ) -> None:
        if self.ctx is None or idx < 0 or idx >= len(self.ctx.bboxes):
            return
        old = self.ctx.bboxes[idx]
        # Mutate the existing BBox object in place so OcrResult/TranslationResult
        # references (which use ``is`` comparison) keep matching.
        old.x, old.y, old.w, old.h = int(x), int(y), int(w), int(h)
        old.area = int(w) * int(h)
        # Geometry changed → cleaned/final become stale.
        self.ctx.cleaned = None
        self.ctx.final = None
        self.status_bar.showMessage(
            f"Bbox #{idx} → ({x}, {y}, {w}×{h}) — press Render to refresh",
            5000,
        )
        self._auto_save_metadata()

    def _on_bbox_add(self) -> None:
        if self.ctx is None or self.ctx.source is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        h, w = self.ctx.source.shape[:2]
        bw, bh = min(160, w // 4), min(80, h // 6)
        x = max(0, (w - bw) // 2)
        y = max(0, (h - bh) // 2)
        self.ctx.bboxes.append(BBox(x=x, y=y, w=bw, h=bh, area=bw * bh))
        # New bbox isn't translated yet → invalidate downstream.
        self.ctx.cleaned = None
        self.ctx.final = None
        self._refresh_tabs(self.ctx)
        self.status_bar.showMessage(
            f"Added bbox at ({x}, {y}) {bw}×{bh} — toggle Edit to drag/resize",
            6000,
        )
        self._auto_save_metadata()

    def _on_text_offset_changed(self, idx: int, ox: int, oy: int) -> None:
        if self.ctx is None or idx < 0 or idx >= len(self.ctx.translations):
            return
        tr = self.ctx.translations[idx]
        tr.text_offset_x = int(ox)
        tr.text_offset_y = int(oy)
        # Offset only affects rendering; cleaned image is still valid.
        self.ctx.final = None
        self.status_bar.showMessage(
            f"Translation #{idx} offset → ({ox}, {oy}) — press Render to refresh",
            5000,
        )
        self._auto_save_metadata()

    def _auto_save_metadata(self) -> None:
        if self.ctx is None or self._source_path is None:
            return
        try:
            persistence.save_context(self.ctx, self._source_path)
        except Exception:  # noqa: BLE001
            pass

    def _on_bbox_delete(self, idx: int) -> None:
        if self.ctx is None or idx < 0 or idx >= len(self.ctx.bboxes):
            return
        bbox = self.ctx.bboxes.pop(idx)
        self.ctx.ocr = [r for r in self.ctx.ocr if r.bbox is not bbox]
        self.ctx.translations = [
            t for t in self.ctx.translations if t.bbox is not bbox
        ]
        # Cleaned no longer matches the new bbox set; force rebuild on next Render.
        self.ctx.cleaned = None
        self.ctx.final = None
        self._refresh_tabs(self.ctx)
        msg = f"Deleted bbox #{idx} — {len(self.ctx.bboxes)} remaining"
        if self.ctx.translations:
            msg += " — press Render to refresh the final image"
        self.status_bar.showMessage(msg, 6000)

    def _on_translation_edit(self, idx: int) -> None:
        if self.ctx is None or idx < 0 or idx >= len(self.ctx.translations):
            return
        tr = self.ctx.translations[idx]
        dlg = TranslationEditDialog(
            tr.text_ja,
            tr.text_ko,
            ignore_boundary=True,
            font_path=tr.font_path,
            font_pt=tr.font_pt,
            text_align=getattr(tr, "text_align", "center") or "center",
            text_rotation=int(getattr(tr, "text_rotation", 0) or 0),
            available_fonts=self.config.fonts,
            default_font_pt=self.config.step5.outside_pt,
            parent=self,
        )
        if dlg.exec():
            current_align = getattr(tr, "text_align", "center") or "center"
            current_rotation = int(getattr(tr, "text_rotation", 0) or 0)
            changed = (
                dlg.korean != tr.text_ko
                or dlg.ignore_boundary != tr.ignore_boundary
                or dlg.font_path != tr.font_path
                or dlg.font_pt != tr.font_pt
                or dlg.text_align != current_align
                or dlg.text_rotation != current_rotation
            )
            if changed:
                tr.text_ko = dlg.korean
                tr.ignore_boundary = dlg.ignore_boundary
                tr.font_path = dlg.font_path
                tr.font_pt = dlg.font_pt
                tr.text_align = dlg.text_align
                tr.text_rotation = dlg.text_rotation
                # Render only — keep cached ctx.cleaned (mask unchanged).
                self.ctx.final = None
                self._refresh_tabs(self.ctx)
                self._launch_thread(step=5)

    # ----------------------------------------------------------------- refresh

    def _refresh_tabs(self, ctx: PageContext) -> None:
        self.source_tab.update_from_context(ctx)
        self.detect_tab.update_from_context(ctx)
        self.translate_tab.update_from_context(ctx)

    def _save_config(self) -> None:
        try:
            self.config.save()
        except Exception:
            pass

    def _push_render_defaults(self) -> None:
        """Tell the Translate tab which font/size the renderer would use, so
        Move-Text previews match the actual rendering."""
        font_path = self.config.step5.font_path or None
        if not font_path:
            try:
                fp = find_default_font()
                font_path = str(fp) if fp is not None else None
            except Exception:  # noqa: BLE001
                font_path = None
        self.translate_tab.set_render_defaults(
            font_path=font_path,
            default_pt=int(self.config.step5.outside_pt),
        )

    def closeEvent(self, event):  # type: ignore[override]
        self._save_config()
        super().closeEvent(event)
