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
from ..i18n import set_language, tr
from ..models import BBox, PageContext
from ..utils.fonts import find_default_font
from ..utils.image_io import load_rgb, save_image
from ..utils.secrets import get_anthropic_key
from .dialogs import TranslationEditDialog
from .explorer import ExplorerPanel
from .language_dialog import LanguageDialog
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
    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.setWindowTitle(tr("app.title"))
        self.resize(1400, 900)

        self.config = config if config is not None else AppConfig.load()
        self.ctx: Optional[PageContext] = None
        self._source_path: Optional[Path] = None
        self.worker = PipelineWorker()
        self._thread: Optional[QThread] = None
        # While True, _load_image_path skips its auto-save step. Queue runs
        # set this so the worker's per-step save_fn writes are not clobbered
        # by the main window auto-saving stale ``self.ctx`` snapshots when
        # switching the displayed image as queue progress events arrive.
        self._queue_active = False
        # Tracks the user's preferred phase for the upcoming queue items so
        # ``_on_queue_item_started`` can jump to the right tab even before
        # the new image has any phase-specific data attached.
        self._queue_target_tab: Optional[int] = None

        self._build_tabs()
        self._build_explorer()
        self._build_side_panel()
        self._build_toolbar()
        self._build_status_bar()
        self._wire_signals()

        self.side_panel.set_api_key_status(get_anthropic_key() is not None)
        self._push_render_defaults()
        self.status_bar.showMessage(
            tr("status.device_ready", device=auto_device())
        )

    # ------------------------------------------------------------------ build

    def _build_tabs(self) -> None:
        self.tabs = QTabWidget(self)
        self.source_tab = SourceTab()
        self.detect_tab = DetectTab()
        self.translate_tab = TranslateTab()
        self.tabs.addTab(self.source_tab, tr("tab.original"))
        self.tabs.addTab(self.detect_tab, tr("tab.detect"))
        self.tabs.addTab(self.translate_tab, tr("tab.translate"))
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

        open_act = QAction(tr("toolbar.open"), self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open)
        tb.addAction(open_act)

        save_act = QAction(tr("toolbar.save_final"), self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._on_save)
        tb.addAction(save_act)

        tb.addSeparator()

        run_all_act = QAction(tr("toolbar.run_all"), self)
        run_all_act.setShortcut("Ctrl+R")
        run_all_act.triggered.connect(self._on_run_all)
        tb.addAction(run_all_act)

        cancel_act = QAction(tr("toolbar.cancel"), self)
        cancel_act.triggered.connect(lambda: self.worker.cancel())
        tb.addAction(cancel_act)

        tb.addSeparator()

        api_act = QAction(tr("toolbar.api_key"), self)
        api_act.triggered.connect(self._on_set_api_key)
        tb.addAction(api_act)

        lang_act = QAction(tr("toolbar.language"), self)
        lang_act.triggered.connect(self._on_change_language)
        tb.addAction(lang_act)

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
        self.worker.page_updated.connect(self._on_page_updated)
        self.worker.phase_finished.connect(self._on_phase_finished)
        self.worker.queue_started.connect(self._on_queue_started)
        self.worker.queue_item_started.connect(self._on_queue_item_started)
        self.worker.queue_item_finished.connect(self._on_queue_item_finished)
        self.worker.queue_finished.connect(self._on_queue_finished)

        # Arrow-key navigation is handled at the window level so the same
        # behaviour applies whether the user has the source / detect /
        # translate view focused, or even the original-image side view.
        for tab in (self.source_tab, self.detect_tab, self.translate_tab):
            for view in (tab.view, tab.original_view):
                view.nav_left.connect(self._on_nav_prev_tab)
                view.nav_right.connect(self._on_nav_next_tab)
                view.nav_up.connect(self._on_nav_prev_image)
                view.nav_down.connect(self._on_nav_next_image)

    # ----------------------------------------------------------------- file IO

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("dialog.open_image"), "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        self._load_image_path(Path(path))

    def _on_explorer_image_selected(self, path: Path) -> None:
        self._load_image_path(path)

    def _load_image_path(self, path: Path) -> None:
        # Auto-save the previously-displayed image. Skip while a queue is
        # running because the worker's per-step save_fn is the source of
        # truth there — overwriting with the main window's stale snapshot
        # would erase translations added inside the queue worker.
        if (
            not self._queue_active
            and self.ctx is not None
            and self._source_path is not None
            and self._source_path != path
        ):
            try:
                persistence.save_context(self.ctx, self._source_path)
            except Exception:  # noqa: BLE001
                pass

        try:
            ctx = persistence.load_context(path)
            if ctx is None:
                ctx = PageContext(source=load_rgb(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, tr("dialog.open_failed"), str(e))
            return
        self.ctx = ctx
        self._source_path = path
        self._refresh_tabs(ctx)

        if ctx.final is not None or ctx.translations:
            target = 2
        elif ctx.bboxes or ctx.mask is not None:
            target = 1
        else:
            target = 0
        self.tabs.setCurrentIndex(target)

        h, w = ctx.source.shape[:2]
        suffix = (
            tr("status.restored_suffix")
            if (ctx.bboxes or ctx.translations or ctx.mask is not None)
            else ""
        )
        self.status_bar.showMessage(
            tr("status.loaded", name=path.name, w=w, h=h, suffix=suffix)
        )

    def _on_save(self) -> None:
        if self.ctx is None or self.ctx.final is None:
            QMessageBox.information(
                self, tr("dialog.nothing_to_save_title"), tr("dialog.nothing_to_save_body")
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("dialog.save_final"), "translated.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        save_image(self.ctx.final, path)
        self.status_bar.showMessage(tr("status.saved", path=path), 5000)

    def _on_save_overwrite(self) -> None:
        target = self._default_save_target()
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            save_image(self.ctx.final, target)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, tr("dialog.save_failed"), str(e))
            return
        self.status_bar.showMessage(tr("status.saved", path=target), 5000)

    def _on_save_as(self) -> None:
        target = self._default_save_target()
        if target is None:
            return
        out_dir = target.parent
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(
                self, tr("dialog.cant_create_folder"), f"{out_dir}\n{e}"
            )
            return

        new_name, ok = QInputDialog.getText(
            self,
            tr("dialog.save_as_title"),
            tr("dialog.save_as_prompt", folder=out_dir),
            text=target.name,
        )
        if not ok or not new_name.strip():
            self.status_bar.showMessage(tr("status.save_cancelled"), 4000)
            return
        new_target = out_dir / new_name.strip()
        try:
            save_image(self.ctx.final, new_target)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, tr("dialog.save_failed"), str(e))
            return
        self.status_bar.showMessage(tr("status.saved", path=new_target), 5000)

    def _default_save_target(self) -> Optional[Path]:
        if self.ctx is None or self.ctx.final is None:
            QMessageBox.information(
                self, tr("dialog.nothing_to_save_title"), tr("dialog.nothing_to_save_body")
            )
            return None
        if self._source_path is None:
            QMessageBox.warning(
                self, tr("dialog.no_source_title"), tr("dialog.no_source_body")
            )
            return None
        return self._source_path.parent / "translated" / self._source_path.name

    def _on_set_api_key(self) -> None:
        dlg = ApiKeyDialog(self)
        if dlg.exec():
            self.side_panel.set_api_key_status(get_anthropic_key() is not None)

    def _on_change_language(self) -> None:
        dlg = LanguageDialog(current=self.config.ui_language, parent=self)
        if dlg.exec():
            new_lang = dlg.selected
            if new_lang != self.config.ui_language:
                self.config.ui_language = new_lang
                set_language(new_lang)
                self._save_config()
                QMessageBox.information(
                    self,
                    tr("toolbar.language"),
                    {
                        "ko": "변경된 언어는 다음 실행부터 완전히 적용됩니다.",
                        "en": "The new language will fully apply on next launch.",
                    }.get(new_lang, "Restart to fully apply the new language."),
                )

    # ----------------------------------------------------------------- pipeline

    def _on_run_phase(self, phase: str) -> None:
        if self.ctx is None:
            QMessageBox.information(
                self, tr("dialog.no_image_title"), tr("dialog.no_image_body")
            )
            return
        if phase == PHASE_TRANSLATE and not get_anthropic_key():
            self._on_set_api_key()
            if not get_anthropic_key():
                return
        self._launch_thread(phase=phase)

    def _on_run_all(self) -> None:
        if self.ctx is None:
            QMessageBox.information(
                self, tr("dialog.no_image_title"), tr("dialog.no_image_body")
            )
            return
        if not get_anthropic_key():
            self._on_set_api_key()
            if not get_anthropic_key():
                return
        self._launch_thread(all_phases=True)

    def _on_render(self) -> None:
        if self.ctx is None:
            QMessageBox.information(
                self, tr("dialog.no_image_title"), tr("dialog.no_image_body")
            )
            return
        if not self.ctx.translations:
            QMessageBox.information(
                self,
                tr("dialog.nothing_to_render_title"),
                tr("dialog.nothing_to_render_body"),
            )
            return
        self._launch_thread(step=5)

    def _on_queue_process(self, paths: list, phases_label: str) -> None:
        if not paths:
            return
        needs_translate = (
            phases_label in ("translate", "all")
            and not self.config.step4.skip_translation
        )
        if needs_translate and not get_anthropic_key():
            self._on_set_api_key()
            if not get_anthropic_key():
                return

        if self.ctx is not None and self._source_path is not None:
            try:
                persistence.save_context(self.ctx, self._source_path)
            except Exception:  # noqa: BLE001
                pass

        if not self._confirm_queue_overwrite(paths, phases_label):
            return

        if phases_label == "detect":
            phases: tuple[str, ...] = (PHASE_DETECT,)
            self._queue_target_tab = 1
        elif phases_label == "translate":
            phases = (PHASE_TRANSLATE,)
            self._queue_target_tab = 2
        else:
            phases = (PHASE_DETECT, PHASE_TRANSLATE)
            # End of an end-to-end run lands on Translate.
            self._queue_target_tab = 2

        self._launch_thread(
            queue=[Path(p) for p in paths],
            queue_phases=phases,
        )

    def _confirm_queue_overwrite(self, paths: list, phases_label: str) -> bool:
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
        action = tr(f"dialog.queue_action.{phases_label}")
        reply = QMessageBox.question(
            self,
            tr("dialog.queue_overwrite_title"),
            tr(
                "dialog.queue_overwrite_body",
                count=len(affected),
                sample=sample,
                action=action,
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_queue_save_all(self, paths: list) -> None:
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
            tr("status.save_all_done", saved=saved, skipped=skipped),
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
            QMessageBox.information(self, tr("dialog.busy_title"), tr("dialog.busy_body"))
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
        QMessageBox.critical(self, tr("dialog.step_failed", idx=idx), msg)

    def _on_phase_finished(self, phase: str, ok: bool, msg: str) -> None:
        tab = self.detect_tab if phase == PHASE_DETECT else self.translate_tab
        tab.set_status(
            msg or (tr("tabs.status_done") if ok else tr("tabs.status_failed")),
            ok=ok,
        )
        if ok:
            target = 1 if phase == PHASE_DETECT else 2
            self.tabs.setCurrentIndex(target)
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
        self._queue_active = True
        self.status_bar.showMessage(tr("status.queue_progress", total=total))

    def _on_queue_item_started(self, path: object, idx: int, total: int) -> None:
        p = Path(str(path))
        self.explorer.mark_queue_status(p, "running")
        # _queue_active is True, so _load_image_path will skip its
        # auto-save and only refresh the view + self.ctx pointer.
        self._load_image_path(p)
        # Jump to the tab that matches what the queue is about to produce
        # (Detect → tab 1, Translate / Run All → tab 2). Without this the
        # previous _load_image_path call may leave us on tab 0/1 because
        # the freshly-loaded ctx has no translations yet.
        if self._queue_target_tab is not None:
            self.tabs.setCurrentIndex(self._queue_target_tab)

    def _on_queue_item_finished(self, path: object, ok: bool, msg: str) -> None:
        p = Path(str(path))
        self.explorer.mark_queue_status(p, "done" if ok else "failed")
        self._queue_done = getattr(self, "_queue_done", 0) + 1
        total = getattr(self, "_queue_total", 1)
        self.progress.setValue(self._queue_done)
        self.status_bar.showMessage(
            tr(
                "status.queue_item",
                done=self._queue_done,
                total=total,
                name=p.name,
                status="OK" if ok else "FAIL",
                msg=msg,
            )
        )

    def _on_queue_finished(self) -> None:
        self.progress.setVisible(False)
        total = getattr(self, "_queue_total", 0)
        self.status_bar.showMessage(tr("status.queue_done", total=total), 8000)
        # The queue worker has been writing the up-to-date state to disk via
        # save_fn after every step; refresh the currently-displayed image
        # from disk so self.ctx matches what's on disk before the user
        # navigates away (which would otherwise clobber it with the stale
        # snapshot loaded at queue_item_started time).
        self._queue_active = False
        self._queue_target_tab = None
        if self._source_path is not None:
            self._load_image_path(self._source_path)

    # ----------------------------------------------------------------- editing

    def _on_bbox_geometry_changed(
        self, idx: int, x: int, y: int, w: int, h: int
    ) -> None:
        if self.ctx is None or idx < 0 or idx >= len(self.ctx.bboxes):
            return
        old = self.ctx.bboxes[idx]
        old.x, old.y, old.w, old.h = int(x), int(y), int(w), int(h)
        old.area = int(w) * int(h)
        self.ctx.cleaned = None
        self.ctx.final = None
        self.status_bar.showMessage(
            tr("status.bbox_changed", idx=idx, x=x, y=y, w=w, h=h),
            5000,
        )
        self._auto_save_metadata()

    def _on_bbox_add(self) -> None:
        if self.ctx is None or self.ctx.source is None:
            QMessageBox.information(
                self, tr("dialog.no_image_title"), tr("dialog.no_image_body")
            )
            return
        h, w = self.ctx.source.shape[:2]
        bw, bh = min(160, w // 4), min(80, h // 6)
        x = max(0, (w - bw) // 2)
        y = max(0, (h - bh) // 2)
        self.ctx.bboxes.append(BBox(x=x, y=y, w=bw, h=bh, area=bw * bh))
        self.ctx.cleaned = None
        self.ctx.final = None
        self._refresh_tabs(self.ctx)
        self.status_bar.showMessage(
            tr("status.bbox_added", x=x, y=y, w=bw, h=bh),
            6000,
        )
        self._auto_save_metadata()

    def _on_text_offset_changed(self, idx: int, ox: int, oy: int) -> None:
        if self.ctx is None or idx < 0 or idx >= len(self.ctx.translations):
            return
        tr_item = self.ctx.translations[idx]
        tr_item.text_offset_x = int(ox)
        tr_item.text_offset_y = int(oy)
        self.ctx.final = None
        self.status_bar.showMessage(
            tr("status.translation_offset", idx=idx, ox=ox, oy=oy),
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
        self.ctx.cleaned = None
        self.ctx.final = None
        self._refresh_tabs(self.ctx)
        msg = tr(
            "status.bbox_deleted", idx=idx, remaining=len(self.ctx.bboxes)
        )
        if self.ctx.translations:
            msg += tr("status.bbox_deleted_render_hint")
        self.status_bar.showMessage(msg, 6000)

    def _on_translation_edit(self, idx: int) -> None:
        if self.ctx is None or idx < 0 or idx >= len(self.ctx.translations):
            return
        tr_item = self.ctx.translations[idx]
        dlg = TranslationEditDialog(
            tr_item.text_ja,
            tr_item.text_ko,
            font_path=tr_item.font_path,
            font_pt=tr_item.font_pt,
            text_align=getattr(tr_item, "text_align", "center") or "center",
            text_rotation=int(getattr(tr_item, "text_rotation", 0) or 0),
            available_fonts=self.side_panel.known_fonts,
            default_font_pt=self.config.step5.outside_pt,
            parent=self,
        )
        if dlg.exec():
            current_align = getattr(tr_item, "text_align", "center") or "center"
            current_rotation = int(getattr(tr_item, "text_rotation", 0) or 0)
            changed = (
                dlg.korean != tr_item.text_ko
                or dlg.font_path != tr_item.font_path
                or dlg.font_pt != tr_item.font_pt
                or dlg.text_align != current_align
                or dlg.text_rotation != current_rotation
            )
            if changed:
                tr_item.text_ko = dlg.korean
                # Always centered on bbox now.
                tr_item.ignore_boundary = True
                tr_item.font_path = dlg.font_path
                tr_item.font_pt = dlg.font_pt
                tr_item.text_align = dlg.text_align
                tr_item.text_rotation = dlg.text_rotation
                self.ctx.final = None
                self._refresh_tabs(self.ctx)
                self._launch_thread(step=5)

    # ----------------------------------------------------------------- refresh

    def _on_page_updated(self, ctx: PageContext) -> None:
        """Worker emits this after every step. While a queue is running we
        keep ``self.ctx`` pointed at the worker's mutated context so it
        doesn't get overwritten by a stale snapshot when the user later
        navigates away."""
        if self._queue_active:
            self.ctx = ctx
        self._refresh_tabs(ctx)

    def _refresh_tabs(self, ctx: PageContext) -> None:
        self.source_tab.update_from_context(ctx)
        self.detect_tab.update_from_context(ctx)
        self.translate_tab.update_from_context(ctx)

    # ----------------------------------------------------------------- nav

    def _on_nav_prev_tab(self) -> None:
        idx = self.tabs.currentIndex()
        if idx > 0:
            self.tabs.setCurrentIndex(idx - 1)

    def _on_nav_next_tab(self) -> None:
        idx = self.tabs.currentIndex()
        if idx < self.tabs.count() - 1:
            self.tabs.setCurrentIndex(idx + 1)

    def _on_nav_prev_image(self) -> None:
        # Disable navigation while the queue worker is writing — switching
        # images mid-run would race with save_fn.
        if self._queue_active:
            return
        self.explorer.select_relative(-1, current=self._source_path)

    def _on_nav_next_image(self) -> None:
        if self._queue_active:
            return
        self.explorer.select_relative(+1, current=self._source_path)

    def _save_config(self) -> None:
        try:
            self.config.save()
        except Exception:
            pass

    def _push_render_defaults(self) -> None:
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
