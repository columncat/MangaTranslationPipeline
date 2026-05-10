from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
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
from ..history import HistoryManager
from ..i18n import set_language, tr
from ..models import BBox, PageContext, TranslationResult
from ..utils.fonts import find_default_font
from ..utils.image_io import load_rgb, save_image
from ..ai import AIProvider
from ..utils.secrets import get_api_key
from .dialogs import TranslationEditDialog
from .explorer import ExplorerPanel
from .history_dock import HistoryDock
from .language_dialog import LanguageDialog
from .mask_editor import MaskEditorDialog
from .save_all_dialog import SaveAllProgressDialog
from .settings_dialog import ApiKeyDialog
from .side_panel import SidePanel
from .tabs import DetectTab, SourceTab, TranslateTab
from .workers import (
    PHASE_DETECT,
    PHASE_RENDER,
    PHASE_TRANSLATE,
    PipelineThread,
    PipelineWorker,
)


class MainWindow(QMainWindow):
    # Image extensions accepted via drag-and-drop. Anything else is ignored.
    _DND_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.setWindowTitle(tr("app.title"))
        self.resize(1400, 900)
        # Accept dropped files / folders at the window level so the user
        # can drop anywhere — main view, sidebar, toolbar, etc.
        self.setAcceptDrops(True)

        self.config = config if config is not None else AppConfig.load()
        self.ctx: Optional[PageContext] = None
        self._source_path: Optional[Path] = None
        self.worker = PipelineWorker()
        self._thread: Optional[QThread] = None
        # Per-image undo / redo. Reset in _load_image_path so each image
        # gets a fresh timeline rooted at its persisted state.
        self.history = HistoryManager()
        # Set to True while history.undo / .redo is restoring state, so
        # the various editing handlers (which would otherwise try to
        # record a fresh entry on every model change) skip recording.
        self._restoring_history = False
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
        self._build_history_dock()
        self._build_toolbar()
        self._build_status_bar()
        self._wire_signals()

        self.side_panel.set_api_key_status(self._current_api_key_present())
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
        self.translate_tab.bbox_geometry_changed.connect(self._on_bbox_geometry_changed)
        self.translate_tab.translation_edit_requested.connect(self._on_translation_edit)
        self.translate_tab.text_offset_changed.connect(self._on_text_offset_changed)
        self.translate_tab.render_requested.connect(self._on_render)
        self.translate_tab.add_text_requested.connect(self._on_add_translation_text)
        self.translate_tab.mask_edit_requested.connect(self._on_edit_bbox_mask)

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
        # Re-evaluate API-key status whenever the user switches backends.
        self.side_panel.config_changed.connect(
            lambda: self.side_panel.set_api_key_status(
                self._current_api_key_present()
            )
        )

    def _build_history_dock(self) -> None:
        self.history_dock = HistoryDock()
        self.history_dock.set_manager(self.history)
        dock = QDockWidget(tr("history.title"), self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setWidget(self.history_dock)
        # Tab the history dock with the settings dock on the right side
        # so users can flip between them without losing screen real estate.
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        # Keep a reference so we can find/raise the dock if the user
        # closes it inadvertently.
        self._history_dock_widget = dock

        self.history_dock.undo_requested.connect(self._on_undo)
        self.history_dock.redo_requested.connect(self._on_redo)
        self.history_dock.jump_requested.connect(self._on_history_jump)

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

        # Undo / Redo. Save references on the window so we can
        # enable / disable them whenever the history changes.
        self.undo_act = QAction(tr("toolbar.undo"), self)
        self.undo_act.setShortcut(QKeySequence.StandardKey.Undo)  # Ctrl+Z
        self.undo_act.triggered.connect(self._on_undo)
        self.undo_act.setEnabled(False)
        tb.addAction(self.undo_act)

        self.redo_act = QAction(tr("toolbar.redo"), self)
        # Ctrl+Y for the Windows-style redo, plus Ctrl+Shift+Z as a
        # secondary so muscle-memory from Photoshop / Linux apps works too.
        self.redo_act.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.redo_act.triggered.connect(self._on_redo)
        self.redo_act.setEnabled(False)
        tb.addAction(self.redo_act)

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
        # Seed a fresh undo history rooted at the persisted state.
        self.history.reset(ctx, label=tr("history.label.loaded", name=path.name))
        self._refresh_history_ui()

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

    # ---- per-provider API-key plumbing ----

    # Backends that need an API key managed by the keyring.
    _KEYED_PROVIDERS = (
        AIProvider.ANTHROPIC,
        AIProvider.OPENAI_COMPAT,
        AIProvider.GEMINI,
    )

    def _current_provider(self) -> str:
        return getattr(self.config.step4, "provider", AIProvider.ANTHROPIC)

    def _current_api_key_present(self) -> bool:
        provider = self._current_provider()
        if provider not in self._KEYED_PROVIDERS:
            # Local backends don't use a key — treat as "always present"
            # so the side panel doesn't flag a false missing-key warning.
            return True
        return get_api_key(provider) is not None

    def _on_set_api_key(self) -> None:
        provider = self._current_provider()
        # If the user has picked a local backend, opening the dialog
        # would make no sense — surface a short hint instead.
        if provider not in self._KEYED_PROVIDERS:
            QMessageBox.information(
                self,
                tr("apikey.title"),
                tr("apikey.local_backend_no_key"),
            )
            return
        dlg = ApiKeyDialog(provider=provider, parent=self)
        if dlg.exec():
            self.side_panel.set_api_key_status(self._current_api_key_present())

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
        if phase == PHASE_TRANSLATE and not self._current_api_key_present():
            self._on_set_api_key()
            if not self._current_api_key_present():
                return
        self._launch_thread(phase=phase)

    def _on_run_all(self) -> None:
        if self.ctx is None:
            QMessageBox.information(
                self, tr("dialog.no_image_title"), tr("dialog.no_image_body")
            )
            return
        if not self._current_api_key_present():
            self._on_set_api_key()
            if not self._current_api_key_present():
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
        if needs_translate and not self._current_api_key_present():
            self._on_set_api_key()
            if not self._current_api_key_present():
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
        elif phases_label == "render":
            phases = (PHASE_RENDER,)
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
        path_objs = [Path(str(p)) for p in paths]
        dlg = SaveAllProgressDialog(path_objs, parent=self)
        dlg.start()
        dlg.exec()

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
        # Single-image phase runs become history entries so undo can roll
        # back past a Detect / Translate / Render execution. Skip while a
        # queue is running because each queued item triggers its own
        # phase_finished events that are not relevant to the active doc.
        if ok and not self._queue_active and self.ctx is not None:
            label_key = {
                PHASE_DETECT: "history.label.detect",
                PHASE_TRANSLATE: "history.label.translate",
                PHASE_RENDER: "history.label.render",
            }.get(phase)
            if label_key:
                self._record_history(tr(label_key))

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
        # The bbox object is shared with TranslationResult.bbox, so the
        # data is already in sync — but the Translate-tab QGraphicsScene
        # still holds the old overlay rect. Force a refresh so switching
        # tabs (or simply repainting after the user releases the drag)
        # shows the new geometry instead of the stale one.
        self.ctx.cleaned = None
        self.ctx.final = None
        self._refresh_tabs(self.ctx)
        self.status_bar.showMessage(
            tr("status.bbox_changed", idx=idx, x=x, y=y, w=w, h=h),
            5000,
        )
        self._record_history(tr("history.label.bbox_moved", idx=idx))
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
        self._record_history(tr("history.label.bbox_added"))
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
        self._record_history(tr("history.label.text_moved", idx=idx))
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
        self._record_history(tr("history.label.bbox_deleted", idx=idx))
        self._auto_save_metadata()

    def _on_edit_bbox_mask(self, idx: int) -> None:
        """Open the per-bbox mask editor and write the result back.

        The crop is taken from the cleaned image when present (so the user
        sees what's already been inpainted) and otherwise from the source.
        Cancelling leaves the existing mask untouched; clicking Reset
        removes the per-bbox mask so Step 5 falls back to the rectangle.
        """
        if self.ctx is None or idx < 0 or idx >= len(self.ctx.translations):
            return
        tr_item = self.ctx.translations[idx]
        bbox = tr_item.bbox

        base_img = (
            self.ctx.source if self.ctx.cleaned is None else self.ctx.cleaned
        )
        if base_img is None:
            QMessageBox.information(
                self, tr("dialog.no_image_title"), tr("dialog.no_image_body")
            )
            return
        h, w = base_img.shape[:2]
        x0, y0 = max(0, bbox.x), max(0, bbox.y)
        x1, y1 = min(w, bbox.x + bbox.w), min(h, bbox.y + bbox.h)
        if x1 <= x0 or y1 <= y0:
            return
        crop = base_img[y0:y1, x0:x1].copy()

        # Pre-fill the editor with whichever of these is most useful:
        # 1) the existing per-bbox mask if any
        # 2) otherwise the global text mask cropped to this bbox (if Step 1
        #    detected something inside the bbox), so the user starts with
        #    "what the auto-detector thought" and tweaks from there.
        initial = None
        if tr_item.bbox_mask is not None and tr_item.bbox_mask.shape[:2] == (
            bbox.h, bbox.w,
        ):
            initial = tr_item.bbox_mask
        elif self.ctx.mask is not None and self.ctx.mask.shape[:2] == (h, w):
            initial = self.ctx.mask[y0:y1, x0:x1].copy()

        dlg = MaskEditorDialog(crop_rgb=crop, initial_mask=initial, parent=self)
        if dlg.exec():
            tr_item.bbox_mask = dlg.result_mask
            # cleaned/final become stale.
            self.ctx.cleaned = None
            self.ctx.final = None
            self._refresh_tabs(self.ctx)
            self._record_history(tr("history.label.mask_edited", idx=idx))
            self._auto_save_metadata()
            # Rebuild final immediately so the user sees the new inpaint.
            self._launch_thread(step=5)

    def _on_add_translation_text(self) -> None:
        """Insert a fresh user-authored text bubble into the page.

        Adds a default-sized bbox at the image center plus an empty
        TranslationResult tied to it, then opens the edit dialog so the
        user can type the text immediately. After the dialog closes the
        page re-renders with the new bubble in place. The user can then
        switch on Move-Text to drag the bubble to its final position.
        """
        if self.ctx is None or self.ctx.source is None:
            QMessageBox.information(
                self, tr("dialog.no_image_title"), tr("dialog.no_image_body")
            )
            return
        # If there are no existing translations and no cleaned image, the
        # user hasn't run Translate yet. We still allow the action — the
        # renderer will inpaint Step-1 mask intersections with the new
        # bbox, but if there's no mask either we'll just paint on top of
        # the source. The user can adjust manually from there.
        h, w = self.ctx.source.shape[:2]
        bw = min(200, max(40, w // 4))
        bh = min(80, max(20, h // 8))
        x = max(0, (w - bw) // 2)
        y = max(0, (h - bh) // 2)
        bbox = BBox(x=x, y=y, w=bw, h=bh, area=bw * bh)
        self.ctx.bboxes.append(bbox)
        # An empty text_ja keeps Step 4 from re-translating it later if
        # the user runs Translate again — empty Japanese is a clear marker
        # of "this is a user-authored bubble, not OCR output".
        new_tr = TranslationResult(
            bbox=bbox,
            text_ja="",
            text_ko="",
        )
        self.ctx.translations.append(new_tr)
        # Force final to be regenerated when the user re-renders.
        self.ctx.final = None
        new_idx = len(self.ctx.translations) - 1
        self._refresh_tabs(self.ctx)
        # Snapshot the bbox-add itself so cancelling the edit dialog
        # below still leaves an undoable trail.
        self._record_history(tr("history.label.text_added"))
        self._auto_save_metadata()
        # Open the edit dialog right away so the user can type the text.
        # If they save changes there, _on_translation_edit records its
        # own snapshot.
        self._on_translation_edit(new_idx)

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
            fill_rgb=getattr(tr_item, "fill_rgb", None),
            stroke_rgb=getattr(tr_item, "stroke_rgb", None),
            bg_fill_enabled=bool(getattr(tr_item, "bg_fill_enabled", False)),
            bg_fill_rgb=tuple(getattr(tr_item, "bg_fill_rgb", (255, 255, 255))),
            bg_fill_pad=int(getattr(tr_item, "bg_fill_pad", 6)),
            available_fonts=self.side_panel.known_fonts,
            default_font_pt=self.config.step5.outside_pt,
            default_fill_rgb=tuple(self.config.step5.fill_rgb),
            default_stroke_rgb=tuple(self.config.step5.stroke_rgb),
            parent=self,
        )
        if dlg.exec():
            current_align = getattr(tr_item, "text_align", "center") or "center"
            current_rotation = int(getattr(tr_item, "text_rotation", 0) or 0)
            current_fill = getattr(tr_item, "fill_rgb", None)
            current_stroke = getattr(tr_item, "stroke_rgb", None)
            current_bg_on = bool(getattr(tr_item, "bg_fill_enabled", False))
            current_bg_rgb = tuple(getattr(tr_item, "bg_fill_rgb", (255, 255, 255)))
            current_bg_pad = int(getattr(tr_item, "bg_fill_pad", 6))
            changed = (
                dlg.korean != tr_item.text_ko
                or dlg.font_path != tr_item.font_path
                or dlg.font_pt != tr_item.font_pt
                or dlg.text_align != current_align
                or dlg.text_rotation != current_rotation
                or dlg.fill_rgb != current_fill
                or dlg.stroke_rgb != current_stroke
                or dlg.bg_fill_enabled != current_bg_on
                or dlg.bg_fill_rgb != current_bg_rgb
                or dlg.bg_fill_pad != current_bg_pad
            )
            if changed:
                tr_item.text_ko = dlg.korean
                # Always centered on bbox now.
                tr_item.ignore_boundary = True
                tr_item.font_path = dlg.font_path
                tr_item.font_pt = dlg.font_pt
                tr_item.text_align = dlg.text_align
                tr_item.text_rotation = dlg.text_rotation
                tr_item.fill_rgb = dlg.fill_rgb
                tr_item.stroke_rgb = dlg.stroke_rgb
                tr_item.bg_fill_enabled = dlg.bg_fill_enabled
                tr_item.bg_fill_rgb = dlg.bg_fill_rgb
                tr_item.bg_fill_pad = dlg.bg_fill_pad
                self.ctx.final = None
                self._refresh_tabs(self.ctx)
                self._record_history(tr("history.label.translation_edited", idx=idx))
                self._launch_thread(step=5)

    # ----------------------------------------------------------------- history

    def _record_history(self, label: str) -> None:
        """Snapshot the current ctx into the undo stack.

        Skipped while a queue is running (queue edits aren't user
        actions on the displayed page) or while we're in the middle of
        applying an undo/redo (otherwise undo would clear redo).
        """
        if self.ctx is None:
            return
        if self._queue_active or self._restoring_history:
            return
        self.history.record(self.ctx, label)
        self._refresh_history_ui()

    def _refresh_history_ui(self) -> None:
        self.history_dock.refresh()
        self.undo_act.setEnabled(self.history.can_undo())
        self.redo_act.setEnabled(self.history.can_redo())

    def _on_undo(self) -> None:
        if self.ctx is None or not self.history.can_undo():
            return
        self._restoring_history = True
        try:
            entry = self.history.undo(self.ctx)
        finally:
            self._restoring_history = False
        if entry is None:
            return
        self._post_history_apply(tr("status.undo", label=entry.label))

    def _on_redo(self) -> None:
        if self.ctx is None or not self.history.can_redo():
            return
        self._restoring_history = True
        try:
            entry = self.history.redo(self.ctx)
        finally:
            self._restoring_history = False
        if entry is None:
            return
        self._post_history_apply(tr("status.redo", label=entry.label))

    def _post_history_apply(self, status_msg: str) -> None:
        """Common tail for undo/redo/jump: refresh UI, save, re-render.

        Re-rendering is conditional on having translations to render —
        otherwise Step 5 would just error out. The launch is silent and
        non-blocking so the user immediately sees the bbox-level state
        change, with the final image catching up shortly after.
        """
        self._refresh_tabs(self.ctx)
        self._refresh_history_ui()
        self._auto_save_metadata()
        self.status_bar.showMessage(status_msg, 4000)
        # Re-render so the visible Translate-tab final image matches
        # the restored translation/bbox state instead of waiting for
        # the user to press Render manually. Skipped when there's
        # nothing to render or while the worker is busy.
        if (
            self.ctx is not None
            and self.ctx.translations
            and not self._queue_active
            and (self._thread is None or not self._thread.isRunning())
        ):
            self._launch_thread(step=5)

    def _on_history_jump(self, target_idx: int) -> None:
        """Step through undo / redo until the dock-clicked entry is current.

        Bounded by the union of undo + redo stacks; ignored when the
        doc is empty or while a queue is running.
        """
        if self.ctx is None or self._queue_active:
            return
        steps = target_idx - self.history.current_index
        if steps == 0:
            return
        self._restoring_history = True
        try:
            if steps < 0:
                for _ in range(-steps):
                    if not self.history.can_undo():
                        break
                    self.history.undo(self.ctx)
            else:
                for _ in range(steps):
                    if not self.history.can_redo():
                        break
                    self.history.redo(self.ctx)
        finally:
            self._restoring_history = False
        direction = (
            tr("status.redo", label="…") if steps > 0
            else tr("status.undo", label="…")
        )
        self._post_history_apply(direction)

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
        self._nav_image(delta=-1)

    def _on_nav_next_image(self) -> None:
        self._nav_image(delta=+1)

    def _nav_image(self, *, delta: int) -> None:
        """Up/Down arrow navigation.

        If the currently-shown image is in the work queue, navigation is
        constrained to the queue (so a user processing 30 chapter pages
        doesn't wander off into unrelated files). Otherwise it walks the
        folder browser. Disabled while the queue worker is writing.
        """
        if self._queue_active:
            return
        if (
            self._source_path is not None
            and self.explorer.is_in_queue(self._source_path)
        ):
            self.explorer.select_relative_in_queue(delta, current=self._source_path)
        else:
            self.explorer.select_relative(delta, current=self._source_path)

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

    # ----------------------------------------------------------------- DnD

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            # Any local URL (file or folder) is fair game; the actual
            # filtering happens in dropEvent.
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if not urls:
            event.ignore()
            return

        first_path = Path(urls[0].toLocalFile())
        # Folder drop: point the explorer at it and load the first image.
        if first_path.is_dir():
            self.explorer.set_folder(first_path)
            for p in sorted(first_path.iterdir(), key=lambda x: x.name.lower()):
                if (
                    p.is_file()
                    and p.suffix.lower() in self._DND_IMAGE_EXTS
                ):
                    self._load_image_path(p)
                    break
            event.acceptProposedAction()
            return

        # File drop: only image files. Open the first matching file and
        # set the explorer to its parent folder so prev/next nav works.
        image_files = [
            Path(u.toLocalFile())
            for u in urls
            if Path(u.toLocalFile()).is_file()
            and Path(u.toLocalFile()).suffix.lower() in self._DND_IMAGE_EXTS
        ]
        if not image_files:
            event.ignore()
            return
        target = image_files[0]
        self.explorer.set_folder(target.parent)
        self._load_image_path(target)
        event.acceptProposedAction()

    def closeEvent(self, event):  # type: ignore[override]
        self._save_config()
        super().closeEvent(event)
