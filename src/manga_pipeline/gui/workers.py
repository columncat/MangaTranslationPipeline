from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal

from ..config import AppConfig
from ..models import PageContext
from ..pipeline.base import PipelineStep, StepResult
from ..pipeline.step1_mask import Step1Mask
from ..pipeline.step2_bboxes import Step2BBoxes
from ..pipeline.step3_ocr import Step3Ocr
from ..pipeline.step4_translate import Step4Translate
from ..pipeline.step5_render import Step5Render

PHASE_DETECT = "detect"
PHASE_TRANSLATE = "translate"

PHASE_STEPS = {
    PHASE_DETECT: (1, 2),
    PHASE_TRANSLATE: (3, 4, 5),
}

QUEUE_MODE_SEQUENTIAL = "sequential"
QUEUE_MODE_PER_STEP = "per_step"

LoadFn = Callable[[Path], Optional[PageContext]]
SaveFn = Callable[[PageContext, Path], None]


class PipelineWorker(QObject):
    started = Signal(int, str)
    progress = Signal(int, int, int, str)
    finished = Signal(int, bool, str)
    failed = Signal(int, str)
    page_updated = Signal(object)
    phase_started = Signal(str)
    phase_finished = Signal(str, bool, str)

    queue_started = Signal(int)                  # total
    queue_item_started = Signal(object, int, int)   # path, idx, total
    queue_item_finished = Signal(object, bool, str)  # path, ok, message
    queue_finished = Signal()

    STEP_NAMES = {
        1: "Step 1 — Mask",
        2: "Step 2 — Bounding Boxes",
        3: "Step 3 — OCR",
        4: "Step 4 — Translate",
        5: "Step 5 — Render",
    }

    PHASE_LABELS = {
        PHASE_DETECT: "Detect (mask + bboxes)",
        PHASE_TRANSLATE: "Translate (OCR + translate + render)",
    }

    def __init__(self):
        super().__init__()
        self._steps: dict[int, PipelineStep] = {}
        self._cancelled = False

    def _step_for(self, idx: int) -> PipelineStep:
        if idx not in self._steps:
            if idx == 1:
                self._steps[1] = Step1Mask()
            elif idx == 2:
                self._steps[2] = Step2BBoxes()
            elif idx == 3:
                self._steps[3] = Step3Ocr()
            elif idx == 4:
                self._steps[4] = Step4Translate()
            elif idx == 5:
                self._steps[5] = Step5Render()
            else:
                raise ValueError(f"unknown step index {idx}")
        return self._steps[idx]

    def cancel(self) -> None:
        self._cancelled = True

    def reset_cancel(self) -> None:
        self._cancelled = False

    def _params_for(self, idx: int, config: AppConfig):
        return {
            1: config.step1,
            2: config.step2,
            3: None,
            4: config.step4,
            5: config.step5,
        }[idx]

    def _run_step_no_view_update(
        self, idx: int, ctx: PageContext, config: AppConfig
    ) -> StepResult:
        """Run one step and emit started/progress/finished, but NOT page_updated."""
        self.started.emit(idx, self.STEP_NAMES.get(idx, f"step {idx}"))

        def cb(cur: int, total: int, msg: str) -> None:
            self.progress.emit(idx, cur, total, msg)

        try:
            step = self._step_for(idx)
            params = self._params_for(idx, config)
            result = step.run(ctx, params, cb)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(idx, str(e))
            return StepResult(ok=False, message=str(e), error=e)

        self.finished.emit(idx, result.ok, result.message)
        return result

    def run_step(self, idx: int, ctx: PageContext, config: AppConfig) -> StepResult:
        result = self._run_step_no_view_update(idx, ctx, config)
        self.page_updated.emit(ctx)
        return result

    def run_phase(
        self, phase: str, ctx: PageContext, config: AppConfig
    ) -> StepResult:
        """Run one composite phase.

        - ``detect``: steps 1 then 2. View update only fires once at the end.
        - ``translate``: steps 3, 4, 5. View update fires after each substep
          so the user gets progressive feedback (OCR → translation → final).
        """
        steps = PHASE_STEPS.get(phase)
        if steps is None:
            return StepResult(ok=False, message=f"unknown phase {phase!r}")

        self.reset_cancel()
        self.phase_started.emit(phase)

        if phase == PHASE_DETECT:
            ctx.bboxes = []
            ctx.ocr = []
            ctx.translations = []
            ctx.cleaned = None
            ctx.final = None
        else:  # translate
            ctx.ocr = []
            ctx.translations = []
            ctx.cleaned = None
            ctx.final = None

        last: StepResult = StepResult(ok=True, message="")
        for idx in steps:
            if self._cancelled:
                last = StepResult(ok=False, message="cancelled")
                break
            last = self._run_step_no_view_update(idx, ctx, config)
            if phase == PHASE_TRANSLATE:
                self.page_updated.emit(ctx)
            if not last.ok:
                break

        if phase == PHASE_DETECT:
            self.page_updated.emit(ctx)

        self.phase_finished.emit(phase, last.ok, last.message)
        return last

    def run_all_phases(self, ctx: PageContext, config: AppConfig) -> None:
        for phase in (PHASE_DETECT, PHASE_TRANSLATE):
            if self._cancelled:
                break
            r = self.run_phase(phase, ctx, config)
            if not r.ok:
                break

    # ---- queue ----

    def run_queue(
        self,
        paths: list[Path],
        mode: str,
        config: AppConfig,
        load_fn: LoadFn,
        save_fn: SaveFn,
        *,
        phases: tuple[str, ...] = (PHASE_DETECT, PHASE_TRANSLATE),
    ) -> None:
        self.reset_cancel()
        total = len(paths)
        self.queue_started.emit(total)

        try:
            if mode == QUEUE_MODE_PER_STEP:
                self._run_queue_per_step(paths, config, load_fn, save_fn, phases)
            else:
                self._run_queue_sequential(paths, config, load_fn, save_fn, phases)
        finally:
            self.queue_finished.emit()

    def _run_queue_sequential(
        self,
        paths: list[Path],
        config: AppConfig,
        load_fn: LoadFn,
        save_fn: SaveFn,
        phases: tuple[str, ...],
    ) -> None:
        total = len(paths)
        for i, p in enumerate(paths):
            if self._cancelled:
                break
            self.queue_item_started.emit(p, i, total)
            ctx = load_fn(p)
            if ctx is None:
                self.queue_item_finished.emit(p, False, "load failed")
                continue
            ok = True
            message = "done"
            for phase in phases:
                if self._cancelled:
                    ok = False
                    message = "cancelled"
                    break
                r = self.run_phase(phase, ctx, config)
                self._safe_save(ctx, p, save_fn)
                if not r.ok:
                    ok = False
                    message = r.message or f"{phase} failed"
                    break
            self.queue_item_finished.emit(p, ok, message)

    def _run_queue_per_step(
        self,
        paths: list[Path],
        config: AppConfig,
        load_fn: LoadFn,
        save_fn: SaveFn,
        phases: tuple[str, ...],
    ) -> None:
        # Compute the actual step indices from the requested phases.
        step_seq: list[int] = []
        for phase in phases:
            step_seq.extend(PHASE_STEPS.get(phase, ()))

        # Load all sources up front so each step only has to do its work.
        contexts: list[Optional[PageContext]] = []
        for i, p in enumerate(paths):
            self.queue_item_started.emit(p, i, len(paths))
            ctx = load_fn(p)
            contexts.append(ctx)
            if ctx is None:
                self.queue_item_finished.emit(p, False, "load failed")

        # Cascade-clear for the phases we're about to run.
        for ctx in contexts:
            if ctx is None:
                continue
            if PHASE_DETECT in phases:
                ctx.bboxes = []
            ctx.ocr = []
            ctx.translations = []
            ctx.cleaned = None
            ctx.final = None

        failed: set[int] = {i for i, c in enumerate(contexts) if c is None}

        for step_idx in step_seq:
            for i, (ctx, path) in enumerate(zip(contexts, paths)):
                if i in failed or ctx is None:
                    continue
                if self._cancelled:
                    failed.add(i)
                    self.queue_item_finished.emit(path, False, "cancelled")
                    continue
                result = self._run_step_no_view_update(step_idx, ctx, config)
                self.page_updated.emit(ctx)
                self._safe_save(ctx, path, save_fn)
                if not result.ok:
                    failed.add(i)
                    msg = result.message or f"step {step_idx} failed"
                    self.queue_item_finished.emit(path, False, msg)

        # Anything not already reported as failed is done.
        for i, path in enumerate(paths):
            if i in failed:
                continue
            self.queue_item_finished.emit(path, True, "done")

    @staticmethod
    def _safe_save(ctx: PageContext, path: Path, save_fn: SaveFn) -> None:
        try:
            save_fn(ctx, path)
        except Exception:  # noqa: BLE001 — saving must never break the queue
            pass


class PipelineThread(QThread):
    def __init__(
        self,
        worker: PipelineWorker,
        ctx: Optional[PageContext],
        config: AppConfig,
        *,
        step: Optional[int] = None,
        phase: Optional[str] = None,
        all_phases: bool = False,
        queue: Optional[list[Path]] = None,
        queue_mode: Optional[str] = None,
        queue_phases: tuple[str, ...] = (PHASE_DETECT, PHASE_TRANSLATE),
        load_fn: Optional[LoadFn] = None,
        save_fn: Optional[SaveFn] = None,
    ):
        super().__init__()
        self.worker = worker
        self.ctx = ctx
        self.config = config
        self.step = step
        self.phase = phase
        self.all_phases = all_phases
        self.queue = queue
        # Per-step batch is the only mode wired up in the GUI now; the
        # parameter is kept for callers that may want to override.
        self.queue_mode = queue_mode or QUEUE_MODE_PER_STEP
        self.queue_phases = queue_phases
        self.load_fn = load_fn
        self.save_fn = save_fn

    def run(self) -> None:  # type: ignore[override]
        if self.queue is not None:
            assert self.load_fn is not None and self.save_fn is not None
            self.worker.run_queue(
                self.queue,
                self.queue_mode,
                self.config,
                self.load_fn,
                self.save_fn,
                phases=self.queue_phases,
            )
            return
        if self.ctx is None:
            return
        if self.all_phases:
            self.worker.run_all_phases(self.ctx, self.config)
        elif self.phase is not None:
            self.worker.run_phase(self.phase, self.ctx, self.config)
        elif self.step is not None:
            self.worker.run_step(self.step, self.ctx, self.config)
