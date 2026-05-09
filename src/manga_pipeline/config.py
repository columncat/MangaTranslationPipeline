from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from .paths import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Step1Params(BaseModel):
    mask_threshold: float = 0.3
    mask_dilate_px: int = 3
    inpaint_dilate_px: int = 7


class Step2Params(BaseModel):
    kernel_w: int = 15
    kernel_h: int = 15
    iterations: int = 1
    min_area: int = 200
    max_area_ratio: float = 0.7


class Step4Params(BaseModel):
    model: str = "claude-sonnet-4-6"
    glossary: str = ""
    style_notes: str = "natural Korean manga dialogue, preserve tone and honorifics"
    max_tokens: int = 4096
    skip_translation: bool = False


class Step5Params(BaseModel):
    """Render parameters.

    Translations are always rendered with ``ignore_boundary`` semantics
    (centered on the bbox, only user newlines split lines), so the
    in-bbox auto-fit fields (``min_pt``, ``max_pt``, ``padding``) have
    been removed. ``outside_pt`` is the single default font size.
    """

    font_path: str = ""
    outside_pt: int = 25
    line_spacing: float = 1.10
    stroke_px: int = 2
    fill_rgb: tuple[int, int, int] = (0, 0, 0)
    stroke_rgb: tuple[int, int, int] = (255, 255, 255)


class AppConfig(BaseModel):
    step1: Step1Params = Field(default_factory=Step1Params)
    step2: Step2Params = Field(default_factory=Step2Params)
    step4: Step4Params = Field(default_factory=Step4Params)
    step5: Step5Params = Field(default_factory=Step5Params)
    # External fonts (outside fonts/ folder) added by the user. Files inside
    # fonts/ are auto-discovered each launch so they don't need to be tracked
    # here. Absolute paths only.
    external_fonts: list[str] = Field(default_factory=list)
    # ``None`` means "ask on next launch" — the language chooser dialog
    # only appears while this field is unset.
    ui_language: Optional[str] = None
    # Toggled to True after the first-run download dialog has finished
    # successfully. Suppresses re-showing the popup on subsequent launches
    # (weights still re-download themselves silently if missing).
    first_run_done: bool = False

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AppConfig":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # ``fonts`` was renamed to ``external_fonts`` — preserve old configs.
        if "fonts" in data and "external_fonts" not in data:
            data["external_fonts"] = data.pop("fonts")
        return cls.model_validate(data)

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(
            yaml.safe_dump(self.model_dump(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
