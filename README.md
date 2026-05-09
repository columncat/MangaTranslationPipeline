# 만화 번역 파이프라인 (Manga Translation Pipeline)

일본어 만화를 한국어(혹은 다른 언어)로 번역하는 5단계 AI 파이프라인 데스크톱 앱.  
*A 5-stage AI pipeline desktop app that translates Japanese manga into Korean (or other languages).*

---

## 시연 영상 / Demo

<!-- 시연 영상이 준비되면 아래 주석을 해제하고 YOUR_VIDEO_ID 자리를 채우세요. -->
<!-- [![Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://youtu.be/YOUR_VIDEO_ID) -->

> 시연 영상 속 만화 페이지는 *Manga109-s* 데이터셋의 작품을 사용했습니다.  
> Manga pages shown in the demo are **Courtesy of NAOKO ETO, Manga109-s**.

---

# 한국어 가이드

## 설치

### 사전 요구사항

- Python **3.10 – 3.12** (3.11.9에서 작동 확인)
- CUDA 지원 GPU (선택 사항이지만 강력히 권장)
- Anthropic API 키 ([console.anthropic.com](https://console.anthropic.com))
- 한글 폰트 파일 (기본 폰트가 포함되어 있으나 사용자 지정 폰트를 사용하고 싶다면 `.ttf` 또는 `.otf` 파일을 `fonts/` 폴더에 넣으세요.)

### 설치 절차

저장소 클론 (서브모듈 포함):

```bash
git clone --recurse-submodules https://github.com/columncat/MangaTranslationPipeline.git
cd MangaTranslationPipeline
```

가상환경 설정 후 CUDA 지원 PyTorch 및 의존성을 설치합니다:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -r vendor/comic_text_detector/requirements.txt
```

## 실행

다음 중 한 가지 방법을 사용합니다:

- 파일 탐색기에서 `run.bat` 더블 클릭
- 명령행에서 `.venv\Scripts\python.exe -m src.manga_pipeline`

첫 실행 시 필요한 AI 모델들 (약 800MB)이 자동으로 다운로드됩니다. 진행 팝업이 1회 표시되며 이후 실행부터는 즉시 시작됩니다.

## 사용법

1. **언어 선택** — 최초 실행 시 한국어/영어 중 인터페이스 언어를 선택합니다 (도구 모음의 **언어…** 로 언제든 변경 가능).
2. **API 키 입력** — 사이드패널의 **설정 → API 키** 에서 Anthropic API 키를 입력합니다 (OS 키링에 안전 보관).
3. **폴더 열기** — 왼쪽 탐색 패널에서 만화 이미지 폴더를 선택하거나, 폴더/이미지를 창에 드래그합니다.
4. **작업 대기열 추가** — 처리할 이미지를 작업 대기열에 추가합니다 (Shift / Ctrl+클릭으로 다중 선택 가능).
5. **검출** — 텍스트 마스크와 박스를 자동 추출합니다.
6. **박스 편집** *(선택)* — 검출 탭의 **편집** 모드에서 박스를 드래그·리사이즈합니다.
7. **번역** — OCR → Claude 번역 → 렌더링이 순차 수행됩니다.
8. **번역 수정** *(선택)* — 번역 탭에서 박스를 더블클릭해 한국어 텍스트·정렬·회전·폰트를 편집합니다.
9. **텍스트 위치 조정** *(선택)* — **텍스트 이동** 모드로 대사 위치를 드래그로 조정합니다 (모드 종료 시 자동 재렌더).
10. **저장** — 사이드패널 **저장** 버튼으로 결과 PNG와 메타데이터를 `<원본_폴더>/translated/<원본_이름>` 에 저장합니다.

### 키보드 단축키

| 키 | 동작 |
|---|---|
| **← / →** | 탭 전환 (원본 / 검출 / 번역) |
| **↑ / ↓** | 이전 / 다음 이미지 (현재 이미지가 작업 대기열에 있으면 대기열 안에서만 이동) |
| **Ctrl + R** | 전체 실행 |
| **Ctrl + O** | 파일 열기 |
| **Ctrl + S** | 최종 이미지 저장 |

---

# English Documentation

A 5-stage AI pipeline desktop app that translates Japanese manga into Korean (or other languages).  
Each stage's result can be reviewed in its own tab, and any stage can be re-run independently.

> Destination languages can be changed by modifying the AI prompt under `src/manga_pipeline/pipeline/step4_translate.py`.

> **License**: This project is distributed under [GPL-3.0](LICENSE) due to its runtime dependency on [comic-text-detector](https://github.com/dmMaze/comic-text-detector) (GPL-3.0).

## Pipeline

```
Source manga image
     │
     ▼
┌─────────────┐
│  1. Mask    │  comic-text-detector → binary text mask
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  2. BBoxes  │  OpenCV morphology → bounding box extraction
│             │  (add / delete / move / resize in GUI)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  3. OCR     │  manga-ocr → Japanese text recognition
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ 4. Translate │  Claude API → Korean translation
│              │  (glossary, style notes, prompt caching)
└──────┬───────┘
       │
       ▼
┌─────────────┐
│  5. Render  │  LaMa inpainting + PIL Korean text rendering
└─────────────┘
     │
     ▼
Final translated PNG
```

## Features

| Feature | Description |
|---------|-------------|
| **Per-stage tabs** | Visual preview of each stage's result |
| **Selective re-run** | Adjust parameters and re-run any single stage |
| **BBox editing** | Add, delete, drag-move, and corner-resize boxes in the GUI |
| **Text repositioning** | Move-Text mode: drag dialogue to a new position; auto re-renders on exit |
| **Alignment & rotation** | Per-bubble text alignment (left / center / right) and rotation |
| **Glossary & style notes** | Proper-noun dictionary + free-form style notes in the side panel |
| **Batch queue** | Process whole folders in per-stage batch mode (models stay loaded) |
| **Metadata persistence** | Save / load work state as JSON + PNG sidecars under `<dir>/metadata/` |
| **Side-by-side compare** | View Original splitter for before / after comparison |
| **Drag-and-drop** | Drop a folder or image file onto the window to open it |
| **Bilingual UI** | Korean / English with first-run language picker |
| **Arrow-key navigation** | Left / Right switches tabs, Up / Down walks the queue or folder |

## Installation

### Requirements

- Python **3.10 – 3.12** (3.11.9 confirmed)
- CUDA-capable GPU (optional but strongly recommended)
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Steps

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/columncat/MangaTranslationPipeline.git
cd MangaTranslationPipeline
```

Create a virtual environment, install CUDA-capable PyTorch, then the rest:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -r vendor/comic_text_detector/requirements.txt
```

### Korean font (optional)

The `fonts/` folder ships with five Naver Nanum fonts (OFL-licensed), so the app works out of the box.  
Drop your own `.ttf` / `.otf` into `fonts/` if you want a custom face — it will appear in the side panel's font picker on the next launch (or after pressing **Refresh** in the Fonts library).

### Model weights

Downloaded automatically on first launch (~800 MB total) into `models/` and the HuggingFace cache:

| Model | Size | Provider |
|-------|------|----------|
| `comictextdetector.pt` | ~200 MB | comic-text-detector release on `manga-image-translator` |
| LaMa inpainting | ~200 MB | `simple-lama-inpainting` package |
| `kha-white/manga-ocr-base` | ~400 MB | HuggingFace cache |

A modal popup with a per-model progress bar appears on the first run; subsequent launches skip the popup unless a weight file is missing.

## Usage

Launch the app:

- Double-click `run.bat` (Windows convenience), or
- run `.venv\Scripts\python.exe -m src.manga_pipeline` from a shell.

Workflow:

1. **Pick a language** — first launch shows a Korean / English picker. Change later from the toolbar's **Language…** action.
2. **Set the API key** — side panel **Settings → API Key**; the value is stored in your OS keyring.
3. **Open a folder** — left explorer panel, or drag a folder / image onto the window.
4. **Queue images** — Shift / Ctrl+click to multi-select rows then click *Add → Queue*.
5. **Run Detect** — extracts the text mask and bounding boxes.
6. **Edit boxes** *(optional)* — toggle **Edit** in the Detect tab to drag-move and corner-resize.
7. **Run Translate** — OCR → Claude translation → render in sequence.
8. **Edit translations** *(optional)* — double-click any box in the Translate tab to edit the Korean text, alignment, rotation, font, and font size.
9. **Reposition text** *(optional)* — toggle **Move Text** to drag a dialogue line to a new position; the final image automatically re-renders when you turn the mode off.
10. **Save** — **Save** in the side panel writes the metadata sidecars and the final PNG to `<source_dir>/translated/<source_name>`.

### Keyboard shortcuts

| Key | Action |
|---|---|
| **← / →** | Switch tab (Original → Detect → Translate) |
| **↑ / ↓** | Previous / next image (constrained to the work queue when the current image is queued, otherwise the whole folder) |
| **Ctrl + R** | Run all |
| **Ctrl + O** | Open file |
| **Ctrl + S** | Save final image |

## Project layout

```
MangaTranslationPipeline/
├── src/manga_pipeline/
│   ├── app.py                  # entry point
│   ├── config.py               # Pydantic settings (persisted to config.yaml)
│   ├── i18n.py                 # bilingual UI string table
│   ├── models.py               # PageContext, BBox, OcrResult, TranslationResult
│   ├── persistence.py          # JSON + PNG metadata sidecar IO
│   ├── pipeline/
│   │   ├── step1_mask.py       # text mask generation
│   │   ├── step2_bboxes.py     # bounding box extraction
│   │   ├── step3_ocr.py        # Japanese OCR
│   │   ├── step4_translate.py  # Claude translation (edit prompt to change target language)
│   │   └── step5_render.py     # LaMa inpainting + Korean text rendering
│   ├── gui/
│   │   ├── main_window.py      # main window, drag-and-drop, arrow-key nav
│   │   ├── tabs.py             # source / detect / translate tabs
│   │   ├── image_view.py       # zoom-pan QGraphicsView
│   │   ├── items.py            # editable bbox + draggable text overlays
│   │   ├── explorer.py         # file browser + work queue
│   │   ├── side_panel.py       # parameter / fonts / API-key dock
│   │   ├── dialogs.py          # translation edit dialog
│   │   ├── settings_dialog.py  # API key dialog
│   │   ├── language_dialog.py  # first-run language picker
│   │   ├── download_dialog.py  # first-run model download progress popup
│   │   ├── save_all_dialog.py  # batch save progress popup
│   │   └── workers.py          # QThread pipeline runner + queue worker
│   ├── ml/
│   │   ├── text_detector.py    # comic-text-detector wrapper
│   │   ├── inpainter.py        # LaMa wrapper
│   │   └── weights.py          # weight download helpers
│   └── utils/
│       ├── fonts.py            # font discovery (project + system)
│       ├── image_io.py         # PIL ↔ ndarray helpers
│       └── secrets.py          # OS keyring access
├── vendor/
│   └── comic_text_detector/    # git submodule (GPL-3.0)
├── fonts/                      # bundled Nanum fonts; drop your own here
├── samples/                    # drop your own manga pages here
├── models/                     # auto-populated weight cache (gitignored at runtime)
├── requirements.txt
├── run.bat                     # Windows launcher
└── README.md
```

## License & Dependency Notice

| Component | License |
|-----------|---------|
| This project | GPL-3.0 |
| [comic-text-detector](https://github.com/dmMaze/comic-text-detector) | GPL-3.0 |
| [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) | MIT |
| [manga-ocr](https://github.com/kha-white/manga-ocr) | Apache 2.0 |
| [Naver Nanum fonts](https://hangeul.naver.com/font) | SIL OFL 1.1 |
| PySide6 | LGPL v3 |
| PyTorch | BSD |

This project is distributed under **GPL-3.0** because comic-text-detector is GPL-3.0; downstream redistributors must therefore comply with GPL-3.0 terms.
