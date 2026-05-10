# 만화 번역 파이프라인 (Manga Translation Pipeline) — v1.1

일본어 만화를 한국어(혹은 다른 언어)로 번역하는 5단계 AI 파이프라인 데스크톱 앱.  
*A 5-stage AI pipeline desktop app that translates Japanese manga into Korean (or other languages).*

---

## 시연 영상 / Demo

<!-- 시연 영상이 준비되면 아래 주석을 해제하고 YOUR_VIDEO_ID 자리를 채우세요. -->
<!-- [![Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://youtu.be/YOUR_VIDEO_ID) -->

> 📽️ **시연 영상 준비 중입니다 / Demo video coming soon.**
>
> 시연 영상 속 만화 페이지는 *Manga109-s* 데이터셋의 작품을 사용했습니다.  
> Manga pages shown in the demo are **Courtesy of NAOKO ETO, Manga109-s**.

---

## v1.1 주요 변경 / What's new in v1.1

- **AI 백엔드 선택지 6종** — Anthropic Claude / OpenAI 호환 (vLLM·LM Studio·OpenRouter·OpenAI) / Google Gemini / Ollama / llama.cpp (GGUF) / **내장 Gemma 4 E4B (자동 다운로드)**
- **Undo / Redo + 작업 히스토리 도크** (Ctrl+Z / Ctrl+Y), redo 가능 항목 연하게 표시
- **bbox별 인페인트 마스크 편집** (브러시·지우개·전체 채우기/지우기/반전)
- **글자 색·윤곽선 색·타원 배경** per-translation 컨트롤
- **번역 탭에서 텍스트 직접 추가** (자동 검출되지 않은 대사용)
- **번역 탭에서도 bbox 편집** (Detect 탭과 동일한 드래그·리사이즈)
- **작업 대기열에 일괄 렌더 버튼** (번역 결과 유지하며 Step 5만 다시)
- 모델 응답 개행 정확도 개선을 위한 **few-shot 프롬프트**
- 사이드패널 실행 버튼이 클릭 즉시 해당 탭으로 자동 전환

---

# 한국어 가이드

## 설치

### 사전 요구사항

- Python **3.10 – 3.12** (3.11.9에서 작동 확인)
- CUDA 지원 GPU (선택 사항이지만 강력히 권장)
- 사용할 AI 백엔드의 API 키 — Anthropic / OpenAI / Google AI Studio (로컬·내장 백엔드는 키 불필요)

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

선택: **Anthropic Claude 외 다른 AI 백엔드를 사용하려면** 추가 SDK 설치:

```bash
pip install -r requirements-ai.txt
```

내장 **Gemma 4 E4B** 백엔드용 GPU 가속 wheel은 `requirements-ai.txt` 첫 줄의 `--extra-index-url`에서 자동으로 가져옵니다 (Windows / Linux + NVIDIA CUDA).

### 폰트 (선택)

`fonts/` 폴더에 Naver Nanum 5종이 기본 포함되어 있어 별도 설치 없이 동작합니다.  
사용자 지정 폰트를 사용하고 싶다면 `.ttf` 또는 `.otf` 파일을 `fonts/` 폴더에 넣으세요.

## 실행

다음 중 한 가지 방법을 사용합니다:

- 파일 탐색기에서 `run.bat` 더블 클릭
- 명령행에서 `.venv\Scripts\python.exe -m src.manga_pipeline`

첫 실행 시 필요한 AI 모델들 (약 800MB)이 자동으로 다운로드됩니다. 진행 팝업이 1회 표시되며 이후 실행부터는 즉시 시작됩니다.  
**내장 Gemma 4 E4B** 백엔드를 처음 선택하면 GGUF 가중치 (~5GB)가 추가로 다운로드됩니다.

## 사용법

1. **언어 선택** — 최초 실행 시 한국어/영어 중 인터페이스 언어를 선택합니다 (도구 모음 **언어…** 로 언제든 변경).
2. **AI 백엔드 선택** — 사이드패널 **Step 4 → 백엔드** 콤보에서 6개 옵션 중 선택. 클라우드 백엔드(Anthropic·OpenAI 호환·Gemini)는 **API 키…** 버튼으로 키 입력 (OS 키링 저장).
3. **폴더 열기** — 왼쪽 탐색 패널에서 만화 이미지 폴더를 선택하거나, 폴더/이미지를 창에 드래그.
4. **이미지 큐잉** — 처리할 이미지를 작업 대기열에 추가 (Shift / Ctrl+클릭으로 다중 선택).
5. **검출 실행** — 텍스트 마스크와 박스를 자동 추출.
6. **박스 편집** *(선택)* — 검출 또는 번역 탭의 **편집** / **박스 편집** 모드에서 박스 드래그·리사이즈.
7. **번역 실행** — OCR → 선택한 AI 백엔드 → 렌더링 순차 수행.
8. **번역 수정** *(선택)* — 번역 탭에서 박스를 더블클릭해 한국어 텍스트·정렬·회전·폰트·**글자 색·윤곽선 색·타원 배경** 편집.
9. **마스크 편집** *(선택)* — **마스크 편집** 토글 후 박스 더블클릭으로 인페인트 영역을 자유 곡선으로 그림.
10. **텍스트 추가** *(선택)* — **텍스트 추가** 버튼으로 자동 검출되지 않은 대사를 수동 추가.
11. **텍스트 위치 조정** *(선택)* — **텍스트 이동** 모드로 대사 위치 드래그 (모드 종료 시 자동 재렌더).
12. **저장** — 사이드패널 **저장** 버튼으로 결과 PNG와 메타데이터를 `<원본_폴더>/translated/<원본_이름>` 에 저장.

### 키보드 단축키

| 키 | 동작 |
|---|---|
| **← / →** | 탭 전환 (원본 / 검출 / 번역) |
| **↑ / ↓** | 이전 / 다음 이미지 (현재 이미지가 작업 대기열에 있으면 대기열 안에서만 이동) |
| **Ctrl + Z** | 되돌리기 |
| **Ctrl + Y** | 다시 실행 (Ctrl + Shift + Z 도 가능) |
| **Ctrl + R** | 전체 실행 |
| **Ctrl + O** | 파일 열기 |
| **Ctrl + S** | 최종 이미지 저장 |

---

# English Documentation

A 5-stage AI pipeline desktop app that translates Japanese manga into Korean (or other languages).  
Each stage's result can be reviewed in its own tab, and any stage can be re-run independently.

> Destination languages can be changed by modifying the AI prompt under `src/manga_pipeline/ai/base.py` (`SYSTEM_TEMPLATE`).

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
│  4. Translate│  AI backend (Claude / GPT / Gemini / Ollama /
│              │  llama.cpp / embedded Gemma 4) → Korean
│              │  with glossary, style notes, prompt caching
└──────┬───────┘
       │
       ▼
┌─────────────┐
│  5. Render  │  LaMa inpainting + PIL Korean text rendering
│             │  (per-bbox custom mask, color, ellipse backdrop)
└─────────────┘
     │
     ▼
Final translated PNG
```

## AI backends

Pick a translator backend in the side panel's **Step 4 → Backend** dropdown. The rest of the pipeline is identical regardless of which one you choose.

| Backend | Provider | Notes |
|---|---|---|
| **Anthropic** | Claude (Sonnet / Opus / Haiku) | Highest quality. Uses prompt caching to keep batch costs down. |
| **OpenAI-compatible** | OpenAI / OpenRouter / vLLM / LM Studio | Anything that speaks `/v1/chat/completions`. Custom Base URL supported. Auto-falls back across `json_object` → `json_schema` → no-format-hint, and reads `reasoning_content` for Qwen3 / o1 / R1 style models. |
| **Google Gemini** | Gemini 2.0 / 2.5 | Uses `response_mime_type='application/json'` for strict output. |
| **Ollama** | Local Ollama server | Default `http://localhost:11434`. The user just needs to `ollama pull qwen3:8b` (or any other text model) once. |
| **llama.cpp (GGUF)** | Local `.gguf` file | Direct in-process load via `llama-cpp-python`. Pick any GGUF you have on disk. `n_ctx` and `n_gpu_layers` configurable. |
| **Embedded — Gemma 4 E4B** | Bundled GGUF (auto-downloaded) | Zero-config: pick the option, hit Translate, the app downloads `gemma-4-E4B-it-Q4_K_M.gguf` (~5 GB) the first time and runs it through llama-cpp-python with CUDA-prebuilt wheels. Apache 2.0. |

The same system prompt (with few-shot line-break examples) is shared across every backend so output formatting stays consistent.

## Features

| Feature | Description |
|---------|-------------|
| **Per-stage tabs** | Visual preview of each stage's result; the side-panel run buttons jump to the destination tab on click. |
| **Selective re-run** | Adjust parameters and re-run any single stage. Per-image batch render available from the queue. |
| **BBox editing** | Add, delete, drag-move, and corner-resize boxes in either the Detect or Translate tab. |
| **Free-form text bubbles** | Add Text button inserts a new bubble that wasn't auto-detected. |
| **Per-bbox inpaint mask** | Edit Mask mode opens a freehand painter to shape the inpaint region instead of using the bbox rectangle. |
| **Translation editor** | Double-click a bubble for Korean text, font, font size, alignment, rotation, **fill colour, stroke colour, optional ellipse background**. |
| **Text repositioning** | Move-Text mode: drag dialogue to a new position; auto re-renders on exit. |
| **Glossary & style notes** | Proper-noun dictionary + free-form style notes in the side panel. |
| **Multi-backend translation** | Six AI backends including a zero-config embedded Gemma 4 model. |
| **Undo / Redo + history dock** | Ctrl+Z / Ctrl+Y. The dock shows the timeline (redo'able entries dimmed); double-click to jump. |
| **Batch queue** | Process whole folders in per-stage batch mode (models stay loaded). Includes a separate "Render-only" pass. |
| **Save-all popup** | Modal progress dialog saves every queued image's final PNG with auto-close countdown. |
| **Metadata persistence** | Save / load work state as JSON + PNG sidecars under `<dir>/metadata/`. |
| **Side-by-side compare** | View Original splitter for before / after comparison. |
| **Drag-and-drop** | Drop a folder or image file onto the window to open it. |
| **Bilingual UI** | Korean / English with first-run language picker. |
| **Arrow-key navigation** | Left / Right switches tabs, Up / Down walks the queue or folder. |

## Installation

### Requirements

- Python **3.10 – 3.12** (3.11.9 confirmed)
- CUDA-capable GPU (optional but strongly recommended)
- API key for whichever cloud backend you intend to use (local / embedded backends need no key)

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

If you want to use any backend other than Anthropic Claude, also:

```bash
pip install -r requirements-ai.txt
```

This pulls in `openai`, `google-genai`, and `llama-cpp-python` (the prebuilt CUDA wheel via the `--extra-index-url` line at the top of the file). Ollama is reached over plain HTTP through `requests` and needs no extra package.

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

### Embedded translation LLM (optional, on demand)

If you switch the Step 4 backend to **Embedded — Gemma 4 E4B**, the app downloads one extra weight file the first time you run Translate:

| Model | Size | Source |
|-------|------|--------|
| `gemma-4-E4B-it-Q4_K_M.gguf` | ~5 GB | [`unsloth/gemma-4-E4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) on HuggingFace |

A separate progress dialog handles this, and the file is cached at `models/gemma-4-E4B-it-Q4_K_M.gguf` for subsequent runs.

## Usage

Launch the app:

- Double-click `run.bat` (Windows convenience), or
- run `.venv\Scripts\python.exe -m src.manga_pipeline` from a shell.

Workflow:

1. **Pick a language** — first launch shows a Korean / English picker. Change later from the toolbar's **Language…** action.
2. **Pick an AI backend** — side panel Step 4 → Backend. For cloud backends, set the matching API key via **API key…**; the value is stored in your OS keyring per-provider.
3. **Open a folder** — left explorer panel, or drag a folder / image onto the window.
4. **Queue images** — Shift / Ctrl+click to multi-select rows then click *Add → Queue*.
5. **Run Detect** — extracts the text mask and bounding boxes. The Detect tab is opened automatically.
6. **Edit boxes** *(optional)* — toggle **Edit** in either the Detect tab or the Translate tab to drag-move and corner-resize.
7. **Run Translate** — OCR → AI translation → render in sequence. Lands on the Translate tab.
8. **Edit translations** *(optional)* — double-click any box in the Translate tab to edit the Korean text, alignment, rotation, font, font size, fill / stroke colour, and optional ellipse background.
9. **Edit per-bbox mask** *(optional)* — toggle **Edit Mask**, double-click a bubble to paint the inpaint region freehand instead of using its rectangle.
10. **Add a free-form text bubble** *(optional)* — Translate-tab **Add Text** button inserts a new bbox + empty translation at the image centre.
11. **Reposition text** *(optional)* — toggle **Move Text** to drag a dialogue line; the final image automatically re-renders when you turn the mode off.
12. **Save** — **Save** in the side panel writes the metadata sidecars and the final PNG to `<source_dir>/translated/<source_name>`.

### Undo / Redo

Every editing action (bbox add / delete / move, translation edit, text move, mask paint, free-form text insert, single-image phase runs) is recorded in a per-image timeline shown in the right-side **History** dock. Up to 50 entries are kept; opening a different image resets the timeline.

- **Ctrl + Z** — undo
- **Ctrl + Y** or **Ctrl + Shift + Z** — redo
- Double-clicking an entry in the dock jumps directly to that state. Redo'able entries are listed dimmed past the current state.
- Undo / redo automatically re-runs Step 5 so the visible final image stays in sync with the restored translation/bbox state.

### Keyboard shortcuts

| Key | Action |
|---|---|
| **← / →** | Switch tab (Original → Detect → Translate) |
| **↑ / ↓** | Previous / next image (constrained to the work queue when the current image is queued, otherwise the whole folder) |
| **Ctrl + Z** | Undo |
| **Ctrl + Y** / **Ctrl + Shift + Z** | Redo |
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
│   ├── history.py              # per-image undo / redo manager
│   ├── ai/                     # AI backend abstraction layer
│   │   ├── base.py             # Translator protocol, factory, prompts
│   │   ├── anthropic_backend.py
│   │   ├── openai_compat.py
│   │   ├── gemini.py
│   │   ├── ollama.py
│   │   ├── llamacpp.py
│   │   └── embedded.py         # Gemma 4 E4B auto-download wrapper
│   ├── pipeline/
│   │   ├── step1_mask.py       # text mask generation
│   │   ├── step2_bboxes.py     # bounding box extraction
│   │   ├── step3_ocr.py        # Japanese OCR
│   │   ├── step4_translate.py  # adapter onto manga_pipeline.ai
│   │   └── step5_render.py     # LaMa inpainting + Korean text rendering
│   ├── gui/
│   │   ├── main_window.py      # main window, drag-and-drop, arrow-key nav
│   │   ├── tabs.py             # source / detect / translate tabs
│   │   ├── image_view.py       # zoom-pan QGraphicsView
│   │   ├── items.py            # editable bbox + draggable text overlays
│   │   ├── explorer.py         # file browser + work queue
│   │   ├── side_panel.py       # parameter / fonts / API-key dock
│   │   ├── dialogs.py          # translation edit dialog (text / colour / bg)
│   │   ├── settings_dialog.py  # per-provider API key dialog
│   │   ├── language_dialog.py  # first-run language picker
│   │   ├── download_dialog.py  # first-run model download progress popup
│   │   ├── embedded_download_dialog.py  # Gemma 4 GGUF download popup
│   │   ├── mask_editor.py      # freehand per-bbox mask painter
│   │   ├── history_dock.py     # undo / redo timeline dock
│   │   ├── save_all_dialog.py  # batch save progress popup
│   │   └── workers.py          # QThread pipeline runner + queue worker
│   ├── ml/
│   │   ├── text_detector.py    # comic-text-detector wrapper
│   │   ├── inpainter.py        # LaMa wrapper
│   │   └── weights.py          # weight download helpers (incl. embedded LLM)
│   └── utils/
│       ├── fonts.py            # font discovery (project + system)
│       ├── image_io.py         # PIL ↔ ndarray helpers
│       └── secrets.py          # per-provider OS keyring access
├── vendor/
│   └── comic_text_detector/    # git submodule (GPL-3.0)
├── fonts/                      # bundled Nanum fonts; drop your own here
├── samples/                    # drop your own manga pages here
├── models/                     # auto-populated weight cache (gitignored at runtime)
├── requirements.txt            # base dependencies (Anthropic Claude only)
├── requirements-ai.txt         # extra SDKs for OpenAI / Gemini / llama.cpp
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
| [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (embedded LLM) | Apache 2.0 + Gemma terms of use |
| [Naver Nanum fonts](https://hangeul.naver.com/font) | SIL OFL 1.1 |
| PySide6 | LGPL v3 |
| PyTorch | BSD |
| `openai`, `google-genai`, `llama-cpp-python` (optional) | Apache 2.0 / MIT |

This project is distributed under **GPL-3.0** because comic-text-detector is GPL-3.0; downstream redistributors must therefore comply with GPL-3.0 terms.
