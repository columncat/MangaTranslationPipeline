# 만화 번역 파이프라인 (Manga Translation Pipeline)

일본어 만화를 한국어(혹은 다른 언어)로 번역하는 5단계 AI 파이프라인 데스크톱 앱.  
각 단계의 결과를 탭별로 시각 확인하고, 원하는 단계만 골라 재실행할 수 있습니다.

> **라이선스**: 이 프로젝트는 [GPL-3.0](LICENSE) 라이선스로 배포됩니다.  
> 런타임 의존성인 [comic-text-detector](https://github.com/dmMaze/comic-text-detector)가 GPL-3.0이기 때문입니다.

---

## 시연 영상

<!-- 시연 영상을 여기에 삽입하세요 -->
<!-- 방법 1: YouTube 링크 -->
<!-- [![시연 영상](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://youtu.be/YOUR_VIDEO_ID) -->

<!-- 방법 2: 로컬 GIF / WebM (저장소 내 assets/ 폴더에 업로드 후 경로 교체) -->
<!-- ![시연 GIF](assets/demo.gif) -->

> 📽️ **시연 영상 준비 중입니다.**

---

## 설치

### 0. 사전 요구사항

- Python **3.10 – 3.12** (3.11.9에서 작동 확인)
- CUDA 지원 GPU (선택 사항이지만 강력히 권장)
- Anthropic API 키 ([console.anthropic.com](https://console.anthropic.com))
- 한글 폰트 파일 (.ttf 혹은 .otf)

### 1. 설치

레포지터리 클론

```bash
git clone --recurse-submodules https://github.com/columncat/MangaTranslationPipeline.git
cd MangaTranslationPipeline
```

가상환경 설정 후 CUDA 지원 Pytorch 및 의존성 설치

```bash
python -m venv .venv
.venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -r vendor/comic_text_detector/requirements.txt
```

## 실행

파일 탐색기에서 run.bat 더블 클릭 혹은

```bash
.venv\Scripts\python.exe -m src.manga_pipeline
```

첫 실행 시 필요한 AI 모델들이 자동으로 다운로드됩니다.

---

## 사용법

1. **폴더 열기** — 왼쪽 탐색 패널에서 만화 이미지가 있는 폴더를 선택합니다.
2. **이미지 선택** — 처리할 이미지를 큐에 추가합니다.
3. **Detect 실행** — 텍스트 마스크·박스를 자동 추출합니다.
4. **박스 편집** (선택) — Detect 탭에서 Edit 모드로 박스를 조정합니다.
5. **Translate 실행** — OCR·번역·렌더링을 순차적으로 수행합니다.
6. **번역 수정** (선택) — 자동 생성된 번역을 편집합니다.
7. **결과 확인** — Render를 눌러 최종 이미지를 확인합니다.
8. **저장** — 사이드패널 **Save** 버튼으로 메타데이터와 결과를 저장합니다.

---
---

# Manga Translation Pipeline

A 5-stage AI pipeline desktop app that translates Japanese manga into Korean(or other languages).  
Each stage result can be reviewed in its own tab, and any stage can be re-run independently.

> **License**: This project is distributed under [GPL-3.0](LICENSE) due to its runtime dependency on [comic-text-detector](https://github.com/dmMaze/comic-text-detector) (GPL-3.0).

---

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
┌─────────────┐
│  4. Translate│  Claude API → Korean translation
│             │  (glossary, style notes, prompt caching)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  5. Render  │  LaMa inpainting + PIL Korean text rendering
└─────────────┘
     │
     ▼
Final translated PNG
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Per-stage tabs** | Visual preview of each stage result |
| **Selective re-run** | Adjust parameters and re-run any single stage |
| **BBox editing** | Add, delete, drag-move, and corner-resize boxes in the GUI |
| **Text repositioning** | Move Text mode: drag translated text to a new position |
| **Alignment & rotation** | Per-bubble text alignment (left/center/right) and rotation |
| **Glossary** | Proper noun dictionary + style notes in the side panel |
| **Batch queue** | Process entire folders in per-stage batch mode |
| **Metadata persistence** | Save/load work state as JSON + PNG sidecars |
| **Side-by-side compare** | View Original splitter for before/after comparison |

---

## Installation

### 0. Requirements

- Python **3.10 – 3.12** (3.11 recommended)
- CUDA-capable GPU (optional — CPU works, but Stage 1 & 5 will be slower)
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### 1. Clone

```bash
git clone --recurse-submodules https://github.com/columncat/MangaTranslationPipeline.git
cd MangaTranslationPipeline
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r vendor/comic_text_detector/requirements.txt
```

### 3. API key

Launch the app and enter your key under **Settings → API Key** in the side panel — it will be stored securely in your OS keyring.

### 4. Korean font

Place a Korean TTF/OTF font file in the `fonts/` folder. The app scans that folder on startup and lists available fonts in the side panel.  
If no font is placed there, it will search for a Korean-capable font installed on your system.

### 5. Model weights

Downloaded automatically on first run (~800 MB total):

| Model | Size | Location |
|-------|------|----------|
| `comictextdetector.pt` | ~200 MB | `models/` |
| LaMa inpainting | ~200 MB | `models/` |
| manga-ocr (HuggingFace) | ~400 MB | HuggingFace cache |

---

## Usage

```bash
python -m manga_pipeline
```

1. **Open folder** — select the folder containing your manga images in the left explorer panel.
2. **Queue images** — add the images you want to process.
3. **Run Detect** — auto-extract text mask and bounding boxes.
4. **Edit boxes** (optional) — switch to Edit mode in the Detect tab to adjust boxes.
5. **Run Translate** — perform OCR → translation → rendering in sequence.
6. **Review** — inspect the final image in the Translate tab.
7. **Save** — click **Save** in the side panel to persist metadata and output.

---

## Project layout

```
MangaTranslationPipeline/
├── src/manga_pipeline/
│   ├── app.py               # entry point
│   ├── config.py            # Pydantic settings
│   ├── models.py            # PageContext, BBox, OcrResult, TranslationResult
│   ├── persistence.py       # JSON + PNG serialisation
│   ├── pipeline/
│   │   ├── step1_mask.py    # text mask generation
│   │   ├── step2_bboxes.py  # bounding box extraction
│   │   ├── step3_ocr.py     # Japanese OCR
│   │   ├── step4_translate.py  # Claude translation
│   │   └── step5_render.py  # Korean text rendering
│   ├── gui/
│   │   ├── main_window.py   # main window
│   │   ├── tabs.py          # Detect / Translate tabs
│   │   ├── explorer.py      # file explorer + queue
│   │   ├── side_panel.py    # settings side panel
│   │   ├── dialogs.py       # translation edit dialog
│   │   └── workers.py       # QThread pipeline runner
│   └── ml/
│       ├── text_detector.py # comic-text-detector wrapper
│       └── inpainter.py     # LaMa wrapper
├── vendor/
│   └── comic_text_detector/ # git submodule (GPL-3.0)
├── tests/
└── requirements.txt
```

---

## Tests

```bash
pytest                  # unit tests (37 cases)
pytest -m gpu           # GPU integration tests (requires model weights)
```

---

## License & Dependency Notice

| Component | License |
|-----------|---------|
| This project | GPL-3.0 |
| [comic-text-detector](https://github.com/dmMaze/comic-text-detector) | GPL-3.0 |
| [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) | MIT |
| [manga-ocr](https://github.com/kha-white/manga-ocr) | Apache 2.0 |
| PySide6 | LGPL v3 |
| PyTorch | BSD |
