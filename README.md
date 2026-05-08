# 만화 번역 파이프라인 (Manga Translation Pipeline)

일본어 만화를 한국어로 번역하는 5단계 AI 파이프라인 데스크톱 앱.  
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

## 파이프라인 구조

```
원본 만화 이미지
     │
     ▼
┌─────────────┐
│  1. 텍스트  │  comic-text-detector → 텍스트 마스크 생성
│  마스크     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  2. 박스    │  OpenCV 모폴로지 → 바운딩 박스 추출
│  추출       │  (GUI에서 직접 추가/삭제/이동 가능)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  3. OCR     │  manga-ocr → 일본어 텍스트 인식
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  4. 번역    │  Claude API → 한국어 번역
│             │  (사전·스타일 지정, 프롬프트 캐싱)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  5. 렌더    │  simple-lama-inpainting 인페인팅 →
│  + 합성     │  PIL 한국어 텍스트 렌더링·합성
└─────────────┘
     │
     ▼
최종 번역본 PNG
```

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **단계별 탭 시각화** | Detect / Translate 탭에서 각 단계 결과 즉시 확인 |
| **단계 재실행** | 파라미터 조정 후 원하는 단계만 선택적으로 재실행 |
| **BBox 편집** | GUI에서 박스 추가·삭제·드래그 이동·코너 리사이즈 |
| **텍스트 위치 조정** | Move Text 모드로 번역문 위치를 드래그로 이동 |
| **정렬·회전** | 말풍선별 텍스트 정렬(좌/중/우)과 회전 각도 설정 |
| **번역 사전** | 고유명사 사전 + 스타일 노트를 사이드패널에서 입력 |
| **작업 큐** | 폴더 내 여러 이미지를 일괄 처리 (단계별 배치 모드) |
| **메타데이터 저장** | 작업 결과를 JSON + PNG 사이드카로 저장·불러오기 |
| **View Original** | 스플리터로 원본과 결과를 나란히 비교 |

---

## 설치

### 0. 사전 요구사항

- Python **3.10 – 3.12** (3.11 권장)
- CUDA 지원 GPU (선택 사항 — CPU 동작 가능, 단 Step 1·5 속도 저하)
- Anthropic API 키 ([api.anthropic.com](https://console.anthropic.com))

### 1. 저장소 클론 (서브모듈 포함)

```bash
git clone --recurse-submodules https://github.com/columncat/MangaTranslationPipeline.git
cd MangaTranslationPipeline
```

이미 클론했다면 서브모듈만 따로 초기화:

```bash
git submodule update --init --recursive
```

### 2. 가상환경 + 의존성 설치

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. API 키 설정

```bash
# 환경변수 (임시)
set ANTHROPIC_API_KEY=sk-ant-...       # Windows
export ANTHROPIC_API_KEY=sk-ant-...    # macOS / Linux
```

또는 앱 실행 후 사이드패널 **Settings → API Key** 에서 입력하면 OS 키링에 안전하게 저장됩니다.

### 4. 모델 가중치

첫 실행 시 자동으로 다운로드됩니다 (총 약 800 MB):

| 모델 | 크기 | 저장 위치 |
|------|------|-----------|
| `comictextdetector.pt` | ~200 MB | `models/` |
| LaMa inpainting | ~200 MB | `models/` |
| manga-ocr (HuggingFace) | ~400 MB | HuggingFace 캐시 |

---

## 실행

```bash
python -m manga_pipeline
```

---

## 사용법

1. **폴더 열기** — 왼쪽 탐색 패널에서 만화 이미지가 있는 폴더를 선택합니다.
2. **이미지 선택** — 처리할 이미지를 큐에 추가합니다.
3. **Detect 실행** — 텍스트 마스크·박스를 자동 추출합니다.
4. **박스 편집** (선택) — Detect 탭에서 Edit 모드로 박스를 조정합니다.
5. **Translate 실행** — OCR·번역·렌더링을 순차적으로 수행합니다.
6. **결과 확인** — Translate 탭에서 최종 이미지를 확인합니다.
7. **저장** — 사이드패널 **Save** 버튼으로 메타데이터와 결과를 저장합니다.

---

## 프로젝트 구조

```
MangaTranslationPipeline/
├── src/manga_pipeline/
│   ├── app.py               # 진입점
│   ├── config.py            # 파라미터 설정 (Pydantic)
│   ├── models.py            # PageContext, BBox, OcrResult, TranslationResult
│   ├── persistence.py       # JSON + PNG 직렬화/역직렬화
│   ├── pipeline/
│   │   ├── step1_mask.py    # 텍스트 마스크 생성
│   │   ├── step2_bboxes.py  # 바운딩 박스 추출
│   │   ├── step3_ocr.py     # 일본어 OCR
│   │   ├── step4_translate.py  # Claude 번역
│   │   └── step5_render.py  # 한국어 렌더링
│   ├── gui/
│   │   ├── main_window.py   # 메인 윈도우
│   │   ├── tabs.py          # Detect / Translate 탭
│   │   ├── explorer.py      # 파일 탐색 + 작업 큐
│   │   ├── side_panel.py    # 설정 사이드패널
│   │   ├── dialogs.py       # 번역 편집 다이얼로그
│   │   └── workers.py       # QThread 파이프라인 실행
│   └── ml/
│       ├── text_detector.py # comic-text-detector 래퍼
│       └── inpainter.py     # LaMa 래퍼
├── vendor/
│   └── comic_text_detector/ # git submodule (GPL-3.0)
├── fonts/                   # 번들 한글 폰트 (NanumGothic 등)
├── tests/
└── requirements.txt
```

---

## 테스트

```bash
pytest                  # 단위 테스트 (37개)
pytest -m gpu           # GPU 통합 테스트 (가중치 필요)
```

---

## 라이선스 및 의존성 고지

| 구성요소 | 라이선스 |
|----------|----------|
| 이 프로젝트 | GPL-3.0 |
| [comic-text-detector](https://github.com/dmMaze/comic-text-detector) | GPL-3.0 |
| [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) | MIT |
| [manga-ocr](https://github.com/kha-white/manga-ocr) | Apache 2.0 |
| PySide6 | LGPL v3 |
| PyTorch | BSD |

---
---

# Manga Translation Pipeline

A 5-stage AI pipeline desktop app that translates Japanese manga into Korean.  
Each stage result can be reviewed in its own tab, and any stage can be re-run independently.

> **License**: This project is distributed under [GPL-3.0](LICENSE) due to its runtime dependency on [comic-text-detector](https://github.com/dmMaze/comic-text-detector) (GPL-3.0).

---

## Demo

<!-- Insert your demo video here -->
<!-- Option 1: YouTube -->
<!-- [![Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)](https://youtu.be/YOUR_VIDEO_ID) -->

<!-- Option 2: Local GIF / WebM (upload to assets/ and replace path) -->
<!-- ![Demo GIF](assets/demo.gif) -->

> 📽️ **Demo video coming soon.**

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

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/columncat/MangaTranslationPipeline.git
cd MangaTranslationPipeline
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Virtual environment & dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. API key

```bash
# Temporary (environment variable)
set ANTHROPIC_API_KEY=sk-ant-...       # Windows
export ANTHROPIC_API_KEY=sk-ant-...    # macOS / Linux
```

Or open the app and enter your key under **Settings → API Key** in the side panel — it will be stored securely in your OS keyring.

### 4. Model weights

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
├── fonts/                   # bundled Korean fonts (NanumGothic etc.)
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
