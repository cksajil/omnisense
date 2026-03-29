---
title: OmniSense Temporal Search
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: omnisense/app.py
pinned: false
license: mit
short_description: Find exactly when something was said in any video
---

# 🔍 OmniSense — Temporal Video Search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/sajilck/omnisense)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Find exactly *when* something was said in a video. No more scrubbing.**

Upload any video → it gets transcribed → search in plain English → click a result → video jumps directly to that moment.

---

## ✨ What it does

You remember a speaker mentioned something specific in a long video, but can't find where. OmniSense lets you search the video like a document and jumps you straight to the right moment.

```
Upload 1-hour lecture
    ↓
Search: "transformer architecture"
    ↓
#1  [12:34 → 13:02]  94% match
    "...the transformer architecture introduced in Attention is All You Need..."
#2  [38:17 → 38:51]  71% match
    "...unlike RNNs, transformers process all tokens in parallel..."
    ↓
Click #1 → video seeks to 12:34 and plays ▶
```

---

## 🏗 Architecture

```
Upload video
     │
     ▼
extract_audio()          ffmpeg → 16kHz mono WAV
     │
     ▼
transcribe()             faster-whisper (int8, CPU)
     │                   → List[TranscriptChunk(text, start, end)]
     ▼
TranscriptSearchIndex    MiniLM encode → FAISS IndexFlatIP
     │
     ▼
.search(query)           cosine similarity → ranked SearchHits
     │
     ▼
Gradio UI                result cards + video seek-to-timestamp
```

---

## 🚀 Quickstart (local)

```bash
git clone https://github.com/cksajil/omnisense.git
cd omnisense
git checkout feat/temporal-search

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# ffmpeg must be installed on your system:
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS:         brew install ffmpeg
# Windows:       https://ffmpeg.org/download.html

python -m omnisense.app
# Open http://localhost:7860
```

---

## 📁 Project structure (this branch)

```
omnisense/
├── omnisense/
│   ├── pipelines/
│   │   ├── audio.py       # faster-whisper transcription → TranscriptChunk
│   │   └── search.py      # MiniLM + FAISS → SearchHit
│   └── app.py             # Gradio UI entry point
├── tests/
│   └── test_search.py
├── requirements.txt        # CPU-only, HF Spaces compatible
└── README.md
```

---

## ⚙️ Model guide

| Model | Speed on CPU (per hr of audio) | Accuracy |
|-------|-------------------------------|----------|
| tiny   | ~2–4 min  | Rough — good for quick demos |
| **base** | **~5–8 min**  | **Good — recommended default ✓** |
| small  | ~10–15 min | Better |
| medium | ~25–35 min | Best CPU-feasible accuracy |

---

## 🤗 Deploy to HuggingFace Spaces

```bash
# Add your HF Space as a git remote
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/omnisense

# Push this branch to the Space's main
git push space feat/temporal-search:main
```

Or: HuggingFace → New Space → Gradio → link GitHub repo → select this branch.

---

## 🧪 Running tests

```bash
pytest tests/
```

---

## 📄 License

MIT © [Sajil C. K.](https://github.com/cksajil)
