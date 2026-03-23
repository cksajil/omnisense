# 🎬 OmniSense — Multimodal AI Media Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> An end-to-end multimodal AI pipeline that analyzes video and audio using 6+ open-source HuggingFace models — transcription, summarization, NER, image captioning, object detection, and semantic search — all in a single interactive dashboard.

---

## ✨ Features

| Module | Models | Output |
|--------|--------|--------|
| 🎙 **Audio** | `openai/whisper-base` | Timestamped transcript |
| 📝 **NLP** | `facebook/bart-large-cnn`, `bert-base-NER` | Summary, entities, topics |
| 🖼 **Vision** | `openai/clip-vit-base-patch32`, `Salesforce/blip` | Frame captions, object labels |
| 🔍 **Search** | `sentence-transformers/all-MiniLM-L6-v2` + FAISS | Natural language retrieval |

---

## 🏗 Architecture
```
Input (video/audio/URL)
        │
        ▼
┌───────────────────┐
│  Extraction Layer │  FFmpeg frames + audio split
└─────────┬─────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
AudioPipeline  VisionPipeline
(Whisper)     (CLIP + BLIP-2)
    │            │
    └─────┬──────┘
          ▼
    NLPPipeline (BART + NER)
          │
          ▼
    SearchPipeline (MiniLM + FAISS)
          │
          ▼
    Gradio Dashboard
```

---

## 🚀 Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/omnisense.git
cd omnisense
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your HF_TOKEN
python -m omnisense.app
```

Open http://localhost:7860

---

## 📁 Project Structure
```
omnisense/
├── omnisense/
│   ├── pipelines/
│   │   ├── audio.py       # Whisper transcription
│   │   ├── vision.py      # CLIP + BLIP-2 + DETR
│   │   ├── nlp.py         # BART + NER + zero-shot
│   │   └── search.py      # Sentence transformers + FAISS
│   ├── models/            # Model loader helpers
│   ├── utils/
│   │   └── logger.py      # Centralised logging
│   ├── api/               # FastAPI routes
│   ├── config.py          # All settings via env vars
│   └── app.py             # Gradio entry point
├── tests/
├── notebooks/             # EDA and prototyping
├── assets/                # Sample media files
├── docs/
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## 🧪 Running Tests
```bash
pytest
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit with conventional commits: `git commit -m "feat: add speaker diarization"`
4. Push and open a PR

---

## 📄 License

MIT © [Sajil C. K.](https://github.com/cksajil)
