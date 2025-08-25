# MDE.Learn.Chatbot

Local end-to-end RAG + LoRA fine-tuning pipeline for Microsoft Defender for Endpoint (MDE) docs:
- Crawls and cleans MDE docs
- Chunks and builds a vector index (Chroma + Sentence-Transformers)
- Fine-tunes Qwen2.5-7B-Instruct-4bit on Apple Silicon with MLX + LoRA
- Serves a Streamlit chatbot with modes: rag | ft | rag_ft
- Optional background update service with HTTP `/update`

Runs fully on macOS Apple Silicon. Base model downloads locally on first use.

Public documentation: https://learn.microsoft.com/en-us/defender-endpoint/

## Features

- Crawler: robots-aware, domain/path constrained to MDE docs
- Processing: HTML → clean text → chunking
- Indexing: persistent Chroma store with MiniLM embeddings
- Training: MLX LoRA finetuning + optional merge to standalone weights
- Inference: adapter-first loading with fallback to merged or base
- App: Streamlit UI with token streaming and source attributions
- Auto-updates: scheduled pipeline + HTTP trigger

## Key Concepts and Links

- MLX: Apple’s array framework optimized for Apple Silicon. [Docs](https://ml-explore.github.io/mlx/) • [GitHub](https://github.com/ml-explore/mlx)
- mlx-lm: Utilities for running/fine-tuning LLMs with MLX. [Repo](https://github.com/ml-explore/mlx-examples/tree/main/llms) • [PyPI](https://pypi.org/project/mlx-lm/)
- LoRA: Low-Rank Adaptation for efficient fine-tuning of large models. [Paper](https://arxiv.org/abs/2106.09685)
- RAG: Retrieval-Augmented Generation to ground answers in external knowledge. [Paper](https://arxiv.org/abs/2005.11401)
- Chroma: Open-source vector database used for indexing and retrieval. [Docs](https://docs.trychroma.com/)
- Sentence-Transformers: Embedding models for semantic search. [Website](https://www.sbert.net/) • [GitHub](https://github.com/UKPLab/sentence-transformers)
- Qwen2.5-7B-Instruct: Instruction-tuned base model. [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Qwen2.5-7B-Instruct-4bit (MLX): Quantized variant used locally. [Hugging Face](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit)
- Streamlit: App framework for data/ML apps. [Website](https://streamlit.io/) • [Docs](https://docs.streamlit.io/)

## Repository Structure

```text
configs/
  config.yaml
  prompts/
    system.txt
data/                       # generated; gitignored
models/                     # adapters/merges; gitignored
outputs/                    # optional logs; gitignored
scripts/
  setup_initial.sh          # crawl → process → index → prepare dataset
  finetune_and_merge.sh     # finetune → merge
src/
  app/app.py                # Streamlit UI
  app/updater.py            # background updater + HTTP /update
  crawler/crawler.py        # crawler
  processing/process.py     # cleaning + chunking
  indexing/build_index.py   # build Chroma index
  inference/retriever.py    # retrieve context
  inference/generate.py     # load model and generate
  training/prepare_dataset.py
  training/finetune_mlx.py
  utils/config.py
  main.py                   # unified CLI entrypoint
requirements.txt
```

## Requirements

- macOS on Apple Silicon (M-series)  
- Python 3.10+
- Disk space for model, index, datasets
- Xcode Command Line Tools recommended for wheels

## Installation

Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:
```bash
pip install -U pip setuptools wheel
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Primary settings live in `configs/config.yaml`. Important keys:

```yaml
project:
  name: mde_learn_chatbot
data:
  raw_html_dir: data/raw/html
  processed_dir: data/processed
  dataset_dir: data/datasets
crawl:
  base_url: https://learn.microsoft.com/en-us/defender-endpoint/
  max_pages: 5000
processing:
  max_chunk_chars: 1200
index:
  chroma_persist_dir: data/index/chroma
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
model:
  base_id: mlx-community/Qwen2.5-7B-Instruct-4bit
  system_prompt_path: configs/prompts/system.txt
finetune:
  out_dir: models/adapters/qwen2_5_mde_lora
merge:
  out_dir: models/merges/qwen2_5_mde_merged
infer:
  max_tokens: 512
app:
  mode: rag_ft    # rag | ft | rag_ft
update:
  enabled: true
  api_host: 127.0.0.1
  api_port: 8799
  min_interval_hours: 24
```

System prompt: `configs/prompts/system.txt`.

## Quickstart

Run from the repo root with your virtual environment active.

Setup → Train → Run (using scripts):
```bash
chmod +x scripts/setup_initial.sh scripts/finetune_and_merge.sh
./scripts/setup_initial.sh
./scripts/finetune_and_merge.sh
python -m streamlit run src/app/app.py
```

Python-only alternative:

Setup:
```bash
python -m src.main crawl
python -m src.main process
python -m src.main index
```

Training:
```bash
python -m src.main prepare-dataset
python -m src.main finetune
```

Optional:
```bash
python -m src.main merge
```

Run:
```bash
python -m streamlit run src/app/app.py
```

App URL: http://localhost:8501. Modes: rag, ft, rag_ft.



Notes:
- First finetune/inference will download the base model locally (Hugging Face).
- Adapters saved to `models/adapters/...`; merged weights to `models/merges/...`.

### Adapter Loading Order (Inference)

1) Load LoRA adapter from `finetune.out_dir` if present  
2) Else use merged weights from `merge.out_dir`  
3) Else fall back to `model.base_id`

## Background Updates and HTTP /update

If `update.enabled: true`, the app starts:
- an HTTP server at `update.api_host:update.api_port` (accepts `POST /update`)
- a scheduler that enforces `min_interval_hours` using `last_run_file`

On update, the pipeline runs: Crawl → Process → Index → Prepare dataset → Finetune.

Manual trigger while the app is running:
```bash
curl -X POST http://127.0.0.1:8799/update
```

- If an update is too soon or already running, server returns 429.
- Accepted requests return 202 and log progress lines prefixed with `[app.updater]`.


## Troubleshooting

- Missing/empty index: ensure steps crawl → process → index completed; index at `data/index/chroma`.
- Model load issues: confirm `mlx` and `mlx-lm` installed; allow base model download on first run.
- Torch wheels on macOS: CPU-only embeddings are fine; upgrade pip and retry if needed.

## Clean / Reset

Warning: removes generated artifacts.

```bash
rm -rf data/raw data/processed data/index/chroma data/datasets models/adapters models/merges outputs
```
