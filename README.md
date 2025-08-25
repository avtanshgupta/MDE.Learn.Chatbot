# MDE.Learn.Chatbot

Local end-to-end RAG + LoRA fine-tuning pipeline for Microsoft Defender for Endpoint (MDE) docs:
- Crawls and cleans MDE docs
- Chunks and builds a vector index (Chroma + Sentence-Transformers)
- Fine-tunes Qwen2.5-7B-Instruct-4bit on Apple Silicon with MLX + LoRA
- Serves a Streamlit chatbot with modes: rag | ft | rag_ft
- Optional background update service with HTTP /update

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
- Basic test suite (pytest) and CI workflows

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
# bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools:
```bash
# bash
pip install -U pip setuptools wheel
```

Install dependencies:
```bash
# bash
pip install -r requirements.txt
```


## Configuration

Primary settings live in `configs/config.yaml`. Important keys (full example):

```yaml
# yaml
project:
  name: mde_learn_chatbot
  seed: 42

data:
  raw_html_dir: data/raw/html
  processed_dir: data/processed
  chunks_path: data/processed/chunks.jsonl
  url_manifest: data/processed/urls.json
  dataset_dir: data/datasets
  finetune_train: data/datasets/finetune.train.jsonl
  finetune_val: data/datasets/finetune.val.jsonl

crawl:
  base_url: https://learn.microsoft.com/en-us/defender-endpoint/
  allowed_domain: learn.microsoft.com
  allowed_path_prefix: /en-us/defender-endpoint
  user_agent: MDE-Learn-Chatbot/1.0 (+https://github.com/avtanshgupta/MDE.Learn.Chatbot)
  max_pages: 50000
  request_timeout_sec: 20
  sleep_between_requests_sec: 0.2
  respect_robots_txt: true
  same_language_only: true
  include_filetypes: [html]
  exclude_url_patterns:
    - "?view="

processing:
  min_section_chars: 200
  max_chunk_chars: 1200
  chunk_overlap_chars: 200
  keep_headings: true

index:
  vector_store: chroma
  chroma_persist_dir: data/index/chroma
  collection: defender-endpoint
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  embedding_batch: 64
  top_k: 5

model:
  base_id: mlx-community/Qwen2.5-7B-Instruct-4bit
  system_prompt_path: configs/prompts/system.txt

finetune:
  out_dir: models/adapters/qwen2_5_mde_lora
  epochs: 1
  batch_size: 1
  accumulate_steps: 32
  lr: 1.0e-5
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
  val_ratio: 0.05
  max_train_samples: null

merge:
  out_dir: models/merges/qwen2_5_mde_merged

infer:
  max_tokens: 512
  temperature: 0.2
  top_p: 0.95
  use_streaming: true
  retrieval_top_k: 5

app:
  host: 0.0.0.0
  port: 8501
  mode: rag_ft  # options: rag, ft, rag_ft

update:
  enabled: true
  api_host: 127.0.0.1
  api_port: 8799
  min_interval_hours: 24
  last_run_file: data/processed/last_update.json
```

System prompt: `configs/prompts/system.txt`.

Notes:
- If fine-tuned adapters or merged weights are found, the app overrides a default `rag` mode to `rag_ft`.
- Sidebar shows non-interactive knobs sourced from config: retrieval_top_k, max_tokens, temperature.

## Quickstart

Run from the repo root with your virtual environment active.

Setup → Train → Run (using scripts):
```bash
# bash
chmod +x scripts/setup_initial.sh scripts/finetune_and_merge.sh
./scripts/setup_initial.sh
./scripts/finetune_and_merge.sh
python -m streamlit run src/app/app.py
```

Python-only alternative (CLI entrypoint):
```bash
# bash
python -m src.main crawl
python -m src.main process
python -m src.main index
python -m src.main prepare-dataset
python -m src.main finetune
python -m src.main merge   # optional
python -m src.main app     # starts Streamlit
```

App URL (default): http://localhost:8501  
Modes: `rag`, `ft`, `rag_ft`.

## Pipeline: Crawl → Process → Index → Dataset → Finetune

1) Crawl (fetch raw HTML and URL manifest based on `configs.crawl`)
```bash
# bash
python -m src.crawler.crawler
```

2) Process (parse HTML, clean text, and chunk)
```bash
# bash
python -m src.processing.process
```

3) Index (embed chunks and build a persistent Chroma index)
```bash
# bash
python -m src.indexing.build_index
```

4) Dataset (create JSONL text for continual pretraining)
```bash
# bash
python -m src.training.prepare_dataset
```

5) Finetune (MLX LoRA finetuning; adapters saved under `models/adapters/...`)
```bash
# bash
python -m src.training.finetune_mlx
```

Optional: merge LoRA into standalone weights
```bash
# bash
python -m src.training.finetune_mlx --merge-only
```

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
# bash
curl -X POST http://127.0.0.1:8799/update
```

- Auto-trigger: If no fine-tuned weights exist, the app auto-triggers a fine-tune once.
- If an update is too soon or already running, server returns 429.
- Accepted requests return 202 and log progress lines prefixed with `[app.updater]`.

## Logging

Default log level is INFO.

Enable DEBUG logs:
```bash
# bash
# Shell scripts:
./scripts/setup_initial.sh --debug
./scripts/finetune_and_merge.sh --debug

# Python CLI entrypoint:
python -m src.main --debug crawl
python -m src.main --debug process
python -m src.main --debug index
python -m src.main --debug prepare-dataset
python -m src.main --debug finetune
python -m src.main --debug merge

# Environment variable (works with Streamlit too):
LOG_LEVEL=DEBUG python -m streamlit run src/app/app.py
```

To persist logs, redirect output:
```bash
# bash
./scripts/setup_initial.sh --debug | tee outputs/setup_initial.log
```

## Clean / Reset

Warning: removes generated artifacts.

Recommended:
```bash
# bash
chmod +x scripts/clean_reset.sh
./scripts/clean_reset.sh --yes
```

Options:
- `--dry-run` to preview deletions
- `--debug` for verbose output
- `--yes` to skip confirmation

Manual alternative:
```bash
# bash
rm -rf data/raw data/processed data/index/chroma data/datasets models/adapters models/merges outputs
```

## Tests

Run the test suite:
```bash
# bash
pytest -q
```

CI workflows are defined under `.github/workflows/`.

## Notes and trade-offs

- Apple Silicon focus: Uses MLX; training and inference are optimized for M-series hardware.
- Model download: Base model is pulled by mlx_lm on first use (Hugging Face cache). Large initial download; slow networks may appear “stuck” while fetching large shards.
- Embeddings run on CPU by default; indexing speed depends on MiniLM and CPU throughput.
- Crawler respects robots.txt and is constrained to `learn.microsoft.com/en-us/defender-endpoint`.

## Known issues / risks

- Long first-time model download (multi-GB). Ensure enough disk space and stable network.
- If mlx_lm API changes, adapter loading falls back to merged/base; guarded in `src/inference/generate.py`.
