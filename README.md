# MDE.Learn.Chatbot

End-to-end local RAG + LoRA fine-tuning pipeline for Microsoft Defender for Endpoint (MDE) docs on Apple Silicon. It crawls the official docs, builds a vector index, fine-tunes Qwen2.5 with MLX + LoRA, and serves a Streamlit chatbot with modes: rag | ft | rag_ft.

Public documentation: https://learn.microsoft.com/en-us/defender-endpoint/

## Features

- Crawler: robots-aware and constrained to `learn.microsoft.com/en-us/defender-endpoint`
- Processing: HTML → clean text → chunking with overlap
- Indexing: Chroma persistent store + MiniLM embeddings
- Training: MLX LoRA fine-tuning, optional merge to standalone weights
- Inference: adapter-first loading, fallback to merged or base
- App: Streamlit UI with token streaming and source attributions
- Auto-updates: scheduled pipeline and HTTP trigger (`POST /update`)

Runs fully offline after initial downloads. Base model is retrieved on first use and cached locally.

## Requirements

- macOS with Apple Silicon (M-series)
- Python 3.10+
- Sufficient disk space for model, index, and datasets
- Xcode Command Line Tools recommended for building some wheels

## Installation

Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade packaging tools and install dependencies:
```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Quickstart

One-time setup → fine-tune → launch app:
```bash
chmod +x scripts/setup_initial.sh scripts/finetune_and_merge.sh
./scripts/setup_initial.sh
./scripts/finetune_and_merge.sh
python -m streamlit run src/app/app.py
```

Python-only CLI alternative:
```bash
python -m src.main crawl
python -m src.main process
python -m src.main index
python -m src.main prepare-dataset
python -m src.main finetune
python -m src.main merge    # optional
python -m src.main app      # starts Streamlit
```

App URL (default): http://localhost:8501  
Modes: `rag`, `ft`, `rag_ft`.

## Pipeline & CLI Reference

Entrypoint: `python -m src.main [--debug] <command>`

End-to-end sequence:
- Crawl — Crawl MDE docs  
  Command: `python -m src.main crawl`
- Process — Convert HTML to clean text chunks  
  Command: `python -m src.main process`
- Index — Build Chroma index  
  Command: `python -m src.main index`
- Prepare dataset — Create JSONL dataset for fine-tuning  
  Command: `python -m src.main prepare-dataset`
- Finetune — Run MLX LoRA fine-tuning  
  Command: `python -m src.main finetune`
- Merge (optional) — Merge LoRA adapters into standalone weights  
  Command: `python -m src.main merge`
- App — Start Streamlit app  
  Command: `python -m src.main app`

Examples:
```bash
python -m src.main --debug crawl
python -m src.main index
python -m src.main app
```

Debug logging:
```bash
LOG_LEVEL=DEBUG python -m streamlit run src/app/app.py
```

### Adapter Loading Order (Inference)

1) Load LoRA adapter from `finetune.out_dir` if present  
2) Else use merged weights from `merge.out_dir`  
3) Else fall back to `model.base_id`


## Background Updates and HTTP /update

If `update.enabled: true`, the app runs:
- an HTTP server at `update.api_host:update.api_port` (accepts `POST /update`)
- a scheduler enforcing `min_interval_hours` via `last_run_file`

On update, the pipeline runs: Crawl → Process → Index → Prepare dataset → Finetune.

Manual trigger while the app is running:
```bash
curl -X POST http://127.0.0.1:8799/update
```

Behavior:
- If no fine-tuned weights exist, the app auto-triggers a fine-tune once.
- If an update is too soon or already running, server returns 429.
- Accepted requests return 202 and stream progress logs prefixed with `[app.updater]`.

## Logging

Default level: INFO.

Enable DEBUG:
```bash
./scripts/setup_initial.sh --debug
./scripts/finetune_and_merge.sh --debug

python -m src.main --debug crawl
python -m src.main --debug process
python -m src.main --debug index
python -m src.main --debug prepare-dataset
python -m src.main --debug finetune
python -m src.main --debug merge
```

Persist logs:
```bash
./scripts/setup_initial.sh --debug | tee outputs/setup_initial.log
```

## Clean / Reset

Removes generated artifacts. Use with care.

Recommended:
```bash
chmod +x scripts/clean_reset.sh
./scripts/clean_reset.sh --yes
```

Manual alternative:
```bash
rm -rf data/raw data/processed data/index/chroma data/datasets models/adapters models/merges outputs
```

## Configuration

Primary settings in `configs/config.yaml`. Summary:

Project
| Key  | Default           | Description                                   |
|------|-------------------|-----------------------------------------------|
| name | mde_learn_chatbot | Project identifier used in logs/artifacts.    |
| seed | 42                | Global random seed for reproducibility.       |

Data
| Key            | Default                              | Description                                         |
|----------------|--------------------------------------|-----------------------------------------------------|
| raw_html_dir   | data/raw/html                         | Directory for raw crawled HTML.                     |
| processed_dir  | data/processed                        | Root directory for processed artifacts.             |
| chunks_path    | data/processed/chunks.jsonl           | JSONL with chunked text used for indexing/RAG.      |
| url_manifest   | data/processed/urls.json              | JSON manifest of discovered/crawled URLs.           |
| dataset_dir    | data/datasets                         | Output directory for generated training datasets.   |
| finetune_train | data/datasets/finetune.train.jsonl    | JSONL training split path for LoRA finetuning.      |
| finetune_val   | data/datasets/finetune.val.jsonl      | JSONL validation split path for LoRA finetuning.    |

Crawl
| Key                        | Default                                                                 | Description                                                                 |
|----------------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| base_url                   | https://learn.microsoft.com/en-us/defender-endpoint/                    | Starting URL for the crawl.                                                 |
| allowed_domain             | learn.microsoft.com                                                     | Domain constraint for crawler.                                              |
| allowed_path_prefix        | /en-us/defender-endpoint                                                | Path prefix constraint for crawler.                                         |
| user_agent                 | MDE-Learn-Chatbot/1.0 (+https://github.com/avtanshgupta/MDE.Learn.Chatbot) | User-Agent string used by the crawler.                                      |
| max_pages                  | 50000                                                                   | Maximum number of pages to crawl.                                           |
| request_timeout_sec        | 20                                                                      | Per-request timeout in seconds.                                             |
| sleep_between_requests_sec | 0.2                                                                     | Delay between HTTP requests in seconds.                                     |
| respect_robots_txt         | true                                                                    | Obey robots.txt directives when crawling.                                   |
| same_language_only         | true                                                                    | Restrict to the same language as the start URL.                             |
| include_filetypes          | [html]                                                                  | Allowed file extensions to fetch.                                           |
| exclude_url_patterns       | ["?view="]                                                              | Substrings/patterns to skip when enqueueing URLs.                           |

Processing
| Key                 | Default | Description                                                                       |
|---------------------|---------|-----------------------------------------------------------------------------------|
| min_section_chars   | 200     | Minimum section length before chunking.                                           |
| max_chunk_chars     | 1200    | Target maximum characters per chunk.                                              |
| chunk_overlap_chars | 200     | Character overlap between consecutive chunks.                                     |
| keep_headings       | true    | Preserve headings in chunk output to aid attribution.                             |

Index
| Key                | Default                                  | Description                                              |
|--------------------|------------------------------------------|----------------------------------------------------------|
| vector_store       | chroma                                   | Vector DB backend.                                       |
| chroma_persist_dir | data/index/chroma                        | Directory for Chroma persistent storage.                 |
| collection         | defender-endpoint                         | Chroma collection name.                                  |
| embedding_model    | sentence-transformers/all-MiniLM-L6-v2    | Sentence-Transformers model for embeddings.              |
| embedding_batch    | 64                                       | Batch size for embedding computation.                    |
| top_k              | 5                                        | Default top-k for retrieval queries.                     |

Model
| Key                | Default                               | Description                                        |
|--------------------|---------------------------------------|----------------------------------------------------|
| base_id            | mlx-community/Qwen2.5-7B-Instruct-4bit | MLX-compatible base model identifier (HF).         |
| system_prompt_path | configs/prompts/system.txt            | Path to system prompt used during inference.       |

Finetune
| Key              | Default                          | Description                                  |
|------------------|----------------------------------|----------------------------------------------|
| out_dir          | models/adapters/qwen2_5_mde_lora | Output directory for LoRA adapters.          |
| epochs           | 1                                | Number of training epochs.                   |
| batch_size       | 1                                | Micro-batch size per step.                   |
| accumulate_steps | 32                               | Gradient accumulation steps.                 |
| lr               | 1.0e-5                           | Learning rate.                               |
| lora.enabled     | true                             | Enable LoRA adapters.                        |
| lora.r           | 16                               | LoRA rank.                                   |
| lora.alpha       | 32                               | LoRA alpha scaling factor.                   |
| lora.dropout     | 0.05                             | LoRA dropout probability.                    |
| val_ratio        | 0.05                             | Fraction of data reserved for validation.    |
| max_train_samples| null                             | Limit training samples (null = use all).     |

Merge
| Key    | Default                          | Description                               |
|--------|----------------------------------|-------------------------------------------|
| out_dir| models/merges/qwen2_5_mde_merged | Output directory for merged full weights. |

Infer
| Key             | Default | Description                                      |
|-----------------|---------|--------------------------------------------------|
| max_tokens      | 512     | Maximum new tokens to generate.                  |
| temperature     | 0.2     | Sampling temperature.                            |
| top_p           | 0.95    | Nucleus sampling probability mass.               |
| use_streaming   | true    | Stream tokens to the UI as they are generated.   |
| retrieval_top_k | 5       | Top-k documents to retrieve for RAG.             |

App
| Key  | Default | Description                        |
|------|---------|------------------------------------|
| host | 0.0.0.0 | Streamlit server host.             |
| port | 8501    | Streamlit server port.             |
| mode | rag_ft  | Run mode: rag, ft, or rag_ft.      |

Update
| Key                | Default                         | Description                                                  |
|--------------------|---------------------------------|--------------------------------------------------------------|
| enabled            | true                            | Enable background updater service.                           |
| api_host           | 127.0.0.1                       | HTTP host for update endpoint.                               |
| api_port           | 8799                            | HTTP port for update endpoint.                               |
| min_interval_hours | 24                              | Minimum hours between allowed update runs.                   |
| last_run_file      | data/processed/last_update.json | File storing last update timestamp/metadata.                 |

System prompt: `configs/prompts/system.txt`.

Notes:
- If fine-tuned adapters or merged weights are found, the app overrides a default `rag` mode to `rag_ft`.
- Sidebar shows non-interactive knobs sourced from config: retrieval_top_k, max_tokens, temperature.

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

## Troubleshooting

- First run downloads multi-GB model shards. Ensure stable network and disk space. Progress may appear stalled during large downloads.
- If embeddings are slow, remember MiniLM runs on CPU by default; overall speed depends on CPU throughput.
- If mlx_lm APIs change, inference falls back to merged/base weights per guards in `src/inference/generate.py`.
- If Streamlit fails to start, check that your venv is active and port 8501 is free; try `python -m src.main app`.

## Key Concepts, Links, and Acknowledgments

- MLX: https://ml-explore.github.io/mlx/ • https://github.com/ml-explore/mlx
- mlx-lm: https://github.com/ml-explore/mlx-examples/tree/main/llms • https://pypi.org/project/mlx-lm/
- LoRA (paper): https://arxiv.org/abs/2106.09685
- RAG (paper): https://arxiv.org/abs/2005.11401
- Chroma: https://docs.trychroma.com/
- Sentence-Transformers: https://www.sbert.net/ • https://github.com/UKPLab/sentence-transformers
- Qwen2.5-7B-Instruct (base): https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen2.5-7B-Instruct-4bit (MLX): https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit
- Streamlit: https://streamlit.io/ • https://docs.streamlit.io/
