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

## Support Matrix
| Component | Supported |
|----------|-----------|
| OS       | macOS (Apple Silicon M1/M2/M3) |
| Python   | 3.10+ |
| CPU/GPU  | CPU inference; MLX utilizes Apple Silicon accelerators |
| RAM      | 8 GB+ recommended |
| Disk     | 15–25 GB free for models, index, datasets (first run) |

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

Python CLI:
```bash
python -m src.main crawl
python -m src.main process
python -m src.main index
python -m src.main prepare-dataset
python -m src.main finetune
python -m src.main merge    # optional
python -m src.main app      # starts Streamlit
```

App URL (default): `http://localhost:8501`  
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

### CLI Help
```bash
python -m src.main -h
python -m src.main crawl -h
```

### Adapter Loading Order (Inference)

1) Load LoRA adapter from `finetune.out_dir` if present  
2) Else use merged weights from `merge.out_dir`  
3) Else fall back to `model.base_id`

### Run Modes
- rag — Retrieve top-k docs and prompt the base model.
- ft — Use the fine-tuned model only (no retrieval).
- rag_ft — Retrieve top-k docs and use the fine-tuned model.

### Programmatic Inference (Python)

Use the model runner directly from Python:
```python
from src.inference.generate import ModelRunner

runner = ModelRunner(mode="rag_ft")  # "rag", "ft", or "rag_ft"
query = "How do I onboard macOS devices to MDE?"
stream, sources = runner.generate(query, stream=True)  # stream=False returns full text
answer = "".join(tok for tok in stream)
print(answer)
for s in sources or []:
    print(f"[{s.get('rank')}] {s.get('title')} -> {s.get('url')} (distance={s.get('distance')})")
```


### Customize Streamlit Server

Override host/port when running Streamlit:
```bash
python -m streamlit run src/app/app.py --server.address 0.0.0.0 --server.port 8502
```

## Common Workflows

- RAG only (no training):
  - Commands:
    ```bash
    python -m src.main crawl
    python -m src.main process
    python -m src.main index
    python -m src.main app
    ```
  - Or set `app.mode: rag` in `configs/config.yaml`.

- Fine-tuned only (ft):
  - Commands:
    ```bash
    python -m src.main prepare-dataset
    python -m src.main finetune
    python -m src.main app
    ```

- Hybrid (rag_ft):
  - Ensure index and fine-tuned weights exist, then:
    ```bash
    python -m src.main app
    ```


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

All settings are defined in `configs/config.yaml`. Top-level keys (one-line summary):
- project — project name and seed for reproducibility.
- data — directories and filenames for raw/processed data, chunks, datasets, and URL manifest.
- crawl — start URL, domain/path constraints, robots respect, request pacing, and limits.
- processing — text cleanup and chunking parameters.
- index — vector store backend, persistence dir, collection, embedding model, batch size, and default top_k.
- model — MLX base_id and system_prompt_path.
- finetune — LoRA output directory and training hyperparameters.
- merge — output directory for merged full weights.
- infer — generation settings (max_tokens, temperature, top_p) and retrieval_top_k.
- app — Streamlit host/port and default mode (rag | ft | rag_ft).
- update — background updater toggle and HTTP endpoint settings (host, port, min_interval_hours, last_run_file).

See Artifacts & Paths for where outputs are written. System prompt file: `configs/prompts/system.txt`.

## Artifacts & Paths

- data/raw/html — raw crawled pages
- data/processed — processed artifacts (chunks.jsonl, urls.json, last_update.json)
- data/index/chroma — Chroma persistent index
- data/datasets — generated training datasets
- models/adapters/... — LoRA adapter weights (finetune.out_dir)
- models/merges/... — merged full weights (merge.out_dir)
- outputs/ — optional logs when piping command output


## Data Formats

- data/processed/chunks.jsonl
  - One JSON object per line with keys: `title` (str), `url` (str), `text` (str).
- data/datasets/finetune.*.jsonl
  - One JSON object per line: `{"text": "..."}`
  - Prepared for continual pretraining from processed chunks.

## Configuration Tips

- Change base model: edit `configs/config.yaml` → `model.base_id`.
- Adjust generation and retrieval: `infer.max_tokens`, `infer.temperature`, `infer.retrieval_top_k`.
- Switch default UI mode: `app.mode` (overrides to `rag_ft` if fine-tuned weights are found).
- Enable/disable background updates: `update.enabled` (and tweak `api_host`, `api_port`, `min_interval_hours`).
- Streamlit port/host: `app.host` (default 0.0.0.0), `app.port` (default 8501).
- Update API: `update.api_host` (default 127.0.0.1), `update.api_port` (default 8799).

### Environment Variables

- LOG_LEVEL — controls logging verbosity (INFO by default). Examples:
  ```bash
  LOG_LEVEL=DEBUG python -m src.main index
  LOG_LEVEL=DEBUG python -m streamlit run src/app/app.py
  ```

## Customize Crawl Target

To adapt the pipeline to a different documentation site, update these keys in `configs/config.yaml` and rerun the pipeline (crawl → process → index):
```yaml
crawl:
  base_url: https://example.com/docs/
  allowed_domain: example.com
  allowed_path_prefix: /docs
  same_language_only: true
  exclude_url_patterns: ["?view="]
```
Then run:
```bash
python -m src.main crawl
python -m src.main process
python -m src.main index
```

## Security & Privacy

- Runs fully local after initial model/downloads; no external telemetry is sent by this project.
- Crawler respects robots.txt and is constrained via `configs/config.yaml` (domain and path prefix).
- The update HTTP server binds to `127.0.0.1` by default; avoid exposing it publicly or add network controls.
- Review third‑party dependencies in `requirements.txt` before production use.

## Known Limitations

- macOS Apple Silicon focus; not tested on Intel macOS or Windows.
- First run downloads multi-GB model shards; requires stable network and ample disk space.
- MiniLM embeddings run on CPU; retrieval speed depends on CPU throughput.
- Streamlit is single-process; long background updates can momentarily affect responsiveness.
- Upstream `mlx_lm` APIs may change; fallbacks exist but pin versions if necessary.

## Troubleshooting

- First run downloads multi-GB model shards. Ensure stable network and disk space. Progress may appear stalled during large downloads.
- If embeddings are slow, remember MiniLM runs on CPU by default; overall speed depends on CPU throughput.
- If mlx_lm APIs change, inference falls back to merged/base weights per guards in `src/inference/generate.py`.
- If Streamlit fails to start, check that your venv is active and port 8501 is free; try `python -m src.main app`.

## Performance Tips

- Retrieval: lower `infer.retrieval_top_k` (e.g., 3–5) to reduce retrieval and generation latency.
- Generation: reduce `infer.max_tokens` and set lower `infer.temperature` for faster, more deterministic outputs.
- Indexing: decrease `index.embedding_batch` if you hit memory pressure; increase it to speed up on larger CPUs.
- Training: tune `finetune.batch_size` and `finetune.accumulate_steps` based on memory; fewer epochs for quick iterations.
- App: set `update.enabled: false` to avoid background work during demos.

## Reproducibility

- Set a fixed seed in `configs/config.yaml` → `project.seed` (default 42).
- Capture exact package set:
  ```bash
  python --version
  pip freeze > requirements-lock.txt
  ```
- Prefer consistent hardware (same Apple Silicon gen) for comparable timings.

## FAQ

- Chroma collection not found:
  - Error like "collection 'defender-endpoint' not found". Run:
    ```bash
    python -m src.main index
    ```
- Port 8501 in use:
  - Start Streamlit on another port:
    ```bash
    python -m streamlit run src/app/app.py --server.port 8502
    ```
- Update endpoint returns 429:
  - Respecting `min_interval_hours`. Use the sidebar "Run update now" button (forces) or wait for the window.
- Start from a clean slate:
  - Reset artifacts and rebuild:
    ```bash
    chmod +x scripts/clean_reset.sh
    ./scripts/clean_reset.sh --yes
    python -m src.main crawl
    python -m src.main process
    python -m src.main index
    ```

## Contributing

- Use Python 3.10+ and create a virtualenv.
- Before opening a PR, run tests and linters:
  - Lint: `ruff check .`
- Keep README and config summary in sync with code changes.

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
