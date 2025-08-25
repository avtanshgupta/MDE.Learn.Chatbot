# MDE.Learn.Chatbot

End-to-end local RAG + LoRA fine-tuning pipeline for Microsoft Defender for Endpoint (MDE) docs on Apple Silicon. It crawls the official docs, builds a vector index, fine-tunes Qwen2.5 with MLX + LoRA, and serves a Streamlit chatbot with modes: rag | ft | rag_ft.

Public documentation: https://learn.microsoft.com/en-us/defender-endpoint/

## Features

- Crawler: robots-aware, constrained to `learn.microsoft.com/en-us/defender-endpoint`
- Processing: HTML → clean text → chunking with overlap
- Indexing: Chroma persistent store + MiniLM embeddings
- Training: MLX LoRA fine-tuning (optional merge to full weights)
- Inference: adapter-first, fallback to merged or base
- App: Streamlit UI with token streaming + source attributions
- Auto-updates: scheduled pipeline and HTTP trigger (`POST /update`)
- Fully local after initial downloads

## Requirements

- macOS on Apple Silicon (M1/M2/M3)
- Python 3.10+
- Disk: 15–25 GB free (first run)
- Xcode Command Line Tools recommended

## Install

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Quickstart

One-time setup with finetune+merge, then launch the app:
```bash
chmod +x scripts/setup_initial.sh scripts/finetune.sh
./scripts/setup_initial.sh
./scripts/finetune.sh --merge      # one-time finetune + merge
python -m streamlit run src/app/app.py
```

Finetune only (no merge):
```bash
./scripts/finetune.sh
```

Alternatively, use the Python CLI:
```bash
python -m src.main crawl
python -m src.main process
python -m src.main index
python -m src.main prepare-dataset
python -m src.main finetune
python -m src.main merge           # optional
python -m src.main app             # starts Streamlit
```

App URL (default): http://localhost:8501  
Modes: `rag`, `ft`, `rag_ft`.

## CLI Reference

Entrypoint:
```bash
python -m src.main [--debug] <command>
```

End-to-end sequence:
- Crawl — `python -m src.main crawl`
- Process — `python -m src.main process`
- Index — `python -m src.main index`
- Prepare dataset — `python -m src.main prepare-dataset`
- Finetune (LoRA) — `python -m src.main finetune`
- Merge (optional) — `python -m src.main merge`
- App — `python -m src.main app`

Help:
```bash
python -m src.main -h
python -m src.main crawl -h
```

### Run Modes

- rag — Retrieve top-k docs, prompt base model
- ft — Use fine-tuned model only (no retrieval)
- rag_ft — Retrieve top-k docs and use fine-tuned model

### Adapter Loading Order

1) Load LoRA adapter from `finetune.out_dir` if present  
2) Else use merged weights from `merge.out_dir`  
3) Else fall back to `model.base_id`

### Programmatic Inference

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

## Configuration

All settings: `configs/config.yaml`. Key sections:
- project — name, seed
- data — raw/processed dirs, chunk and dataset paths
- crawl — base_url, domain/path constraints, pacing, robots
- processing — cleanup + chunking parameters
- index — vector store, persist dir, collection, embedding model/batch, top_k
- model — MLX base model id and system prompt path
- finetune — LoRA output dir and hyperparameters
- merge — output dir for merged weights
- infer — generation params and `retrieval_top_k`
- app — Streamlit host/port and default mode (rag | ft | rag_ft)
- update — background updater toggle and HTTP endpoint settings

System prompt file: `configs/prompts/system.txt`.

## Artifacts & Paths

- data/raw/html — crawled pages
- data/processed — chunks.jsonl, urls.json, last_update.json
- data/index/chroma — Chroma index
- data/datasets — training datasets
- models/adapters/... — LoRA adapter weights (`finetune.out_dir`)
- models/merges/... — merged full weights (`merge.out_dir`)
- outputs/ — optional logs when piping command output

## Background Updates

If `update.enabled: true`, the app exposes `POST /update` at `update.api_host:update.api_port` and enforces `min_interval_hours`.

Manual trigger:
```bash
curl -X POST http://127.0.0.1:8799/update
```

- If no fine-tuned weights exist, the app may auto-trigger a one-time finetune.
- 429 on too-frequent runs; 202 when accepted with logs prefixed `[app.updater]`.

## Logging

Default level: INFO. Enable DEBUG via CLI `--debug` or env:
```bash
LOG_LEVEL=DEBUG python -m src.main index
LOG_LEVEL=DEBUG python -m streamlit run src/app/app.py
```

## Clean Reset

Removes generated artifacts. Use with care.
```bash
chmod +x scripts/clean_reset.sh
./scripts/clean_reset.sh --yes
```

Manual alternative:
```bash
rm -rf data/raw data/processed data/index/chroma data/datasets models/adapters models/merges outputs
```

## Customize Crawl Target

Update `configs/config.yaml` and rerun crawl → process → index:
```yaml
crawl:
  base_url: https://example.com/docs/
  allowed_domain: example.com
  allowed_path_prefix: /docs
  same_language_only: true
  exclude_url_patterns: ["?view="]
```
```bash
python -m src.main crawl
python -m src.main process
python -m src.main index
```

## Performance Tips

- Retrieval: lower `infer.retrieval_top_k` (e.g., 3–5)
- Generation: reduce `infer.max_tokens`, lower `infer.temperature`
- Indexing: tune `index.embedding_batch` for memory/throughput
- Training: tune `finetune.batch_size` and `finetune.accumulate_steps`
- Demos: set `update.enabled: false`

## Troubleshooting

- First run downloads multi-GB model shards: ensure stable network + disk space
- Slow embeddings: MiniLM runs on CPU by default
- Streamlit port in use: `python -m streamlit run src/app/app.py --server.port 8502`
- Chroma collection missing: `python -m src.main index`
- mlx_lm API changes: inference falls back per guards in `src/inference/generate.py`

## Reproducibility

- Fixed seed: `project.seed` (default 42)
- Record packages:
```bash
python --version
pip freeze > requirements-lock.txt
```

## Security & Privacy

- Fully local after initial downloads; no telemetry sent by this project
- Crawler respects robots.txt and domain/path constraints
- Update server binds to `127.0.0.1` by default; avoid exposing publicly

## Known Limitations

- Focused on macOS Apple Silicon
- Large first-time downloads
- MiniLM embeddings on CPU
- Streamlit is single-process; long updates can affect responsiveness
- Upstream `mlx_lm` changes possible; pin versions if required

## Contributing

- Python 3.10+ in a virtualenv
- Before PRs: run tests and linters
  - Lint: `ruff check .`

## References

- MLX — https://ml-explore.github.io/mlx/ • https://github.com/ml-explore/mlx
- mlx-lm — https://github.com/ml-explore/mlx-examples/tree/main/llms • https://pypi.org/project/mlx-lm/
- LoRA — https://arxiv.org/abs/2106.09685
- RAG — https://arxiv.org/abs/2005.11401
- Chroma — https://docs.trychroma.com/
- Sentence-Transformers — https://www.sbert.net/ • https://github.com/UKPLab/sentence-transformers
- Qwen2.5-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen2.5-7B-Instruct-4bit (MLX) — https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit
- Streamlit — https://streamlit.io/ • https://docs.streamlit.io/
