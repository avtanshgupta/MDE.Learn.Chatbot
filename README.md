# MDE.Learn.Chatbot

End-to-end local project that:

- Crawls and extracts all documentation under <https://learn.microsoft.com/en-us/defender-endpoint/>
- Processes and chunks content, builds a vector index (Chroma + Sentence-Transformers)
- Fine-tunes Qwen2.5-7B-Instruct-4bit locally on Apple Silicon using MLX + LoRA
- Serves a Streamlit chatbot with RAG for freshness (retrieval) and FT for domain tone

Tested on macOS Apple Silicon. The base model is downloaded locally; fine-tuning and inference run locally using MLX.

## Project Structure

```text
configs/
  config.yaml                  # Central config
  prompts/
    system.txt                 # System prompt used by inference/app
data/
  raw/html/                    # Crawled HTML (ignored by git)
  processed/                   # Extracted chunks and manifests (ignored)
  datasets/                    # Fine-tune datasets (ignored)
  index/chroma/                # Chroma DB persistence (ignored)
models/
  adapters/                    # LoRA adapter output (ignored)
  merges/                      # (Optional) merged model weights (ignored)
  base/                        # (Optional) local base copy (ignored)
outputs/                       # Any additional outputs/logs (ignored)
src/
  app/app.py                   # Streamlit UI
  crawler/crawler.py           # Crawler (robots-aware)
  processing/process.py        # HTML -> clean text, chunking
  indexing/build_index.py      # Build Chroma index with embeddings
  inference/retriever.py       # Query index, format context
  inference/generate.py        # Load model (base/adapter/merged) and generate
  training/prepare_dataset.py  # Create JSONL {"text": "..."} datasets
  training/finetune_mlx.py     # Run MLX LoRA finetune and optional merge
  utils/config.py              # Config helpers
requirements.txt
```

## Setup

1. Prerequisites

- Python 3.10+
- macOS on Apple Silicon (M-series; MLX uses Apple GPUs/NPUs)
- Sufficient disk space (model + index + datasets)
- Xcode Command Line Tools may help with wheels

2. Create environment and install dependencies

```bash
# create and activate venv
python -m venv .venv
source .venv/bin/activate

# upgrade basics
pip install -U pip setuptools wheel

# install project deps
pip install -r requirements.txt
```

## Configuration

All knobs live in `configs/config.yaml`. Key sections:

```yaml
# configs/config.yaml (excerpt)
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
finetune:
  out_dir: models/adapters/qwen2_5_mde_lora
infer:
  max_tokens: 512
app:
  mode: rag_ft  # rag | ft | rag_ft
```

System prompt is in `configs/prompts/system.txt`.

## Pipeline: Crawl → Process → Index → Dataset → Finetune

Run these from the repo root with your venv activated.

1. Crawl docs (respects robots.txt, filters to defender-endpoint path)

```bash
python -m src.crawler.crawler
```

2. Process HTML to clean text and chunks

```bash
python -m src.processing.process
```

3. Build vector index (Chroma + Sentence-Transformers)

```bash
python -m src.indexing.build_index
```

4. Prepare fine-tune datasets (JSONL {"text": ...})

```bash
python -m src.training.prepare_dataset
```

5. Fine-tune Qwen2.5-7B-Instruct-4bit locally with MLX LoRA

```bash
python -m src.training.finetune_mlx
```

6. (Optional) Merge LoRA into standalone weights

```bash
python -m src.training.finetune_mlx --merge-only
```

Notes:

- The first finetune/inference call will download the base model locally (Hugging Face).
- LoRA adapters are saved to `models/adapters/...`. Merge outputs to `models/merges/...`.

## Streamlit Chatbot

Start the app:

```bash
streamlit run src/app/app.py
```

Sidebar “Mode”:

- rag: Only retrieval-augmented generation over the Chroma index
- ft: Only the fine-tuned model (no retrieval)
- rag_ft: Use RAG to ground the FT’d model with the latest indexed content

The app streams tokens and shows sources (URLs with chunk ranks) for transparency.

## RAG + Fine-tune Design

- RAG
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
  - Vector store: Chroma persistent collection at `data/index/chroma`
  - Top-k passages injected into the prompt. Sources displayed back to user.
- Fine-tune (LoRA, MLX)
  - Task: continual pretraining on domain chunks as plain text
  - Format: JSONL with {"text": "..."} created from processed chunks
  - Adapter loaded at inference time, or merged weights used if desired

This pairing gives:

- Freshness via retrieval from the latest crawl/index
- Domain fluency via LoRA adapter

## Safety, Use, and Policies

- Only answers grounded in Microsoft Defender for Endpoint docs should be returned. If missing/uncertain, the model is prompted to say it doesn’t know based on indexed documentation.
- The crawler respects robots.txt and filters to the Defender for Endpoint section.
- Do not use the system to generate harmful or policy-violating content.

## Troubleshooting

- Chroma not found / empty index
  - Ensure steps 1–3 completed successfully; index lives at `data/index/chroma`.
- Model load errors
  - Ensure `mlx` and `mlx-lm` installed; on first run the base model downloads locally.
- Torch install on macOS
  - CPU-only embeddings are fine. If wheels fail, try upgrading pip and retry.

## Clean/Reset Artifacts

```bash
# WARNING: This removes local artifacts (rebuild as needed)
rm -rf data/raw data/processed data/index/chroma data/datasets models/adapters models/merges outputs
```

## License and Content

- Microsoft Defender for Endpoint documentation is owned by Microsoft. Use this project for personal/educational purposes and respect the site’s terms and robots.txt. The repository contains only code to crawl/process/index content locally; no proprietary content is committed.
