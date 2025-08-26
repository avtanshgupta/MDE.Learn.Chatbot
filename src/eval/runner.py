import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.eval.dataset import QAItem, load_eval_set
from src.eval.metrics import bootstrap_ci, compute_all_metrics
from src.eval.timing import count_tokens, timer
from src.inference.generate import ModelRunner
from src.inference.retriever import Retriever, format_context
from src.utils.config import ensure_dir, load_config

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VariantSpec:
    name: str
    mode: str          # "rag" | "ft" | "rag_ft"
    force_base: bool   # True => base weights only

# Preset variants for convenience
VARIANTS_PRESET: Dict[str, VariantSpec] = {
    "base_ft": VariantSpec("base_ft", "ft", True),
    "base_rag": VariantSpec("base_rag", "rag", True),
    "ft_only": VariantSpec("ft_only", "ft", False),
    "rag_ft": VariantSpec("rag_ft", "rag_ft", False),
}

def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _shuffle(items: List[Any], seed: int) -> List[Any]:
    rng = random.Random(seed)
    arr = list(items)
    rng.shuffle(arr)
    return arr

def _build_frozen_rag_prompt(question: str, retriever: Retriever, top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
    """Replicates ModelRunner._build_rag_prompt for frozen-RAG caching."""
    docs = retriever.query(question, top_k)
    context, sources = format_context(docs)
    instruction = (
        f"Use the following retrieved MDE documentation excerpts to answer the question.\n"
        f"Cite sources by index when helpful.\n\nContext:\n{context}\n\nQuestion: {question}\n"
    )
    return instruction, sources

def _needs_retrieval(mode: str) -> bool:
    return mode in ("rag", "rag_ft")

def _first_token_and_full_text(it: Iterable[str]) -> Tuple[float, str]:
    """Consume a token stream iterator measuring time to first token and collecting full text.
    Returns (t_first_since_call, full_text). t_first_since_call is 0.0 if no streaming (single yield)."""
    t0 = time.perf_counter()
    got_first = False
    t_first = 0.0
    chunks: List[str] = []
    for tok in it:
        if not got_first:
            t_first = time.perf_counter() - t0
            got_first = True
        chunks.append(tok)
    full = "".join(chunks)
    # If non-streaming fallback yielded once at end, t_first approximates total
    return t_first, full

def run_evaluation(
    dataset_path: str,
    variant_names: List[str],
    out_dir: Optional[str] = None,
    live_rag: bool = True,
    frozen_rag: bool = False,
    max_items: Optional[int] = None,
    warmup: int = 1,
    seed: Optional[int] = None,
    auto_index: bool = False,
) -> str:
    """Runs evaluation and writes per-variant JSONL files. Returns run directory."""
    cfg = load_config()
    if seed is None:
        seed = int(cfg.get("project", {}).get("seed", 42))

    # Resolve run directory
    if not out_dir:
        out_dir = os.path.join("outputs", "eval", "runs", _now_ts())
    ensure_dir(out_dir)

    # Record run config
    run_cfg = {
        "dataset_path": dataset_path,
        "variant_names": variant_names,
        "live_rag": bool(live_rag),
        "frozen_rag": bool(frozen_rag),
        "max_items": max_items,
        "warmup": warmup,
        "seed": seed,
        "model": cfg.get("model", {}),
        "infer": cfg.get("infer", {}),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2, ensure_ascii=False)

    # Load dataset
    items = load_eval_set(dataset_path, max_items=max_items)
    items = _shuffle(items, seed=seed)

    # Ensure retrieval availability if any variant requires RAG
    needs_retrieval = any(_needs_retrieval(VARIANTS_PRESET[v].mode) for v in variant_names)
    if needs_retrieval:
        def _try_retriever() -> None:
            _r = Retriever()
            del _r
        try:
            _try_retriever()
        except Exception as e:
            if auto_index:
                logger.info("RAG index missing; --auto-index enabled. Building index...")
                try:
                    from src.indexing import build_index as _build_index
                    _build_index.build_index()
                    _try_retriever()
                    logger.info("Index built successfully; continuing evaluation.")
                except Exception as e2:
                    raise RuntimeError(
                        "Auto-index failed. Run 'python -m src.main index' before evaluating RAG variants "
                        "or evaluate non-RAG variants (base_ft,ft_only)."
                    ) from e2
            else:
                raise RuntimeError(
                    "RAG retrieval requires a built Chroma index. Run 'python -m src.main index' first, "
                    "or evaluate non-RAG variants (base_ft,ft_only)."
                ) from e

    # Optional frozen-RAG cache
    frozen_cache: Dict[str, Dict[str, Any]] = {}
    shared_retriever: Optional[Retriever] = None
    if frozen_rag:
        logger.info("Preparing frozen-RAG retrieval cache...")
        shared_retriever = Retriever()
        top_k = int(cfg.get("infer", {}).get("retrieval_top_k", 5))
        for it in items:
            with timer("retrieval"):
                user_prompt, sources = _build_frozen_rag_prompt(it.question, shared_retriever, top_k)
            frozen_cache[it.id] = {
                "user_prompt": user_prompt,
                "sources": sources,
                "retrieval_time": None,  # Measured per-variant as 0 when frozen; original measured above but not per-variant
            }
        logger.info("Frozen-RAG cache prepared: %d entries", len(frozen_cache))

    # Iterate variants
    for vn in variant_names:
        if vn not in VARIANTS_PRESET:
            raise ValueError(f"Unknown variant '{vn}'. Valid: {list(VARIANTS_PRESET.keys())}")
        spec = VARIANTS_PRESET[vn]
        out_path = os.path.join(out_dir, f"{spec.name}.jsonl")
        logger.info("Starting variant: %s -> mode=%s force_base=%s", spec.name, spec.mode, spec.force_base)

        # Initialize model and time load
        times: Dict[str, float] = {}
        with timer("model_load", times):
            runner = ModelRunner(mode=spec.mode, force_base=spec.force_base)

        # Warmup
        for _ in range(max(0, int(warmup))):
            try:
                _stream, _ = runner.generate("Warmup prompt. Ignore.", stream=True)
                # Consume minimally to trigger model
                try:
                    next(_stream)
                except StopIteration:
                    pass
            except Exception as e:
                logger.debug("Warmup failed (ignored): %s", e)

        # Per-sample evaluation
        with open(out_path, "w", encoding="utf-8") as f:
            for it in items:
                rec: Dict[str, Any] = {
                    "id": it.id,
                    "variant": spec.name,
                    "mode": spec.mode,
                    "force_base": spec.force_base,
                    "question": it.question,
                    "gold_answer": it.answer,
                    "gold_urls": it.gold_urls,
                }

                # Retrieval and prompt composition
                retrieval_time = 0.0
                user_prompt: str
                sources: List[Dict[str, Any]] = []
                if _needs_retrieval(spec.mode):
                    if frozen_rag and it.id in frozen_cache:
                        cached = frozen_cache[it.id]
                        user_prompt = cached["user_prompt"]
                        sources = cached["sources"]
                        retrieval_time = 0.0
                    elif live_rag:
                        _times: Dict[str, float] = {}
                        with timer("retrieval", _times):
                            user_prompt, sources = runner._build_rag_prompt(it.question)  # type: ignore[attr-defined]
                        retrieval_time = float(_times.get("retrieval", 0.0))
                    else:
                        # Default to live retrieval if neither specified
                        t0 = time.perf_counter()
                        user_prompt, sources = runner._build_rag_prompt(it.question)  # type: ignore[attr-defined]
                        retrieval_time = time.perf_counter() - t0
                else:
                    user_prompt = it.question
                    sources = []

                rec["retrieval_time_sec"] = retrieval_time
                rec["sources"] = sources

                # Generation with timing
                t0 = time.perf_counter()
                stream, _ = runner.generate_from_user_prompt(user_prompt, sources, stream=True)
                t_first, text = _first_token_and_full_text(stream)
                t_total = time.perf_counter() - t0

                # Token counts and throughput
                try:
                    out_tokens = count_tokens(runner.tokenizer, text)
                except Exception:
                    out_tokens = max(1, len(text.split()))
                toks_per_sec = float(out_tokens) / max(1e-6, (t_total))

                rec.update(
                    {
                        "first_token_latency_sec": t_first,
                        "gen_latency_sec": t_total,
                        "total_latency_sec": retrieval_time + t_total + times.get("model_load", 0.0) * 0.0,  # model load excluded per sample
                        "output_text": text,
                        "output_tokens": out_tokens,
                        "tokens_per_sec": toks_per_sec,
                    }
                )

                # Quality metrics
                metrics = compute_all_metrics(text, it.answer)
                rec["metrics"] = metrics

                # Write record
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info("Variant complete -> %s", out_path)

    logger.info("All variants completed. Run dir: %s", out_dir)
    return out_dir

def summarize_run(run_dir: str) -> Dict[str, Any]:
    """Aggregate basic stats across variant JSONLs in a run directory."""
    def _iter_records(path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    summaries: Dict[str, Any] = {}
    for fname in os.listdir(run_dir):
        if not fname.endswith(".jsonl"):
            continue
        variant = fname[:-6]  # strip .jsonl
        path = os.path.join(run_dir, fname)
        recs = list(_iter_records(path))
        if not recs:
            continue

        def collect(field: str) -> List[float]:
            vals = []
            for r in recs:
                v = r.get(field)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return vals

        def collect_metric(name: str) -> List[float]:
            vals = []
            for r in recs:
                m = r.get("metrics", {})
                v = m.get(name)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return vals

        agg = {
            "count": len(recs),
            "gen_latency_sec_mean": (sum(collect("gen_latency_sec")) / len(recs)),
            "first_token_latency_sec_mean": (sum(collect("first_token_latency_sec")) / len(recs)),
            "retrieval_time_sec_mean": (sum(collect("retrieval_time_sec")) / len(recs)),
            "tokens_per_sec_mean": (sum(collect("tokens_per_sec")) / len(recs)),
            "f1_mean": (sum(collect_metric("f1")) / len(recs)),
            "rouge_l_f1_mean": (sum(collect_metric("rouge_l_f1")) / len(recs)),
            "exact_match_mean": (sum(collect_metric("exact_match")) / len(recs)),
        }

        # Simple bootstrap CIs for tokens/sec and f1
        for key, src in [("tokens_per_sec_ci95", collect("tokens_per_sec")), ("f1_ci95", collect_metric("f1"))]:
            ci = bootstrap_ci(src, n=1000, alpha=0.05, seed=42)
            agg[key] = ci

        summaries[variant] = agg

    # Save a machine-readable summary JSON
    out_path = os.path.join(run_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Wrote summary -> %s", out_path)
    return summaries

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate model variants on an eval dataset.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to eval JSONL with fields: question, answer, gold_urls?")
    ap.add_argument("--variants", type=str, default="base_rag,rag_ft", help=f"Comma-separated from: {','.join(VARIANTS_PRESET.keys())}")
    ap.add_argument("--out-dir", type=str, default="", help="Output run directory (defaults to outputs/eval/runs/<ts>)")
    ap.add_argument("--frozen-rag", action="store_true", help="Use frozen-RAG (cache retrieved context per question)")
    ap.add_argument("--live-rag", action="store_true", help="Use live retrieval timing (default if neither is set)")
    ap.add_argument("--max-items", type=int, default=None, help="Limit number of eval items")
    ap.add_argument("--warmup", type=int, default=1, help="Number of warmup prompts per variant")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")

    args = ap.parse_args()
    variant_list = [v.strip() for v in args.variants.split(",") if v.strip()]

    run_dir = run_evaluation(
        dataset_path=args.dataset,
        variant_names=variant_list,
        out_dir=(args.out_dir or None),
        live_rag=bool(args.live_rag),
        frozen_rag=bool(args.frozen_rag),
        max_items=args.max_items,
        warmup=args.warmup,
        seed=args.seed,
    )
    summarize_run(run_dir)
