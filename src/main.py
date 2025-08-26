import argparse
import logging
import os
import subprocess
import sys

from src.utils.logging_setup import setup_logging

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

setup_logging()
logger = logging.getLogger(__name__)


def run_cmd(cmd: list[str]) -> int:
    logger.info("Running command: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.communicate()
    code = proc.returncode
    logger.info("Command exited with code=%s", code)
    return code


def cmd_crawl(args: argparse.Namespace) -> None:
    logger.info("Starting crawl step")
    import src.crawler.crawl as crawler

    crawler.crawl()
    logger.info("Crawl step complete")


def cmd_process(args: argparse.Namespace) -> None:
    logger.info("Starting process step")
    import src.processing.process_docs as process

    process.process()
    logger.info("Process step complete")


def cmd_index(args: argparse.Namespace) -> None:
    logger.info("Starting index step")
    import src.indexing.build_index as build_index

    build_index.build_index()
    logger.info("Index step complete")


def cmd_prepare_dataset(args: argparse.Namespace) -> None:
    logger.info("Starting dataset preparation step")
    import src.training.prepare_dataset as prep

    prep.main()
    logger.info("Dataset preparation step complete")


def cmd_finetune(args: argparse.Namespace) -> None:
    logger.info("Starting finetune step")
    import src.training.finetune_mlx as ft

    ft.finetune()
    logger.info("Finetune step complete")


def cmd_merge(args: argparse.Namespace) -> None:
    logger.info("Starting merge step")
    import src.training.finetune_mlx as ft

    ft.merge()
    logger.info("Merge step complete")


def cmd_app(args: argparse.Namespace) -> None:
    # Launch Streamlit app
    logger.info("Launching Streamlit app via run_cmd")
    code = run_cmd([sys.executable, "-m", "streamlit", "run", "src/app/streamlit_app.py"])
    if code != 0:
        logger.error("Streamlit exited with code=%s", code)
        sys.exit(code)
    else:
        logger.info("Streamlit exited cleanly")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.eval.runner import run_evaluation, summarize_run

    variant_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    # Default to live retrieval unless user explicitly disables it via flags (frozen only).
    live_flag = bool(args.live_rag) or (not bool(args.frozen_rag))
    run_dir = run_evaluation(
        dataset_path=args.dataset,
        variant_names=variant_list,
        out_dir=(args.out_dir or None),
        live_rag=live_flag,
        frozen_rag=bool(args.frozen_rag),
        max_items=args.max_items,
        warmup=args.warmup,
        seed=args.seed,
        auto_index=bool(args.auto_index),
    )
    summarize_run(run_dir)

def main() -> None:
    logger.info("Parsing CLI arguments")
    logger.debug("argv: %s", " ".join(sys.argv))
    ap = argparse.ArgumentParser(description="MDE Learn Chatbot pipeline")
    ap.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("crawl", help="Crawl MDE docs").set_defaults(fn=cmd_crawl)
    sub.add_parser("process", help="Process HTML to text chunks").set_defaults(fn=cmd_process)
    sub.add_parser("index", help="Build Chroma index").set_defaults(fn=cmd_index)
    sub.add_parser("prepare-dataset", help="Create JSONL text dataset for finetuning").set_defaults(fn=cmd_prepare_dataset)
    sub.add_parser("finetune", help="Run MLX LoRA finetuning").set_defaults(fn=cmd_finetune)
    sub.add_parser("merge", help="Merge LoRA into standalone weights").set_defaults(fn=cmd_merge)
    sub.add_parser("app", help="Start Streamlit app").set_defaults(fn=cmd_app)

    eval_p = sub.add_parser("evaluate", help="Evaluate model variants on a QA dataset")
    eval_p.add_argument("--dataset", type=str, required=True, help="Path to eval JSONL with fields: question, answer, gold_urls?")
    eval_p.add_argument("--variants", type=str, default="base_rag,rag_ft", help="Comma-separated from: base_ft,base_rag,ft_only,rag_ft")
    eval_p.add_argument("--out-dir", type=str, default="", help="Output run directory (defaults to outputs/eval/runs/<ts>)")
    eval_p.add_argument("--frozen-rag", action="store_true", help="Use frozen-RAG (cache retrieved context per question)")
    eval_p.add_argument("--live-rag", action="store_true", help="Use live retrieval timing (default if neither is set)")
    eval_p.add_argument("--max-items", type=int, default=None, help="Limit number of eval items")
    eval_p.add_argument("--warmup", type=int, default=1, help="Warmup prompts per variant")
    eval_p.add_argument("--seed", type=int, default=None, help="Random seed")
    eval_p.add_argument("--auto-index", action="store_true", help="Automatically build the Chroma index if missing (for RAG variants)")
    eval_p.set_defaults(fn=cmd_evaluate)

    args = ap.parse_args()
    if getattr(args, "debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    logger.info("Selected command: %s", args.cmd)
    logger.info("Dispatching to subcommand handler")
    try:
        args.fn(args)
        logger.info("Subcommand completed successfully")
    except Exception:
        logger.exception("Subcommand raised an exception")
        raise


if __name__ == "__main__":
    main()
