import os
import sys
import subprocess
import argparse
import logging
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
    import src.crawler.crawler as crawler
    crawler.crawl()
    logger.info("Crawl step complete")

def cmd_process(args: argparse.Namespace) -> None:
    logger.info("Starting process step")
    import src.processing.process as process
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
    code = run_cmd([sys.executable, "-m", "streamlit", "run", "src/app/app.py"])
    if code != 0:
        logger.error("Streamlit exited with code=%s", code)
        sys.exit(code)
    else:
        logger.info("Streamlit exited cleanly")

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
