import os
import sys
import subprocess
import argparse

# Ensure project root on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_cmd(cmd: list[str]) -> int:
    print(f"[main] Running command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    proc.communicate()
    code = proc.returncode
    print(f"[main] Command exited with code={code}")
    return code

def cmd_crawl(args: argparse.Namespace) -> None:
    print("[main] Starting crawl step")
    import src.crawler.crawler as crawler
    crawler.crawl()
    print("[main] Crawl step complete")

def cmd_process(args: argparse.Namespace) -> None:
    print("[main] Starting process step")
    import src.processing.process as process
    process.process()
    print("[main] Process step complete")

def cmd_index(args: argparse.Namespace) -> None:
    print("[main] Starting index step")
    import src.indexing.build_index as build_index
    build_index.build_index()
    print("[main] Index step complete")

def cmd_prepare_dataset(args: argparse.Namespace) -> None:
    print("[main] Starting dataset preparation step")
    import src.training.prepare_dataset as prep
    prep.main()
    print("[main] Dataset preparation step complete")

def cmd_finetune(args: argparse.Namespace) -> None:
    print("[main] Starting finetune step")
    import src.training.finetune_mlx as ft
    ft.finetune()
    print("[main] Finetune step complete")

def cmd_merge(args: argparse.Namespace) -> None:
    print("[main] Starting merge step")
    import src.training.finetune_mlx as ft
    ft.merge()
    print("[main] Merge step complete")

def cmd_app(args: argparse.Namespace) -> None:
    # Launch Streamlit app
    print("[main] Launching Streamlit app via run_cmd")
    code = run_cmd([sys.executable, "-m", "streamlit", "run", "src/app/app.py"])
    if code != 0:
        print(f"[main] Streamlit exited with code={code}")
        sys.exit(code)
    else:
        print("[main] Streamlit exited cleanly")

def main() -> None:
    print("[main] Parsing CLI arguments")
    print(f"[main] argv: {' '.join(sys.argv)}")
    ap = argparse.ArgumentParser(description="MDE Learn Chatbot pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("crawl", help="Crawl MDE docs").set_defaults(fn=cmd_crawl)
    sub.add_parser("process", help="Process HTML to text chunks").set_defaults(fn=cmd_process)
    sub.add_parser("index", help="Build Chroma index").set_defaults(fn=cmd_index)
    sub.add_parser("prepare-dataset", help="Create JSONL text dataset for finetuning").set_defaults(fn=cmd_prepare_dataset)
    sub.add_parser("finetune", help="Run MLX LoRA finetuning").set_defaults(fn=cmd_finetune)
    sub.add_parser("merge", help="Merge LoRA into standalone weights").set_defaults(fn=cmd_merge)
    sub.add_parser("app", help="Start Streamlit app").set_defaults(fn=cmd_app)

    args = ap.parse_args()
    print(f"[main] Selected command: {args.cmd}")
    print("[main] Dispatching to subcommand handler")
    try:
        args.fn(args)
        print("[main] Subcommand completed successfully")
    except Exception as e:
        print(f"[main] Subcommand raised an exception: {e}")
        raise

if __name__ == "__main__":
    main()
