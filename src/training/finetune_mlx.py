import logging
import os
import subprocess
import sys
from typing import List

from src.utils.config import ensure_dir, load_config

logger = logging.getLogger(__name__)


def run(cmd: List[str]) -> int:
    logger.info("Running command: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.communicate()
    code = proc.returncode
    logger.info("Command finished with code=%s", code)
    return code


def finetune() -> None:
    cfg = load_config()

    base_id = cfg["model"]["base_id"]
    train_path = cfg["data"]["finetune_train"]
    val_path = cfg["data"]["finetune_val"]
    out_dir = cfg["finetune"]["out_dir"]

    epochs = str(cfg["finetune"]["epochs"])
    batch_size = str(cfg["finetune"]["batch_size"])
    accumulate = str(cfg["finetune"]["accumulate_steps"])
    lr = str(cfg["finetune"]["lr"])

    lora = cfg["finetune"]["lora"]
    r = str(lora["r"])
    alpha = str(lora["alpha"])
    dropout = str(lora["dropout"])

    ensure_dir(out_dir)

    logger.info("Finetune start ->")
    logger.info("  model=%s", base_id)
    logger.info("  train=%s", train_path)
    logger.info("  val=%s", val_path)
    logger.info("  out_dir=%s", out_dir)
    logger.info("  epochs=%s batch=%s accumulate=%s lr=%s", epochs, batch_size, accumulate, lr)
    logger.info("  lora: r=%s alpha=%s dropout=%s", r, alpha, dropout)

    # Prepare mlx_lm lora dataset directory with expected filenames
    lora_data_dir = os.path.join(os.path.dirname(train_path), "lora_data")
    ensure_dir(lora_data_dir)
    train_link = os.path.join(lora_data_dir, "train.jsonl")
    valid_link = os.path.join(lora_data_dir, "valid.jsonl")
    # link or copy train/val into expected names
    for src, dst in [(train_path, train_link), (val_path, valid_link)]:
        try:
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            logger.debug("Symlinked %s -> %s", dst, src)
        except Exception as e:
            import shutil

            shutil.copyfile(src, dst)
            logger.debug("Copied %s -> %s (symlink failed: %s)", src, dst, e)

    # Uses mlx_lm lora subcommand (LoRA fine-tuning)
    # Note: Recent mlx-lm versions use 'lora' with --data directory (containing train.jsonl/valid.jsonl)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        base_id,
        "--train",
        "--data",
        lora_data_dir,
        "--adapter-path",
        out_dir,
        "--batch-size",
        batch_size,
        "--learning-rate",
        lr,
        "--save-every",
        "1000",
    ]

    code = run(cmd)
    if code != 0:
        logger.error("Finetune failed with code=%s", code)
        sys.exit(code)
    logger.info("Finetune DONE. LoRA adapter saved to: %s", out_dir)


def merge() -> None:
    cfg = load_config()
    base_id = cfg["model"]["base_id"]
    adapter_dir = cfg["finetune"]["out_dir"]
    out_dir = cfg["merge"]["out_dir"]

    ensure_dir(out_dir)

    logger.info("Merge start ->")
    logger.info("  base=%s", base_id)
    logger.info("  adapter=%s", adapter_dir)
    logger.info("  out_dir=%s", out_dir)

    # Merge LoRA into a new model folder for standalone inference (optional; for FT-only mode)
    cmd = [sys.executable, "-m", "mlx_lm", "fuse", "--model", base_id, "--adapter-path", adapter_dir, "--save-path", out_dir]
    code = run(cmd)
    if code != 0:
        logger.error("Merge failed with code=%s", code)
        sys.exit(code)
    logger.info("Merge DONE. Merged model saved to: %s", out_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only", action="store_true", help="Only run LoRA merge step.")
    args = ap.parse_args()

    if args.merge_only:
        merge()
    else:
        finetune()
