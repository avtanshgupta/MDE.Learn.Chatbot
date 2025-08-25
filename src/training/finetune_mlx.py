import os
import subprocess
import sys
from typing import List

from src.utils.config import load_config, ensure_dir

print("[training.finetune_mlx] Module loaded")

def run(cmd: List[str]) -> int:
    print("[training.finetune_mlx] Running command:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.communicate()
    code = proc.returncode
    print(f"[training.finetune_mlx] Command finished with code={code}")
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

    print("[training.finetune_mlx] Finetune start ->")
    print(f"  model={base_id}")
    print(f"  train={train_path}")
    print(f"  val={val_path}")
    print(f"  out_dir={out_dir}")
    print(f"  epochs={epochs} batch={batch_size} accumulate={accumulate} lr={lr}")
    print(f"  lora: r={r} alpha={alpha} dropout={dropout}")

    # Uses mlx-lm finetune module with LoRA on JSONL {"text": "..."} format
    # Trains locally on Apple Silicon (MLX).
    cmd = [
        sys.executable, "-m", "mlx_lm.finetune",
        "--model", base_id,
        "--train", train_path,
        "--val", val_path,
        "--output", out_dir,
        "--epochs", epochs,
        "--batch-size", batch_size,
        "--accumulate", accumulate,
        "--lr", lr,
        "--lora-r", r,
        "--lora-alpha", alpha,
        "--lora-dropout", dropout,
        "--format", "text",           # our dataset has {"text": "..."}
        "--dtype", "bfloat16",        # good default for Apple Silicon
        "--warmup-steps", "50",
        "--save-every", "0",          # only save at the end
    ]

    code = run(cmd)
    if code != 0:
        print("[training.finetune_mlx] Finetune failed.")
        sys.exit(code)
    print(f"[training.finetune_mlx] Finetune DONE. LoRA adapter saved to: {out_dir}")

def merge() -> None:
    cfg = load_config()
    base_id = cfg["model"]["base_id"]
    adapter_dir = cfg["finetune"]["out_dir"]
    out_dir = cfg["merge"]["out_dir"]

    ensure_dir(out_dir)

    print("[training.finetune_mlx] Merge start ->")
    print(f"  base={base_id}")
    print(f"  adapter={adapter_dir}")
    print(f"  out_dir={out_dir}")

    # Merge LoRA into a new model folder for standalone inference (optional; for FT-only mode)
    cmd = [
        sys.executable, "-m", "mlx_lm.merge",
        "--model", base_id,
        "--adapter", adapter_dir,
        "--out", out_dir
    ]
    code = run(cmd)
    if code != 0:
        print("[training.finetune_mlx] Merge failed.")
        sys.exit(code)
    print(f"[training.finetune_mlx] Merge DONE. Merged model saved to: {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only", action="store_true", help="Only run LoRA merge step.")
    args = ap.parse_args()

    if args.merge_only:
        merge()
    else:
        finetune()
