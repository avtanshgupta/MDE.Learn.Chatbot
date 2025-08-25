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
            print(f"[training.finetune_mlx] Symlinked {dst} -> {src}")
        except Exception as e:
            import shutil
            shutil.copyfile(src, dst)
            print(f"[training.finetune_mlx] Copied {src} -> {dst} (symlink failed: {e})")

    # Uses mlx_lm lora subcommand (LoRA fine-tuning)
    # Note: Recent mlx-lm versions use 'lora' with --data directory (containing train.jsonl/valid.jsonl)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_id,
        "--train",
        "--data", lora_data_dir,
        "--adapter-path", out_dir,
        "--batch-size", batch_size,
        "--learning-rate", lr,
        "--save-every", "1000",
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
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", base_id,
        "--adapter-path", adapter_dir,
        "--save-path", out_dir
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
