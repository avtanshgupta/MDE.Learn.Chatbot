import json
import os
import random
from typing import List, Dict, Any, Tuple

from src.utils.config import load_config, ensure_dir, read_json

print("[training.prepare_dataset] Module loaded")

def load_chunks(path: str) -> List[Dict[str, Any]]:
    print(f"[training.prepare_dataset] load_chunks <- {path}")
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        print(f"[training.prepare_dataset] chunks file missing: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("text"):
                    records.append(rec)
            except Exception as e:
                print(f"[training.prepare_dataset] bad json line skipped: {e}")
                continue
    print(f"[training.prepare_dataset] load_chunks -> {len(records)} records")
    return records

def build_clm_text(rec: Dict[str, Any]) -> str:
    title = rec.get("title", "").strip()
    url = rec.get("url", "").strip()
    text = rec.get("text", "").strip()
    header = []
    if title:
        header.append(f"Title: {title}")
    if url:
        header.append(f"URL: {url}")
    prefix = ("\n".join(header) + "\n\n") if header else ""
    return f"{prefix}{text}\n\n"

def split_dataset(records: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    print(f"[training.prepare_dataset] split_dataset: total={len(records)} val_ratio={val_ratio} seed={seed}")
    random.Random(seed).shuffle(records)
    n = len(records)
    n_val = int(n * val_ratio)
    val = records[:n_val]
    train = records[n_val:]
    train_texts = [build_clm_text(r) for r in train]
    val_texts = [build_clm_text(r) for r in val]
    print(f"[training.prepare_dataset] split -> train={len(train_texts)} val={len(val_texts)}")
    return train_texts, val_texts

def cap_samples(samples: List[str], cap: int | None) -> List[str]:
    if cap is None:
        return samples
    capped = samples[: max(0, cap)]
    print(f"[training.prepare_dataset] cap_samples: cap={cap} -> {len(capped)}")
    return capped

def write_jsonl_texts(texts: List[str], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    print(f"[training.prepare_dataset] write_jsonl_texts -> {path} count={len(texts)}")
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None
    print(f"[training.prepare_dataset] wrote {len(texts)} lines (bytes={size}) to {path}")

def main() -> None:
    cfg = load_config()
    chunks_path = cfg["data"]["chunks_path"]
    out_train = cfg["data"]["finetune_train"]
    out_val = cfg["data"]["finetune_val"]
    val_ratio = float(cfg["finetune"]["val_ratio"])
    max_train = cfg["finetune"].get("max_train_samples")
    seed = int(cfg["project"]["seed"])

    print(f"[training.prepare_dataset] start -> chunks={chunks_path}")
    chunks = load_chunks(chunks_path)
    if not chunks:
        print("[training.prepare_dataset] No chunks found. Run crawler and processing first.")
        return

    train_texts, val_texts = split_dataset(chunks, val_ratio, seed)
    train_texts = cap_samples(train_texts, max_train if isinstance(max_train, int) else None)

    print(f"[training.prepare_dataset] writing train ({len(train_texts)}) -> {out_train}")
    write_jsonl_texts(train_texts, out_train)

    print(f"[training.prepare_dataset] writing val ({len(val_texts)}) -> {out_val}")
    write_jsonl_texts(val_texts, out_val)

    print("[training.prepare_dataset] DONE")

if __name__ == "__main__":
    main()
