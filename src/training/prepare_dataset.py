import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

from src.utils.config import ensure_dir, load_config

logger = logging.getLogger(__name__)


def load_chunks(path: str) -> List[Dict[str, Any]]:
    logger.info("load_chunks <- %s", path)
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        logger.warning("chunks file missing: %s", path)
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
                logger.debug("bad json line skipped: %s", e)
                continue
    logger.info("load_chunks -> %d records", len(records))
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
    logger.info("split_dataset: total=%d val_ratio=%.3f seed=%d", len(records), val_ratio, seed)
    random.Random(seed).shuffle(records)
    n = len(records)
    n_val = int(n * val_ratio)
    val = records[:n_val]
    train = records[n_val:]
    train_texts = [build_clm_text(r) for r in train]
    val_texts = [build_clm_text(r) for r in val]
    logger.info("split -> train=%d val=%d", len(train_texts), len(val_texts))
    return train_texts, val_texts


def cap_samples(samples: List[str], cap: int | None) -> List[str]:
    if cap is None:
        return samples
    capped = samples[: max(0, cap)]
    logger.info("cap_samples: cap=%s -> %d", cap, len(capped))
    return capped


def write_jsonl_texts(texts: List[str], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    logger.info("write_jsonl_texts -> %s count=%d", path, len(texts))
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None
    logger.info("wrote %d lines (bytes=%s) to %s", len(texts), size, path)


def main() -> None:
    cfg = load_config()
    chunks_path = cfg["data"]["chunks_path"]
    out_train = cfg["data"]["finetune_train"]
    out_val = cfg["data"]["finetune_val"]
    val_ratio = float(cfg["finetune"]["val_ratio"])
    max_train = cfg["finetune"].get("max_train_samples")
    seed = int(cfg["project"]["seed"])

    logger.info("start -> chunks=%s", chunks_path)
    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.warning("No chunks found. Run crawler and processing first.")
        return

    train_texts, val_texts = split_dataset(chunks, val_ratio, seed)
    train_texts = cap_samples(train_texts, max_train if isinstance(max_train, int) else None)

    logger.info("writing train (%d) -> %s", len(train_texts), out_train)
    write_jsonl_texts(train_texts, out_train)

    logger.info("writing val (%d) -> %s", len(val_texts), out_val)
    write_jsonl_texts(val_texts, out_val)

    logger.info("DONE")


if __name__ == "__main__":
    main()
