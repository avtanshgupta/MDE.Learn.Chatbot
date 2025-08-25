import os
import json
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config, ensure_dir

logger = logging.getLogger(__name__)

def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    logger.info("load_chunks <- %s", chunks_path)
    chunks: List[Dict[str, Any]] = []
    if not os.path.exists(chunks_path):
        logger.warning("chunks file missing: %s", chunks_path)
        return chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "text" in rec and rec["text"]:
                    chunks.append(rec)
            except Exception as e:
                logger.debug("bad json line skipped: %s", e)
                continue
    logger.info("load_chunks -> %d records", len(chunks))
    return chunks

def build_index() -> None:
    cfg = load_config()
    persist_dir = cfg["index"]["chroma_persist_dir"]
    collection_name = cfg["index"]["collection"]
    embed_model_id = cfg["index"]["embedding_model"]
    batch = int(cfg["index"]["embedding_batch"])

    ensure_dir(persist_dir)
    logger.info("build_index -> persist_dir=%s collection=%s batch=%d", persist_dir, collection_name, batch)

    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
    try:
        collection = client.get_collection(collection_name)
        logger.info("using existing collection: %s", collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        logger.info("created new collection: %s", collection_name)

    # Load data
    chunks_path = cfg["data"]["chunks_path"]
    records = load_chunks(chunks_path)
    if not records:
        logger.warning("No chunks found at %s. Run processing first.", chunks_path)
        return

    # Embedding model
    logger.info("Loading embedding model: %s", embed_model_id)
    embedder = SentenceTransformer(embed_model_id)
    logger.info("Embedding model loaded.")

    # Prepare and upsert in batches
    ids, texts, metas = [], [], []
    logger.info("Collecting %d records for upsert", len(records))
    for rec in tqdm(records, desc="[indexing] Collecting"):
        uid = f'{rec["url"]}#chunk-{rec["chunk_id"]}'
        ids.append(uid)
        texts.append(rec["text"])
        metas.append({
            "url": rec.get("url", ""),
            "title": rec.get("title", ""),
            "source_path": rec.get("source_path", ""),
            "chunk_id": rec.get("chunk_id", 0),
        })

    total = len(texts)
    logger.info("Embedding & upserting total=%d batch=%d", total, batch)
    for i in tqdm(range(0, total, batch), desc="[indexing] Embedding"):
        batch_texts = texts[i:i+batch]
        batch_ids = ids[i:i+batch]
        batch_metas = metas[i:i+batch]
        embs = embedder.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
        embs_list = [e.astype(np.float32).tolist() for e in embs]
        collection.upsert(ids=batch_ids, embeddings=embs_list, metadatas=batch_metas, documents=batch_texts)
        logger.debug("upserted: %d..%d", i, i + len(batch_texts) - 1)

    logger.info("Indexing complete.")

if __name__ == "__main__":
    build_index()
