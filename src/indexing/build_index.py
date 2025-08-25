import os
import json
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config, ensure_dir

print("[indexing] Module loaded")

def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    print(f"[indexing] load_chunks <- {chunks_path}")
    chunks: List[Dict[str, Any]] = []
    if not os.path.exists(chunks_path):
        print(f"[indexing] chunks file missing: {chunks_path}")
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
                print(f"[indexing] bad json line skipped: {e}")
                continue
    print(f"[indexing] load_chunks -> {len(chunks)} records")
    return chunks

def build_index() -> None:
    cfg = load_config()
    persist_dir = cfg["index"]["chroma_persist_dir"]
    collection_name = cfg["index"]["collection"]
    embed_model_id = cfg["index"]["embedding_model"]
    batch = int(cfg["index"]["embedding_batch"])

    ensure_dir(persist_dir)
    print(f"[indexing] build_index -> persist_dir={persist_dir} collection={collection_name} batch={batch}")

    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
    try:
        collection = client.get_collection(collection_name)
        print(f"[indexing] using existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        print(f"[indexing] created new collection: {collection_name}")

    # Load data
    chunks_path = cfg["data"]["chunks_path"]
    records = load_chunks(chunks_path)
    if not records:
        print(f"[indexing] No chunks found at {chunks_path}. Run processing first.")
        return

    # Embedding model
    print(f"[indexing] Loading embedding model: {embed_model_id}")
    embedder = SentenceTransformer(embed_model_id)
    print(f"[indexing] Embedding model loaded.")

    # Prepare and upsert in batches
    ids, texts, metas = [], [], []
    print(f"[indexing] Collecting {len(records)} records for upsert")
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
    print(f"[indexing] Embedding & upserting total={total} batch={batch}")
    for i in tqdm(range(0, total, batch), desc="[indexing] Embedding"):
        batch_texts = texts[i:i+batch]
        batch_ids = ids[i:i+batch]
        batch_metas = metas[i:i+batch]
        embs = embedder.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
        embs_list = [e.astype(np.float32).tolist() for e in embs]
        collection.upsert(ids=batch_ids, embeddings=embs_list, metadatas=batch_metas, documents=batch_texts)
        print(f"[indexing] upserted: {i}..{i+len(batch_texts)-1}")

    print("[indexing] Indexing complete.")

if __name__ == "__main__":
    build_index()
