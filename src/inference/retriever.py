import os
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from src.utils.config import load_config

class Retriever:
    def __init__(self) -> None:
        print("[inference.retriever] Initializing Retriever")
        cfg = load_config()
        self.persist_dir = cfg["index"]["chroma_persist_dir"]
        self.collection_name = cfg["index"]["collection"]
        self.embed_model_id = cfg["index"]["embedding_model"]
        print(f"[inference.retriever] Config: persist_dir={self.persist_dir}, collection={self.collection_name}, embed_model={self.embed_model_id}")

        print("[inference.retriever] Connecting to Chroma PersistentClient...")
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(allow_reset=False))
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"[inference.retriever] Loaded collection '{self.collection_name}'")
        except Exception as e:
            print(f"[inference.retriever] Failed to load collection '{self.collection_name}' at {self.persist_dir}: {e}")
            raise RuntimeError(
                f"Chroma collection '{self.collection_name}' not found at {self.persist_dir}. "
                "Build the index first."
            ) from e

        print(f"[inference.retriever] Loading embedding model: {self.embed_model_id}")
        self.embedder = SentenceTransformer(self.embed_model_id)
        print("[inference.retriever] Embedding model loaded")

    def embed(self, texts: List[str]) -> List[List[float]]:
        print(f"[inference.retriever] Embedding {len(texts)} text(s)")
        embs = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        try:
            shape = getattr(embs, "shape", None)
            dtype = getattr(embs, "dtype", None)
            print(f"[inference.retriever] Embeddings computed shape={shape} dtype={dtype}")
        except Exception:
            pass
        out = [e.astype(np.float32).tolist() for e in embs]
        print(f"[inference.retriever] Embeddings converted to float32 lists with count={len(out)}")
        return out

    def query(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        print(f"[inference.retriever] Querying collection '{self.collection_name}' with top_k={top_k}")
        print(f"[inference.retriever] Query text: {query[:120].replace('\\n', ' ')}{'...' if len(query) > 120 else ''}")
        embs = self.embed([query])
        print("[inference.retriever] Calling Chroma .query(...)")
        res = self.collection.query(query_embeddings=embs, n_results=top_k, include=["metadatas", "documents", "distances", "embeddings"])
        docs: List[Dict[str, Any]] = []
        n_hits = 0
        try:
            n_hits = len(res.get("ids", [[]])[0])
        except Exception:
            pass
        print(f"[inference.retriever] Retrieved {n_hits} hit(s)")
        # Chroma returns lists of lists (per query)
        for i in range(n_hits):
            docs.append({
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
        print(f"[inference.retriever] Assembled {len(docs)} document dict(s)")
        return docs

def format_context(docs: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return formatted context string and simplified sources list."""
    print(f"[inference.retriever] Formatting context for {len(docs)} doc(s)")
    lines: List[str] = []
    sources: List[Dict[str, Any]] = []
    for idx, d in enumerate(docs, start=1):
        meta = d.get("metadata", {}) or {}
        title = meta.get("title", "") or "Untitled"
        url = meta.get("url", "") or ""
        chunk_id = meta.get("chunk_id", 0)
        lines.append(f"[{idx}] Title: {title}\nURL: {url}\nChunk: {chunk_id}\nContent:\n{d.get('text','')}\n")
        sources.append({"rank": idx, "title": title, "url": url, "chunk_id": chunk_id, "distance": d.get("distance", None)})
    context = "\n\n".join(lines)
    print(f"[inference.retriever] Built context length={len(context)} chars; sources={len(sources)}")
    return context, sources
