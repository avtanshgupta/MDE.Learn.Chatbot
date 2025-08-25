import os
import logging
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from src.utils.config import load_config

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self) -> None:
        logger.info("Initializing Retriever")
        cfg = load_config()
        self.persist_dir = cfg["index"]["chroma_persist_dir"]
        self.collection_name = cfg["index"]["collection"]
        self.embed_model_id = cfg["index"]["embedding_model"]
        logger.info(
            "Config: persist_dir=%s, collection=%s, embed_model=%s",
            self.persist_dir, self.collection_name, self.embed_model_id
        )

        logger.info("Connecting to Chroma PersistentClient...")
        self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(allow_reset=False))
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info("Loaded collection '%s'", self.collection_name)
        except Exception as e:
            logger.exception(
                "Failed to load collection '%s' at %s: %s",
                self.collection_name, self.persist_dir, e
            )
            raise RuntimeError(
                f"Chroma collection '{self.collection_name}' not found at {self.persist_dir}. "
                "Build the index first."
            ) from e

        logger.info("Loading embedding model: %s", self.embed_model_id)
        self.embedder = SentenceTransformer(self.embed_model_id)
        logger.info("Embedding model loaded")

    def embed(self, texts: List[str]) -> List[List[float]]:
        logger.debug("Embedding %d text(s)", len(texts))
        embs = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        try:
            shape = getattr(embs, "shape", None)
            dtype = getattr(embs, "dtype", None)
            logger.debug("Embeddings computed shape=%s dtype=%s", shape, dtype)
        except Exception:
            pass
        out = [e.astype(np.float32).tolist() for e in embs]
        logger.debug("Embeddings converted to float32 lists with count=%d", len(out))
        return out

    def query(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        logger.info("Querying collection '%s' with top_k=%d", self.collection_name, top_k)
        q_preview = query.replace("\n", " ")[:120]
        logger.debug("Query text preview: %s%s", q_preview, "..." if len(query) > 120 else "")
        embs = self.embed([query])
        logger.debug("Calling Chroma .query(...)")
        res = self.collection.query(query_embeddings=embs, n_results=top_k, include=["metadatas", "documents", "distances", "embeddings"])
        docs: List[Dict[str, Any]] = []
        n_hits = 0
        try:
            n_hits = len(res.get("ids", [[]])[0])
        except Exception:
            pass
        logger.info("Retrieved %d hit(s)", n_hits)
        # Chroma returns lists of lists (per query)
        for i in range(n_hits):
            docs.append({
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
        logger.debug("Assembled %d document dict(s)", len(docs))
        return docs

def format_context(docs: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return formatted context string and simplified sources list."""
    logger.debug("Formatting context for %d doc(s)", len(docs))
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
    logger.debug("Built context length=%d chars; sources=%d", len(context), len(sources))
    return context, sources
