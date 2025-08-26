import logging
import math
import re
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.config import load_config

logger = logging.getLogger(__name__)

# Lazy singleton for sentence-transformers (used for cosine similarity)
_EMBEDDER = None
def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            cfg = load_config()
            model_id = cfg.get("index", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Loading sentence-transformers model for eval metrics: %s", model_id)
            _EMBEDDER = SentenceTransformer(model_id)
        except Exception as e:
            logger.warning("Failed to load sentence-transformers for cosine similarity: %s", e)
            _EMBEDDER = False  # sentinel for unavailable
    return _EMBEDDER

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")

def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    s = s or ""
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []

def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalized strings match exactly else 0.0."""
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def token_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """Returns (precision, recall, f1) at token level."""
    p_tokens = _tokenize(pred)
    g_tokens = _tokenize(gold)
    if not p_tokens and not g_tokens:
        return 1.0, 1.0, 1.0
    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0
    from collections import Counter
    p_cnt = Counter(p_tokens)
    g_cnt = Counter(g_tokens)
    overlap = sum((p_cnt & g_cnt).values())
    precision = overlap / max(1, sum(p_cnt.values()))
    recall = overlap / max(1, sum(g_cnt.values()))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        ai = a[i]
        row = dp[i]
        row_next = dp[i+1]
        for j in range(m):
            if ai == b[j]:
                row_next[j+1] = row[j] + 1
            else:
                row_next[j+1] = max(row_next[j], row[j+1])
    return dp[n][m]

def rouge_l(pred: str, gold: str) -> Dict[str, float]:
    """Compute ROUGE-L (F1-style) via LCS over tokens. Returns dict with p, r, f1."""
    p_tokens = _tokenize(pred)
    g_tokens = _tokenize(gold)
    if not p_tokens and not g_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not p_tokens or not g_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_len(p_tokens, g_tokens)
    p = lcs / max(1, len(p_tokens))
    r = lcs / max(1, len(g_tokens))
    f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
    return {"precision": p, "recall": r, "f1": f1}

def cosine_similarity(pred: str, gold: str) -> Optional[float]:
    """Cosine similarity between embeddings of pred and gold. Returns None if unavailable."""
    embedder = _get_embedder()
    if not embedder:
        return None
    try:
        embs = embedder.encode([pred or "", gold or ""], show_progress_bar=False, normalize_embeddings=True)
        import numpy as np
        a = embs[0].astype(np.float32)
        b = embs[1].astype(np.float32)
        sim = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        return sim
    except Exception as e:
        logger.warning("cosine_similarity failed: %s", e)
        return None

def summarize(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return {"mean": None, "min": None, "max": None}
    return {"mean": sum(vals)/len(vals), "min": min(vals), "max": max(vals)}

def bootstrap_ci(values: List[float], n: int = 1000, alpha: float = 0.05, seed: int = 42) -> Optional[Tuple[float, float]]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    rng = random.Random(seed)
    samples = []
    k = len(vals)
    for _ in range(max(10, n)):
        bs = [vals[rng.randrange(k)] for _ in range(k)]
        samples.append(sum(bs)/len(bs))
    samples.sort()
    lo_idx = int((alpha/2) * len(samples))
    hi_idx = int((1 - alpha/2) * len(samples)) - 1
    lo = samples[max(0, min(lo_idx, len(samples)-1))]
    hi = samples[max(0, min(hi_idx, len(samples)-1))]
    return (lo, hi)

def compute_all_metrics(pred: str, gold: str) -> Dict[str, Any]:
    em = exact_match(pred, gold)
    p, r, f1 = token_f1(pred, gold)
    rouge = rouge_l(pred, gold)
    cos = cosine_similarity(pred, gold)
    return {
        "exact_match": em,
        "precision": p,
        "recall": r,
        "f1": f1,
        "rouge_l_f1": rouge.get("f1", 0.0),
        "cosine_sim": cos,
    }
