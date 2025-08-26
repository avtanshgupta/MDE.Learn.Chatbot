import contextlib
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def timer(name: str, store: Optional[Dict[str, float]] = None, key: Optional[str] = None):
    """
    Context manager to measure elapsed wall time in seconds.
    If 'store' is provided, records under store[key or name].
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if store is not None:
            store[key or name] = dt
        logger.debug("Timer '%s' -> %.4fs", name, dt)

def count_tokens(tokenizer, text: str) -> int:
    """
    Best-effort token count using available tokenizer API.
    Falls back to character length if encode is unavailable.
    """
    try:
        # HuggingFace-style
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text or ""))
        # Some tokenizers expose __call__ returning input_ids
        if callable(tokenizer):
            out = tokenizer(text or "", return_tensors=None)
            if isinstance(out, dict) and "input_ids" in out:
                return len(out["input_ids"])
    except Exception as e:
        logger.debug("count_tokens fallback due to: %s", e)
    # Fallback heuristic
    return max(1, len((text or "").split()))
