import os
import json
import hashlib
from typing import Any, Dict, Optional

import yaml
import logging
from src.utils.logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load and cache the YAML config."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        logger.info("Loading config from: %s", path)
        with open(path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f) or {}
        logger.debug("Config loaded with top-level keys: %s", list(_CONFIG_CACHE.keys()))
    else:
        logger.debug("Using cached config")
    return _CONFIG_CACHE

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    if path:
        os.makedirs(path, exist_ok=True)
        logger.debug("ensure_dir -> %s", path)
    return path

def url_to_filename(url: str, ext: str = "html") -> str:
    """Create a stable filename for a URL using SHA256 hash prefix."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    fname = f"{h}.{ext}"
    logger.debug("url_to_filename: %s -> %s", url, fname)
    return fname

def save_json(obj: Any, path: str) -> None:
    """Write JSON to disk with UTF-8 and nice formatting."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    size = None
    try:
        size = os.path.getsize(path)
    except Exception:
        pass
    kind = type(obj).__name__
    logger.info("save_json -> %s (type=%s, bytes=%s)", path, kind, size)

def read_json(path: str) -> Any:
    """Read JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    kind = type(data).__name__
    logger.debug("read_json <- %s (type=%s)", path, kind)
    return data
