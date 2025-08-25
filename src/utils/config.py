import os
import json
import hashlib
from typing import Any, Dict, Optional

import yaml

print("[utils.config] Module loaded")

_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load and cache the YAML config."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        print(f"[utils.config] Loading config from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f) or {}
        print(f"[utils.config] Config loaded with top-level keys: {list(_CONFIG_CACHE.keys())}")
    else:
        print("[utils.config] Using cached config")
    return _CONFIG_CACHE

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    if path:
        os.makedirs(path, exist_ok=True)
        print(f"[utils.config] ensure_dir -> {path}")
    return path

def url_to_filename(url: str, ext: str = "html") -> str:
    """Create a stable filename for a URL using SHA256 hash prefix."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    fname = f"{h}.{ext}"
    print(f"[utils.config] url_to_filename: {url} -> {fname}")
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
    print(f"[utils.config] save_json -> {path} (type={kind}, bytes={size})")

def read_json(path: str) -> Any:
    """Read JSON from disk."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    kind = type(data).__name__
    print(f"[utils.config] read_json <- {path} (type={kind})")
    return data
