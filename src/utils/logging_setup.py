import logging
import os
from typing import Optional

_INITIALIZED = False


def setup_logging(level: Optional[str] = None) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    # Resolve level: explicit arg > env LOG_LEVEL > INFO
    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    level_value = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level_value,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _INITIALIZED = True
