import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

@dataclass
class QAItem:
    id: str
    question: str
    answer: str
    gold_urls: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _coerce_item(obj: Dict[str, Any], idx: int) -> Optional[QAItem]:
    if not isinstance(obj, dict):
        logger.warning("Skipping non-dict item at line=%d", idx)
        return None
    q = (obj.get("question") or "").strip()
    a = (obj.get("answer") or "").strip()
    if not q or not a:
        logger.warning("Skipping item missing question/answer at line=%d", idx)
        return None
    gold_urls = obj.get("gold_urls") or []
    if not isinstance(gold_urls, list):
        gold_urls = []
    _id = (obj.get("id") or f"item_{idx}").strip()
    return QAItem(id=_id, question=q, answer=a, gold_urls=[str(u).strip() for u in gold_urls if str(u).strip()])

def load_eval_set(path: str, max_items: Optional[int] = None) -> List[QAItem]:
    """
    Load a JSONL eval set with schema:
      {"id": str?, "question": str, "answer": str, "gold_urls": [str]?}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    items: List[QAItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                logger.warning("Skipping bad JSONL line=%d: %s", i, e)
                continue
            item = _coerce_item(obj, i)
            if item:
                items.append(item)
            if isinstance(max_items, int) and len(items) >= max_items:
                break
    logger.info("Loaded eval set: %s (count=%d)", path, len(items))
    return items
