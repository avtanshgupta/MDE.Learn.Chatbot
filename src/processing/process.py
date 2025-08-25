import os
import re
import json
from typing import List, Dict, Any, Tuple
import logging

from bs4 import BeautifulSoup
from readability import Document

from src.utils.config import load_config, ensure_dir, read_json

logger = logging.getLogger(__name__)

BLOCK_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "aside", "form"}

def html_to_text(html: str) -> Tuple[str, str]:
    """Extract main content text and title from HTML using readability with bs4 cleanup."""
    try:
        doc = Document(html)
        title = doc.short_title() or ""
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
        logger.debug("readability extracted title='%s'", title[:80])
    except Exception as e:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.text.strip() if soup.title else ""
        logger.debug("readability failed, fallback bs4 title='%s', err=%s", title[:80], e)

    # remove non-content blocks
    removed = 0
    for tag in soup.find_all(BLOCK_TAGS):
        tag.decompose()
        removed += 1
    if removed:
        logger.debug("removed non-content blocks: %d", removed)

    # get visible text
    text_parts: List[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code"]):
        t = node.get_text(separator=" ", strip=True)
        if t:
            text_parts.append(t)

    text = "\n".join(text_parts)
    text = normalize_whitespace(text)
    logger.debug("html_to_text: paragraphs=%d chars=%d", len(text_parts), len(text))
    return title, text

def normalize_whitespace(s: str) -> str:
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, max_len: int, overlap: int) -> List[str]:
    """Greedy paragraph-based chunking with character limits and overlap."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    logger.debug("chunk_text: paras=%d max_len=%d overlap=%d", len(paras), max_len, overlap)

    for p in paras:
        p_len = len(p) + 2  # account for newline join
        if cur_len + p_len <= max_len or not cur:
            cur.append(p)
            cur_len += p_len
        else:
            chunks.append("\n\n".join(cur))
            # start new chunk with overlap from end of previous
            if overlap > 0 and chunks[-1]:
                prev = chunks[-1]
                tail = prev[max(0, len(prev) - overlap):]
                # try to align to paragraph boundary
                tail_paras = tail.split("\n\n")
                tail_keep = tail_paras[-3:] if len(tail_paras) > 3 else tail_paras
                cur = ["\n\n".join(tail_keep), p]
                cur_len = sum(len(x) + 2 for x in cur)
            else:
                cur = [p]
                cur_len = len(p) + 2

    if cur:
        chunks.append("\n\n".join(cur))

    # final trim to max_len
    trimmed: List[str] = []
    for c in chunks:
        if len(c) <= max_len:
            trimmed.append(c)
        else:
            # hard split if a single paragraph overflowed
            step = max_len - overlap if overlap < max_len else max_len
            for i in range(0, len(c), step):
                trimmed.append(c[i:i + max_len])

    logger.debug("chunk_text: chunks=%d", len(trimmed))
    return trimmed

def process() -> None:
    cfg = load_config()
    manifest_path = cfg["data"]["url_manifest"]
    raw_dir = cfg["data"]["raw_html_dir"]
    chunks_path = cfg["data"]["chunks_path"]

    min_section = int(cfg["processing"]["min_section_chars"])
    max_chunk = int(cfg["processing"]["max_chunk_chars"])
    overlap = int(cfg["processing"]["chunk_overlap_chars"])

    ensure_dir(os.path.dirname(chunks_path))
    logger.info("start -> manifest=%s chunks_out=%s", manifest_path, chunks_path)

    manifest: List[Dict[str, Any]] = read_json(manifest_path) if os.path.exists(manifest_path) else []
    logger.info("manifest entries: %d", len(manifest))

    out_f = open(chunks_path, "w", encoding="utf-8")
    count_pages, count_chunks = 0, 0

    for item in manifest:
        url = item["url"]
        path = item["path"]
        if not os.path.exists(path):
            logger.debug("skip missing file: %s", path)
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        logger.debug("page -> url=%s", url)
        title, text = html_to_text(html)
        if not text or len(text) < min_section:
            logger.debug("skip short page: chars=%s url=%s", len(text) if text else 0, url)
            continue

        chunks = chunk_text(text, max_chunk, overlap)
        for idx, chunk in enumerate(chunks):
            rec = {
                "url": url,
                "title": title,
                "source_path": path,
                "chunk_id": idx,
                "text": chunk,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count_chunks += 1

        count_pages += 1
        logger.debug("page done: url=%s chunks=%d", url, len(chunks))

    out_f.close()
    logger.info("DONE pages=%d chunks_total=%d -> %s", count_pages, count_chunks, chunks_path)

if __name__ == "__main__":
    process()
