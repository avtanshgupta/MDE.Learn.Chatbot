import os
import re
import json
from typing import List, Dict, Any, Tuple

from bs4 import BeautifulSoup
from readability import Document

from src.utils.config import load_config, ensure_dir, read_json

print("[processing] Module loaded")

BLOCK_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "aside", "form"}

def html_to_text(html: str) -> Tuple[str, str]:
    """Extract main content text and title from HTML using readability with bs4 cleanup."""
    try:
        doc = Document(html)
        title = doc.short_title() or ""
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
        print(f"[processing] readability extracted title='{title[:80]}'")
    except Exception as e:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.text.strip() if soup.title else ""
        print(f"[processing] readability failed, fallback bs4 title='{title[:80]}', err={e}")

    # remove non-content blocks
    removed = 0
    for tag in soup.find_all(BLOCK_TAGS):
        tag.decompose()
        removed += 1
    if removed:
        print(f"[processing] removed non-content blocks: {removed}")

    # get visible text
    text_parts: List[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code"]):
        t = node.get_text(separator=" ", strip=True)
        if t:
            text_parts.append(t)

    text = "\n".join(text_parts)
    text = normalize_whitespace(text)
    print(f"[processing] html_to_text: paragraphs={len(text_parts)} chars={len(text)}")
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
    print(f"[processing] chunk_text: paras={len(paras)} max_len={max_len} overlap={overlap}")

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
            for i in range(0, len(c), max_len - overlap if overlap < max_len else max_len):
                trimmed.append(c[i:i + max_len])

    print(f"[processing] chunk_text: chunks={len(trimmed)}")
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
    print(f"[processing] start -> manifest={manifest_path} chunks_out={chunks_path}")

    manifest: List[Dict[str, Any]] = read_json(manifest_path) if os.path.exists(manifest_path) else []
    print(f"[processing] manifest entries: {len(manifest)}")

    out_f = open(chunks_path, "w", encoding="utf-8")
    count_pages, count_chunks = 0, 0

    for item in manifest:
        url = item["url"]
        path = item["path"]
        if not os.path.exists(path):
            print(f"[processing] skip missing file: {path}")
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        print(f"[processing] page -> url={url}")
        title, text = html_to_text(html)
        if not text or len(text) < min_section:
            print(f"[processing] skip short page: chars={len(text) if text else 0} url={url}")
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
        print(f"[processing] page done: url={url} chunks={len(chunks)}")

    out_f.close()
    print(f"[processing] DONE pages={count_pages} chunks_total={count_chunks} -> {chunks_path}")

if __name__ == "__main__":
    process()
