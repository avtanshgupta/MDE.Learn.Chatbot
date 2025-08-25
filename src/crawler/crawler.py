import re
import time
import httpx
import urllib.parse as urlparse
from urllib.parse import urljoin, urlunparse
from urllib import robotparser
from bs4 import BeautifulSoup
from typing import Set, List, Dict, Tuple

from src.utils.config import load_config, ensure_dir, url_to_filename, save_json

print("[crawler] Module loaded")

def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments and normalizing scheme/host."""
    parsed = urlparse.urlparse(url)
    normalized = parsed._replace(fragment="")
    netloc = normalized.netloc.lower()
    scheme = normalized.scheme.lower()
    out = urlunparse((scheme, netloc, normalized.path, normalized.params, normalized.query, ""))
    # Debug (comment to reduce noise if needed)
    # print(f"[crawler] normalize_url: {url} -> {out}")
    return out

def is_allowed(url: str, cfg: dict, rp: robotparser.RobotFileParser) -> bool:
    """Check domain/path filters, language, patterns, and robots.txt."""
    allowed_domain = cfg["crawl"]["allowed_domain"]
    allowed_prefix = cfg["crawl"]["allowed_path_prefix"]
    same_lang = cfg["crawl"]["same_language_only"]
    exclude_patterns = cfg["crawl"]["exclude_url_patterns"]
    include_filetypes = cfg["crawl"]["include_filetypes"]

    parsed = urlparse.urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    if parsed.netloc != allowed_domain:
        return False

    if not parsed.path.startswith(allowed_prefix):
        return False

    if same_lang and not parsed.path.startswith(allowed_prefix):
        return False

    # Exclude patterns
    for pat in exclude_patterns or []:
        if pat and pat in url:
            return False

    # Filetype check (simple)
    if "." in parsed.path.split("/")[-1]:
        ext = parsed.path.split("/")[-1].split(".")[-1].lower()
        if ext not in include_filetypes:
            return False

    # robots.txt
    if cfg["crawl"]["respect_robots_txt"] and not rp.can_fetch(cfg["crawl"]["user_agent"], url):
        return False

    return True

def extract_links(html: str, base_url: str) -> List[str]:
    """Extract and normalize anchor links from an HTML page."""
    soup = BeautifulSoup(html, "lxml")
    hrefs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        abs_url = urljoin(base_url, href)
        hrefs.append(normalize_url(abs_url))
    print(f"[crawler] extract_links: found={len(hrefs)} base={base_url}")
    return hrefs

def fetch(client: httpx.Client, url: str, timeout: int) -> Tuple[int, str]:
    """Fetch a URL and return (status_code, text)."""
    try:
        r = client.get(url, timeout=timeout, follow_redirects=True)
        print(f"[crawler] fetch: {url} -> status={r.status_code} bytes={len(r.content) if r.content else 0}")
        return r.status_code, r.text if r.status_code == 200 else ""
    except Exception as e:
        print(f"[crawler] fetch error: {url} -> {e}")
        return 0, ""

def crawl() -> None:
    cfg = load_config()
    base_url = cfg["crawl"]["base_url"]
    ua = cfg["crawl"]["user_agent"]
    max_pages = int(cfg["crawl"]["max_pages"])
    timeout_sec = int(cfg["crawl"]["request_timeout_sec"])
    sleep_s = float(cfg["crawl"]["sleep_between_requests_sec"])

    raw_html_dir = cfg["data"]["raw_html_dir"]
    url_manifest_path = cfg["data"]["url_manifest"]
    ensure_dir(raw_html_dir)
    ensure_dir(url_manifest_path.rsplit("/", 1)[0])

    print(f"[crawler] Start crawl base={base_url} max_pages={max_pages} timeout={timeout_sec}s sleep={sleep_s}s")

    # robots.txt
    rp = robotparser.RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    try:
        rp.set_url(robots_url)
        rp.read()
        print(f"[crawler] robots.txt loaded: {robots_url}")
    except Exception as e:
        print(f"[crawler] robots.txt load failed: {robots_url} -> {e}")

    start = normalize_url(base_url)
    queue: List[str] = [start]
    seen: Set[str] = set()
    manifest: List[Dict[str, str]] = []

    headers = {"User-Agent": ua, "Accept": "text/html,application/xhtml+xml"}

    with httpx.Client(headers=headers, http2=True) as client:
        while queue and len(seen) < max_pages:
            url = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)

            if not is_allowed(url, cfg, rp):
                # print(f"[crawler] skip (not allowed): {url}")
                continue

            print(f"[crawler] Visiting ({len(seen)}/{max_pages}): {url}")

            status, html = fetch(client, url, timeout_sec)
            if status != 200 or not html:
                print(f"[crawler] skip (bad status or empty): {url}")
                continue

            # store raw html
            fname = url_to_filename(url, "html")
            fpath = f"{raw_html_dir}/{fname}"
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[crawler] saved: {fpath}")

            manifest.append({"url": url, "path": fpath})

            # enqueue new links
            links = extract_links(html, url)
            added = 0
            for link in links:
                if link not in seen:
                    queue.append(link)
                    added += 1
            print(f"[crawler] queued +{added}, queue_size={len(queue)}")

            # polite crawling
            if sleep_s > 0:
                print(f"[crawler] sleep {sleep_s}s")
                time.sleep(sleep_s)

    # Save manifest
    save_json(manifest, url_manifest_path)
    print(f"[crawler] DONE. pages_saved={len(manifest)} manifest={url_manifest_path}")

if __name__ == "__main__":
    crawl()
