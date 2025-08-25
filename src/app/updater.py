import os
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Tuple, Optional

# Utilities for last-run tracking

def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _read_last_run(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data.get("last_run", None))
    except Exception:
        return None

def _write_last_run(path: str, ts: float) -> None:
    try:
        _ensure_parent(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"last_run": ts}, f)
    except Exception as e:
        print(f"[app.updater] Failed to write last_run file: {e}")

def _now() -> float:
    return time.time()

# Public helpers

def get_last_run_info(cfg) -> Optional[float]:
    return _read_last_run(cfg["update"]["last_run_file"])

def seconds_until_next(cfg) -> float:
    last = get_last_run_info(cfg)
    interval_h = float(cfg["update"]["min_interval_hours"])
    if not last:
        return 0.0
    delta = (last + interval_h * 3600.0) - _now()
    return max(0.0, delta)

# Update execution

_RUNNING_LOCK = threading.Lock()
_RUNNING_FLAG = False

def _set_running(flag: bool) -> None:
    global _RUNNING_FLAG
    with _RUNNING_LOCK:
        _RUNNING_FLAG = flag

def _is_running() -> bool:
    with _RUNNING_LOCK:
        return _RUNNING_FLAG

def can_run_update(cfg) -> Tuple[bool, float]:
    """
    Returns (can_run, seconds_remaining)
    """
    if _is_running():
        return False, seconds_until_next(cfg)
    rem = seconds_until_next(cfg)
    return rem <= 0.0, rem

def is_update_running() -> bool:
    """Return True if an update pipeline is currently running."""
    return _is_running()

def _run_update_pipeline(cfg) -> None:
    print("[app.updater] Update pipeline start")
    # Lazy imports to avoid heavy startup
    from src.crawler.crawler import crawl
    from src.processing.process import process as process_docs
    from src.indexing.build_index import build_index
    from src.training.prepare_dataset import main as prepare_dataset
    from src.training.finetune_mlx import finetune

    try:
        print("[app.updater] Step: crawl")
        crawl()
    except Exception as e:
        print(f"[app.updater] crawl failed: {e}")

    try:
        print("[app.updater] Step: process")
        process_docs()
    except Exception as e:
        print(f"[app.updater] process failed: {e}")

    try:
        print("[app.updater] Step: index")
        build_index()
    except Exception as e:
        print(f"[app.updater] index failed: {e}")

    try:
        print("[app.updater] Step: prepare_dataset")
        prepare_dataset()
    except Exception as e:
        print(f"[app.updater] prepare_dataset failed: {e}")

    try:
        print("[app.updater] Step: finetune (LoRA)")
        finetune()
    except Exception as e:
        print(f"[app.updater] finetune failed: {e}")

    print("[app.updater] Update pipeline end")

def _run_thread(cfg, on_complete: Optional[Callable[[], None]]) -> None:
    _set_running(True)
    try:
        _run_update_pipeline(cfg)
        _write_last_run(cfg["update"]["last_run_file"], _now())
        if on_complete:
            try:
                on_complete()
            except Exception as e:
                print(f"[app.updater] on_complete failed: {e}")
    finally:
        _set_running(False)

def trigger_update(cfg, on_complete: Optional[Callable[[], None]] = None, force: bool = False) -> Tuple[bool, str]:
    # Always prevent concurrent runs
    if _is_running():
        return False, "Update already running."
    # Respect rate limit unless forced
    if not force:
        ok, rem = can_run_update(cfg)
        if not ok:
            return False, f"Update not allowed yet. Try again in ~{int(rem // 60)} min."
    t = threading.Thread(target=_run_thread, args=(cfg, on_complete), daemon=True)
    t.start()
    return True, "Update accepted and running in background."

def trigger_update_force(cfg, on_complete: Optional[Callable[[], None]] = None) -> Tuple[bool, str]:
    return trigger_update(cfg, on_complete=on_complete, force=True)

# HTTP server for external trigger

class _UpdateHandler(BaseHTTPRequestHandler):
    cfg = None
    on_complete = None

    def do_POST(self):
        if self.path != "/update":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return
        accepted, msg = trigger_update(self.cfg, self.on_complete)
        if accepted:
            self.send_response(202)
        else:
            self.send_response(429)
        self.end_headers()
        self.wfile.write(msg.encode("utf-8"))

    def log_message(self, format, *args):
        # Silence default noisy logging
        print(f"[app.updater] HTTP: " + format % args)

def start_http_server(cfg, on_complete: Optional[Callable[[], None]] = None) -> None:
    host = cfg["update"]["api_host"]
    port = int(cfg["update"]["api_port"])
    def _srv():
        try:
            server = HTTPServer((host, port), _UpdateHandler)
            _UpdateHandler.cfg = cfg
            _UpdateHandler.on_complete = on_complete
            print(f"[app.updater] HTTP update server on http://{host}:{port} (POST /update)")
            server.serve_forever()
        except Exception as e:
            print(f"[app.updater] HTTP server failed: {e}")
    th = threading.Thread(target=_srv, daemon=True)
    th.start()

def start_scheduler(cfg, on_complete: Optional[Callable[[], None]] = None) -> None:
    def _loop():
        print("[app.updater] Daily scheduler started")
        while True:
            # If last_run is missing but fine-tuned weights already exist, set a baseline
            try:
                last = get_last_run_info(cfg)
                adapter_dir = cfg.get("finetune", {}).get("out_dir")
                merged_dir = cfg.get("merge", {}).get("out_dir")
                adapter_present = bool(adapter_dir) and os.path.isdir(adapter_dir) and bool(os.listdir(adapter_dir))
                merged_present = bool(merged_dir) and os.path.isdir(merged_dir) and bool(os.listdir(merged_dir))
                if last is None and (adapter_present or merged_present):
                    print("[app.updater] Baseline last_run set at startup (pre-trained weights present); waiting for next interval")
                    _write_last_run(cfg["update"]["last_run_file"], _now())
            except Exception as e:
                print(f"[app.updater] Scheduler pre-check failed: {e}")

            ok, rem = can_run_update(cfg)
            if ok:
                print("[app.updater] Scheduler triggering update")
                trigger_update(cfg, on_complete)
                # Sleep at least min_interval to avoid immediate re-run
                sleep_sec = max(3600, int(float(cfg["update"]["min_interval_hours"]) * 3600))
            else:
                sleep_sec = int(min(3600, max(60, rem)))
            time.sleep(sleep_sec)
    th = threading.Thread(target=_loop, daemon=True)
    th.start()

def start_background_services(cfg, on_complete: Optional[Callable[[], None]] = None) -> None:
    if not cfg.get("update", {}).get("enabled", False):
        print("[app.updater] Update services disabled via config")
        return
    start_http_server(cfg, on_complete)
    start_scheduler(cfg, on_complete)
