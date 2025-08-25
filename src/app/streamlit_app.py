import os
import sys
import time
from typing import Any, Dict, List

import streamlit as st

# Ensure project root on sys.path when running "streamlit run src/app/streamlit_app.py"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.updater import (
    can_run_update,
    get_last_run_info,
    is_update_running,
    seconds_until_next,
    start_background_services,
    trigger_update,
)
from src.inference.generate import ModelRunner
from src.utils.config import load_config

st.set_page_config(page_title="MDE Learn Chatbot", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

cfg = load_config()
print("[app.app] Config loaded")
adapter_dir = cfg["finetune"]["out_dir"]
merged_dir = cfg["merge"]["out_dir"]
adapter_present = os.path.isdir(adapter_dir) and bool(os.listdir(adapter_dir))
merged_present = os.path.isdir(merged_dir) and bool(os.listdir(merged_dir))
default_mode = cfg["app"]["mode"]
if (adapter_present or merged_present) and default_mode == "rag":
    print("[app.app] Pre-trained weights detected; overriding default mode 'rag' to 'rag_ft'")
    default_mode = "rag_ft"
print(f"[app.app] Default mode={default_mode} (adapter_present={adapter_present}, merged_present={merged_present})")
modes = ["rag", "ft", "rag_ft"]
print(f"[app.app] Supported modes={modes}; base model={cfg['model']['base_id']}")
# Start background update services once per session
if "services_started" not in st.session_state:
    print("[app.app] Starting background update services")
    try:

        def _on_update_complete():
            try:
                print("[app.app] Update complete callback: invalidating runners cache and switching mode to rag_ft")
                st.session_state["runners"] = {}
                st.session_state["mode"] = "rag_ft"
            except Exception as e:
                print(f"[app.app] Update complete callback error: {e}")

        start_background_services(cfg, on_complete=_on_update_complete)
        st.session_state["services_started"] = True
    except Exception as e:
        print(f"[app.app] Failed to start background services: {e}")

with st.sidebar:
    st.title("Settings")
    # Non-interactive details (no controls)
    top_k = int(cfg["infer"]["retrieval_top_k"])
    max_tokens = int(cfg["infer"]["max_tokens"])
    temperature = float(cfg["infer"]["temperature"])
    # Initialize app mode once, based on available fine-tuned weights
    if "mode" not in st.session_state:
        st.session_state["mode"] = default_mode
        print(f"[app.app] Initialized session mode to {st.session_state['mode']}")
    st.markdown(f"**Mode:** {st.session_state['mode']}")
    st.markdown(f"**Retrieval top_k:** {top_k}")
    st.markdown(f"**Max tokens:** {max_tokens}")
    st.markdown(f"**Temperature:** {temperature:.2f}")
    st.markdown("---")
    st.caption("Model: " + cfg["model"]["base_id"])
    print(f"[app.app] Sidebar details: mode={st.session_state['mode']}, top_k={top_k}, max_tokens={max_tokens}, temperature={temperature}")

    st.markdown("---")
    st.subheader("Data Updates")
    if cfg.get("update", {}).get("enabled", False):
        running = False
        try:
            running = is_update_running()
            print(f"[app.app] Update running status: {running}")
        except Exception as e:
            print(f"[app.app] Failed to read running status: {e}")
        # Last run info
        try:
            last_ts = get_last_run_info(cfg)
            if last_ts:
                try:
                    human = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts))
                    st.caption(f"Last update: {human}")
                except Exception:
                    st.caption(f"Last update timestamp: {last_ts}")
        except Exception as e:
            print(f"[app.app] Failed to read last run info: {e}")
        # Next allowed window
        rem = 0
        try:
            rem = seconds_until_next(cfg)
        except Exception as e:
            print(f"[app.app] Failed to compute next update window: {e}")

        # Auto-trigger first fine-tune if no pre-trained weights exist
        try:
            if not (adapter_present or merged_present):
                if not st.session_state.get("auto_update_triggered", False):
                    can, rem2 = can_run_update(cfg)
                    if can:
                        st.caption("No pre-trained weights found. Auto-triggering fine-tune update…")
                        accepted, msg = trigger_update(cfg)
                        print(f"[app.app] Auto update trigger: accepted={accepted}; msg={msg}")
                        st.session_state["auto_update_triggered"] = True
                    else:
                        rem2 = max(0, int(rem2))
                        hh2 = rem2 // 3600
                        mm2 = (rem2 % 3600) // 60
                        st.caption(f"No pre-trained weights found. Next Update: {hh2:02d}:{mm2:02d}")
        except Exception as e:
            print(f"[app.app] Auto-trigger check failed: {e}")

        if running:
            st.warning("Update is currently running…")
            st.button("Run update now", disabled=True)
            st.caption("Status will refresh automatically while the update runs.")
            try:
                time.sleep(2)
                st.experimental_rerun()
            except Exception as _:
                pass
        else:
            rem = max(0, int(rem))
            hh = rem // 3600
            mm = (rem % 3600) // 60
            st.caption(f"Next Update: {hh:02d}:{mm:02d}")
            # Allow manual update even within the interval (will force-run)
            if st.button("Run update now", disabled=False):
                if rem > 0:
                    accepted, msg = trigger_update(cfg, force=True)
                else:
                    accepted, msg = trigger_update(cfg)
                st.info(msg)
                print(f"[app.app] Manual update trigger: accepted={accepted}; msg={msg}")
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
    else:
        st.caption("Updates disabled via config")

st.title("Microsoft Defender for Endpoint Chatbot")
st.caption("Grounded on learn.microsoft.com MDE docs with optional LoRA fine-tuning (MLX) and RAG.")

# Cache a runner per mode to avoid reloading on every rerun
if "runners" not in st.session_state:
    print("[app.app] Initializing runners cache in session_state")
    st.session_state["runners"] = {}
else:
    print(f"[app.app] Runners cache present with modes: {list(st.session_state['runners'].keys())}")


def get_runner(m: str) -> ModelRunner:
    print(f"[app.app] get_runner called for mode={m}")
    runner = st.session_state["runners"].get(m)
    if runner is None:
        print(f"[app.app] Creating new ModelRunner for mode={m}")
        runner = ModelRunner(mode=m)
        st.session_state["runners"][m] = runner
    else:
        print(f"[app.app] Reusing existing ModelRunner for mode={m}")
    # update runtime knobs
    runner.retrieval_top_k = top_k
    runner.max_tokens = max_tokens
    runner.temperature = temperature
    print(f"[app.app] Updated runner knobs: top_k={runner.retrieval_top_k}, max_tokens={runner.max_tokens}, temperature={runner.temperature}")
    return runner


# Pre-warm the selected mode to load pre-trained weights on app load
try:
    if "prewarm_done" not in st.session_state:
        if ('adapter_present' in globals() and adapter_present) or ('merged_present' in globals() and merged_present):
            print(f"[app.app] Pre-warming runner for mode={st.session_state['mode']}")
            _ = get_runner(st.session_state["mode"])
        else:
            print("[app.app] No pre-trained weights detected; skipping pre-warm")
        st.session_state["prewarm_done"] = True
except Exception as e:
    print(f"[app.app] Pre-warm failed: {e}")

# Chat UI
if "history" not in st.session_state:
    print("[app.app] Initializing chat history")
    st.session_state["history"] = []  # list of {"role": "user"/"assistant", "content": str, "sources": [...]}
else:
    print(f"[app.app] Chat history present with {len(st.session_state['history'])} turn(s)")

# Input
with st.form(key="chat-form", clear_on_submit=False):
    user_msg = st.text_area("Ask a question about Microsoft Defender for Endpoint:", height=120, placeholder="e.g. How do I onboard macOS devices to MDE?")
    submitted = st.form_submit_button("Send")

# Render existing history
print(f"[app.app] Rendering history with {len(st.session_state['history'])} turn(s)")
for turn in st.session_state["history"]:
    if turn["role"] == "user":
        st.markdown(f"**You:** {turn['content']}")
    else:
        st.markdown("**Assistant:**")
        st.write(turn["content"])
        sources: List[Dict[str, Any]] = turn.get("sources", [])
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    title = s.get("title") or "Untitled"
                    url = s.get("url") or ""
                    rank = s.get("rank")
                    dist = s.get("distance")
                    st.markdown(f"- [{rank}] [{title}]({url}) (distance={dist})")

# Handle new query
if submitted and user_msg and user_msg.strip():
    msg = user_msg.strip()
    print(f"[app.app] Submit clicked. Mode={st.session_state['mode']}. User message length={len(msg)}")
    st.session_state["history"].append({"role": "user", "content": msg})
    st.markdown(f"**You:** {msg}")

    runner = get_runner(st.session_state["mode"])
    st.markdown("**Assistant:**")
    placeholder = st.empty()
    out_text = ""
    tok_count = 0

    try:
        print("[app.app] Calling runner.generate(stream=True)")
        stream, sources = runner.generate(msg, stream=True)
        for tok in stream:
            out_text += tok
            tok_count += 1
            if tok_count % 50 == 0:
                print(f"[app.app] Streamed {tok_count} token chunks so far")
            placeholder.markdown(out_text)
        print(f"[app.app] Streaming complete. Total token chunks={tok_count}. Output length={len(out_text)}")
        # Finalize assistant message
        st.session_state["history"].append({"role": "assistant", "content": out_text, "sources": sources})
        if sources:
            print(f"[app.app] Rendering {len(sources)} source(s)")
            with st.expander("Sources"):
                for s in sources:
                    title = s.get("title") or "Untitled"
                    url = s.get("url") or ""
                    rank = s.get("rank")
                    dist = s.get("distance")
                    st.markdown(f"- [{rank}] [{title}]({url}) (distance={dist})")
        else:
            print("[app.app] No sources returned")
    except Exception as e:
        print(f"[app.app] Inference failed: {e}")
        st.error(f"Inference failed: {e}")
