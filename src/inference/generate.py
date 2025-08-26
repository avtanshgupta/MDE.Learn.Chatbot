import logging
import os
from typing import Dict, Iterator, List, Optional, Tuple

from mlx_lm import generate as mlx_generate

# MLX LM
from mlx_lm import load as mlx_load

from src.inference.retriever import Retriever, format_context
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """
    Build chat messages in OpenAI-style format, which Qwen2.5 supports via chat template.
    """
    logger.debug("Building chat messages")
    messages: List[Dict[str, str]] = []
    if system_prompt:
        logger.debug("System prompt length=%d", len(system_prompt))
        messages.append({"role": "system", "content": system_prompt})
    else:
        logger.debug("No system prompt provided")
    up_preview = user_prompt.replace("\n", " ")[:120]
    logger.debug("User prompt preview: %s%s", up_preview, "..." if len(user_prompt) > 120 else "")
    messages.append({"role": "user", "content": user_prompt})
    logger.debug("Messages built count=%d", len(messages))
    return messages


def build_chat_messages(system_prompt: str, history: Optional[List[Dict[str, str]]], latest_user: str) -> List[Dict[str, str]]:
    """
    Build chat messages including prior history plus the latest user turn.
    Only 'user' and 'assistant' roles are kept from history.
    """
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # Keep a reasonable tail of history to preserve context while bounding length
    if history:
        tail = history[-16:]
        for m in tail:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": latest_user})
    return messages

class ModelRunner:
    def __init__(self, mode: Optional[str] = None) -> None:
        """
        mode: "rag" | "ft" | "rag_ft"
        """
        logger.info("Initializing ModelRunner with mode=%s", mode)
        cfg = load_config()
        self.cfg = cfg
        self.mode = mode or cfg["app"]["mode"]
        logger.info("Effective mode=%s", self.mode)

        self.model_id = cfg["model"]["base_id"]
        self.adapter_dir = cfg["finetune"]["out_dir"]
        self.merged_dir = cfg["merge"]["out_dir"]
        logger.info("Model config: base_id=%s, adapter_dir=%s, merged_dir=%s", self.model_id, self.adapter_dir, self.merged_dir)

        self.max_tokens = int(cfg["infer"]["max_tokens"])
        self.temperature = float(cfg["infer"]["temperature"])
        self.top_p = float(cfg["infer"]["top_p"])
        self.use_streaming = bool(cfg["infer"]["use_streaming"])
        self.retrieval_top_k = int(cfg["infer"]["retrieval_top_k"])
        logger.debug("Inference params: max_tokens=%d, temperature=%.4f, top_p=%.4f, use_streaming=%s", self.max_tokens, self.temperature, self.top_p, self.use_streaming)
        logger.debug("Retrieval top_k=%d", self.retrieval_top_k)

        self.system_prompt = ""
        # load system prompt file
        try:
            spp = cfg["model"]["system_prompt_path"]
            if spp and os.path.exists(spp):
                with open(spp, "r", encoding="utf-8") as f:
                    self.system_prompt = f.read().strip()
                logger.debug("Loaded system prompt from %s length=%d", spp, len(self.system_prompt))
            else:
                logger.debug("No system prompt file found at %s", spp)
        except Exception as e:
            logger.warning("Failed to load system prompt: %s", e)
            self.system_prompt = ""

        # RAG retriever (lazy)
        self._retriever: Optional[Retriever] = None
        logger.debug("Retriever will be created lazily on first RAG call")

        # Load model/tokenizer
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        """
        Load base model with optional LoRA adapter or merged weights using mlx_lm.
        Preference:
          - For ft/rag_ft: try adapter_dir if exists, else try merged_dir, else base.
          - For rag: load base only.
        """
        adapter: Optional[str] = None
        model_path_or_id = self.model_id

        if self.mode in ("ft", "rag_ft"):
            # Prefer loading adapter live if present
            if os.path.isdir(self.adapter_dir) and os.listdir(self.adapter_dir):
                adapter = self.adapter_dir
            elif os.path.isdir(self.merged_dir) and os.listdir(self.merged_dir):
                # Use merged as the base
                model_path_or_id = self.merged_dir
            # else fall back to base_id

        # For rag only, use base model (no adapter)
        logger.info("Loading model: %s (adapter=%s)", model_path_or_id, adapter)
        try:
            if adapter:
                try:
                    model, tokenizer = mlx_load(model_path_or_id, adapter=adapter, lazy=True)
                    logger.info("Model and tokenizer loaded with adapter")
                except TypeError:
                    logger.debug("'adapter' kw not supported in this mlx_lm version; attempting fallback")
                    # Prefer merged weights if available
                    if os.path.isdir(self.merged_dir) and os.listdir(self.merged_dir):
                        model_path_or_id = self.merged_dir
                        logger.info("Falling back to merged weights at %s", model_path_or_id)
                    else:
                        logger.info("Merged weights unavailable; falling back to base model")
                    model, tokenizer = mlx_load(model_path_or_id, lazy=True)
                    logger.info("Model and tokenizer loaded without adapter")
            else:
                model, tokenizer = mlx_load(model_path_or_id, lazy=True)
                logger.info("Model and tokenizer loaded")
        except Exception as e:
            logger.exception("Model load failed: %s", e)
            raise
        return model, tokenizer

    def _build_rag_prompt(self, question: str) -> Tuple[str, List[Dict[str, any]]]:
        """
        Retrieve context and build a grounded prompt.
        """
        if self._retriever is None:
            logger.debug("Creating Retriever for RAG")
            self._retriever = Retriever()
        q_preview = question.replace("\n", " ")[:120]
        logger.debug("Retrieving context for question: %s%s (top_k=%d)", q_preview, "..." if len(question) > 120 else "", self.retrieval_top_k)
        docs = self._retriever.query(question, self.retrieval_top_k)
        context, sources = format_context(docs)
        logger.debug("Retrieved %d doc(s); context length=%d; sources=%d", len(docs), len(context), len(sources))

        instruction = (
            f"Use the following retrieved MDE documentation excerpts to answer the question.\nCite sources by index when helpful.\n\nContext:\n{context}\n\nQuestion: {question}\n"
        )
        return instruction, sources

    def _compose_user_prompt(self, question: str) -> Tuple[str, List[Dict[str, any]]]:
        """
        Compose the final user prompt string depending on mode.
        """
        if self.mode == "rag":
            logger.debug("Mode=rag; building RAG prompt")
            return self._build_rag_prompt(question)
        elif self.mode == "ft":
            # FT only, no retrieval
            logger.debug("Mode=ft; using raw question without retrieval")
            return question, []
        else:
            # rag_ft
            logger.debug("Mode=rag_ft; building RAG prompt for fine-tuned model")
            return self._build_rag_prompt(question)

    def generate(self, question: str, stream: bool = True, history: Optional[List[Dict[str, str]]] = None) -> Tuple[Iterator[str], List[Dict[str, any]]]:
        """
        Return an iterator over generated text chunks and the list of sources (for RAG).
        Includes prior chat history for conversational context.
        """
        q_preview = question.replace("\n", " ")[:120]
        logger.info("Generate called. Question preview: %s%s", q_preview, "..." if len(question) > 120 else "")
        user_prompt, sources = self._compose_user_prompt(question)
        messages = build_chat_messages(self.system_prompt, history, user_prompt)
        logger.debug("Messages prepared count=%d", len(messages))

        # Prefer using chat template for Qwen2.5
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.debug("Chat template applied. Prompt length=%d", len(prompt))

        # Build kwargs with compatibility for different mlx_lm versions.
        # Prefer 'temperature'; fall back to 'temp' if needed.
        base_kwargs = {
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        logger.debug("Base generation kwargs (no temperature): %s", base_kwargs)

        def call_generate(streaming: bool):
            import re
            # Start with common kwargs; we'll dynamically drop unsupported ones.
            active_kwargs = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            if streaming:
                active_kwargs["stream"] = True

            while True:
                try:
                    result = mlx_generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        **active_kwargs,
                    )
                    is_stream = bool(active_kwargs.get("stream", False))
                    return result, is_stream
                except TypeError as e:
                    msg = str(e)
                    logger.debug("mlx_lm.generate TypeError: %s; kwargs=%s", msg, active_kwargs)
                    # Remove unexpected kwargs as reported by error message
                    m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                    if m:
                        bad = m.group(1)
                        if bad in active_kwargs:
                            logger.debug("Removing unsupported kwarg: %s", bad)
                            active_kwargs.pop(bad)
                            continue
                    # Try max_new_tokens in place of max_tokens
                    if "max_tokens" in active_kwargs:
                        val = active_kwargs.pop("max_tokens")
                        active_kwargs["max_new_tokens"] = val
                        logger.debug("Switched to 'max_new_tokens'")
                        continue
                    # If 'stream' likely unsupported, drop it and retry non-streaming
                    if "stream" in active_kwargs:
                        logger.debug("Dropping 'stream' and retrying non-streaming")
                        active_kwargs.pop("stream", None)
                        continue
                    # If nothing else helps, drop optional sampling args
                    dropped = False
                    for k in ("temperature", "top_p"):
                        if k in active_kwargs:
                            logger.debug("Dropping optional kwarg: %s", k)
                            active_kwargs.pop(k)
                            dropped = True
                    if dropped:
                        continue
                    logger.exception("mlx_lm.generate failed and no further fallbacks available")
                    raise

        if stream and self.use_streaming:
            logger.info("Starting streaming generation...")

            def streamer():
                res, is_stream = call_generate(streaming=True)
                if is_stream and not isinstance(res, str):
                    for token in res:
                        yield token
                else:
                    yield res

            return streamer(), sources
        else:
            logger.info("Starting non-streaming generation...")
            res, is_stream = call_generate(streaming=False)
            # Ensure we produce a single full text string
            if is_stream and not isinstance(res, str):
                text = "".join(list(res))
            else:
                text = res
            logger.debug("Non-streaming generation complete. Text length=%d", len(text))

            def single():
                yield text

            return single(), sources
