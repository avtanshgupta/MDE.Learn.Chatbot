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

    def generate(self, question: str, stream: bool = True) -> Tuple[Iterator[str], List[Dict[str, any]]]:
        """
        Return an iterator over generated text chunks and the list of sources (for RAG).
        """
        q_preview = question.replace("\n", " ")[:120]
        logger.info("Generate called. Question preview: %s%s", q_preview, "..." if len(question) > 120 else "")
        user_prompt, sources = self._compose_user_prompt(question)
        messages = build_messages(self.system_prompt, user_prompt)
        logger.debug("Messages prepared count=%d", len(messages))

        # Prefer using chat template for Qwen2.5
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        logger.debug("Chat template applied. Prompt length=%d", len(prompt))

        gen_kwargs = {
            "temp": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        logger.debug("Generation kwargs: %s", gen_kwargs)

        if stream and self.use_streaming:
            logger.info("Starting streaming generation...")

            def streamer():
                try:
                    for token in mlx_generate(self.model, self.tokenizer, prompt=prompt, stream=True, **gen_kwargs):
                        yield token
                except TypeError:
                    logger.debug("'stream' kw not supported by mlx_lm.generate; falling back to non-streaming")
                    text = mlx_generate(self.model, self.tokenizer, prompt=prompt, **gen_kwargs)
                    yield text

            return streamer(), sources
        else:
            logger.info("Starting non-streaming generation...")
            text = mlx_generate(self.model, self.tokenizer, prompt=prompt, **gen_kwargs)
            logger.debug("Non-streaming generation complete. Text length=%d", len(text))

            def single():
                yield text

            return single(), sources
