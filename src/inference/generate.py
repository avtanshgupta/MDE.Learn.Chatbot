import os
from typing import Dict, List, Iterator, Optional, Tuple

from src.utils.config import load_config
from src.inference.retriever import Retriever, format_context

# MLX LM
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """
    Build chat messages in OpenAI-style format, which Qwen2.5 supports via chat template.
    """
    print("[inference.generate] Building chat messages")
    messages: List[Dict[str, str]] = []
    if system_prompt:
        print(f"[inference.generate] System prompt length={len(system_prompt)}")
        messages.append({"role": "system", "content": system_prompt})
    else:
        print("[inference.generate] No system prompt provided")
    up_preview = user_prompt.replace("\n", " ")[:120]
    print(f"[inference.generate] User prompt preview: {up_preview}{'...' if len(user_prompt) > 120 else ''}")
    messages.append({"role": "user", "content": user_prompt})
    print(f"[inference.generate] Messages built count={len(messages)}")
    return messages


class ModelRunner:
    def __init__(self, mode: Optional[str] = None) -> None:
        """
        mode: "rag" | "ft" | "rag_ft"
        """
        print(f"[inference.generate] Initializing ModelRunner with mode={mode}")
        cfg = load_config()
        self.cfg = cfg
        self.mode = mode or cfg["app"]["mode"]
        print(f"[inference.generate] Effective mode={self.mode}")

        self.model_id = cfg["model"]["base_id"]
        self.adapter_dir = cfg["finetune"]["out_dir"]
        self.merged_dir = cfg["merge"]["out_dir"]
        print(f"[inference.generate] Model config: base_id={self.model_id}, adapter_dir={self.adapter_dir}, merged_dir={self.merged_dir}")

        self.max_tokens = int(cfg["infer"]["max_tokens"])
        self.temperature = float(cfg["infer"]["temperature"])
        self.top_p = float(cfg["infer"]["top_p"])
        self.use_streaming = bool(cfg["infer"]["use_streaming"])
        self.retrieval_top_k = int(cfg["infer"]["retrieval_top_k"])
        print(f"[inference.generate] Inference params: max_tokens={self.max_tokens}, temperature={self.temperature}, top_p={self.top_p}, use_streaming={self.use_streaming}")
        print(f"[inference.generate] Retrieval top_k={self.retrieval_top_k}")

        self.system_prompt = ""
        # load system prompt file
        try:
            spp = cfg["model"]["system_prompt_path"]
            if spp and os.path.exists(spp):
                with open(spp, "r", encoding="utf-8") as f:
                    self.system_prompt = f.read().strip()
                print(f"[inference.generate] Loaded system prompt from {spp} length={len(self.system_prompt)}")
            else:
                print(f"[inference.generate] No system prompt file found at {spp}")
        except Exception as e:
            print(f"[inference.generate] Failed to load system prompt: {e}")
            self.system_prompt = ""

        # RAG retriever (lazy)
        self._retriever: Optional[Retriever] = None
        print("[inference.generate] Retriever will be created lazily on first RAG call")

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
        print(f"[inference.generate] Loading model: {model_path_or_id} (adapter={adapter})")
        try:
            if adapter:
                try:
                    model, tokenizer = mlx_load(model_path_or_id, adapter=adapter, lazy=True)
                    print("[inference.generate] Model and tokenizer loaded with adapter")
                except TypeError:
                    print("[inference.generate] 'adapter' kw not supported in this mlx_lm version; attempting fallback")
                    # Prefer merged weights if available
                    if os.path.isdir(self.merged_dir) and os.listdir(self.merged_dir):
                        model_path_or_id = self.merged_dir
                        print(f"[inference.generate] Falling back to merged weights at {model_path_or_id}")
                    else:
                        print("[inference.generate] Merged weights unavailable; falling back to base model")
                    model, tokenizer = mlx_load(model_path_or_id, lazy=True)
                    print("[inference.generate] Model and tokenizer loaded without adapter")
            else:
                model, tokenizer = mlx_load(model_path_or_id, lazy=True)
                print("[inference.generate] Model and tokenizer loaded")
        except Exception as e:
            print(f"[inference.generate] Model load failed: {e}")
            raise
        return model, tokenizer

    def _build_rag_prompt(self, question: str) -> Tuple[str, List[Dict[str, any]]]:
        """
        Retrieve context and build a grounded prompt.
        """
        if self._retriever is None:
            print("[inference.generate] Creating Retriever for RAG")
            self._retriever = Retriever()
        q_preview = question.replace("\n", " ")[:120]
        print(f"[inference.generate] Retrieving context for question: {q_preview}{'...' if len(question) > 120 else ''} (top_k={self.retrieval_top_k})")
        docs = self._retriever.query(question, self.retrieval_top_k)
        context, sources = format_context(docs)
        print(f"[inference.generate] Retrieved {len(docs)} doc(s); context length={len(context)}; sources={len(sources)}")

        instruction = (
            "Use the following retrieved MDE documentation excerpts to answer the question.\n"
            "Cite sources by index when helpful.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
        )
        return instruction, sources

    def _compose_user_prompt(self, question: str) -> Tuple[str, List[Dict[str, any]]]:
        """
        Compose the final user prompt string depending on mode.
        """
        if self.mode == "rag":
            print("[inference.generate] Mode=rag; building RAG prompt")
            return self._build_rag_prompt(question)
        elif self.mode == "ft":
            # FT only, no retrieval
            print("[inference.generate] Mode=ft; using raw question without retrieval")
            return question, []
        else:
            # rag_ft
            print("[inference.generate] Mode=rag_ft; building RAG prompt for fine-tuned model")
            return self._build_rag_prompt(question)

    def generate(self, question: str, stream: bool = True) -> Tuple[Iterator[str], List[Dict[str, any]]]:
        """
        Return an iterator over generated text chunks and the list of sources (for RAG).
        """
        q_preview = question.replace("\n", " ")[:120]
        print(f"[inference.generate] Generate called. Question preview: {q_preview}{'...' if len(question) > 120 else ''}")
        user_prompt, sources = self._compose_user_prompt(question)
        messages = build_messages(self.system_prompt, user_prompt)
        print(f"[inference.generate] Messages prepared count={len(messages)}")

        # Prefer using chat template for Qwen2.5
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"[inference.generate] Chat template applied. Prompt length={len(prompt)}")

        gen_kwargs = {
            "temp": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        print(f"[inference.generate] Generation kwargs: {gen_kwargs}")

        if stream and self.use_streaming:
            print("[inference.generate] Starting streaming generation...")
            def streamer():
                try:
                    for token in mlx_generate(self.model, self.tokenizer, prompt=prompt, stream=True, **gen_kwargs):
                        yield token
                except TypeError:
                    print("[inference.generate] 'stream' kw not supported by mlx_lm.generate; falling back to non-streaming")
                    text = mlx_generate(self.model, self.tokenizer, prompt=prompt, **gen_kwargs)
                    yield text
            return streamer(), sources
        else:
            print("[inference.generate] Starting non-streaming generation...")
            text = mlx_generate(self.model, self.tokenizer, prompt=prompt, **gen_kwargs)
            print(f"[inference.generate] Non-streaming generation complete. Text length={len(text)}")
            def single():
                yield text
            return single(), sources
