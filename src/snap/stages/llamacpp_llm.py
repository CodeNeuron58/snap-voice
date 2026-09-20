"""LLM via llama-cpp-python (GGUF) — the CPU spike baseline, with token streaming.

NPU path: swap this class for a GenieX/QAIRT-backed one with the same
stream_reply() signature. Nothing else in the pipeline may care.
"""

from collections.abc import Callable

from snap import config
from snap.timing import TIMINGS

Message = dict  # {"role": "...", "content": "..."}


class LlamaCppLLM:
    def __init__(self) -> None:
        from llama_cpp import Llama  # heavy import: keep lazy

        self._llm = Llama(
            model_path=str(config.LLM_GGUF),
            n_ctx=config.LLM_N_CTX,
            n_threads=config.LLM_N_THREADS,
            verbose=False,
        )

    def stream_reply(
        self, messages: list[Message], on_token: Callable[[str], None]
    ) -> str:
        chunks: list[str] = []
        with TIMINGS.stage("llm_first_token"):
            stream = self._llm.create_chat_completion(
                messages=messages, max_tokens=config.LLM_MAX_TOKENS, stream=True
            )
            first = next(iter(stream), None)
            if first:
                delta = first["choices"][0]["delta"].get("content") or ""
                chunks.append(delta)
                on_token(delta)
        with TIMINGS.stage("llm_rest"):
            for part in stream:
                delta = part["choices"][0]["delta"].get("content") or ""
                if delta:
                    chunks.append(delta)
                    on_token(delta)
        return "".join(chunks)

    def warm_up(self) -> None:
        """One 1-token completion so chat-template + compute-graph caches are hot."""
        for _part in self._llm.create_chat_completion(
            messages=[{"role": "user", "content": "hi"}], max_tokens=1
        ):
            break
