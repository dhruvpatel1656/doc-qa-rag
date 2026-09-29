"""Answer generation: Anthropic's Claude API, with an offline extractive fallback."""
import os
from typing import Dict, List

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered source passages provided. "
    "Cite the passages you used like [1] or [2]. If the sources do not contain "
    "the answer, say you could not find it in the documents."
)


class AnthropicLLM:
    def __init__(self, model: str = None):
        import anthropic  # lazy import

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def generate(self, question: str, passages: List[Dict]) -> str:
        context = "\n\n".join(
            f"[{i}] ({p['source']}) {p['text']}" for i, p in enumerate(passages, start=1)
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


class ExtractiveLLM:
    """No-API fallback: returns the best-matching passage instead of a generated answer."""

    def generate(self, question: str, passages: List[Dict]) -> str:
        if not passages:
            return "I could not find anything relevant in the uploaded documents."
        best = passages[0]
        return f"Most relevant passage (from {best['source']}): {best['text']} [1]"


def get_llm():
    """Use Claude when an API key is configured, otherwise fall back to extraction."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
    return ExtractiveLLM()
