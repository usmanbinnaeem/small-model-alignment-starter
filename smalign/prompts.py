"""The single prompt format shared by SFT, DPO, and generation.

ChatML, the format Qwen2.5 tokenizers ship with, built as a plain string so the
exact text is fixed and does not depend on a tokenizer's chat template. The
response that follows is terminated by the tokenizer's EOS token, which the
batch builders append.
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a helpful assistant."


def format_prompt(instruction: str, system: str | None = SYSTEM_PROMPT) -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)
