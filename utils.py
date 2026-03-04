"""
Shared utilities for 4o with Memory.

Consolidates helpers that were duplicated across memory.py and smart_memory.py.
"""

import tiktoken


def estimate_tokens(text: str) -> int:
    """Accurately estimate token count using GPT-4o encoding (o200k_base)."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4
