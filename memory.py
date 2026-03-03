"""
Memory System for 4o with Memory

Components:
1. estimate_tokens() - Accurate token counting via tiktoken
2. MnemoMemoryManager - App-facing interface for Streamlit session state

MnemoClient has been extracted to mnemo_client.py for reuse across
context_engine.py, metadata_loops.py, and session_store.py.
"""

import tiktoken
from typing import Dict
from mnemo_client import MnemoClient


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Accurately estimate token count using GPT-4o encoding."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


# =============================================================================
# MNEMO MEMORY MANAGER (App Interface)
# =============================================================================

class MnemoMemoryManager:
    """
    App-facing memory manager.

    Provides the interface expected by the Streamlit app:
    - Holds cross-session toggle state
    - Wraps MnemoClient for direct access
    """

    def __init__(self,
                 openrouter_key: str = None,
                 hf_key: str = None,
                 openai_key: str = None,
                 user_id: str = "default",
                 cross_session_enabled: bool = True):

        self.user_id = user_id
        self.cross_session_enabled = cross_session_enabled
        self.mnemo = MnemoClient(
            base_url="https://athelaperk-mnemo-mcp.hf.space",
            token=hf_key
        )

    def toggle_cross_session(self, enabled: bool):
        """Toggle cross-session memory on/off"""
        self.cross_session_enabled = enabled

    def get_stats(self) -> Dict:
        """Get memory statistics for display"""
        mnemo_stats = self.mnemo.get_stats()
        return {
            "mnemo": {
                "available": mnemo_stats.get("available", False),
                "stats": mnemo_stats
            },
            "cross_session_enabled": self.cross_session_enabled
        }
