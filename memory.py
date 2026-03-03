"""
Memory System for 4o with Memory

Components:
1. estimate_tokens() - Accurate token counting via tiktoken
2. MnemoClient - REST client for Mnemo v4 cloud memory
3. MnemoMemoryManager - App-facing interface for Streamlit session state
"""

import tiktoken
from typing import List, Dict, Optional, Tuple
import requests


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Accurately estimate token count using GPT-4o's encoding."""
    if not text:
        return 0
    try:
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


# =============================================================================
# MNEMO v4 REST CLIENT
# =============================================================================

class MnemoClient:
    """
    Client for Mnemo v4 Server - Advanced memory system with:
    - Three-tiered memory hierarchy (semantic, episodic, working)
    - Neural link pathways (8 types)
    - Memory utility predictor
    - Self-tuning parameters

    Hosted at: https://athelaperk-mnemo-mcp.hf.space
    """

    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = (base_url or "https://athelaperk-mnemo-mcp.hf.space").rstrip('/')
        self.token = token
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.available = self._check_availability()
        self.last_error = None

    def _check_availability(self) -> bool:
        """Check if Mnemo server is available"""
        try:
            response = requests.get(f"{self.base_url}/", headers=self.headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("name", "").startswith("Mnemo")
            return False
        except Exception as e:
            self.last_error = str(e)
            return False

    def add(self, content: str, metadata: dict = None, namespace: str = "default") -> Optional[str]:
        """Add a memory to Mnemo"""
        if not self.available:
            return None
        try:
            payload = {"content": content, "namespace": namespace}
            if metadata:
                payload["metadata"] = metadata
            response = requests.post(
                f"{self.base_url}/add",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("memory_id")
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def search(self, query: str, limit: int = 10, namespace: str = "default") -> List[Dict]:
        """Search memories"""
        if not self.available:
            return []
        try:
            response = requests.post(
                f"{self.base_url}/search",
                headers=self.headers,
                json={"query": query, "limit": limit, "namespace": namespace},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            return []
        except Exception:
            return []

    def get_context(self, query: str, namespace: str = "default") -> str:
        """Get formatted context for LLM injection"""
        if not self.available:
            return ""
        try:
            response = requests.post(
                f"{self.base_url}/get_context",
                headers=self.headers,
                json={"query": query, "namespace": namespace},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("context", "")
            return ""
        except Exception:
            return ""

    def should_inject(self, query: str) -> Tuple[bool, str]:
        """Check if memory should be injected for this query"""
        if not self.available:
            return False, "mnemo_unavailable"
        try:
            response = requests.post(
                f"{self.base_url}/should_inject",
                headers=self.headers,
                json={"query": query},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("should_inject", False), data.get("reason", "unknown")
            return False, "api_error"
        except Exception:
            return False, "exception"

    def list_memories(self, namespace: str = "default") -> List[Dict]:
        """List all memories"""
        if not self.available:
            return []
        try:
            response = requests.get(
                f"{self.base_url}/list",
                headers=self.headers,
                params={"namespace": namespace},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("memories", [])
            return []
        except Exception:
            return []

    def get_stats(self) -> Dict:
        """Get Mnemo statistics"""
        if not self.available:
            return {"available": False, "error": self.last_error}
        try:
            response = requests.get(
                f"{self.base_url}/stats",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                stats = data.get("stats", {})
                stats["available"] = True
                return stats
            return {"available": False}
        except Exception:
            return {"available": False}

    def clear(self, namespace: str = "default") -> bool:
        """Clear all memories in namespace"""
        if not self.available:
            return False
        try:
            response = requests.post(
                f"{self.base_url}/clear",
                headers=self.headers,
                json={"namespace": namespace, "confirm": True},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False


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
