"""
Mnemo v4 REST Client

Standalone client for the Mnemo v4 memory server.
Uses requests.Session for connection pooling (TCP reuse across calls).

Hosted at: https://athelaperk-mnemo-mcp.hf.space

Extracted from memory.py so that context_engine.py, metadata_loops.py,
and session_store.py can all import a single client instead of each
building their own inline requests calls.
"""

import requests
from typing import List, Dict, Optional, Tuple

DEFAULT_MNEMO_URL = "https://athelaperk-mnemo-mcp.hf.space"


class MnemoClient:
    """
    Client for Mnemo v4 Server -- three-tiered memory hierarchy,
    neural link pathways, memory utility predictor, self-tuning params.

    Uses requests.Session for connection pooling so that repeated
    calls to the same host reuse the underlying TCP connection.
    """

    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = (base_url or DEFAULT_MNEMO_URL).rstrip("/")
        self.token = token

        # Shared session -- connection pooling + default headers
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

        self.available = self._check_availability()
        self.last_error = None

    def _check_availability(self) -> bool:
        """Check if Mnemo server is reachable and identifies itself."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                return response.json().get("name", "").startswith("Mnemo")
            return False
        except Exception as e:
            self.last_error = str(e)
            return False

    def add(self, content: str, metadata: dict = None, namespace: str = "default") -> Optional[str]:
        """Store a memory. Returns memory_id or None."""
        if not self.available:
            return None
        try:
            payload = {"content": content, "namespace": namespace}
            if metadata:
                payload["metadata"] = metadata
            response = self.session.post(f"{self.base_url}/add", json=payload, timeout=10)
            if response.status_code == 200:
                return response.json().get("memory_id")
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def get(self, memory_id: str) -> Optional[Dict]:
        """Fetch a single memory by ID. Returns full memory dict or None."""
        if not self.available:
            return None
        try:
            response = self.session.get(f"{self.base_url}/get/{memory_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def get_content(self, memory_id: str) -> Optional[str]:
        """Fetch just the content string for a memory. Convenience wrapper."""
        data = self.get(memory_id)
        if data:
            return data.get("content", "")
        return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if not self.available:
            return False
        try:
            response = self.session.delete(f"{self.base_url}/delete/{memory_id}", timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.last_error = str(e)
            return False

    def search(self, query: str, limit: int = 10, namespace: str = "default") -> List[Dict]:
        """Search memories by semantic similarity."""
        if not self.available:
            return []
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json={"query": query, "limit": limit, "namespace": namespace},
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except Exception:
            return []

    def get_context(self, query: str, namespace: str = "default") -> str:
        """Get formatted context string for LLM injection."""
        if not self.available:
            return ""
        try:
            response = self.session.post(
                f"{self.base_url}/get_context",
                json={"query": query, "namespace": namespace},
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("context", "")
            return ""
        except Exception:
            return ""

    def should_inject(self, query: str) -> Tuple[bool, str]:
        """Ask Mnemo whether memory should be injected for this query."""
        if not self.available:
            return False, "mnemo_unavailable"
        try:
            response = self.session.post(
                f"{self.base_url}/should_inject",
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
        """List all memories in a namespace."""
        if not self.available:
            return []
        try:
            response = self.session.get(
                f"{self.base_url}/list",
                params={"namespace": namespace},
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("memories", [])
            return []
        except Exception:
            return []

    def clear(self, namespace: str = "default") -> bool:
        """Clear all memories in a namespace."""
        if not self.available:
            return False
        try:
            response = self.session.post(
                f"{self.base_url}/clear",
                json={"namespace": namespace, "confirm": True},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_stats(self) -> Dict:
        """Get Mnemo server statistics."""
        if not self.available:
            return {"available": False, "error": self.last_error}
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json().get("stats", {})
                stats["available"] = True
                return stats
            return {"available": False}
        except Exception:
            return {"available": False}
