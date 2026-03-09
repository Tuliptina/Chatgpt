"""
Mnemo Client - Gradio API Client for Streamlit (v6.0)

Connects to the Mnemo Gradio Space (AthelaPerk/mnemo) which serves as
both the interactive demo AND the MCP/API server.

v6.0 changes:
- REWRITTEN for Gradio HTTP API format (was Docker REST API)
- Base URL: athelaperk-mnemo.hf.space (was mnemo-mcp)
- Call format: POST /gradio_api/call/{endpoint} → event_id → GET result
- All gr.api endpoints return JSON strings (parsed by client)
- No Docker server dependency — single Gradio Space handles everything

Transport format:
  POST /gradio_api/call/{api_name}
    body: {"data": [arg1, arg2, ...]}
    returns: {"event_id": "abc123"}

  GET /gradio_api/call/{api_name}/{event_id}
    returns SSE stream:
      event: complete
      data: ["json_string_result"]

All 27 methods preserved with identical signatures. Callers (app.py,
context_engine.py, session_store.py, etc.) require zero changes.
"""

import json
import time
import requests
from typing import List, Dict, Optional, Tuple, Any


# =============================================================================
# GRADIO API TRANSPORT
# =============================================================================

class GradioTransport:
    """Low-level Gradio HTTP API caller.

    Handles the two-step call/result pattern and SSE response parsing.
    All gr.api endpoints return a JSON string as the first element of
    the data array. This class handles parsing that automatically.
    """

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_base = f"{self.base_url}/gradio_api/call"
        self.timeout = timeout

    def call(self, endpoint: str, *args, parse_json: bool = True) -> Any:
        """Call a Gradio API endpoint and return the result.

        Args:
            endpoint: The api_name (e.g. "add_api", "search_api")
            *args: Positional arguments passed in the data array
            parse_json: If True, parse the first result as JSON string

        Returns:
            Parsed result (dict/list) if parse_json=True, raw string otherwise

        Raises:
            RuntimeError: If the call fails or times out
        """
        # Step 1: POST to initiate call
        url = f"{self.api_base}/{endpoint}"
        resp = requests.post(
            url,
            json={"data": list(args)},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gradio call failed: {resp.status_code} {resp.text[:200]}")

        event_id = resp.json().get("event_id")
        if not event_id:
            raise RuntimeError(f"No event_id in response: {resp.text[:200]}")

        # Step 2: GET the result via SSE
        result_url = f"{url}/{event_id}"
        resp2 = requests.get(result_url, timeout=self.timeout)
        if resp2.status_code != 200:
            raise RuntimeError(f"Gradio result failed: {resp2.status_code}")

        # Parse SSE response
        raw = self._parse_sse(resp2.text)

        if parse_json and raw and isinstance(raw, str):
            return json.loads(raw)
        return raw

    def _parse_sse(self, text: str) -> Any:
        """Parse Gradio SSE response.

        Format:
            event: complete
            data: ["result_string"]

        Returns the first element of the data array.
        """
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]  # Strip "data: " prefix
                try:
                    data_array = json.loads(data_str)
                    if isinstance(data_array, list) and data_array:
                        return data_array[0]
                    return data_array
                except json.JSONDecodeError:
                    return data_str
        return None

    def ping(self) -> bool:
        """Check if the Gradio server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


# =============================================================================
# MNEMO CLIENT
# =============================================================================

class MnemoClient:
    """
    Client for the Mnemo Gradio Space.

    All methods have identical signatures to the v5.3 Docker REST client.
    Callers (app.py, context_engine.py, etc.) require zero changes.

    Usage:
        client = MnemoClient(base_url="https://athelaperk-mnemo.hf.space")
        client.add("Sebastian fears the dark", metadata={"type": "character"})
        results = client.search("Sebastian fears")
    """

    DEFAULT_URL = "https://athelaperk-mnemo.hf.space"

    def __init__(self, base_url: str = None, token: str = None):
        self.base_url = (base_url or self.DEFAULT_URL).rstrip("/")
        self.token = token
        self.transport = GradioTransport(self.base_url)

        # Check availability
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self.transport.ping()
        return self._available

    def _call(self, endpoint: str, *args, parse_json: bool = True) -> Any:
        """Call a Gradio gr.api endpoint with error handling."""
        try:
            return self.transport.call(endpoint, *args, parse_json=parse_json)
        except Exception as e:
            print(f"MnemoClient error ({endpoint}): {e}")
            return None

    # =========================================================================
    # BLOB MEMORY OPERATIONS (v5.3 compat)
    # =========================================================================

    def add(self, content: str, namespace: str = "default",
            metadata: dict = None, priority: float = 1.0) -> Optional[str]:
        """Add a memory. Returns memory ID or None."""
        meta_json = json.dumps(metadata or {})
        result = self._call("add_api", content, namespace, meta_json, str(priority))
        if result and isinstance(result, dict):
            return result.get("id")
        return None

    def search(self, query: str, limit: int = 15,
               namespace: str = None) -> List[Dict]:
        """Search memories. Returns list of result dicts."""
        result = self._call("search_api", query, str(limit), namespace or "")
        if result and isinstance(result, list):
            return result
        return []

    def should_inject(self, query: str, context: str = "",
                      conversation_history: str = "") -> Tuple[bool, str, float]:
        """Check if memory injection is recommended.
        Returns (should_inject, reason, confidence).
        """
        result = self._call("should_inject_api", query, context, conversation_history)
        if result and isinstance(result, dict):
            return (
                result.get("should_inject", False),
                result.get("reason", ""),
                result.get("confidence", 0.0),
            )
        return False, "client error", 0.0

    def get_context(self, query: str, limit: int = 15) -> str:
        """Get injection context for a query. Returns context string."""
        result = self._call("get_context_api", query, str(limit))
        if result and isinstance(result, dict):
            return result.get("context", "")
        return ""

    def get(self, memory_id: str) -> Optional[Dict]:
        """Get a single memory by ID."""
        result = self._call("get_memory", memory_id)
        if result and isinstance(result, dict) and "error" not in result:
            return result
        return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        result = self._call("delete_memory_api", memory_id)
        if result and isinstance(result, dict):
            return result.get("deleted", False)
        return False

    def list_memories(self, namespace: str = None) -> List[Dict]:
        """List all memories."""
        result = self._call("list_memories", namespace or "")
        if result and isinstance(result, list):
            return result
        return []

    def get_stats(self) -> Dict:
        """Get system statistics."""
        result = self._call("stats_api")
        if result and isinstance(result, dict):
            return result
        return {}

    def maintenance(self) -> Dict:
        """Run maintenance (decay/prune)."""
        result = self._call("maintenance_api")
        if result and isinstance(result, dict):
            return result
        return {}

    def clear(self) -> bool:
        """Clear all memories."""
        result = self._call("clear_api", "true")
        if result and isinstance(result, dict):
            return result.get("cleared", False)
        return False

    # =========================================================================
    # CONNECTION POINT OPERATIONS (v6.0)
    # =========================================================================

    def add_point(self, entity: str, point_type: str, value: str = "",
                  connects_to: str = "", reason: str = "",
                  weight: float = 0.5, category: str = "fact",
                  session_id: str = "", source: str = "auto_extract",
                  thread_id: str = "", position: int = -1, 
                  namespace: str = "default") -> Optional[str]:
        """Add a ConnectionPoint. Returns CP ID or None."""
        result = self._call(
            "add_point_api",
            entity, point_type, value, connects_to, reason,
            str(weight), category, session_id, source,
            thread_id, str(position), namespace
        )
        if result and isinstance(result, dict):
            return result.get("id")
        return None

    def add_points_batch(self, points: List[Dict]) -> List[str]:
        """Add multiple ConnectionPoints. Returns list of CP IDs."""
        result = self._call("add_points_batch", json.dumps(points))
        if result and isinstance(result, dict):
            return result.get("created", [])
        return []

    def graph_search(self, query: str, top_k: int = 15, active_sessions: List[str] = None) -> List[Dict]:
        """Graph search for ConnectionPoints. Returns list of CP dicts."""
        sessions_str = json.dumps(active_sessions) if active_sessions else "[]"
        
        # Pass the 3rd argument to the API
        result = self._call("graph_search_api", query, str(top_k), sessions_str)
        if result and isinstance(result, list):
            return result
        return []

    def entity_lookup(self, entity: str) -> List[Dict]:
        """Look up all CPs for an entity. Returns list of CP dicts."""
        result = self._call("entity_lookup_api", entity)
        if result and isinstance(result, list):
            return result
        return []

    def get_point(self, cp_id: str) -> Optional[Dict]:
        """Get a single ConnectionPoint by ID."""
        result = self._call("get_point", cp_id)
        if result and isinstance(result, dict) and "error" not in result:
            return result
        return None

    def delete_point(self, cp_id: str) -> bool:
        """Delete a ConnectionPoint by ID."""
        result = self._call("delete_point_api", cp_id)
        if result and isinstance(result, dict):
            return result.get("deleted", False)
        return False

    def list_points(self, limit: int = 200) -> List[Dict]:
        """List all ConnectionPoints."""
        result = self._call("list_points", str(limit))
        if result and isinstance(result, list):
            return result
        return []

    # =========================================================================
    # THREAD OPERATIONS (v6.0)
    # =========================================================================

    def add_thread(self, thread_id: str, name: str, entity: str = "",
                   thread_type: str = "plot_line",
                   session_id: str = "") -> Optional[str]:
        """Create a narrative thread. Returns thread ID or None."""
        result = self._call(
            "create_thread_api",
            thread_id, name, entity, thread_type, session_id,
        )
        if result and isinstance(result, dict):
            return result.get("id")
        return None

    def advance_thread(self, thread_id: str, new_point_id: str = None,
                       tension_delta: float = 0.0,
                       tone_update: str = None) -> Optional[Dict]:
        """Advance a thread's position.
        Note: current server endpoint only supports position change.
        Tension/tone updates via ConnectionPoints.
        """
        # User's endpoint takes (thread_id, new_position)
        # -1 means advance by one step
        result = self._call("advance_thread", thread_id, "-1")
        if result and isinstance(result, dict) and "error" not in result:
            return result
        return None

    def trace_thread(self, thread_id: str, direction: str = "back",
                     steps: int = 5, from_position: int = -1) -> List[Dict]:
        """Trace a thread forward or back. Returns list of CP dicts."""
        result = self._call(
            "trace_thread_api",
            thread_id, direction, str(steps), str(from_position),
        )
        if result and isinstance(result, list):
            return result
        return []

    def get_active_threads(self) -> List[Dict]:
        """Get all active threads."""
        result = self._call("active_threads_api")
        if result and isinstance(result, list):
            return result
        return []

    def get_thread(self, thread_id: str) -> Optional[Dict]:
        """Get a single thread by ID."""
        result = self._call("get_thread", thread_id)
        if result and isinstance(result, dict) and "error" not in result:
            return result
        return None

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread by ID."""
        result = self._call("delete_thread", thread_id)
        if result and isinstance(result, dict):
            return "deleted" in result
        return False

    # =========================================================================
    # KNOT OPERATIONS (v6.0)
    # =========================================================================

    def add_knot(self, knot_id: str, name: str, thread_ids: List[str],
                 pivot_type: str = "collision", reason: str = "",
                 session_id: str = "") -> Optional[str]:
        """Create a narrative knot. Returns knot ID or None."""
        result = self._call(
            "create_knot_api",
            knot_id, name, json.dumps(thread_ids),
            pivot_type, reason, session_id,
        )
        if result and isinstance(result, dict):
            return result.get("id")
        return None

    def get_knot_context(self, knot_id: str) -> Optional[Dict]:
        """Get full multi-thread context at a knot."""
        result = self._call("knot_context_api", knot_id)
        if result and isinstance(result, dict) and "error" not in result:
            return result
        return None

    def list_knots(self) -> List[Dict]:
        """List all knots."""
        result = self._call("list_knots")
        if result and isinstance(result, list):
            return result
        return []

    # =========================================================================
    # SESSION OPERATIONS (v6.0)
    # =========================================================================

    def delete_session(self, session_id: str) -> Optional[Dict]:
        """Delete all data for a session."""
        result = self._call("delete_session", session_id)
        if result and isinstance(result, dict):
            return result
        return None
