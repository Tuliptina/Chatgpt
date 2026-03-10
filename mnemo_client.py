"""
Mnemo Client - Gradio API Client + Local Engine Client for Streamlit (v7.0)

Two client modes:
  MnemoClient      — Remote HTTP client via Gradio API (v6.0 compat, ~530ms/call)
  LocalMnemoClient — Embedded local engine via SQLite (v7.0, ~5ms/call)

Use create_mnemo_client() factory to pick the right one.

v7.0 changes:
- NEW: LocalMnemoClient wraps MnemoEngine directly (zero network latency)
- NEW: create_mnemo_client() factory function
- KEPT: MnemoClient + GradioTransport fully intact as fallback

Transport format (remote mode):
  POST /gradio_api/call/{api_name}
    body: {"data": [arg1, arg2, ...]}
    returns: {"event_id": "abc123"}

  GET /gradio_api/call/{api_name}/{event_id}
    returns SSE stream:
      event: complete
      data: ["json_string_result"]
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
            thread_id, str(position), namespace,
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
                   session_id: str = "", point_ids: List[str] = None) -> Optional[str]:
        """Create a narrative thread. Returns thread ID or None."""
        pts_json = json.dumps(point_ids or [])
        result = self._call(
            "create_thread_api",
            thread_id, name, entity, thread_type, session_id, pts_json,
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


# =============================================================================
# LOCAL ENGINE CLIENT (v7.0 — zero network latency)
# =============================================================================

class LocalMnemoClient:
    """Local embedded client that wraps MnemoEngine directly.

    Same method signatures as MnemoClient. Callers (app.py, context_engine.py,
    session_store.py, etc.) require zero changes when switching from remote
    to local mode.

    All operations hit local SQLite + FAISS + NumPy. Zero network latency.
    Background sync to HF Datasets handled by PersistentMnemo wrapper.

    Usage:
        from mnemo_client import LocalMnemoClient
        client = LocalMnemoClient(db_path="/home/claude/mnemo.db", hf_key="hf_...")
        results = client.graph_search("Alistair relationship", top_k=15)
    """

    def __init__(self, db_path: str = None, hf_key: str = None,
                 enable_hf_sync: bool = True):
        import os
        # Set env vars so PersistentMnemo can find them
        if hf_key:
            os.environ.setdefault("HF_TOKEN", hf_key)
        os.environ.setdefault("DATASET_REPO_ID", "AthelaPerk/Private")

        from mnemo_core import PersistentMnemo
        self._engine = PersistentMnemo(
            db_path=db_path,
            enable_hf_sync=enable_hf_sync,
        )
        self.token = hf_key or ""
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    # =========================================================================
    # BLOB MEMORY OPERATIONS
    # =========================================================================

    def add(self, content: str, namespace: str = "default",
            metadata: dict = None, priority: float = 1.0) -> Optional[str]:
        return self._engine.add(content, namespace, metadata, priority)

    def search(self, query: str, limit: int = 15,
               namespace: str = None) -> List[Dict]:
        results = self._engine.search(query, top_k=limit, namespace=namespace)
        return [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]

    def should_inject(self, query: str, context: str = "",
                      conversation_history: str = "") -> Tuple[bool, str, float]:
        return self._engine.should_inject(query, context, conversation_history)

    def get_context(self, query: str, limit: int = 15) -> str:
        result = self._engine.get_context(query, top_k=limit)
        return result if isinstance(result, str) else ""

    def get(self, memory_id: str) -> Optional[Dict]:
        return self._engine.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        return self._engine.delete(memory_id)

    def list_memories(self, namespace: str = None) -> List[Dict]:
        return self._engine.list_all()

    def get_stats(self) -> Dict:
        return self._engine.get_stats()

    def maintenance(self) -> Dict:
        return self._engine.maintenance()

    def clear(self) -> bool:
        self._engine.clear()
        return True

    # =========================================================================
    # CONNECTION POINT OPERATIONS
    # =========================================================================

    def add_point(self, entity: str, point_type: str, value: str = "",
                  connects_to: str = "", reason: str = "",
                  weight: float = 0.5, category: str = "fact",
                  session_id: str = "", source: str = "auto_extract",
                  thread_id: str = "", position: int = -1,
                  namespace: str = "default") -> Optional[str]:
        return self._engine.add_point(
            entity=entity, point_type=point_type, value=value,
            connects_to=connects_to, reason=reason, weight=weight,
            category=category, session_id=session_id, source=source,
            thread_id=thread_id, position=position, namespace=namespace,
        )

    def add_points_batch(self, points: List[Dict]) -> List[str]:
        return self._engine.add_points_batch(points)

    def graph_search(self, query: str, top_k: int = 15,
                     active_sessions: List[str] = None) -> List[Dict]:
        return self._engine.graph_search(query, top_k=top_k,
                                          active_sessions=active_sessions)

    def entity_lookup(self, entity: str) -> List[Dict]:
        return self._engine.entity_lookup(entity)

    def get_point(self, cp_id: str) -> Optional[Dict]:
        return self._engine.get_point(cp_id)

    def delete_point(self, cp_id: str) -> bool:
        return self._engine.delete_point(cp_id)

    def list_points(self, limit: int = 200) -> List[Dict]:
        return self._engine.list_points(limit=limit)

    # =========================================================================
    # THREAD OPERATIONS
    # =========================================================================

    def add_thread(self, thread_id: str, name: str, entity: str = "",
                   thread_type: str = "plot_line",
                   session_id: str = "", point_ids: List[str] = None) -> Optional[str]:
        return self._engine.add_thread(thread_id, name, entity,
                                        thread_type, session_id, point_ids=point_ids)

    def advance_thread(self, thread_id: str, new_point_id: str = None,
                       tension_delta: float = 0.0,
                       tone_update: str = None) -> Optional[Dict]:
        success = self._engine.advance_thread(thread_id, -1)
        return {"status": "success"} if success else None

    def trace_thread(self, thread_id: str, direction: str = "back",
                     steps: int = 5, from_position: int = -1) -> List[Dict]:
        return self._engine.trace_thread(thread_id, from_position, direction, steps)

    def get_active_threads(self) -> List[Dict]:
        return self._engine.get_active_threads()

    def get_thread(self, thread_id: str) -> Optional[Dict]:
        return self._engine.get_thread(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        return self._engine.delete_thread(thread_id)

    # =========================================================================
    # KNOT OPERATIONS
    # =========================================================================

    def add_knot(self, knot_id: str, name: str, thread_ids: List[str],
                 pivot_type: str = "collision", reason: str = "",
                 session_id: str = "") -> Optional[str]:
        return self._engine.add_knot(knot_id, name, thread_ids,
                                      pivot_type, reason, session_id)

    def get_knot_context(self, knot_id: str) -> Optional[Dict]:
        return self._engine.get_knot_context(knot_id)

    def list_knots(self) -> List[Dict]:
        return self._engine.list_knots()

    def delete_knot(self, knot_id: str) -> bool:
        return self._engine.delete_knot(knot_id)

    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================

    def delete_session(self, session_id: str) -> Optional[Dict]:
        return self._engine.delete_session(session_id)


# =============================================================================
# RESILIENT CLIENT (auto-failover: local → remote)
# =============================================================================

class ResilientMnemoClient:
    """Auto-failover client: tries local engine first, falls back to remote.

    On init:
      1. Try to create LocalMnemoClient (SQLite + FAISS + NumPy)
      2. If that fails (missing deps, disk error, etc.) → fall back to MnemoClient
      3. Log which mode is active

    On runtime errors:
      - If local mode fails on a call, logs the error and retries via remote
      - After 3 consecutive local failures, permanently switches to remote
      - Caller never sees the failover — same interface throughout

    Usage:
        client = create_mnemo_client(token="hf_...", db_path="~/.mnemo/mnemo.db")
        # Works in local mode (~5ms per call)
        # If local fails → transparent switch to remote (~530ms per call)
    """

    MAX_LOCAL_FAILURES = 3  # Switch to remote after this many consecutive failures

    def __init__(self, base_url: str = None, token: str = None,
                 db_path: str = None, enable_hf_sync: bool = True):
        self.token = token or ""
        self._base_url = base_url or MnemoClient.DEFAULT_URL
        self._db_path = db_path
        self._enable_hf_sync = enable_hf_sync

        self._local: Optional[LocalMnemoClient] = None
        self._remote: Optional[MnemoClient] = None
        self._mode = "uninitialized"
        self._consecutive_failures = 0
        self._total_failovers = 0

        # Try local first
        self._try_init_local()

        # If local failed, use remote
        if self._mode != "local":
            self._init_remote()

    def _try_init_local(self):
        """Attempt to initialize local engine."""
        try:
            self._local = LocalMnemoClient(
                db_path=self._db_path, hf_key=self.token,
                enable_hf_sync=self._enable_hf_sync,
            )
            self._mode = "local"
            self._consecutive_failures = 0
            print(f"[MNEMO] ✅ Local engine initialized (SQLite + FAISS + NumPy)")
            print(f"[MNEMO]    db_path: {self._db_path}")
        except Exception as e:
            print(f"[MNEMO] ⚠️  Local engine failed: {type(e).__name__}: {e}")
            print(f"[MNEMO]    Falling back to remote Gradio API...")
            self._local = None

    def _init_remote(self):
        """Initialize remote client as fallback."""
        try:
            self._remote = MnemoClient(base_url=self._base_url, token=self.token)
            if self._mode != "local":  # Only set mode if not already local
                self._mode = "remote"
            print(f"[MNEMO] 🌐 Remote client initialized ({self._base_url})")
        except Exception as e:
            print(f"[MNEMO] ❌ Remote client also failed: {e}")
            if self._mode != "local":
                self._mode = "unavailable"

    def _ensure_remote(self):
        """Lazy-init remote client when needed for failover."""
        if self._remote is None:
            self._init_remote()
        return self._remote

    def _switch_to_remote(self, reason: str):
        """Permanently switch to remote mode after repeated local failures."""
        self._mode = "remote"
        self._total_failovers += 1
        self._ensure_remote()
        print(f"[MNEMO] 🔄 Switched to remote mode: {reason}")
        print(f"[MNEMO]    Total failovers: {self._total_failovers}")

    def _call(self, method_name: str, *args, **kwargs):
        """Route a method call through local → remote failover chain.

        1. If in local mode: call local, on error try remote
        2. If in remote mode: call remote directly
        3. Track consecutive failures for permanent mode switch
        """
        # --- LOCAL MODE ---
        if self._mode == "local" and self._local is not None:
            try:
                result = getattr(self._local, method_name)(*args, **kwargs)
                self._consecutive_failures = 0  # Reset on success
                return result
            except Exception as e:
                self._consecutive_failures += 1
                print(f"[MNEMO] ⚠️  Local .{method_name}() failed "
                      f"({self._consecutive_failures}/{self.MAX_LOCAL_FAILURES}): "
                      f"{type(e).__name__}: {e}")

                if self._consecutive_failures >= self.MAX_LOCAL_FAILURES:
                    self._switch_to_remote(
                        f"{self.MAX_LOCAL_FAILURES} consecutive local failures")

                # Try remote as one-off fallback
                remote = self._ensure_remote()
                if remote:
                    try:
                        return getattr(remote, method_name)(*args, **kwargs)
                    except Exception as re:
                        print(f"[MNEMO] ❌ Remote fallback also failed: {re}")
                        raise  # Re-raise if both fail

                raise  # No remote available

        # --- REMOTE MODE ---
        remote = self._ensure_remote()
        if remote:
            return getattr(remote, method_name)(*args, **kwargs)

        raise RuntimeError(f"[MNEMO] No backend available for .{method_name}()")

    @property
    def available(self) -> bool:
        if self._mode == "local" and self._local:
            return self._local.available
        if self._remote:
            return self._remote.available
        return False

    @property
    def mode(self) -> str:
        """Current operating mode: 'local', 'remote', or 'unavailable'."""
        return self._mode

    def get_client_stats(self) -> dict:
        """Stats about the resilient client itself."""
        return {
            "mode": self._mode,
            "consecutive_failures": self._consecutive_failures,
            "total_failovers": self._total_failovers,
            "local_available": self._local is not None,
            "remote_available": self._remote is not None,
            "max_local_failures": self.MAX_LOCAL_FAILURES,
        }

    # =========================================================================
    # All public methods — delegate through _call() for failover
    # =========================================================================

    # --- Blob memory ---
    def add(self, content, namespace="default", metadata=None, priority=1.0):
        return self._call("add", content, namespace, metadata, priority)
    def search(self, query, limit=15, namespace=None):
        return self._call("search", query, limit, namespace)
    def should_inject(self, query, context="", conversation_history=""):
        return self._call("should_inject", query, context, conversation_history)
    def get_context(self, query, limit=15):
        return self._call("get_context", query, limit)
    def get(self, memory_id):
        return self._call("get", memory_id)
    def delete(self, memory_id):
        return self._call("delete", memory_id)
    def list_memories(self, namespace=None):
        return self._call("list_memories", namespace)
    def get_stats(self):
        return self._call("get_stats")
    def maintenance(self):
        return self._call("maintenance")
    def clear(self):
        return self._call("clear")

    # --- Connection Points ---
    def add_point(self, entity, point_type, value="", connects_to="", reason="",
                  weight=0.5, category="fact", session_id="", source="auto_extract",
                  thread_id="", position=-1, namespace="default"):
        return self._call("add_point", entity, point_type, value, connects_to,
                          reason, weight, category, session_id, source,
                          thread_id, position, namespace)
    def add_points_batch(self, points):
        return self._call("add_points_batch", points)
    def graph_search(self, query, top_k=15, active_sessions=None):
        return self._call("graph_search", query, top_k, active_sessions)
    def entity_lookup(self, entity):
        return self._call("entity_lookup", entity)
    def get_point(self, cp_id):
        return self._call("get_point", cp_id)
    def delete_point(self, cp_id):
        return self._call("delete_point", cp_id)
    def list_points(self, limit=200):
        return self._call("list_points", limit)

    # --- Threads ---
    def add_thread(self, thread_id, name, entity="", thread_type="plot_line",
                   session_id="", point_ids=None):
        return self._call("add_thread", thread_id, name, entity, thread_type,
                          session_id, point_ids)
    def advance_thread(self, thread_id, new_point_id=None, tension_delta=0.0,
                       tone_update=None):
        return self._call("advance_thread", thread_id, new_point_id,
                          tension_delta, tone_update)
    def trace_thread(self, thread_id, direction="back", steps=5, from_position=-1):
        return self._call("trace_thread", thread_id, direction, steps, from_position)
    def get_active_threads(self):
        return self._call("get_active_threads")
    def get_thread(self, thread_id):
        return self._call("get_thread", thread_id)
    def delete_thread(self, thread_id):
        return self._call("delete_thread", thread_id)

    # --- Knots ---
    def add_knot(self, knot_id, name, thread_ids, pivot_type="collision",
                 reason="", session_id=""):
        return self._call("add_knot", knot_id, name, thread_ids, pivot_type,
                          reason, session_id)
    def get_knot_context(self, knot_id):
        return self._call("get_knot_context", knot_id)
    def list_knots(self):
        return self._call("list_knots")
    def delete_knot(self, knot_id):
        return self._call("delete_knot", knot_id)

    # --- Session ---
    def delete_session(self, session_id):
        return self._call("delete_session", session_id)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_mnemo_client(mode: str = "auto", base_url: str = None,
                        token: str = None, db_path: str = None,
                        enable_hf_sync: bool = True):
    """Create a Mnemo client with automatic failover.

    Args:
        mode: "auto" (try local, fall back to remote), "local", or "remote"
        base_url: HF Space URL (remote/auto mode)
        token: HuggingFace token
        db_path: SQLite database path (local/auto mode)
        enable_hf_sync: Whether to sync .db to HF Datasets

    Returns:
        ResilientMnemoClient (auto mode) — tries local, falls back to remote
        LocalMnemoClient (local mode) — local only, fails hard if engine fails
        MnemoClient (remote mode) — remote only, ~530ms per call
    """
    if mode == "auto":
        return ResilientMnemoClient(
            base_url=base_url, token=token, db_path=db_path,
            enable_hf_sync=enable_hf_sync,
        )
    elif mode == "local":
        return LocalMnemoClient(db_path=db_path, hf_key=token,
                                enable_hf_sync=enable_hf_sync)
    else:
        return MnemoClient(base_url=base_url or MnemoClient.DEFAULT_URL,
                           token=token)
