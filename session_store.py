"""
Session Store - Persistent Session & Conversation Storage

Sessions (full message history) -> HuggingFace Dataset repo (AthelaPerk/Private)
Conversation summaries (for search) -> Mnemo v4

Replaces persistent_storage.py which stored session JSON blobs inside Mnemo
memories, polluting the memory space and requiring O(n) scans.
"""

import json
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


# =============================================================================
# SESSION STORE
# =============================================================================

class SessionStore:
    """
    Persistent storage for chat sessions and conversation memory.

    Architecture:
    - Sessions stored as sessions.json in a private HF Dataset repo
    - Conversation summaries stored in Mnemo for semantic search
    - In-memory cache avoids redundant HF API calls
    """

    SESSIONS_FILE = "sessions.json"
    HF_REPO = "AthelaPerk/Private"
    REPO_TYPE = "dataset"

    def __init__(self, hf_key: str, mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space"):
        self.hf_key = hf_key
        self.mnemo_url = mnemo_url.rstrip("/")
        self.hf_api = HfApi(token=hf_key)

        # Mnemo headers (for conversation search)
        self.headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }

        # In-memory cache
        self._sessions = None  # session_id -> session_data
        self._cache_time = None
        self._CACHE_TTL = 120  # seconds before re-downloading

    # =========================================================================
    # SESSION CRUD (HuggingFace Dataset)
    # =========================================================================

    def save_session(self, session_id: str, title: str, messages: List[Dict],
                     timestamp: str = None) -> bool:
        """Save a chat session to HF Dataset storage."""
        if not messages:
            return False

        timestamp = timestamp or datetime.now().isoformat()

        session_data = {
            "id": session_id,
            "title": title,
            "timestamp": timestamp,
            "message_count": len([m for m in messages if m.get("role") == "user"]),
            "preview": self._get_preview(messages),
            "messages": messages
        }

        try:
            sessions = self._load_cache()
            sessions[session_id] = session_data
            self._write_sessions(sessions)
            return True
        except Exception:
            return False

    def load_sessions(self, limit: int = 20) -> List[Dict]:
        """Load all saved sessions, newest first."""
        try:
            sessions = self._load_cache()
            sorted_sessions = sorted(
                sessions.values(),
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            return sorted_sessions[:limit]
        except Exception:
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its conversation summaries from Mnemo."""
        try:
            sessions = self._load_cache()
            if session_id not in sessions:
                return False

            del sessions[session_id]
            self._write_sessions(sessions)

            # Also clean up conversation summaries from Mnemo
            self._delete_mnemo_conversations(session_id)
            return True
        except Exception:
            return False

    def cleanup_stale_sessions(self) -> int:
        """No-op. HF overwrites by filename, no duplicates accumulate."""
        return 0

    # =========================================================================
    # CONVERSATION MEMORY (Mnemo - for semantic search)
    # =========================================================================

    def save_conversation_turn(self, user_message: str, assistant_response: str,
                               session_id: str = None) -> bool:
        """Save a conversation summary to Mnemo for cross-session search."""
        content = f"[CONVERSATION] User asked: {user_message[:200]} | Assistant discussed: {assistant_response[:300]}"
        try:
            response = requests.post(
                f"{self.mnemo_url}/add",
                headers=self.headers,
                json={
                    "content": content,
                    "metadata": {
                        "type": "conversation",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def search_conversations(self, query: str, limit: int = 5) -> List[str]:
        """Search past conversation summaries via Mnemo semantic search."""
        try:
            response = requests.post(
                f"{self.mnemo_url}/search",
                headers=self.headers,
                json={"query": query, "limit": limit * 2},
                timeout=15
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                conversations = []
                for r in results:
                    content = r.get("content", "")
                    if "[CONVERSATION]" in content:
                        conversations.append(content.replace("[CONVERSATION]", "").strip())
                return conversations[:limit]
            return []
        except Exception:
            return []

    # =========================================================================
    # CROSS-SESSION RECALL
    # =========================================================================

    def get_previous_sessions_content(self, current_session_id: str = None,
                                      limit: int = 3) -> List[Dict]:
        """Get summaries of previous sessions for cross-session recall."""
        results = []
        try:
            all_sessions = self.load_sessions(limit=limit + 5)

            for session in all_sessions:
                if current_session_id and session.get("id") == current_session_id:
                    continue

                messages = session.get("messages", [])
                if messages:
                    summary_parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")[:150]
                        if role == "user":
                            summary_parts.append(f"User: {content}")
                        elif role == "assistant":
                            summary_parts.append(f"Assistant: {content}")

                    results.append({
                        "id": session.get("id"),
                        "title": session.get("title"),
                        "timestamp": session.get("timestamp"),
                        "summary": "\n".join(summary_parts[:6])
                    })

                    if len(results) >= limit:
                        break

            return results
        except Exception:
            return []

    def search_sessions(self, query: str, current_session_id: str = None,
                        limit: int = 3) -> List[str]:
        """Search through session content for relevant past discussions."""
        results = []
        query_lower = query.lower()

        try:
            all_sessions = self.load_sessions(limit=15)

            for session in all_sessions:
                if current_session_id and session.get("id") == current_session_id:
                    continue

                messages = session.get("messages", [])
                title = session.get("title", "").lower()

                if "last chat" in title or "previous chat" in title or "what did we" in title:
                    continue

                relevance = 0
                matched_content = []

                if any(word in title for word in query_lower.split()):
                    relevance += 2

                for msg in messages:
                    content = msg.get("content", "")
                    if any(word in content.lower() for word in query_lower.split() if len(word) > 3):
                        relevance += 1
                        role = msg.get("role", "")
                        matched_content.append(f"{role}: {content[:100]}")

                if relevance > 0 or matched_content:
                    summary = f"Chat '{session.get('title', 'Untitled')}':\n"
                    if matched_content:
                        summary += "\n".join(matched_content[:4])
                    else:
                        for msg in messages[:4]:
                            summary += f"\n{msg.get('role')}: {msg.get('content', '')[:80]}"
                    results.append((relevance, summary))
                elif messages:
                    summary = f"Chat '{session.get('title', 'Untitled')}':\n"
                    for msg in messages[:4]:
                        summary += f"\n{msg.get('role')}: {msg.get('content', '')[:80]}"
                    results.append((0, summary))

            results.sort(key=lambda x: x[0], reverse=True)
            return [r[1] for r in results[:limit]]

        except Exception:
            return []

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        try:
            sessions = self._load_cache()
            return {
                "total": len(sessions),
                "sessions": len(sessions),
                "backend": "huggingface"
            }
        except Exception:
            return {"total": 0, "sessions": 0, "backend": "huggingface"}

    # =========================================================================
    # INTERNAL: HF Dataset I/O
    # =========================================================================

    def _load_cache(self) -> Dict[str, Dict]:
        """Load sessions from cache or HF repo."""
        now = time.time()
        if (self._sessions is not None
                and self._cache_time is not None
                and now - self._cache_time < self._CACHE_TTL):
            return self._sessions

        self._sessions = self._download_sessions()
        self._cache_time = now
        return self._sessions

    def _download_sessions(self) -> Dict[str, Dict]:
        """Download sessions.json from HF repo."""
        try:
            path = hf_hub_download(
                repo_id=self.HF_REPO,
                filename=self.SESSIONS_FILE,
                repo_type=self.REPO_TYPE,
                token=self.hf_key,
                force_download=True
            )
            with open(path, "r") as f:
                data = json.load(f)

            # data is a dict of session_id -> session_data
            if isinstance(data, dict):
                return data
            # Handle legacy list format
            if isinstance(data, list):
                return {s["id"]: s for s in data if "id" in s}
            return {}

        except (EntryNotFoundError, RepositoryNotFoundError):
            # File or repo doesnt exist yet - first run
            return {}
        except Exception:
            # Network error, JSON error, etc - return cached or empty
            return self._sessions if self._sessions else {}

    def _write_sessions(self, sessions: Dict[str, Dict]):
        """Upload sessions.json to HF repo."""
        content = json.dumps(sessions, indent=2, default=str)
        self.hf_api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=self.SESSIONS_FILE,
            repo_id=self.HF_REPO,
            repo_type=self.REPO_TYPE,
            commit_message=f"Update sessions ({len(sessions)} sessions)"
        )
        # Update cache
        self._sessions = sessions
        self._cache_time = time.time()

    def _invalidate_cache(self):
        """Force re-download on next access."""
        self._sessions = None
        self._cache_time = None

    # =========================================================================
    # INTERNAL: Helpers
    # =========================================================================

    def _get_preview(self, messages: List[Dict]) -> str:
        """Get preview text from first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")[:100]
        return ""

    def _delete_mnemo_conversations(self, session_id: str):
        """Remove conversation summaries for a deleted session from Mnemo."""
        try:
            response = requests.get(
                f"{self.mnemo_url}/list",
                headers=self.headers,
                timeout=10
            )
            if response.status_code != 200:
                return

            for mem in response.json().get("memories", []):
                metadata = mem.get("metadata", {})
                if metadata.get("session_id") == session_id and metadata.get("type") == "conversation":
                    mem_id = mem.get("id", "")
                    if mem_id:
                        try:
                            requests.delete(
                                f"{self.mnemo_url}/delete/{mem_id}",
                                headers=self.headers,
                                timeout=10
                            )
                        except Exception:
                            pass
        except Exception:
            pass
