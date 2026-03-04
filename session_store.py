"""
Session Store - Persistent Session & Conversation Storage (v5.1.1)

Sessions (full message history) -> HuggingFace Dataset repo (AthelaPerk/Private)
Conversation summaries (for search) -> Mnemo v5.1 via MnemoClient

v5.1.1 changes:
- Added save_folder_state and load_folder_state for persistent UI folder organization
- Split monolithic sessions.json into per-session files:
    sessions/_index.json  — lightweight manifest (no messages)
    sessions/{session_id}.json — individual session with messages
- Eliminates last-write-wins data loss from concurrent tabs
- O(1) save/load per session instead of O(n) for the whole blob
- Auto-migrates from old sessions.json on first load
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from mnemo_client import MnemoClient


# =============================================================================
# SESSION STORE
# =============================================================================

class SessionStore:
    """
    Persistent storage for chat sessions and conversation memory.

    Architecture (v5.1):
    - Index: sessions/_index.json — {session_id: {title, timestamp, count, preview}}
    - Data:  sessions/{session_id}.json — full session with messages
    - Folders: sessions/_folders.json — state map of UI folders
    - Conversation summaries stored in Mnemo for semantic search
    - In-memory cache with TTL for both index and individual sessions
    """

    # HF repo config
    HF_REPO = "AthelaPerk/Private"
    REPO_TYPE = "dataset"

    # File paths
    INDEX_FILE = "sessions/_index.json"
    SESSION_DIR = "sessions"

    # Legacy (for migration)
    LEGACY_FILE = "sessions.json"

    # Limits
    MAX_SESSIONS = 50

    def __init__(self, hf_key: str, mnemo_client: MnemoClient):
        self.hf_key = hf_key
        self.hf_api = HfApi(token=hf_key)
        self.mnemo = mnemo_client

        # In-memory cache
        self._index: Optional[Dict[str, Dict]] = None
        self._index_time: Optional[float] = None
        self._session_cache: Dict[str, Dict] = {}  # session_id -> full session
        self._CACHE_TTL = 120  # seconds

        # One-shot migration flag
        self._migrated = False

    # =========================================================================
    # FOLDER STATE MANAGEMENT
    # =========================================================================

    def save_folder_state(self, folders: Dict[str, List[str]]):
        """Save the session folder organization state."""
        content = json.dumps(folders, indent=2)
        self.hf_api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=f"{self.SESSION_DIR}/_folders.json",
            repo_id=self.HF_REPO,
            repo_type=self.REPO_TYPE,
            commit_message="Update folder state"
        )

    def load_folder_state(self) -> Dict[str, List[str]]:
        """Load the session folder organization state."""
        try:
            path = hf_hub_download(
                repo_id=self.HF_REPO,
                filename=f"{self.SESSION_DIR}/_folders.json",
                repo_type=self.REPO_TYPE,
                token=self.hf_key,
                force_download=True
            )
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {"📁 Default": []}

    # =========================================================================
    # SESSION CRUD
    # =========================================================================

    def save_session(self, session_id: str, title: str,
                     messages: List[Dict], timestamp: str = None) -> bool:
        """Save a single session. Writes two files:
        1. sessions/{session_id}.json — full session data
        2. sessions/_index.json — updated manifest (no messages)
        """
        if not messages:
            return False
        timestamp = timestamp or datetime.now().isoformat()

        session_data = {
            "id": session_id,
            "title": title,
            "timestamp": timestamp,
            "message_count": len([m for m in messages if m.get("role") == "user"]),
            "preview": self._get_preview(messages),
            "messages": messages,
        }

        try:
            # 1. Write the individual session file
            self._write_session_file(session_id, session_data)

            # 2. Update the index (without messages)
            index = self._load_index()
            index[session_id] = {
                "id": session_id,
                "title": title,
                "timestamp": timestamp,
                "message_count": session_data["message_count"],
                "preview": session_data["preview"],
            }
            self._write_index(index)

            # 3. Update local cache
            self._session_cache[session_id] = session_data
            return True
        except Exception:
            return False

    def load_sessions(self, limit: int = 20) -> List[Dict]:
        """Load sessions with messages, sorted by timestamp (newest first)."""
        try:
            index = self._load_index()
            sorted_entries = sorted(
                index.values(),
                key=lambda x: x.get("timestamp", ""),
                reverse=True,
            )[:limit]

            results = []
            for entry in sorted_entries:
                sid = entry.get("id", "")
                # Populate messages from cache or download
                if sid in self._session_cache:
                    session = dict(entry)
                    session["messages"] = self._session_cache[sid].get("messages", [])
                else:
                    full = self._download_session_file(sid)
                    if full:
                        self._session_cache[sid] = full
                        session = dict(entry)
                        session["messages"] = full.get("messages", [])
                    else:
                        session = dict(entry)
                        session["messages"] = []
                results.append(session)
            return results
        except Exception:
            return []

    def load_session_messages(self, session_id: str) -> List[Dict]:
        """Load full messages for a single session (on-demand)."""
        # Check cache first
        if session_id in self._session_cache:
            return self._session_cache[session_id].get("messages", [])

        # Download individual session file
        try:
            session_data = self._download_session_file(session_id)
            if session_data:
                self._session_cache[session_id] = session_data
                return session_data.get("messages", [])
        except Exception:
            pass
        return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session's file and remove from index."""
        try:
            # Remove from index
            index = self._load_index()
            if session_id in index:
                del index[session_id]
                self._write_index(index)

            # Delete the session file
            try:
                self.hf_api.delete_file(
                    path_in_repo=f"{self.SESSION_DIR}/{session_id}.json",
                    repo_id=self.HF_REPO,
                    repo_type=self.REPO_TYPE,
                    commit_message=f"Delete session {session_id}",
                )
            except EntryNotFoundError:
                pass  # Already gone

            # Clean caches
            self._session_cache.pop(session_id, None)

            # Clean Mnemo conversations
            self._delete_mnemo_conversations(session_id)
            return True
        except Exception:
            return False

    def cleanup_stale_sessions(self, keep: int = None) -> int:
        """Remove oldest sessions beyond the limit."""
        keep = keep or self.MAX_SESSIONS
        try:
            index = self._load_index()
            if len(index) <= keep:
                return 0

            sorted_ids = sorted(
                index.keys(),
                key=lambda sid: index[sid].get("timestamp", ""),
                reverse=True,
            )
            to_delete = sorted_ids[keep:]
            deleted = 0
            for sid in to_delete:
                if self.delete_session(sid):
                    deleted += 1
            return deleted
        except Exception:
            return 0

    # =========================================================================
    # CONVERSATION MEMORY (Mnemo via MnemoClient)
    # =========================================================================

    def save_conversation_turn(self, user_message: str,
                               assistant_response: str,
                               session_id: str = None) -> bool:
        content = (
            f"[CONVERSATION] User asked: {user_message[:200]} "
            f"| Assistant discussed: {assistant_response[:300]}"
        )
        meta = {
            "type": "conversation",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }
        return bool(self.mnemo.add(content, metadata=meta))

    def search_conversations(self, query: str, limit: int = 5) -> List[str]:
        results = self.mnemo.search(query, limit=limit * 2)
        conversations = [
            r.get("content", "").replace("[CONVERSATION]", "").strip()
            for r in results
            if "[CONVERSATION]" in r.get("content", "")
        ]
        return conversations[:limit]

    # =========================================================================
    # CROSS-SESSION RECALL
    # =========================================================================

    def get_previous_sessions_content(self, current_session_id: str = None,
                                      limit: int = 3) -> List[Dict]:
        results = []
        try:
            all_sessions = self.load_sessions(limit=limit + 5)
            for session in all_sessions:
                if current_session_id and session.get("id") == current_session_id:
                    continue

                # Load messages on demand if not cached
                sid = session.get("id", "")
                messages = session.get("messages", [])
                if not messages and sid:
                    messages = self.load_session_messages(sid)

                if messages:
                    summary_parts = [
                        f"{msg.get('role', '').title()}: {msg.get('content', '')[:150]}"
                        for msg in messages
                    ]
                    results.append({
                        "id": sid,
                        "title": session.get("title"),
                        "timestamp": session.get("timestamp"),
                        "summary": "\n".join(summary_parts[:6]),
                    })
                    if len(results) >= limit:
                        break
            return results
        except Exception:
            return []

    def search_sessions(self, query: str, current_session_id: str = None,
                        limit: int = 3) -> List[str]:
        results = []
        query_lower = query.lower()
        try:
            all_sessions = self.load_sessions(limit=15)
            for session in all_sessions:
                if current_session_id and session.get("id") == current_session_id:
                    continue

                sid = session.get("id", "")
                messages = session.get("messages", [])
                if not messages and sid:
                    messages = self.load_session_messages(sid)

                title = session.get("title", "").lower()
                if any(x in title for x in ["last chat", "previous chat", "what did we"]):
                    continue

                relevance = 2 if any(word in title for word in query_lower.split()) else 0
                matched_content = []

                for msg in messages:
                    content = msg.get("content", "")
                    if any(word in content.lower() for word in query_lower.split() if len(word) > 3):
                        relevance += 1
                        matched_content.append(f"{msg.get('role')}: {content[:100]}")

                if relevance > 0 or matched_content:
                    summary = (
                        f"Chat '{session.get('title', 'Untitled')}':\n"
                        + ("\n".join(matched_content[:4]) if matched_content
                           else "\n".join([f"{m.get('role')}: {m.get('content', '')[:80]}" for m in messages[:4]]))
                    )
                    results.append((relevance, summary))
                elif messages:
                    results.append((
                        0,
                        f"Chat '{session.get('title', 'Untitled')}':\n"
                        + "\n".join([f"{m.get('role')}: {m.get('content', '')[:80]}" for m in messages[:4]]),
                    ))

            results.sort(key=lambda x: x[0], reverse=True)
            return [r[1] for r in results[:limit]]
        except Exception:
            return []

    def get_stats(self) -> Dict:
        try:
            index = self._load_index()
            return {
                "total": len(index),
                "sessions": len(index),
                "cached": len(self._session_cache),
                "backend": "huggingface_per_session",
            }
        except Exception:
            return {"total": 0, "sessions": 0, "backend": "huggingface_per_session"}

    # =========================================================================
    # INTERNAL: Per-Session File I/O
    # =========================================================================

    def _session_path(self, session_id: str) -> str:
        """Path for a session file inside the repo."""
        return f"{self.SESSION_DIR}/{session_id}.json"

    def _write_session_file(self, session_id: str, data: Dict):
        """Upload a single session file."""
        content = json.dumps(data, indent=2, default=str)
        self.hf_api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=self._session_path(session_id),
            repo_id=self.HF_REPO,
            repo_type=self.REPO_TYPE,
            commit_message=f"Save session {session_id}",
        )

    def _download_session_file(self, session_id: str) -> Optional[Dict]:
        """Download and parse a single session file."""
        try:
            path = hf_hub_download(
                repo_id=self.HF_REPO,
                filename=self._session_path(session_id),
                repo_type=self.REPO_TYPE,
                token=self.hf_key,
                force_download=True,
            )
            with open(path, "r") as f:
                return json.load(f)
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        except Exception:
            return None

    # =========================================================================
    # INTERNAL: Index I/O
    # =========================================================================

    def _load_index(self) -> Dict[str, Dict]:
        """Load the session index, with cache + TTL."""
        now = time.time()
        if (self._index is not None
                and self._index_time is not None
                and now - self._index_time < self._CACHE_TTL):
            return self._index

        self._index = self._download_index()

        # One-shot migration from legacy sessions.json
        if not self._index and not self._migrated:
            self._migrated = True
            self._index = self._migrate_legacy()

        self._index_time = now
        return self._index

    def _download_index(self) -> Dict[str, Dict]:
        """Download and parse the index file."""
        try:
            path = hf_hub_download(
                repo_id=self.HF_REPO,
                filename=self.INDEX_FILE,
                repo_type=self.REPO_TYPE,
                token=self.hf_key,
                force_download=True,
            )
            with open(path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (EntryNotFoundError, RepositoryNotFoundError):
            return {}
        except Exception:
            return self._index if self._index else {}

    def _write_index(self, index: Dict[str, Dict]):
        """Upload the index file."""
        content = json.dumps(index, indent=2, default=str)
        self.hf_api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=self.INDEX_FILE,
            repo_id=self.HF_REPO,
            repo_type=self.REPO_TYPE,
            commit_message="Update session index",
        )
        self._index = index
        self._index_time = time.time()

    # =========================================================================
    # INTERNAL: Legacy Migration
    # =========================================================================

    def _migrate_legacy(self) -> Dict[str, Dict]:
        """Migrate from monolithic sessions.json to per-session files.

        Runs once on first load if no index exists. Downloads the old
        sessions.json, splits each session into its own file, builds the
        index, and uploads everything. The old file is left in place
        (safe to delete manually later).
        """
        try:
            path = hf_hub_download(
                repo_id=self.HF_REPO,
                filename=self.LEGACY_FILE,
                repo_type=self.REPO_TYPE,
                token=self.hf_key,
                force_download=True,
            )
            with open(path, "r") as f:
                data = json.load(f)
        except (EntryNotFoundError, RepositoryNotFoundError):
            return {}
        except Exception:
            return {}

        # Normalize — old format could be dict or list
        if isinstance(data, list):
            sessions = {s["id"]: s for s in data if "id" in s}
        elif isinstance(data, dict):
            sessions = data
        else:
            return {}

        if not sessions:
            return {}

        # Build index and write per-session files
        index = {}
        for session_id, session_data in sessions.items():
            try:
                self._write_session_file(session_id, session_data)
                index[session_id] = {
                    "id": session_id,
                    "title": session_data.get("title", "Untitled"),
                    "timestamp": session_data.get("timestamp", ""),
                    "message_count": session_data.get("message_count", 0),
                    "preview": session_data.get("preview", ""),
                }
                # Populate cache while we're at it
                self._session_cache[session_id] = session_data
            except Exception:
                continue  # Skip individual failures

        # Write the new index
        if index:
            self._write_index(index)

        return index

    # =========================================================================
    # INTERNAL: Helpers
    # =========================================================================

    def _get_preview(self, messages: List[Dict]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")[:100]
        return ""

    def _delete_mnemo_conversations(self, session_id: str):
        """Remove Mnemo conversation memories tied to a session."""
        try:
            memories = self.mnemo.list_memories()
            for mem in memories:
                meta = mem.get("metadata", {})
                if (meta.get("session_id") == session_id
                        and meta.get("type") == "conversation"):
                    self.mnemo.delete(mem.get("id"))
        except Exception:
            pass
