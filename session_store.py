"""
Session Store - Persistent Session & Conversation Storage

Sessions (full message history) -> HuggingFace Dataset repo (AthelaPerk/Private)
Conversation summaries (for search) -> Mnemo v4 via MnemoClient

Replaces persistent_storage.py which stored session JSON blobs inside Mnemo
memories, polluting the memory space and requiring O(n) scans.
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

    Architecture:
    - Sessions stored as sessions.json in a private HF Dataset repo
    - Conversation summaries stored in Mnemo for semantic search via MnemoClient
    - In-memory cache avoids redundant HF API calls
    """

    SESSIONS_FILE = "sessions.json"
    HF_REPO = "AthelaPerk/Private"
    REPO_TYPE = "dataset"

    def __init__(self, hf_key: str, mnemo_client: MnemoClient):
        self.hf_key = hf_key
        self.hf_api = HfApi(token=hf_key)
        self.mnemo = mnemo_client

        # In-memory cache
        self._sessions = None  
        self._cache_time = None
        self._CACHE_TTL = 120  

    # =========================================================================
    # SESSION CRUD (HuggingFace Dataset)
    # =========================================================================

    def save_session(self, session_id: str, title: str, messages: List[Dict], timestamp: str = None) -> bool:
        if not messages: return False
        timestamp = timestamp or datetime.now().isoformat()
        session_data = {
            "id": session_id, "title": title, "timestamp": timestamp,
            "message_count": len([m for m in messages if m.get("role") == "user"]),
            "preview": self._get_preview(messages), "messages": messages
        }
        try:
            sessions = self._load_cache()
            sessions[session_id] = session_data
            self._write_sessions(sessions)
            return True
        except Exception:
            return False

    def load_sessions(self, limit: int = 20) -> List[Dict]:
        try:
            sessions = self._load_cache()
            return sorted(sessions.values(), key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
        except Exception:
            return []

    def delete_session(self, session_id: str) -> bool:
        try:
            sessions = self._load_cache()
            if session_id not in sessions: return False
            del sessions[session_id]
            self._write_sessions(sessions)
            self._delete_mnemo_conversations(session_id)
            return True
        except Exception:
            return False

    def cleanup_stale_sessions(self) -> int:
        return 0

    # =========================================================================
    # CONVERSATION MEMORY (Mnemo via MnemoClient)
    # =========================================================================

    def save_conversation_turn(self, user_message: str, assistant_response: str, session_id: str = None) -> bool:
        content = f"[CONVERSATION] User asked: {user_message[:200]} | Assistant discussed: {assistant_response[:300]}"
        meta = {"type": "conversation", "session_id": session_id, "timestamp": datetime.now().isoformat()}
        return bool(self.mnemo.add(content, metadata=meta))

    def search_conversations(self, query: str, limit: int = 5) -> List[str]:
        results = self.mnemo.search(query, limit=limit * 2)
        conversations = [r.get("content", "").replace("[CONVERSATION]", "").strip() 
                         for r in results if "[CONVERSATION]" in r.get("content", "")]
        return conversations[:limit]

    # =========================================================================
    # CROSS-SESSION RECALL
    # =========================================================================

    def get_previous_sessions_content(self, current_session_id: str = None, limit: int = 3) -> List[Dict]:
        results = []
        try:
            all_sessions = self.load_sessions(limit=limit + 5)
            for session in all_sessions:
                if current_session_id and session.get("id") == current_session_id: continue
                messages = session.get("messages", [])
                if messages:
                    summary_parts = [f"{msg.get('role').title()}: {msg.get('content', '')[:150]}" for msg in messages]
                    results.append({
                        "id": session.get("id"), "title": session.get("title"),
                        "timestamp": session.get("timestamp"), "summary": "\n".join(summary_parts[:6])
                    })
                    if len(results) >= limit: break
            return results
        except Exception:
            return []

    def search_sessions(self, query: str, current_session_id: str = None, limit: int = 3) -> List[str]:
        results = []
        query_lower = query.lower()
        try:
            all_sessions = self.load_sessions(limit=15)
            for session in all_sessions:
                if current_session_id and session.get("id") == current_session_id: continue
                messages = session.get("messages", [])
                title = session.get("title", "").lower()
                if any(x in title for x in ["last chat", "previous chat", "what did we"]): continue

                relevance = 2 if any(word in title for word in query_lower.split()) else 0
                matched_content = []

                for msg in messages:
                    content = msg.get("content", "")
                    if any(word in content.lower() for word in query_lower.split() if len(word) > 3):
                        relevance += 1
                        matched_content.append(f"{msg.get('role')}: {content[:100]}")

                if relevance > 0 or matched_content:
                    summary = f"Chat '{session.get('title', 'Untitled')}':\n" + ("\n".join(matched_content[:4]) if matched_content else "\n".join([f"{m.get('role')}: {m.get('content', '')[:80]}" for m in messages[:4]]))
                    results.append((relevance, summary))
                elif messages:
                    results.append((0, f"Chat '{session.get('title', 'Untitled')}':\n" + "\n".join([f"{m.get('role')}: {m.get('content', '')[:80]}" for m in messages[:4]])))

            results.sort(key=lambda x: x[0], reverse=True)
            return [r[1] for r in results[:limit]]
        except Exception:
            return []

    def get_stats(self) -> Dict:
        try:
            sessions = self._load_cache()
            return {"total": len(sessions), "sessions": len(sessions), "backend": "huggingface"}
        except Exception:
            return {"total": 0, "sessions": 0, "backend": "huggingface"}

    # =========================================================================
    # INTERNAL: HF Dataset I/O
    # =========================================================================

    def _load_cache(self) -> Dict[str, Dict]:
        now = time.time()
        if self._sessions is not None and self._cache_time is not None and now - self._cache_time < self._CACHE_TTL:
            return self._sessions
        self._sessions = self._download_sessions()
        self._cache_time = now
        return self._sessions

    def _download_sessions(self) -> Dict[str, Dict]:
        try:
            path = hf_hub_download(repo_id=self.HF_REPO, filename=self.SESSIONS_FILE, repo_type=self.REPO_TYPE, token=self.hf_key, force_download=True)
            with open(path, "r") as f: data = json.load(f)
            if isinstance(data, dict): return data
            if isinstance(data, list): return {s["id"]: s for s in data if "id" in s}
            return {}
        except (EntryNotFoundError, RepositoryNotFoundError): return {}
        except Exception: return self._sessions if self._sessions else {}

    def _write_sessions(self, sessions: Dict[str, Dict]):
        content = json.dumps(sessions, indent=2, default=str)
        self.hf_api.upload_file(path_or_fileobj=content.encode("utf-8"), path_in_repo=self.SESSIONS_FILE, repo_id=self.HF_REPO, repo_type=self.REPO_TYPE, commit_message=f"Update sessions")
        self._sessions = sessions
        self._cache_time = time.time()

    def _get_preview(self, messages: List[Dict]) -> str:
        for msg in messages:
            if msg.get("role") == "user": return msg.get("content", "")[:100]
        return ""

    def _delete_mnemo_conversations(self, session_id: str):
        try:
            memories = self.mnemo.list_memories()
            for mem in memories:
                meta = mem.get("metadata", {})
                if meta.get("session_id") == session_id and meta.get("type") == "conversation":
                    self.mnemo.delete(mem.get("id"))
        except Exception:
            pass
