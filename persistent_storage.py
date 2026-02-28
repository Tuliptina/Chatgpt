"""
Persistent Storage for Sessions and Conversations

Stores data in Mnemo so it survives browser refresh:
- Session history (list of past chats)
- Conversation content (for cross-session memory)
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Optional


class PersistentStorage:
    """
    Persistent storage using Mnemo cloud backend.
    
    Stores:
    - SESSION_HISTORY: List of all chat sessions
    - CONVERSATION: Actual chat content for memory
    """
    
    def __init__(self, hf_key: str, mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space"):
        self.hf_key = hf_key
        self.mnemo_url = mnemo_url
        self.headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
        # Cache to avoid hammering Mnemo with repeated list_all_memories() calls
        self._memory_cache = None
        self._cache_time = None
        self._CACHE_TTL = 30  # seconds
    
    # =========================================================================
    # SESSION HISTORY (Persistent across browser refresh)
    # =========================================================================
    
    def save_session(self, session_id: str, title: str, messages: List[Dict], timestamp: str = None):
        """Save a chat session to persistent storage INCLUDING full messages.
        
        Uses single API call (add only). Deduplication happens on load.
        Old approach used 3 calls (list + delete + add) — now just 1.
        """
        if not messages:
            return False
        
        timestamp = timestamp or datetime.now().isoformat()
        
        # Create session data with FULL messages
        session_data = {
            "id": session_id,
            "title": title,
            "timestamp": timestamp,
            "message_count": len([m for m in messages if m.get("role") == "user"]),
            "preview": self._get_preview(messages),
            "messages": messages  # Store full messages!
        }
        
        # Store as a special SESSION memory (single API call)
        content = f"[SESSION] {json.dumps(session_data)}"
        
        try:
            response = requests.post(
                f"{self.mnemo_url}/add",
                headers=self.headers,
                json={
                    "content": content,
                    "metadata": {
                        "type": "session",
                        "session_id": session_id,
                        "timestamp": timestamp
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                self.invalidate_cache()
            return response.status_code == 200
        except Exception:
            return False
    
    def load_sessions(self, limit: int = 20) -> List[Dict]:
        """Load all saved sessions from persistent storage.
        
        Deduplicates by session_id, keeping the newest version.
        """
        sessions_by_id = {}  # session_id → session_data (newest wins)
        
        try:
            memories = self.list_all_memories()
            
            for mem in memories:
                content = mem.get("content", "")
                if content.startswith("[SESSION]"):
                    try:
                        json_str = content[9:].strip()  # Remove "[SESSION] "
                        session_data = json.loads(json_str)
                        sid = session_data.get("id", "")
                        ts = session_data.get("timestamp", "")
                        
                        # Keep newest version per session_id
                        if sid not in sessions_by_id or ts > sessions_by_id[sid].get("timestamp", ""):
                            sessions_by_id[sid] = session_data
                    except Exception:
                        pass
            
            # Sort by timestamp (newest first)
            sessions = sorted(sessions_by_id.values(), key=lambda x: x.get("timestamp", ""), reverse=True)
            return sessions[:limit]
            
        except Exception:
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session AND all related memories from persistent storage.
        This includes:
        - [SESSION] entries (may be multiple due to upsert approach)
        - [CONVERSATION] entries with this session_id
        - Any extracted memories with this session_id in metadata
        """
        deleted_count = 0
        
        try:
            memories = self.list_all_memories(use_cache=False)
            
            for mem in memories:
                content = mem.get("content", "")
                metadata = mem.get("metadata", {})
                mem_id = mem.get("id", "")
                
                should_delete = False
                
                # Check if it's a SESSION entry for this session
                if "[SESSION]" in content and session_id in content:
                    should_delete = True
                
                # Check if it's a CONVERSATION entry with this session_id
                elif "[CONVERSATION]" in content and metadata.get("session_id") == session_id:
                    should_delete = True
                
                # Check if it's an extracted memory with this session_id
                elif metadata.get("session_id") == session_id:
                    should_delete = True
                
                if should_delete:
                    try:
                        response = requests.delete(
                            f"{self.mnemo_url}/delete/{mem_id}",
                            headers=self.headers,
                            timeout=10
                        )
                        if response.status_code == 200:
                            deleted_count += 1
                    except Exception:
                        pass
            
            if deleted_count > 0:
                self.invalidate_cache()
            return deleted_count > 0
        except Exception:
            return False
    
    def _get_preview(self, messages: List[Dict]) -> str:
        """Get preview text from messages"""
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")[:100]
        return ""
    
    def cleanup_stale_sessions(self):
        """Remove duplicate session entries, keeping only the newest per session_id.
        
        Since save_session now does add-only (no delete-before-add),
        old session snapshots accumulate. Call this periodically.
        """
        try:
            memories = self.list_all_memories(use_cache=False)
            
            # Group session memories by session_id
            sessions_by_id = {}  # session_id → list of (timestamp, mem_id)
            for mem in memories:
                content = mem.get("content", "")
                if content.startswith("[SESSION]"):
                    try:
                        session_data = json.loads(content[9:].strip())
                        sid = session_data.get("id", "")
                        ts = session_data.get("timestamp", "")
                        mem_id = mem.get("id", "")
                        if sid:
                            sessions_by_id.setdefault(sid, []).append((ts, mem_id))
                    except Exception:
                        pass
            
            deleted = 0
            for sid, entries in sessions_by_id.items():
                if len(entries) <= 1:
                    continue
                # Sort by timestamp descending, keep newest, delete rest
                entries.sort(key=lambda x: x[0], reverse=True)
                for ts, mem_id in entries[1:]:
                    try:
                        requests.delete(
                            f"{self.mnemo_url}/delete/{mem_id}",
                            headers=self.headers,
                            timeout=10
                        )
                        deleted += 1
                    except Exception:
                        pass
            
            if deleted > 0:
                self.invalidate_cache()
            return deleted
        except Exception:
            return 0
    
    # =========================================================================
    # CONVERSATION MEMORY (For cross-session recall)
    # =========================================================================
    
    def save_conversation_turn(self, user_message: str, assistant_response: str, session_id: str = None):
        """
        Save a conversation turn for cross-session memory.
        This allows the AI to remember what was discussed.
        """
        timestamp = datetime.now().isoformat()
        
        # Create a summary of the exchange
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
                        "timestamp": timestamp
                    }
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_recent_conversations(self, limit: int = 10) -> List[str]:
        """Get recent conversation summaries"""
        conversations = []
        
        try:
            memories = self.list_all_memories()
            
            for mem in memories:
                content = mem.get("content", "")
                if content.startswith("[CONVERSATION]"):
                    conversations.append(content[14:].strip())
            
            return conversations[:limit]
        except Exception:
            return []
    
    def search_conversations(self, query: str, limit: int = 5) -> List[str]:
        """Search past conversations"""
        try:
            response = requests.post(
                f"{self.mnemo_url}/search",
                headers=self.headers,
                json={
                    "query": query,
                    "limit": limit * 2  # Get more to filter
                },
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
    
    def get_previous_sessions_content(self, current_session_id: str = None, limit: int = 3) -> List[Dict]:
        """
        Get content from previous sessions for cross-session recall.
        Skips the current session to get ACTUAL previous chats.
        
        Args:
            current_session_id: ID of current session to skip
            limit: Number of previous sessions to return
        """
        sessions = []
        
        try:
            all_sessions = self.load_sessions(limit=limit + 5)  # Get more to filter
            
            for session in all_sessions:
                # Skip current session
                if current_session_id and session.get("id") == current_session_id:
                    continue
                    
                messages = session.get("messages", [])
                if messages:
                    # Create a summary of the conversation
                    summary_parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")[:150]
                        if role == "user":
                            summary_parts.append(f"User: {content}")
                        elif role == "assistant":
                            summary_parts.append(f"Assistant: {content}")
                    
                    sessions.append({
                        "id": session.get("id"),
                        "title": session.get("title"),
                        "timestamp": session.get("timestamp"),
                        "summary": "\n".join(summary_parts[:6])  # First 6 messages
                    })
                    
                    if len(sessions) >= limit:
                        break
            
            return sessions
        except Exception:
            return []
    
    def search_sessions(self, query: str, current_session_id: str = None, limit: int = 3) -> List[str]:
        """
        Search through session content for relevant past discussions.
        Skips the current session.
        
        Args:
            query: Search query
            current_session_id: ID of current session to skip
            limit: Number of results to return
        """
        results = []
        query_lower = query.lower()
        
        try:
            all_sessions = self.load_sessions(limit=15)
            
            for session in all_sessions:
                # Skip current session
                if current_session_id and session.get("id") == current_session_id:
                    continue
                    
                messages = session.get("messages", [])
                title = session.get("title", "").lower()
                
                # Skip sessions that are just asking about previous chats
                if "last chat" in title or "previous chat" in title or "what did we" in title:
                    continue
                
                # Check if query matches title or message content
                relevance = 0
                matched_content = []
                
                # Check title
                if any(word in title for word in query_lower.split()):
                    relevance += 2
                
                # Check messages
                for msg in messages:
                    content = msg.get("content", "")
                    if any(word in content.lower() for word in query_lower.split() if len(word) > 3):
                        relevance += 1
                        role = msg.get("role", "")
                        matched_content.append(f"{role}: {content[:100]}")
                
                if relevance > 0 or matched_content:
                    # Create summary
                    summary = f"Chat '{session.get('title', 'Untitled')}':\n"
                    if matched_content:
                        summary += "\n".join(matched_content[:4])
                    else:
                        # Just show first few messages
                        for msg in messages[:4]:
                            summary += f"\n{msg.get('role')}: {msg.get('content', '')[:80]}"
                    results.append((relevance, summary))
                elif messages:
                    # If no keyword match but has messages, still include with low relevance
                    summary = f"Chat '{session.get('title', 'Untitled')}':\n"
                    for msg in messages[:4]:
                        summary += f"\n{msg.get('role')}: {msg.get('content', '')[:80]}"
                    results.append((0, summary))
            
            # Sort by relevance
            results.sort(key=lambda x: x[0], reverse=True)
            return [r[1] for r in results[:limit]]
            
        except Exception:
            return []
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def list_all_memories(self, use_cache: bool = True) -> List[Dict]:
        """List all memories from Mnemo with short-lived cache"""
        now = time.time()
        
        if use_cache and self._memory_cache is not None and self._cache_time:
            if now - self._cache_time < self._CACHE_TTL:
                return self._memory_cache
        
        try:
            response = requests.get(
                f"{self.mnemo_url}/list",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                self._memory_cache = response.json().get("memories", [])
                self._cache_time = now
                return self._memory_cache
            return []
        except Exception:
            return []
    
    def invalidate_cache(self):
        """Invalidate the memory cache after write operations"""
        self._memory_cache = None
        self._cache_time = None
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        try:
            memories = self.list_all_memories()
            
            sessions = 0
            conversations = 0
            other = 0
            
            for mem in memories:
                content = mem.get("content", "")
                if content.startswith("[SESSION]"):
                    sessions += 1
                elif content.startswith("[CONVERSATION]"):
                    conversations += 1
                else:
                    other += 1
            
            return {
                "total": len(memories),
                "sessions": sessions,
                "conversations": conversations,
                "memories": other
            }
        except Exception:
            return {"total": 0, "sessions": 0, "conversations": 0, "memories": 0}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import os
    
    HF_KEY = os.environ.get("HF_KEY", "")
    
    if not HF_KEY:
        print("Set HF_KEY environment variable to test")
    else:
        print("Testing Persistent Storage...")
        
        storage = PersistentStorage(HF_KEY)
        
        # Test save session
        print("\n1. Saving test session...")
        success = storage.save_session(
            session_id="test_session_001",
            title="Test conversation",
            messages=[
                {"role": "user", "content": "Hello, this is a test"},
                {"role": "assistant", "content": "Hi! I'm responding to your test."}
            ]
        )
        print(f"   Save result: {'✅' if success else '❌'}")
        
        # Test load sessions
        print("\n2. Loading sessions...")
        sessions = storage.load_sessions()
        print(f"   Found {len(sessions)} sessions")
        for s in sessions[:3]:
            print(f"   - {s.get('title', 'Untitled')[:30]}")
        
        # Test save conversation
        print("\n3. Saving conversation turn...")
        success = storage.save_conversation_turn(
            user_message="What's my favorite color?",
            assistant_response="Based on our previous conversations, you mentioned you like blue.",
            session_id="test_session_001"
        )
        print(f"   Save result: {'✅' if success else '❌'}")
        
        # Test stats
        print("\n4. Storage stats:")
        stats = storage.get_stats()
        print(f"   Total: {stats['total']}")
        print(f"   Sessions: {stats['sessions']}")
        print(f"   Conversations: {stats['conversations']}")
        print(f"   Memories: {stats['memories']}")
        
        print("\n✅ Test complete!")
