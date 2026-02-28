"""
Two-Layer Memory System for 4o with Memory

Memory Backends (in order of preference):
1. Mnemo v4 MCP Server (cloud-hosted on HuggingFace, advanced features)
2. OpenAI embeddings + local storage
3. HuggingFace embeddings + local storage  
4. Keyword matching fallback (always works)

Layer 1: Context Memory (in-session, summarized to reduce tokens)
Layer 2: Cross-Session Memory (persistent, Mnemo or local vector storage)
"""

import json
import os
import numpy as np
import tiktoken
from datetime import datetime
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
# MNEMO v4 MCP CLIENT
# =============================================================================

class MnemoClient:
    """
    Client for Mnemo v4 MCP Server - Advanced memory system with:
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
# EMBEDDING PROVIDERS (Fallback when Mnemo unavailable)
# =============================================================================

class EmbeddingProvider:
    """Multi-provider embedding system with automatic fallback"""
    
    def __init__(self, openai_key: str = None, hf_key: str = None):
        self.openai_key = openai_key
        self.hf_key = hf_key
        self.last_provider = None
        
        # OpenAI config
        self.openai_model = "text-embedding-3-small"
        self.openai_url = "https://api.openai.com/v1/embeddings"
        
        # HuggingFace config
        self.hf_model = "intfloat/multilingual-e5-large"
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding using available providers with fallback"""
        
        # Try OpenAI first (most reliable)
        if self.openai_key:
            embedding = self._openai_embedding(text)
            if embedding:
                self.last_provider = "openai"
                return embedding
        
        # Try HuggingFace as fallback
        if self.hf_key:
            embedding = self._hf_embedding(text)
            if embedding:
                self.last_provider = "huggingface"
                return embedding
        
        # No embedding available
        self.last_provider = None
        return None
    
    def _openai_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from OpenAI API"""
        try:
            response = requests.post(
                self.openai_url,
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.openai_model,
                    "input": text[:8000]
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            return None
        except Exception:
            return None
    
    def _hf_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from HuggingFace Inference API"""
        try:
            # UPDATED: Use the correct API inference feature extraction pipeline endpoint
            response = requests.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.hf_model}",
                headers={
                    "Authorization": f"Bearer {self.hf_key}",
                    "Content-Type": "application/json"
                },
                json={"inputs": text[:512], "options": {"wait_for_model": True}},
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        return list(np.mean(result, axis=0))
                    return result
            return None
        except Exception:
            return None


# =============================================================================
# CONTEXT MEMORY (Layer 1 - In-Session)
# =============================================================================

class ContextMemory:
    """
    Layer 1: In-Session Context Memory
    - Keeps sliding window of recent exchanges
    - Summarizes older context to reduce token count
    - Resets on session end
    """
    
    def __init__(self, window_size: int = 4, max_summary_tokens: int = 300):
        self.window_size = window_size
        self.max_summary_tokens = max_summary_tokens
        self.exchanges: List[Dict] = []
        self.summary: str = ""
        self.summary_threshold = 8
    
    def add_exchange(self, user_msg: str, assistant_msg: str):
        """Add a conversation exchange"""
        self.exchanges.append({
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.exchanges) > self.summary_threshold:
            self._compress_old_context()
    
    def _compress_old_context(self):
        """Compress older exchanges into summary"""
        if len(self.exchanges) <= self.window_size:
            return
        
        old_exchanges = self.exchanges[:-self.window_size]
        key_points = []
        for ex in old_exchanges:
            user_snippet = ex["user"][:100]
            key_points.append(f"User discussed: {user_snippet}...")
        
        self.summary = " | ".join(key_points)[:self.max_summary_tokens * 4]
        self.exchanges = self.exchanges[-self.window_size:]
    
    def get_context_for_prompt(self) -> Tuple[str, List[Dict]]:
        """Get optimized context for the prompt"""
        recent = []
        for ex in self.exchanges[-self.window_size:]:
            recent.append({"role": "user", "content": ex["user"]})
            recent.append({"role": "assistant", "content": ex["assistant"]})
        return self.summary, recent
    
    def get_token_estimate(self) -> int:
        """Estimate tokens used by context (Updated to use Tiktoken)"""
        summary_tokens = estimate_tokens(self.summary)
        recent_text = " ".join([ex["user"] + " " + ex["assistant"] for ex in self.exchanges[-self.window_size:]])
        recent_tokens = estimate_tokens(recent_text)
        return summary_tokens + recent_tokens
    
    def clear(self):
        """Clear session context"""
        self.exchanges = []
        self.summary = ""


# =============================================================================
# CROSS-SESSION MEMORY (Layer 2 - Persistent)
# =============================================================================

class CrossSessionMemory:
    """
    Layer 2: Cross-Session Persistent Memory
    
    Uses Mnemo v4 as primary backend (if available), falls back to local storage.
    """
    
    def __init__(self, mnemo_client: MnemoClient = None, 
                 embedding_provider: EmbeddingProvider = None,
                 memory_file: str = "cross_session_memory.json"):
        self.mnemo = mnemo_client
        self.embeddings = embedding_provider
        self.memory_file = memory_file
        self.memories: List[Dict] = self._load_memories()
        self.enabled = True
        self.use_mnemo = mnemo_client and mnemo_client.available
    
    def _load_memories(self) -> List[Dict]:
        """Load memories from local file (fallback)"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_memories(self):
        """Save memories to local file"""
        memories_to_save = []
        for mem in self.memories:
            save_mem = {k: v for k, v in mem.items() if k != "embedding"}
            memories_to_save.append(save_mem)
        with open(self.memory_file, "w") as f:
            json.dump(memories_to_save, f, indent=2)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8))
    
    def _keyword_similarity(self, query_words: set, memory_keywords: List[str]) -> float:
        """Calculate keyword-based similarity"""
        if not memory_keywords:
            return 0.0
        keywords_set = set(memory_keywords)
        overlap = len(query_words & keywords_set)
        return overlap / max(len(keywords_set), 1)
    
    def store(self, content: str, memory_type: str = "general", importance: float = 0.5) -> bool:
        """Store a memory"""
        if not self.enabled:
            return False
        
        # Try Mnemo first
        if self.use_mnemo and self.mnemo:
            memory_id = self.mnemo.add(
                content, 
                metadata={"type": memory_type, "importance": importance}
            )
            if memory_id:
                return True
        
        # Fallback to local storage
        embedding = None
        if self.embeddings:
            embedding = self.embeddings.get_embedding(content)
        
        words = content.lower().split()
        keywords = [w for w in words if len(w) > 4 and w.isalpha()][:15]
        
        memory = {
            "content": content,
            "embedding": embedding,
            "keywords": keywords,
            "type": memory_type,
            "importance": importance,
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "embedding_provider": self.embeddings.last_provider if self.embeddings else None
        }
        
        # Deduplicate
        if embedding:
            for existing in self.memories:
                if existing.get("embedding"):
                    sim = self._cosine_similarity(embedding, existing["embedding"])
                    if sim > 0.9:
                        existing["access_count"] += 1
                        existing["content"] = content
                        self._save_memories()
                        return True
        
        self.memories.append(memory)
        if len(self.memories) > 200:
            self.memories.sort(key=lambda x: x.get("importance", 0) * (1 + x.get("access_count", 0) * 0.1))
            self.memories = self.memories[-200:]
        
        self._save_memories()
        return True
    
    def retrieve(self, query: str, top_k: int = 8, threshold: float = 0.4) -> str:
        """Retrieve relevant memories"""
        if not self.enabled:
            return ""
        
        # Try Mnemo first
        if self.use_mnemo and self.mnemo:
            context = self.mnemo.get_context(query)
            if context:
                return context
        
        # Fallback to local search
        if not self.memories:
            return ""
        
        query_embedding = None
        if self.embeddings:
            query_embedding = self.embeddings.get_embedding(query)
        query_words = set(w.lower() for w in query.split() if len(w) > 3)
        
        scored = []
        for memory in self.memories:
            score = 0
            
            if query_embedding and memory.get("embedding"):
                sim = self._cosine_similarity(query_embedding, memory["embedding"])
                score = sim * (0.7 + 0.3 * memory.get("importance", 0.5))
            else:
                kw_sim = self._keyword_similarity(query_words, memory.get("keywords", []))
                if kw_sim > 0:
                    score = kw_sim * memory.get("importance", 0.5)
            
            if score >= threshold or (not query_embedding and score > 0.1):
                scored.append((score, memory))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        top_memories = scored[:top_k]
        
        if not top_memories:
            return ""
        
        for _, mem in top_memories:
            mem["access_count"] = mem.get("access_count", 0) + 1
        self._save_memories()
        
        memory_parts = []
        for score, mem in top_memories:
            memory_parts.append(f"[{mem.get('type', 'memory')}] {mem['content']}")
        
        return "\n".join(memory_parts)
    
    def extract_and_store_from_conversation(self, user_msg: str, assistant_msg: str):
        """Extract and store memories from conversation"""
        if not self.enabled:
            return
        
        user_lower = user_msg.lower()
        
        # Preference patterns
        preference_signals = ["i like", "i prefer", "i love", "i hate", "i always", "i never", "my favorite", "i enjoy"]
        for signal in preference_signals:
            if signal in user_lower:
                self.store(user_msg[:200], memory_type="preference", importance=0.7)
                return
        
        # Fact patterns
        fact_signals = [
            "i am ", "i'm ", "my name", "i work", "i live", "i have ", 
            "we are ", "we're ", "our company", "our startup", "our product", 
            "we focus", "working on", "i study", "my job", "my role"
        ]
        for signal in fact_signals:
            if signal in user_lower:
                self.store(user_msg[:200], memory_type="fact", importance=0.8)
                return
        
        # Topic patterns
        if len(user_msg) > 80:
            self.store(f"Discussion: {user_msg[:150]}", memory_type="topic", importance=0.4)
    
    def toggle(self, enabled: bool):
        """Toggle cross-session memory"""
        self.enabled = enabled
    
    def clear(self):
        """Clear all memories"""
        self.memories = []
        self._save_memories()
        if self.use_mnemo and self.mnemo:
            self.mnemo.clear()
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        # Check Mnemo stats
        mnemo_stats = {}
        if self.mnemo:
            mnemo_stats = self.mnemo.get_stats()
        
        # Local stats
        embedding_count = sum(1 for m in self.memories if m.get("embedding"))
        
        return {
            "total_memories": len(self.memories),
            "with_embeddings": embedding_count,
            "keyword_only": len(self.memories) - embedding_count,
            "enabled": self.enabled,
            "backend": "mnemo" if (self.use_mnemo and mnemo_stats.get("available")) else "local",
            "mnemo": mnemo_stats,
            "last_provider": self.embeddings.last_provider if self.embeddings else None,
            "by_type": {
                "preference": sum(1 for m in self.memories if m.get("type") == "preference"),
                "fact": sum(1 for m in self.memories if m.get("type") == "fact"),
                "topic": sum(1 for m in self.memories if m.get("type") == "topic"),
                "general": sum(1 for m in self.memories if m.get("type") == "general"),
            }
        }


# =============================================================================
# COMBINED TWO-LAYER MEMORY MANAGER
# =============================================================================

class TwoLayerMemory:
    """Combined Two-Layer Memory Manager with Mnemo integration"""
    
    def __init__(self, 
                 mnemo_url: str = None,
                 mnemo_token: str = None,
                 openai_key: str = None, 
                 hf_key: str = None, 
                 cross_session_enabled: bool = True):
        
        # Initialize Mnemo client
        self.mnemo = MnemoClient(
            base_url=mnemo_url or "https://athelaperk-mnemo-mcp.hf.space",
            token=mnemo_token or hf_key  # Use HF token for Mnemo auth
        )
        
        # Initialize embedding provider (fallback)
        self.embedding_provider = EmbeddingProvider(openai_key=openai_key, hf_key=hf_key)
        
        # Initialize memory layers
        self.context_memory = ContextMemory(window_size=4, max_summary_tokens=300)
        self.cross_session = CrossSessionMemory(
            mnemo_client=self.mnemo,
            embedding_provider=self.embedding_provider
        )
        self.cross_session.toggle(cross_session_enabled)
    
    def process_exchange(self, user_msg: str, assistant_msg: str):
        """Process a completed exchange"""
        self.context_memory.add_exchange(user_msg, assistant_msg)
        self.cross_session.extract_and_store_from_conversation(user_msg, assistant_msg)
    
    def build_context(self, user_query: str) -> Tuple[str, List[Dict], str]:
        """Build optimized context for API call"""
        summary, recent_messages = self.context_memory.get_context_for_prompt()
        cross_session_context = self.cross_session.retrieve(user_query) if self.cross_session.enabled else ""
        return summary, recent_messages, cross_session_context
    
    def get_token_estimate(self) -> Dict:
        """Estimate tokens for both layers (Updated to use Tiktoken)"""
        context_tokens = self.context_memory.get_token_estimate()
        cross_text = self.cross_session.retrieve("test") if self.cross_session.enabled else ""
        cross_tokens = estimate_tokens(cross_text)
        return {
            "context_memory": context_tokens,
            "cross_session": cross_tokens,
            "total": context_tokens + cross_tokens
        }
    
    def toggle_cross_session(self, enabled: bool):
        """Toggle cross-session memory"""
        self.cross_session.toggle(enabled)
    
    def clear_session(self):
        """Clear current session only"""
        self.context_memory.clear()
    
    def clear_all(self):
        """Clear everything"""
        self.context_memory.clear()
        self.cross_session.clear()
    
    def get_stats(self) -> Dict:
        """Get combined stats"""
        return {
            "context_memory": {
                "exchanges": len(self.context_memory.exchanges),
                "has_summary": bool(self.context_memory.summary),
                "token_estimate": self.context_memory.get_token_estimate()
            },
            "cross_session": self.cross_session.get_stats(),
            "mnemo_available": self.mnemo.available,
            "embedding_provider": self.embedding_provider.last_provider
        }


# =============================================================================
# MNEMO MEMORY MANAGER (App Interface)
# =============================================================================

class MnemoMemoryManager:
    """
    App-facing memory manager that wraps TwoLayerMemory.
    Provides the interface expected by the Streamlit app.
    """
    
    def __init__(self, 
                 openrouter_key: str = None,
                 hf_key: str = None,
                 openai_key: str = None,
                 user_id: str = "default",
                 cross_session_enabled: bool = True):
        
        self.user_id = user_id
        self.memory = TwoLayerMemory(
            mnemo_url="https://athelaperk-mnemo-mcp.hf.space",
            mnemo_token=hf_key,
            openai_key=openai_key,
            hf_key=hf_key,
            cross_session_enabled=cross_session_enabled
        )
        self.conversation_history: List[Dict] = []
    
    def build_context(self, user_message: str, system_prompt: str) -> Tuple[List[Dict], Dict]:
        """
        Build context for API call including system prompt and memories.
        
        Returns: (messages_list, context_metadata)
        """
        # Get memory context
        summary, recent_messages, cross_session_context = self.memory.build_context(user_message)
        
        # Build system prompt with memory context
        full_system = system_prompt
        
        if summary:
            full_system += f"\n\n[Earlier in this conversation: {summary}]"
        
        if cross_session_context:
            full_system += f"\n\n[What you know about this person from past conversations:\n{cross_session_context}]"
        
        # Build messages list
        messages = [{"role": "system", "content": full_system}]
        
        # Add recent conversation history
        messages.extend(recent_messages)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Build metadata
        memories_used = 0
        if cross_session_context:
            memories_used = len(cross_session_context.split('\n'))
        
        context_meta = {
            "cross_session_memories_used": memories_used,
            "context_tokens_estimate": self.memory.context_memory.get_token_estimate() + estimate_tokens(cross_session_context),
            "has_summary": bool(summary),
            "mnemo_available": self.memory.mnemo.available
        }
        
        return messages, context_meta
    
    def process_turn(self, user_message: str, assistant_response: str):
        """Process a completed conversation turn"""
        self.memory.process_exchange(user_message, assistant_response)
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
    
    def toggle_cross_session(self, enabled: bool):
        """Toggle cross-session memory on/off"""
        self.memory.toggle_cross_session(enabled)
    
    def clear_session(self):
        """Clear current session context"""
        self.memory.clear_session()
        self.conversation_history = []
    
    def clear_all_memories(self):
        """Clear all memories including cross-session"""
        self.memory.clear_all()
        self.conversation_history = []
    
    def get_stats(self) -> Dict:
        """Get memory statistics for display"""
        raw_stats = self.memory.get_stats()
        
        return {
            "context_memory": {
                "messages_in_window": raw_stats["context_memory"]["exchanges"],
                "total_processed": len(self.conversation_history),
                "has_summary": raw_stats["context_memory"]["has_summary"],
                "token_estimate": raw_stats["context_memory"]["token_estimate"]
            },
            "cross_session_memory": {
                "total_memories": raw_stats["cross_session"]["total_memories"],
                "backend": raw_stats["cross_session"]["backend"],
                "enabled": raw_stats["cross_session"]["enabled"],
                "by_type": raw_stats["cross_session"]["by_type"]
            },
            "mnemo": {
                "available": raw_stats["mnemo_available"],
                "stats": raw_stats["cross_session"].get("mnemo", {})
            }
        }
