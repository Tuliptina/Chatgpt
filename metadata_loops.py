"""
Metadata Loop System - Token-Efficient Context Management

Replaces expensive context window summaries with compressed metadata loops.

TOKEN SAVINGS:
- Full content: ~50-200 tokens per memory
- Metadata loop: ~10-20 tokens per memory
- Savings: 70-90% reduction in context tokens!
"""

import re
import json
import hashlib
import requests
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# CONFIGURATION
# =============================================================================

class LoopConfig:
    DEFAULT_LOOP_CAPACITY = 100
    MAX_LOOP_CAPACITY = 250
    MICRO_LOOP_SIZE = 15
    RELEVANCE_THRESHOLD_FULL = 0.45
    RELEVANCE_THRESHOLD_META = 0.20
    METADATA_TOKEN_BUDGET = 1000
    FULL_CONTENT_TOKEN_BUDGET = 3000
    TOTAL_CONTEXT_BUDGET = 4000
    MAX_KEYWORDS = 5
    MAX_SUMMARY_WORDS = 15
    CYCLE_DECAY_RATE = 0.9


_STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
    'by', 'from', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'again',
    'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'so',
    'yet', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'not', 'only', 'own', 'same', 'than', 'too',
    'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'this', 'that', 'these', 'those',
    'about', 'like', 'out', 'up', 'down', 'if',
    'it', 'its', 'he', 'she', 'his', 'her', 'they', 'them', 'their',
    'me', 'my', 'we', 'our', 'you', 'your', 'who', 'what', 'which',
    'tell', 'show', 'give', 'get', 'find',
    'retrieve', 'everything', 'anything', 'something', 'please',
    'know', 'remember', 'recall'
})


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MetadataToken:
    """Compressed representation of a memory (~10-20 tokens)."""
    id: str
    keywords: List[str]
    category: str
    summary: str
    importance: float
    relevance: float = 0.0
    cycle_count: int = 0
    last_cycled: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    full_content_ref: str = ""

    def to_context_string(self) -> str:
        kw_str = ", ".join(self.keywords[:3])
        return f"[{self.category}] {self.summary} ({kw_str})"

    def estimate_tokens(self) -> int:
        return len(self.to_context_string().split()) + 5

    def to_dict(self) -> dict:
        return {
            "id": self.id, "keywords": self.keywords,
            "category": self.category, "summary": self.summary,
            "importance": self.importance, "relevance": self.relevance,
            "cycle_count": self.cycle_count, "access_count": self.access_count,
            "full_content_ref": self.full_content_ref
        }


@dataclass
class MetadataLoop:
    """A loop of related metadata tokens that cycle through context."""
    id: str
    name: str
    category: str
    tokens: List[MetadataToken] = field(default_factory=list)
    current_position: int = 0
    capacity: int = LoopConfig.DEFAULT_LOOP_CAPACITY
    priority: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    total_cycles: int = 0

    def add_token(self, token: MetadataToken) -> bool:
        if len(self.tokens) >= self.capacity:
            return False
        self.tokens.append(token)
        return True

    def cycle(self, steps: int = 1) -> List[MetadataToken]:
        if not self.tokens:
            return []
        self.total_cycles += 1
        self.current_position = (self.current_position + steps) % len(self.tokens)
        window_size = min(5, len(self.tokens))
        visible = []
        for i in range(window_size):
            idx = (self.current_position + i) % len(self.tokens)
            token = self.tokens[idx]
            token.cycle_count += 1
            token.last_cycled = datetime.now()
            visible.append(token)
        return visible

    def get_by_relevance(self, top_k: int = 5) -> List[MetadataToken]:
        return sorted(self.tokens, key=lambda t: t.relevance, reverse=True)[:top_k]

    def estimate_tokens(self) -> int:
        return sum(t.estimate_tokens() for t in self.tokens)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "category": self.category,
            "token_count": len(self.tokens), "capacity": self.capacity,
            "priority": self.priority, "total_cycles": self.total_cycles
        }


# =============================================================================
# METADATA EXTRACTOR
# =============================================================================

class MetadataExtractor:
    """Extracts compressed metadata from full content."""

    def __init__(self, openrouter_key: str = None):
        self.openrouter_key = openrouter_key

    def extract_simple(self, content: str, category: str = "general") -> MetadataToken:
        """Simple extraction without API call."""
        words = content.lower().split()
        filtered = []
        for w in words:
            clean = re.sub(r'[\[\]\(\)\{\}:,\.!\?"]', '', w)
            if clean and clean not in _STOPWORDS and len(clean) > 2:
                filtered.append(clean)

        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        keywords = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:LoopConfig.MAX_KEYWORDS]
        summary_words = content.split()[:LoopConfig.MAX_SUMMARY_WORDS]
        summary = " ".join(summary_words)
        if len(content.split()) > LoopConfig.MAX_SUMMARY_WORDS:
            summary += "..."

        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return MetadataToken(
            id=f"meta_{content_hash}", keywords=keywords,
            category=category, summary=summary,
            importance=0.5, full_content_ref=content_hash
        )

    def extract_smart(self, content: str, category: str = "general") -> MetadataToken:
        """Smart extraction using GPT-4o with Native JSON Mode."""
        if not self.openrouter_key:
            return self.extract_simple(content, category)

        prompt = (
            f"Extract metadata from this text.\n\nTEXT: {content[:4000]}\n\n"
            'Return ONLY a JSON object: {"keywords": ["max 5"], "summary": "max 15 words", "importance": 0.8}'
        )
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openrouter_key}", "Content-Type": "application/json"},
                json={
                    "model": "openai/gpt-4o-2024-11-20",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1, "max_tokens": 150,
                    "response_format": {"type": "json_object"}
                },
                timeout=10
            )
            if response.status_code != 200:
                return self.extract_simple(content, category)

            parsed = json.loads(response.json()["choices"][0]["message"]["content"])
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            return MetadataToken(
                id=f"meta_{content_hash}",
                keywords=parsed.get("keywords", [])[:LoopConfig.MAX_KEYWORDS],
                category=category,
                summary=parsed.get("summary", content[:50])[:100],
                importance=float(parsed.get("importance", 0.5)),
                full_content_ref=content_hash
            )
        except Exception:
            return self.extract_simple(content, category)



# =============================================================================
# LOOP MANAGER
# =============================================================================

class LoopManager:
    """
    Manages metadata loops for token-efficient context injection.

    COST COMPARISON:
    - Old method (summarize context): 500-2000 tokens per turn
    - Loop method: 100-300 tokens per turn
    - Savings: 60-85% token reduction!
    """

    def __init__(self, openrouter_key: str = None, hf_key: str = None,
                 mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space"):
        self.openrouter_key = openrouter_key
        self.hf_key = hf_key
        self.mnemo_url = mnemo_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
        self.extractor = MetadataExtractor(openrouter_key)
        self.loops: Dict[str, MetadataLoop] = {}
        self.content_cache: Dict[str, str] = {}
        self._MAX_CACHE_SIZE = 200

        for category in ["character", "plot", "setting", "theme", "style", "fact", "general"]:
            self.loops[category] = MetadataLoop(
                id=f"loop_{category}", name=category.title(), category=category
            )

    # =========================================================================
    # MNEMO INTEGRATION
    # =========================================================================

    def _mnemo_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search Mnemo for relevant memories"""
        try:
            response = requests.post(
                f"{self.mnemo_url}/search", headers=self.headers,
                json={"query": query, "limit": limit}, timeout=10
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except Exception:
            return []

    def _mnemo_get(self, memory_id: str) -> Optional[str]:
        """Get full content from Mnemo"""
        try:
            response = requests.get(
                f"{self.mnemo_url}/get/{memory_id}",
                headers=self.headers, timeout=10
            )
            if response.status_code == 200:
                return response.json().get("content", "")
            return None
        except Exception:
            return None

    def _mnemo_list(self) -> List[Dict]:
        """List all memories"""
        try:
            response = requests.get(
                f"{self.mnemo_url}/list",
                headers=self.headers, timeout=10
            )
            if response.status_code == 200:
                return response.json().get("memories", [])
            return []
        except Exception:
            return []

    # =========================================================================
    # LOOP OPERATIONS
    # =========================================================================

    def load_from_mnemo(self, use_smart_extraction: bool = False):
        """Load memories from Mnemo and convert to metadata tokens."""
        memories = self._mnemo_list()
        for mem in memories:
            content = mem.get("content", "")
            mem_id = mem.get("id", "")

            category = "general"
            cl = content.lower()
            if "[character]" in cl: category = "character"
            elif "[plot]" in cl: category = "plot"
            elif "[setting]" in cl: category = "setting"
            elif "[theme]" in cl: category = "theme"
            elif "[style]" in cl: category = "style"
            elif "[fact]" in cl: category = "fact"

            if use_smart_extraction:
                token = self.extractor.extract_smart(content, category)
            else:
                token = self.extractor.extract_simple(content, category)
            token.full_content_ref = mem_id

            if category in self.loops:
                self.loops[category].add_token(token)
            self.content_cache[mem_id] = content
        self._trim_cache()

    def _trim_cache(self):
        """Prevent unbounded cache growth"""
        if len(self.content_cache) > self._MAX_CACHE_SIZE:
            keys = list(self.content_cache.keys())
            for key in keys[:len(keys) // 2]:
                del self.content_cache[key]

    def add_to_loop(self, content: str, category: str = "general",
                    memory_id: str = None, use_smart: bool = False) -> MetadataToken:
        """Add new content to appropriate loop"""
        if use_smart:
            token = self.extractor.extract_smart(content, category)
        else:
            token = self.extractor.extract_simple(content, category)

        if memory_id:
            token.full_content_ref = memory_id
            self.content_cache[memory_id] = content
            self._trim_cache()

        if category not in self.loops:
            self.loops[category] = MetadataLoop(
                id=f"loop_{category}", name=category.title(), category=category
            )
        self.loops[category].add_token(token)
        return token

    def update_relevance(self, query: str):
        """Update relevance scores for all tokens based on query."""
        raw_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
        query_words = raw_words - _STOPWORDS
        if not query_words:
            query_words = raw_words - {'the', 'a', 'an', 'is', 'are', 'it', 'to', 'of'}

        for loop in self.loops.values():
            for token in loop.tokens:
                # Source 1: Keywords + category + summary
                token_words = set(kw.lower() for kw in token.keywords)
                token_words.add(token.category.lower())
                summary_words = set(re.sub(r'[^\w\s]', '', token.summary.lower()).split())
                token_words.update(summary_words)

                # Source 2: Full content (if cached)
                full_content_words = set()
                if token.full_content_ref and token.full_content_ref in self.content_cache:
                    full_text = self.content_cache[token.full_content_ref].lower()
                    full_content_words = set(re.sub(r'[^\w\s]', '', full_text).split()) - _STOPWORDS

                all_matchable = token_words | full_content_words

                # Count matches (including partial/substring)
                matches = 0
                for qw in query_words:
                    if len(qw) < 3:
                        continue
                    for tw in all_matchable:
                        if len(tw) < 3:
                            continue
                        if qw == tw or (len(qw) >= 4 and qw in tw) or (len(tw) >= 4 and tw in qw):
                            matches += 1
                            break

                base_relevance = min(1.0, matches / max(len(query_words), 1)) if query_words else 0.0

                # Boost: character/proper noun in full content
                if full_content_words:
                    for qw in query_words:
                        if len(qw) >= 4 and qw in full_content_words:
                            base_relevance = min(1.0, base_relevance + 0.15)
                            break

                token.relevance = base_relevance

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def build_context(self, query: str) -> Tuple[str, Dict]:
        """
        Build context for LLM injection using metadata loops.

        High relevance (>0.45): Inject FULL content
        Medium relevance (0.20-0.45): Inject metadata only
        Low relevance (<0.20): Skip
        """
        self.update_relevance(query)

        context_parts = []
        metadata = {
            "full_content_injected": 0, "metadata_injected": 0,
            "tokens_estimated": 0, "loops_accessed": []
        }
        full_content_tokens = 0
        metadata_tokens = 0
        high_relevance = []
        medium_relevance = []

        for loop in self.loops.values():
            for token in loop.tokens:
                if token.relevance >= LoopConfig.RELEVANCE_THRESHOLD_FULL:
                    high_relevance.append(token)
                elif token.relevance >= LoopConfig.RELEVANCE_THRESHOLD_META:
                    medium_relevance.append(token)

        high_relevance.sort(key=lambda t: t.relevance, reverse=True)
        medium_relevance.sort(key=lambda t: t.relevance, reverse=True)

        # Inject full content for highest relevance (budget limited)
        if high_relevance:
            context_parts.append("[RELEVANT MEMORIES]")
            for token in high_relevance:
                if full_content_tokens >= LoopConfig.FULL_CONTENT_TOKEN_BUDGET:
                    break
                full_content = self.content_cache.get(token.full_content_ref, "")
                if not full_content and token.full_content_ref:
                    full_content = self._mnemo_get(token.full_content_ref) or ""
                    if full_content:
                        self.content_cache[token.full_content_ref] = full_content
                if full_content:
                    context_parts.append(f"- {full_content}")
                    full_content_tokens += len(full_content.split())
                    metadata["full_content_injected"] += 1
                    token.access_count += 1

        # Inject metadata for medium relevance (budget limited)
        if medium_relevance:
            context_parts.append("\n[RELATED CONTEXT]")
            for token in medium_relevance:
                if metadata_tokens >= LoopConfig.METADATA_TOKEN_BUDGET:
                    break
                meta_str = token.to_context_string()
                context_parts.append(f"- {meta_str}")
                metadata_tokens += token.estimate_tokens()
                metadata["metadata_injected"] += 1

        accessed_loops = set()
        for token in high_relevance + medium_relevance:
            accessed_loops.add(token.category)
        metadata["loops_accessed"] = list(accessed_loops)
        metadata["tokens_estimated"] = full_content_tokens + metadata_tokens

        return "\n".join(context_parts) if context_parts else "", metadata

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get loop system statistics"""
        total_tokens = 0
        total_items = 0
        loop_stats = {}
        for loop_id, loop in self.loops.items():
            loop_stats[loop_id] = {
                "items": len(loop.tokens), "capacity": loop.capacity,
                "usage": f"{len(loop.tokens)/loop.capacity*100:.1f}%",
                "estimated_tokens": loop.estimate_tokens(),
                "total_cycles": loop.total_cycles
            }
            total_tokens += loop.estimate_tokens()
            total_items += len(loop.tokens)

        return {
            "total_loops": len(self.loops), "total_items": total_items,
            "total_metadata_tokens": total_tokens,
            "cached_full_content": len(self.content_cache),
            "config": {
                "metadata_budget": LoopConfig.METADATA_TOKEN_BUDGET,
                "full_content_budget": LoopConfig.FULL_CONTENT_TOKEN_BUDGET,
                "relevance_threshold_full": LoopConfig.RELEVANCE_THRESHOLD_FULL,
                "relevance_threshold_meta": LoopConfig.RELEVANCE_THRESHOLD_META
            },
            "loops": loop_stats
        }
