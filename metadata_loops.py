"""
Metadata Loop System - Token-Efficient Context Management

Replaces expensive context window summaries with compressed metadata loops.

From SLM Architecture:
- Token Memory Loop stores info in efficient token form (not full semantic)
- Loops cycle through context window based on relevance
- Default loop capacity: 100 items, expandable to 250
- Loop merging when similarity > 0.8
- Contains metadata + reference pointers to full content

TOKEN SAVINGS:
- Full content: ~50-200 tokens per memory
- Metadata loop: ~10-20 tokens per memory
- Savings: 70-90% reduction in context tokens!
"""

import os
import json
import re
import hashlib
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class LoopConfig:
    """Loop system configuration from SLM spec"""
    
    # Loop sizes
    DEFAULT_LOOP_CAPACITY = 100
    MAX_LOOP_CAPACITY = 250
    MICRO_LOOP_SIZE = 15  # For focused info clusters
    
    # Thresholds (adjusted for keyword matching)
    LOOP_MERGE_THRESHOLD = 0.80  # Merge loops when similarity exceeds
    RELEVANCE_THRESHOLD_FULL = 0.45  # Inject full content above this (was 0.60)
    RELEVANCE_THRESHOLD_META = 0.20  # Inject metadata above this (was 0.30)
    
    # Token budgets
    METADATA_TOKEN_BUDGET = 1000    # Max tokens for metadata context (was 500)
    FULL_CONTENT_TOKEN_BUDGET = 3000  # Max tokens for full content (was 1500)
    TOTAL_CONTEXT_BUDGET = 4000     # Total context injection limit (was 2000)
    
    # Compression
    MAX_KEYWORDS = 5
    MAX_SUMMARY_WORDS = 15
    
    # Cycling
    CYCLE_DECAY_RATE = 0.9  # Priority decay per cycle


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MetadataToken:
    """
    Compressed representation of a memory.
    Uses ~10-20 tokens instead of 50-200 for full content.
    """
    id: str
    keywords: List[str]  # Max 5 keywords
    category: str
    summary: str  # Max 15 words
    importance: float
    relevance: float = 0.0
    
    # Tracking
    cycle_count: int = 0
    last_cycled: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # Reference to full content
    full_content_ref: str = ""  # Pointer to Mnemo memory
    
    def to_context_string(self) -> str:
        """Convert to minimal context string (~15 tokens)"""
        kw_str = ", ".join(self.keywords[:3])
        return f"[{self.category}] {self.summary} ({kw_str})"
    
    def estimate_tokens(self) -> int:
        """Estimate token count"""
        text = self.to_context_string()
        return len(text.split()) + 5  # Rough estimate
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "keywords": self.keywords,
            "category": self.category,
            "summary": self.summary,
            "importance": self.importance,
            "relevance": self.relevance,
            "cycle_count": self.cycle_count,
            "access_count": self.access_count,
            "full_content_ref": self.full_content_ref
        }


@dataclass
class MetadataLoop:
    """
    A loop of related metadata tokens that cycle through context.
    """
    id: str
    name: str
    category: str
    tokens: List[MetadataToken] = field(default_factory=list)
    
    # Loop state
    current_position: int = 0
    capacity: int = LoopConfig.DEFAULT_LOOP_CAPACITY
    priority: float = 0.5
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    total_cycles: int = 0
    
    def add_token(self, token: MetadataToken) -> bool:
        """Add token to loop, return False if at capacity"""
        if len(self.tokens) >= self.capacity:
            return False
        self.tokens.append(token)
        return True
    
    def cycle(self, steps: int = 1) -> List[MetadataToken]:
        """Cycle through loop, return visible tokens"""
        if not self.tokens:
            return []
        
        self.total_cycles += 1
        self.current_position = (self.current_position + steps) % len(self.tokens)
        
        # Return tokens in current window (5 items)
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
        """Get most relevant tokens"""
        sorted_tokens = sorted(self.tokens, key=lambda t: t.relevance, reverse=True)
        return sorted_tokens[:top_k]
    
    def estimate_tokens(self) -> int:
        """Total tokens for all metadata"""
        return sum(t.estimate_tokens() for t in self.tokens)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "token_count": len(self.tokens),
            "capacity": self.capacity,
            "priority": self.priority,
            "total_cycles": self.total_cycles
        }


# =============================================================================
# METADATA EXTRACTOR
# =============================================================================

class MetadataExtractor:
    """
    Extracts compressed metadata from full content.
    Uses GPT-4o for smart extraction or falls back to simple extraction.
    """
    
    def __init__(self, openrouter_key: str = None):
        self.openrouter_key = openrouter_key
    
    def extract_simple(self, content: str, category: str = "general") -> MetadataToken:
        """
        Simple extraction without API call.
        Fast but less accurate.
        """
        # Extract keywords (simple word frequency)
        words = content.lower().split()
        # Filter common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                    'by', 'from', 'as', 'into', 'through', 'during', 'before',
                    'after', 'above', 'below', 'between', 'under', 'again',
                    'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'so',
                    'yet', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                    'such', 'no', 'not', 'only', 'own', 'same', 'than', 'too',
                    'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where',
                    'why', 'how', 'all', 'any', 'this', 'that', 'these', 'those'}
        
        # Also remove bracketed tags and punctuation
        filtered = []
        for w in words:
            # Clean word
            clean = re.sub(r'[\[\]\(\)\{\}:,\.!?"]', '', w)
            if clean and clean not in stopwords and len(clean) > 2:
                filtered.append(clean)
        
        # Count frequency
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        
        # Top keywords
        keywords = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:LoopConfig.MAX_KEYWORDS]
        
        # Simple summary (first N words)
        summary_words = content.split()[:LoopConfig.MAX_SUMMARY_WORDS]
        summary = " ".join(summary_words)
        if len(content.split()) > LoopConfig.MAX_SUMMARY_WORDS:
            summary += "..."
        
        # Generate ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return MetadataToken(
            id=f"meta_{content_hash}",
            keywords=keywords,
            category=category,
            summary=summary,
            importance=0.5,
            full_content_ref=content_hash
        )
    
    def extract_smart(self, content: str, category: str = "general") -> MetadataToken:
        """
        Smart extraction using GPT-4o and Native JSON Mode.
        More accurate but costs ~$0.001 per extraction.
        """
        if not self.openrouter_key:
            return self.extract_simple(content, category)
        
        prompt = f"""Extract metadata from this text.

TEXT: {content[:4000]}

Return ONLY a JSON object with this exact structure:
{{
  "keywords": ["max 5 key terms"],
  "summary": "max 15 word summary",
  "importance": 0.8
}}"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-2024-11-20",  # Nov 2024 version - $2.5/M tokens
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 150,
                    "response_format": {"type": "json_object"}  # NATIVE JSON MODE
                },
                timeout=10
            )
            
            if response.status_code != 200:
                return self.extract_simple(content, category)
            
            data = response.json()
            raw = data["choices"][0]["message"]["content"]
            
            # Safe JSON parsing without regex
            parsed = json.loads(raw)
            
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            
            return MetadataToken(
                id=f"meta_{content_hash}",
                keywords=parsed.get("keywords", [])[:LoopConfig.MAX_KEYWORDS],
                category=category,
                summary=parsed.get("summary", content[:50])[:100],
                importance=float(parsed.get("importance", 0.5)),
                full_content_ref=content_hash
            )
            
        except Exception as e:
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
    
    def __init__(
        self,
        openrouter_key: str = None,
        hf_key: str = None,
        mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space"
    ):
        self.openrouter_key = openrouter_key
        self.hf_key = hf_key
        self.mnemo_url = mnemo_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
        
        self.extractor = MetadataExtractor(openrouter_key)
        
        # Loops by category
        self.loops: Dict[str, MetadataLoop] = {}
        
        # Full content cache (for high-relevance injection)
        self.content_cache: Dict[str, str] = {}
        self._MAX_CACHE_SIZE = 200  # Prevent unbounded growth
        
        # Initialize default loops
        for category in ["character", "plot", "setting", "theme", "style", "fact", "general"]:
            self.loops[category] = MetadataLoop(
                id=f"loop_{category}",
                name=category.title(),
                category=category
            )
    
    # =========================================================================
    # MNEMO INTEGRATION
    # =========================================================================
    
    def _mnemo_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search Mnemo for relevant memories"""
        try:
            response = requests.post(
                f"{self.mnemo_url}/search",
                headers=self.headers,
                json={"query": query, "limit": limit},
                timeout=10
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
                headers=self.headers,
                timeout=10
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
                headers=self.headers,
                timeout=10
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
        """
        Load memories from Mnemo and convert to metadata tokens.
        """
        memories = self._mnemo_list()
        
        for mem in memories:
            content = mem.get("content", "")
            mem_id = mem.get("id", "")
            
            # Detect category from content
            category = "general"
            content_lower = content.lower()
            if "[character]" in content_lower:
                category = "character"
            elif "[plot]" in content_lower:
                category = "plot"
            elif "[setting]" in content_lower:
                category = "setting"
            elif "[theme]" in content_lower:
                category = "theme"
            elif "[style]" in content_lower:
                category = "style"
            elif "[fact]" in content_lower:
                category = "fact"
            
            # Extract metadata
            if use_smart_extraction:
                token = self.extractor.extract_smart(content, category)
            else:
                token = self.extractor.extract_simple(content, category)
            
            token.full_content_ref = mem_id
            
            # Add to appropriate loop
            if category in self.loops:
                self.loops[category].add_token(token)
            
            # Cache full content
            self.content_cache[mem_id] = content
        
        # Trim cache if too large
        self._trim_cache()
    
    def _trim_cache(self):
        """Prevent unbounded cache growth"""
        if len(self.content_cache) > self._MAX_CACHE_SIZE:
            keys = list(self.content_cache.keys())
            for key in keys[:len(keys) // 2]:
                del self.content_cache[key]
    
    def add_to_loop(
        self,
        content: str,
        category: str = "general",
        memory_id: str = None,
        use_smart: bool = False
    ) -> MetadataToken:
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
                id=f"loop_{category}",
                name=category.title(),
                category=category
            )
        
        self.loops[category].add_token(token)
        return token
    
    def update_relevance(self, query: str):
        """
        Update relevance scores for all tokens based on query.
        Uses keyword matching + full content matching + character-aware boosting.
        """
        # Clean query and remove stopwords for better matching
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'shall', 'to', 'of',
                     'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'about', 'like', 'through', 'after', 'before', 'between',
                     'out', 'up', 'down', 'and', 'but', 'or', 'not', 'no', 'so',
                     'if', 'then', 'than', 'too', 'very', 'just', 'also', 'how',
                     'all', 'each', 'every', 'both', 'few', 'more', 'most', 'some',
                     'such', 'only', 'own', 'same', 'that', 'this', 'these', 'those',
                     'it', 'its', 'he', 'she', 'his', 'her', 'they', 'them', 'their',
                     'me', 'my', 'we', 'our', 'you', 'your', 'who', 'what', 'which',
                     'when', 'where', 'why', 'tell', 'show', 'give', 'get', 'find',
                     'retrieve', 'everything', 'anything', 'something', 'please',
                     'know', 'remember', 'recall'}
        
        raw_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
        query_words = raw_words - stopwords
        
        # If all words were stopwords, fall back to raw (minus very common ones)
        if not query_words:
            query_words = raw_words - {'the', 'a', 'an', 'is', 'are', 'it', 'to', 'of'}
        
        for loop in self.loops.values():
            for token in loop.tokens:
                # === Source 1: Keywords + category + summary ===
                token_words = set(kw.lower() for kw in token.keywords)
                token_words.add(token.category.lower())
                summary_words = set(re.sub(r'[^\w\s]', '', token.summary.lower()).split())
                token_words.update(summary_words)
                
                # === Source 2: Full content (if cached) ===
                full_content_words = set()
                if token.full_content_ref and token.full_content_ref in self.content_cache:
                    full_text = self.content_cache[token.full_content_ref].lower()
                    full_content_words = set(re.sub(r'[^\w\s]', '', full_text).split()) - stopwords
                
                all_matchable = token_words | full_content_words
                
                # === Count matches (including partial/substring) ===
                matches = 0
                for qw in query_words:
                    if len(qw) < 3:
                        continue
                    for tw in all_matchable:
                        if len(tw) < 3:
                            continue
                        # Exact match or substring (both directions)
                        if qw == tw or (len(qw) >= 4 and qw in tw) or (len(tw) >= 4 and tw in qw):
                            matches += 1
                            break
                
                # === Calculate base relevance ===
                if query_words:
                    base_relevance = min(1.0, matches / max(len(query_words), 1))
                else:
                    base_relevance = 0.0
                
                # === Boost: if ANY query word is a character/proper noun in full content ===
                if full_content_words:
                    for qw in query_words:
                        if len(qw) >= 4 and qw in full_content_words:
                            base_relevance = min(1.0, base_relevance + 0.15)
                            break
                
                token.relevance = base_relevance
    
    # =========================================================================
    # CONTEXT BUILDING (THE KEY COST SAVER)
    # =========================================================================
    
    def build_context(self, query: str) -> Tuple[str, Dict]:
        """
        Build context for LLM injection using metadata loops.
        
        STRATEGY:
        1. Update relevance scores based on query
        2. High relevance (>0.85): Inject FULL content
        3. Medium relevance (0.50-0.85): Inject metadata only
        4. Low relevance (<0.50): Skip
        
        Returns: (context_string, metadata)
        """
        # Update relevance scores
        self.update_relevance(query)
        
        context_parts = []
        metadata = {
            "full_content_injected": 0,
            "metadata_injected": 0,
            "tokens_estimated": 0,
            "loops_accessed": []
        }
        
        full_content_tokens = 0
        metadata_tokens = 0
        
        # Collect high-relevance items for full content injection
        high_relevance = []
        medium_relevance = []
        
        for loop in self.loops.values():
            for token in loop.tokens:
                if token.relevance >= LoopConfig.RELEVANCE_THRESHOLD_FULL:
                    high_relevance.append(token)
                elif token.relevance >= LoopConfig.RELEVANCE_THRESHOLD_META:
                    medium_relevance.append(token)
        
        # Sort by relevance
        high_relevance.sort(key=lambda t: t.relevance, reverse=True)
        medium_relevance.sort(key=lambda t: t.relevance, reverse=True)
        
        # Inject full content for highest relevance (budget limited)
        if high_relevance:
            context_parts.append("[RELEVANT MEMORIES]")
            
            for token in high_relevance:
                if full_content_tokens >= LoopConfig.FULL_CONTENT_TOKEN_BUDGET:
                    break
                
                # Get full content
                full_content = self.content_cache.get(token.full_content_ref, "")
                if not full_content and token.full_content_ref:
                    full_content = self._mnemo_get(token.full_content_ref) or ""
                    if full_content:
                        self.content_cache[token.full_content_ref] = full_content
                
                if full_content:
                    context_parts.append(f"• {full_content}")
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
                context_parts.append(f"• {meta_str}")
                metadata_tokens += token.estimate_tokens()
                metadata["metadata_injected"] += 1
        
        # Add loop cycling info
        accessed_loops = set()
        for token in high_relevance + medium_relevance:
            accessed_loops.add(token.category)
        metadata["loops_accessed"] = list(accessed_loops)
        
        # Calculate total tokens
        metadata["tokens_estimated"] = full_content_tokens + metadata_tokens
        
        context = "\n".join(context_parts) if context_parts else ""
        
        return context, metadata
    
    def build_minimal_context(self, query: str, max_tokens: int = 200) -> str:
        """
        Build ultra-minimal context for very tight token budgets.
        Only injects top 3 most relevant metadata items.
        """
        self.update_relevance(query)
        
        # Collect all tokens
        all_tokens = []
        for loop in self.loops.values():
            all_tokens.extend(loop.tokens)
        
        # Sort by relevance
        all_tokens.sort(key=lambda t: t.relevance, reverse=True)
        
        # Take top items
        context_parts = []
        token_count = 0
        
        for token in all_tokens[:5]:
            if token.relevance < 0.3:
                break
            
            meta_str = token.to_context_string()
            est_tokens = token.estimate_tokens()
            
            if token_count + est_tokens > max_tokens:
                break
            
            context_parts.append(meta_str)
            token_count += est_tokens
        
        return " | ".join(context_parts) if context_parts else ""
    
    # =========================================================================
    # LOOP MERGING
    # =========================================================================
    
    def calculate_loop_similarity(self, loop1: MetadataLoop, loop2: MetadataLoop) -> float:
        """Calculate similarity between two loops based on keywords"""
        keywords1 = set()
        for token in loop1.tokens:
            keywords1.update(token.keywords)
        
        keywords2 = set()
        for token in loop2.tokens:
            keywords2.update(token.keywords)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        overlap = len(keywords1 & keywords2)
        total = len(keywords1 | keywords2)
        
        return overlap / total if total > 0 else 0.0
    
    def merge_loops(self, loop1_id: str, loop2_id: str) -> Optional[MetadataLoop]:
        """Merge two loops if similarity > threshold"""
        if loop1_id not in self.loops or loop2_id not in self.loops:
            return None
        
        loop1 = self.loops[loop1_id]
        loop2 = self.loops[loop2_id]
        
        similarity = self.calculate_loop_similarity(loop1, loop2)
        
        if similarity < LoopConfig.LOOP_MERGE_THRESHOLD:
            return None
        
        # Create merged loop
        merged = MetadataLoop(
            id=f"loop_{loop1.category}_{loop2.category}",
            name=f"{loop1.name}+{loop2.name}",
            category=loop1.category,
            capacity=min(loop1.capacity + loop2.capacity, LoopConfig.MAX_LOOP_CAPACITY)
        )
        
        # Add all tokens
        for token in loop1.tokens:
            merged.add_token(token)
        for token in loop2.tokens:
            merged.add_token(token)
        
        # Remove old loops, add merged
        del self.loops[loop1_id]
        del self.loops[loop2_id]
        self.loops[merged.id] = merged
        
        return merged
    
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
                "items": len(loop.tokens),
                "capacity": loop.capacity,
                "usage": f"{len(loop.tokens)/loop.capacity*100:.1f}%",
                "estimated_tokens": loop.estimate_tokens(),
                "total_cycles": loop.total_cycles
            }
            total_tokens += loop.estimate_tokens()
            total_items += len(loop.tokens)
        
        return {
            "total_loops": len(self.loops),
            "total_items": total_items,
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
    
    def estimate_savings(self, query: str) -> Dict:
        """
        Estimate token savings compared to full context injection.
        """
        # Calculate full content cost
        full_content_tokens = 0
        for content in self.content_cache.values():
            full_content_tokens += len(content.split())
        
        # Calculate loop method cost
        context, meta = self.build_context(query)
        loop_tokens = meta["tokens_estimated"]
        
        savings = full_content_tokens - loop_tokens if full_content_tokens > 0 else 0
        savings_pct = (savings / full_content_tokens * 100) if full_content_tokens > 0 else 0
        
        return {
            "full_context_tokens": full_content_tokens,
            "loop_method_tokens": loop_tokens,
            "tokens_saved": savings,
            "savings_percentage": f"{savings_pct:.1f}%",
            "cost_saved_per_message": f"${savings * 2.5 / 1_000_000:.6f}"  # Input token cost
        }


# =============================================================================
# INTEGRATION WITH EXISTING MEMORY SYSTEM
# =============================================================================

class LoopEnabledMemoryManager:
    """
    Memory manager that uses metadata loops instead of full context summaries.
    Drop-in replacement for MnemoMemoryManager.
    """
    
    def __init__(
        self,
        openrouter_key: str,
        hf_key: str,
        mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space"
    ):
        self.openrouter_key = openrouter_key
        self.hf_key = hf_key
        self.mnemo_url = mnemo_url
        
        # Initialize loop manager
        self.loop_manager = LoopManager(
            openrouter_key=openrouter_key,
            hf_key=hf_key,
            mnemo_url=mnemo_url
        )
        
        # Load existing memories into loops
        self.loop_manager.load_from_mnemo(use_smart_extraction=False)
        
        # Conversation history (minimal)
        self.conversation_history: List[Dict] = []
        self.max_history = 4  # Keep last 4 turns (8 messages)
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim to max
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def build_context(self, query: str, system_prompt: str) -> Tuple[List[Dict], Dict]:
        """
        Build messages array with loop-based context injection.
        
        Returns: (messages_list, metadata)
        """
        # Get context from loops
        loop_context, meta = self.loop_manager.build_context(query)
        
        # Build system message with context
        if loop_context:
            enhanced_system = f"""{system_prompt}

{loop_context}"""
        else:
            enhanced_system = system_prompt
        
        # Build messages
        messages = [{"role": "system", "content": enhanced_system}]
        
        # Add recent conversation history
        for msg in self.conversation_history[-self.max_history * 2:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages, meta
    
    def process_turn(self, user_message: str, assistant_response: str):
        """Process a conversation turn"""
        self.add_to_history("user", user_message)
        self.add_to_history("assistant", assistant_response)
    
    def add_memory(self, content: str, category: str = "general") -> bool:
        """Add new memory to loops and Mnemo"""
        # Add to Mnemo
        try:
            response = requests.post(
                f"{self.mnemo_url}/add",
                headers=self.loop_manager.headers,
                json={
                    "content": f"[{category.upper()}] {content}",
                    "metadata": {"category": category}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                mem_id = response.json().get("memory_id")
                
                # Add to loop
                self.loop_manager.add_to_loop(
                    content=content,
                    category=category,
                    memory_id=mem_id,
                    use_smart=False
                )
                return True
            return False
        except Exception:
            return False
    
    def get_stats(self) -> Dict:
        """Get comprehensive stats"""
        loop_stats = self.loop_manager.get_stats()
        
        return {
            "loop_system": loop_stats,
            "conversation_history_length": len(self.conversation_history),
            "max_history": self.max_history
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("METADATA LOOP SYSTEM - TEST")
    print("=" * 60)
    
    HF_KEY = os.environ.get("HF_KEY", "")
    OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
    
    # Test metadata extraction
    print("\n--- Testing Metadata Extractor ---")
    extractor = MetadataExtractor()
    
    test_content = "[CHARACTER] Dr. Marcus Webb is a professor of neuroscience who manipulates his students for corporate research. He fears irrelevance as younger colleagues overtake him."
    
    token = extractor.extract_simple(test_content, "character")
    print(f"Keywords: {token.keywords}")
    print(f"Summary: {token.summary}")
    print(f"Context string: {token.to_context_string()}")
    print(f"Estimated tokens: {token.estimate_tokens()}")
    
    # Test loop manager
    print("\n--- Testing Loop Manager ---")
    manager = LoopManager(hf_key=HF_KEY, openrouter_key=OPENROUTER_KEY)
    
    # Load from Mnemo
    print("Loading memories from Mnemo...")
    manager.load_from_mnemo(use_smart_extraction=False)
    
    # Get stats
    stats = manager.get_stats()
    print(f"Total loops: {stats['total_loops']}")
    print(f"Total items: {stats['total_items']}")
    print(f"Total metadata tokens: {stats['total_metadata_tokens']}")
    
    # Test context building
    print("\n--- Testing Context Building ---")
    query = "Tell me about the mentor and protagonist's relationship"
    
    context, meta = manager.build_context(query)
    print(f"Full content injected: {meta['full_content_injected']}")
    print(f"Metadata injected: {meta['metadata_injected']}")
    print(f"Estimated tokens: {meta['tokens_estimated']}")
    print(f"Loops accessed: {meta['loops_accessed']}")
    
    # Estimate savings
    print("\n--- Token Savings Estimate ---")
    savings = manager.estimate_savings(query)
    print(f"Full context would use: {savings['full_context_tokens']} tokens")
    print(f"Loop method uses: {savings['loop_method_tokens']} tokens")
    print(f"Savings: {savings['savings_percentage']}")
    print(f"Cost saved per message: {savings['cost_saved_per_message']}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
