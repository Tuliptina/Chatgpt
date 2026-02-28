"""
Context Engine - Deep Understanding, Style Matching, and Memory Degradation

Three key features:
1. DEEP CONTEXT: Understand meaning, not just facts
2. STYLE MATCHING: Extract prose samples and writing voice
3. DEGRADATION: Gentle decay and pruning of unused memories

The problem this solves:
- Surface: [CHARACTER] A character was held captive
- AI thinks: Literal chains, basement imprisonment
- Reality: Could be psychological manipulation, addiction, coercion

Solution: Context layers + Neural link clustering + Style samples
"""

import os
import json
import re
import requests
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ContextEngineConfig:
    """Configuration for context engine"""
    
    # Degradation settings (gentle)
    DECAY_RATE = 0.10  # 10% decay per cycle
    DECAY_INTERVAL_HOURS = 24  # Apply decay every 24 hours
    MIN_RELEVANCE_THRESHOLD = 0.05  # Below this = candidate for pruning
    MIN_ACCESS_FOR_KEEP = 2  # Items accessed < this AND low relevance = prune
    
    # Pruning settings
    PRUNE_AFTER_DAYS = 30  # Consider pruning items older than this
    MAX_ITEMS_PER_CATEGORY = 150  # Soft cap before aggressive pruning
    
    # Duplicate merging
    MERGE_SIMILARITY_THRESHOLD = 0.80  # Merge if >80% similar
    
    # Context extraction
    MAX_CONTEXT_ITEMS = 5  # Max context items per query
    NEURAL_LINK_DEPTH = 2  # How many hops to follow neural links
    
    # Style extraction
    MAX_PROSE_SAMPLES = 3  # Number of prose samples to keep
    MAX_SAMPLE_LENGTH = 300  # Max chars per sample


# =============================================================================
# DEGRADATION SYSTEM
# =============================================================================

class MemoryDegradation:
    """
    Gentle degradation system to prevent memory bloat.
    
    Strategy:
    - Decay: Reduce relevance of unused items slowly
    - Prune: Remove truly stale items (low relevance + low access + old)
    - Merge: Combine duplicates to save space
    """
    
    def __init__(self, hf_key: str, mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space"):
        self.hf_key = hf_key
        self.mnemo_url = mnemo_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
        self.config = ContextEngineConfig()
        self.last_decay_time = None
        
    def _list_memories(self) -> List[Dict]:
        """Get all memories from Mnemo"""
        try:
            response = requests.get(
                f"{self.mnemo_url}/list",
                headers=self.headers,
                timeout=15
            )
            if response.status_code == 200:
                return response.json().get("memories", [])
            return []
        except Exception:
            return []
    
    def _delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from Mnemo"""
        try:
            response = requests.delete(
                f"{self.mnemo_url}/delete/{memory_id}",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def _update_memory_metadata(self, memory_id: str, metadata: Dict) -> bool:
        """Update memory metadata (for tracking access, decay, etc.)"""
        # Note: This depends on Mnemo supporting metadata updates
        # For now, we'll track locally
        return True
    
    def apply_decay(self, loop_manager) -> Dict:
        """
        Apply gentle decay to unused items.
        Call this periodically (e.g., every session or daily).
        """
        now = datetime.now()
        
        # Check if enough time has passed
        if self.last_decay_time:
            hours_since = (now - self.last_decay_time).total_seconds() / 3600
            if hours_since < self.config.DECAY_INTERVAL_HOURS:
                return {"skipped": True, "reason": f"Only {hours_since:.1f}h since last decay"}
        
        decayed_count = 0
        
        for loop in loop_manager.loops.values():
            for token in loop.tokens:
                # Only decay items that haven't been accessed recently
                if token.access_count == 0:
                    # Apply decay
                    old_relevance = token.relevance
                    token.relevance *= (1 - self.config.DECAY_RATE)
                    token.importance *= (1 - self.config.DECAY_RATE * 0.5)  # Slower importance decay
                    
                    if old_relevance != token.relevance:
                        decayed_count += 1
        
        self.last_decay_time = now
        
        return {
            "decayed_items": decayed_count,
            "decay_rate": self.config.DECAY_RATE,
            "timestamp": now.isoformat()
        }
    
    def identify_prune_candidates(self, loop_manager) -> List[Dict]:
        """
        Identify items that should be pruned.
        Criteria: Low relevance + Low access + Old
        """
        candidates = []
        now = datetime.now()
        
        for loop in loop_manager.loops.values():
            for token in loop.tokens:
                # Check pruning criteria
                low_relevance = token.relevance < self.config.MIN_RELEVANCE_THRESHOLD
                low_access = token.access_count < self.config.MIN_ACCESS_FOR_KEEP
                
                # Check age
                age_days = 0
                if hasattr(token, 'last_cycled'):
                    age_days = (now - token.last_cycled).days
                
                old_enough = age_days > self.config.PRUNE_AFTER_DAYS
                
                # Must meet ALL criteria for pruning (conservative)
                if low_relevance and low_access and old_enough:
                    candidates.append({
                        "id": token.id,
                        "category": token.category,
                        "summary": token.summary,
                        "relevance": token.relevance,
                        "access_count": token.access_count,
                        "age_days": age_days,
                        "reason": "Low relevance + Low access + Old"
                    })
                    
        return candidates
    
    def prune_memories(self, loop_manager, dry_run: bool = True) -> Dict:
        """
        Prune stale memories.
        Set dry_run=False to actually delete.
        """
        candidates = self.identify_prune_candidates(loop_manager)
        
        if dry_run:
            return {
                "dry_run": True,
                "candidates": len(candidates),
                "items": candidates[:10]  # Show first 10
            }
        
        pruned = 0
        failed = 0
        
        for item in candidates:
            memory_id = item.get("id", "")
            if memory_id and self._delete_memory(memory_id):
                pruned += 1
                # Also remove from loop
                for loop in loop_manager.loops.values():
                    loop.tokens = [t for t in loop.tokens if t.id != memory_id]
            else:
                failed += 1
        
        return {
            "dry_run": False,
            "pruned": pruned,
            "failed": failed,
            "total_candidates": len(candidates)
        }
    
    def find_duplicates(self, loop_manager) -> List[Tuple[Dict, Dict, float]]:
        """
        Find duplicate/similar memories that can be merged.
        Returns list of (item1, item2, similarity) tuples.
        """
        duplicates = []
        
        for loop in loop_manager.loops.values():
            tokens = loop.tokens
            
            for i, token1 in enumerate(tokens):
                for token2 in tokens[i+1:]:
                    similarity = self._calculate_similarity(token1, token2)
                    
                    if similarity >= self.config.MERGE_SIMILARITY_THRESHOLD:
                        duplicates.append((
                            {"id": token1.id, "summary": token1.summary, "keywords": token1.keywords},
                            {"id": token2.id, "summary": token2.summary, "keywords": token2.keywords},
                            similarity
                        ))
        
        return duplicates
    
    def _calculate_similarity(self, token1, token2) -> float:
        """Calculate keyword-based similarity between two tokens"""
        kw1 = set(k.lower() for k in token1.keywords)
        kw2 = set(k.lower() for k in token2.keywords)
        
        if not kw1 or not kw2:
            return 0.0
        
        overlap = len(kw1 & kw2)
        total = len(kw1 | kw2)
        
        return overlap / total if total > 0 else 0.0
    
    def get_health_report(self, loop_manager) -> Dict:
        """
        Generate a health report of the memory system.
        """
        total_items = 0
        low_relevance = 0
        never_accessed = 0
        high_usage_loops = []
        
        for loop_name, loop in loop_manager.loops.items():
            item_count = len(loop.tokens)
            total_items += item_count
            
            for token in loop.tokens:
                if token.relevance < 0.1:
                    low_relevance += 1
                if token.access_count == 0:
                    never_accessed += 1
            
            usage_pct = (item_count / loop.capacity) * 100
            if usage_pct > 70:
                high_usage_loops.append({
                    "name": loop_name,
                    "usage": f"{usage_pct:.1f}%",
                    "items": item_count
                })
        
        prune_candidates = len(self.identify_prune_candidates(loop_manager))
        duplicates = len(self.find_duplicates(loop_manager))
        
        return {
            "total_items": total_items,
            "low_relevance_items": low_relevance,
            "never_accessed_items": never_accessed,
            "prune_candidates": prune_candidates,
            "potential_duplicates": duplicates,
            "high_usage_loops": high_usage_loops,
            "health_score": self._calculate_health_score(
                total_items, low_relevance, never_accessed, prune_candidates
            )
        }
    
    def _calculate_health_score(self, total, low_rel, never_acc, prune_cand) -> str:
        """Calculate overall health score"""
        if total == 0:
            return "EMPTY"
        
        issues = 0
        if low_rel / total > 0.3:
            issues += 1
        if never_acc / total > 0.5:
            issues += 1
        if prune_cand > 20:
            issues += 1
        
        if issues == 0:
            return "HEALTHY ✅"
        elif issues == 1:
            return "FAIR ⚠️"
        else:
            return "NEEDS ATTENTION ❌"


# =============================================================================
# DEEP CONTEXT UNDERSTANDING
# =============================================================================

class DeepContextEngine:
    """
    Builds deep contextual understanding using neural links.
    
    Instead of just returning "[CHARACTER] A character was captive",
    returns a context cluster:
    - The fact
    - What it MEANS (e.g. psychological, not physical)
    - Related context (connected characters, plot threads)
    - What it's NOT (common misinterpretations)
    """
    
    def __init__(self, hf_key: str, openrouter_key: str = None, 
                 mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space"):
        self.hf_key = hf_key
        self.openrouter_key = openrouter_key
        self.mnemo_url = mnemo_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
        self.config = ContextEngineConfig()
        
        # Context cache
        self.context_cache: Dict[str, Dict] = {}
    
    def _search_mnemo(self, query: str, limit: int = 10) -> List[Dict]:
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
    
    def _get_memory_with_links(self, memory_id: str) -> Optional[Dict]:
        """Get a memory with its neural links"""
        try:
            response = requests.get(
                f"{self.mnemo_url}/get/{memory_id}",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def _list_all_memories(self) -> List[Dict]:
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
    
    def build_context_cluster(self, query: str) -> Dict:
        """
        Build a deep context cluster for a query.
        
        Returns:
        {
            "primary_facts": [...],      # Direct matches
            "related_context": [...],    # Via neural links
            "interpretation": "...",     # What this means
            "clarifications": [...]      # Common misunderstandings
        }
        """
        # Get primary results
        primary_results = self._search_mnemo(query, limit=5)
        
        if not primary_results:
            return {
                "primary_facts": [],
                "related_context": [],
                "interpretation": None,
                "clarifications": []
            }
        
        primary_facts = []
        related_context = []
        all_memory_ids = set()
        
        # Process primary results
        for result in primary_results:
            content = result.get("content", "")
            mem_id = result.get("id", "")
            score = result.get("score", 0)
            
            primary_facts.append({
                "content": content,
                "id": mem_id,
                "relevance": score
            })
            all_memory_ids.add(mem_id)
        
        # Follow neural links to get related context
        all_memories = self._list_all_memories()
        memory_map = {m.get("id"): m for m in all_memories}
        
        for primary in primary_facts[:3]:  # Top 3 primary results
            mem_id = primary.get("id")
            if mem_id in memory_map:
                mem = memory_map[mem_id]
                links = mem.get("links", [])
                
                for link in links[:5]:  # Top 5 links per memory
                    linked_id = link.get("target_id", "")
                    if linked_id and linked_id not in all_memory_ids:
                        if linked_id in memory_map:
                            linked_mem = memory_map[linked_id]
                            related_context.append({
                                "content": linked_mem.get("content", ""),
                                "id": linked_id,
                                "link_strength": link.get("strength", 0),
                                "link_type": link.get("type", "related")
                            })
                            all_memory_ids.add(linked_id)
        
        # Look for [CONTEXT] entries specifically
        context_entries = []
        for mem in all_memories:
            content = mem.get("content", "")
            if content.startswith("[CONTEXT]"):
                # Check if relevant to query
                query_words = set(query.lower().split())
                content_words = set(content.lower().split())
                if query_words & content_words:
                    context_entries.append(content)
        
        return {
            "primary_facts": primary_facts,
            "related_context": related_context[:self.config.MAX_CONTEXT_ITEMS],
            "context_entries": context_entries,
            "interpretation": self._generate_interpretation(primary_facts, related_context),
            "query": query
        }
    
    def _generate_interpretation(self, facts: List[Dict], context: List[Dict]) -> Optional[str]:
        """Generate interpretation from facts and context"""
        if not facts:
            return None
        
        # Simple interpretation: combine key information
        all_content = []
        for f in facts[:3]:
            all_content.append(f.get("content", ""))
        for c in context[:3]:
            all_content.append(c.get("content", ""))
        
        if not all_content:
            return None
        
        # Return as context summary
        return " | ".join(c[:100] for c in all_content[:5])
    
    def extract_deep_context(self, content: str, openrouter_key: str = None) -> List[Dict]:
        """
        Extract deep context from content using GPT-4o.
        Returns both facts AND context/meaning.
        """
        api_key = openrouter_key or self.openrouter_key
        if not api_key:
            return []
        
        prompt = f"""Analyze this content and extract BOTH facts AND their deeper meaning/context.

CONTENT:
{content[:3000]}

Extract in these categories:

1. FACTS: What literally happens or is stated
2. CONTEXT: What it MEANS (psychological, thematic, symbolic significance)
3. CLARIFICATIONS: Common misinterpretations to avoid
4. RELATIONSHIPS: How elements connect to each other

Return JSON array:
[
  {{"category": "CHARACTER", "content": "The protagonist was trapped in an abusive dynamic with their mentor"}},
  {{"category": "CONTEXT", "content": "The 'captivity' was psychological manipulation through dependency, NOT literal imprisonment. The mentor used emotional control as leverage."}},
  {{"category": "CLARIFICATION", "content": "The protagonist was NOT physically restrained. Their captivity was through emotional dependency and gaslighting."}},
  {{"category": "RELATIONSHIP", "content": "The protagonist's captivity connects to: the mentor as manipulator, substance dependency as control tool, the organization's methods"}}
]

Be thorough. Extract meaning, not just surface facts."""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-2024-11-20",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                
                # Parse JSON
                json_match = re.search(r'\[.*\]', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            return []
        except Exception as e:
            print(f"Context extraction error: {e}")
            return []
    
    def build_enriched_context(self, query: str, loop_manager=None) -> str:
        """
        Build enriched context for GPT-4o that includes:
        - Primary facts
        - Deep context/meaning
        - Related information via neural links
        - Clarifications
        """
        cluster = self.build_context_cluster(query)
        
        context_parts = []
        
        # Add primary facts
        if cluster["primary_facts"]:
            context_parts.append("[RELEVANT MEMORIES]")
            for fact in cluster["primary_facts"][:3]:
                context_parts.append(f"• {fact['content']}")
        
        # Add context entries (these explain meaning)
        if cluster.get("context_entries"):
            context_parts.append("\n[CONTEXT & MEANING]")
            for ctx in cluster["context_entries"][:3]:
                # Remove [CONTEXT] prefix for cleaner output
                clean = ctx.replace("[CONTEXT]", "").strip()
                context_parts.append(f"• {clean}")
        
        # Add related context from neural links
        if cluster["related_context"]:
            context_parts.append("\n[RELATED INFORMATION]")
            for related in cluster["related_context"][:3]:
                context_parts.append(f"• {related['content']}")
        
        return "\n".join(context_parts) if context_parts else ""


# =============================================================================
# STYLE EXTRACTION
# =============================================================================

class StyleExtractor:
    """
    Extracts writing style from uploaded content.
    
    Captures:
    - PROSE_SAMPLE: Actual paragraphs showing voice
    - DIALOGUE_SAMPLE: Character speech patterns
    - VOICE: POV, tense, tone, atmosphere
    - VOCABULARY: Period/character-specific words
    """
    
    def __init__(self, openrouter_key: str = None):
        self.openrouter_key = openrouter_key
        self.config = ContextEngineConfig()
    
    def extract_style(self, content: str, openrouter_key: str = None) -> List[Dict]:
        """
        Extract style elements from content.
        """
        api_key = openrouter_key or self.openrouter_key
        if not api_key:
            return self._extract_style_simple(content)
        
        return self._extract_style_smart(content, api_key)
    
    def _extract_style_simple(self, content: str) -> List[Dict]:
        """Simple extraction without API"""
        results = []
        
        # Extract prose samples (paragraphs with description)
        paragraphs = content.split('\n\n')
        descriptive_paragraphs = []
        
        for p in paragraphs:
            p = p.strip()
            # Look for descriptive prose (not dialogue-heavy)
            if len(p) > 100 and p.count('"') < 4:
                descriptive_paragraphs.append(p)
        
        # Take best prose samples
        for p in descriptive_paragraphs[:self.config.MAX_PROSE_SAMPLES]:
            sample = p[:self.config.MAX_SAMPLE_LENGTH]
            if len(p) > self.config.MAX_SAMPLE_LENGTH:
                sample += "..."
            results.append({
                "category": "PROSE_SAMPLE",
                "content": sample
            })
        
        # Extract dialogue samples
        dialogues = re.findall(r'"([^"]{20,150})"', content)
        for d in dialogues[:3]:
            results.append({
                "category": "DIALOGUE_SAMPLE", 
                "content": d
            })
        
        return results
    
    def _extract_style_smart(self, content: str, api_key: str) -> List[Dict]:
        """Smart extraction using GPT-4o"""
        
        prompt = f"""Analyze this text and extract STYLE elements to help match the writing voice.

TEXT:
{content[:4000]}

Extract:

1. PROSE_SAMPLE: 2-3 paragraphs that best show the author's descriptive voice (atmosphere, imagery, rhythm). Quote directly, max 300 chars each.

2. DIALOGUE_SAMPLE: 2-3 lines that show how characters speak (rhythm, vocabulary, subtext). Quote directly.

3. VOICE: Describe the narrative voice:
   - POV (first person, close third, omniscient)
   - Tense (past, present)
   - Tone (dark, humorous, melancholic, etc.)
   - Sentence style (short punchy, long flowing, mixed)
   - Atmosphere (gothic, romantic, gritty, etc.)

4. VOCABULARY: List 10-15 distinctive words/phrases the author uses (period-specific, character-specific, recurring imagery)

Return JSON array:
[
  {{"category": "PROSE_SAMPLE", "content": "The rain hammered the tin roof like a thousand tiny fists, and somewhere below, a door slammed shut..."}},
  {{"category": "DIALOGUE_SAMPLE", "content": "You're late. Again."}},
  {{"category": "VOICE", "content": "Close third-person POV, past tense, noir atmosphere, mix of short punchy dialogue and flowing descriptive passages, world-weary tone"}},
  {{"category": "VOCABULARY", "content": "List the actual distinctive words and phrases from the text"}}
]"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-2024-11-20",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                
                # Parse JSON
                json_match = re.search(r'\[.*\]', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            
            return self._extract_style_simple(content)
        except Exception as e:
            print(f"Style extraction error: {e}")
            return self._extract_style_simple(content)


# =============================================================================
# UNIFIED CONTEXT ENGINE
# =============================================================================

class ContextEngine:
    """
    Unified engine combining:
    - Deep context understanding
    - Style matching
    - Memory degradation
    """
    
    def __init__(self, hf_key: str, openrouter_key: str = None,
                 mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space"):
        self.hf_key = hf_key
        self.openrouter_key = openrouter_key
        self.mnemo_url = mnemo_url
        
        # Components
        self.degradation = MemoryDegradation(hf_key, mnemo_url)
        self.context = DeepContextEngine(hf_key, openrouter_key, mnemo_url)
        self.style = StyleExtractor(openrouter_key)
    
    def extract_from_file(self, content: str, filename: str, 
                          openrouter_key: str = None) -> List[Dict]:
        """
        Extract ALL types of memories from a file:
        - Facts (CHARACTER, PLOT, SETTING, etc.)
        - Context (meaning, relationships)
        - Style (prose samples, voice)
        """
        api_key = openrouter_key or self.openrouter_key
        all_memories = []
        
        # Extract deep context (facts + meaning)
        context_memories = self.context.extract_deep_context(content, api_key)
        all_memories.extend(context_memories)
        
        # Extract style
        style_memories = self.style.extract_style(content, api_key)
        all_memories.extend(style_memories)
        
        return all_memories
    
    def build_rich_context(self, query: str, loop_manager=None) -> Tuple[str, Dict]:
        """
        Build enriched context for a query.
        
        Returns:
        - context_string: Ready to inject into prompt
        - metadata: Stats about what was included
        """
        context_parts = []
        metadata = {
            "facts_included": 0,
            "context_included": 0,
            "style_included": 0,
            "neural_links_followed": 0
        }
        
        # Get deep context cluster
        cluster = self.context.build_context_cluster(query)
        
        # Add facts
        if cluster["primary_facts"]:
            for fact in cluster["primary_facts"][:3]:
                context_parts.append(fact["content"])
                metadata["facts_included"] += 1
        
        # Add context entries
        if cluster.get("context_entries"):
            for ctx in cluster["context_entries"][:2]:
                context_parts.append(ctx)
                metadata["context_included"] += 1
        
        # Add related context from neural links
        if cluster["related_context"]:
            for related in cluster["related_context"][:3]:
                context_parts.append(related["content"])
                metadata["neural_links_followed"] += 1
        
        # Check for style memories if writing-related query
        writing_keywords = ["write", "scene", "chapter", "story", "prose", "dialogue", "style"]
        if any(kw in query.lower() for kw in writing_keywords):
            # Search for style memories
            style_results = self.context._search_mnemo("PROSE_SAMPLE VOICE DIALOGUE_SAMPLE", limit=5)
            for style in style_results[:2]:
                content = style.get("content", "")
                if any(tag in content for tag in ["PROSE_SAMPLE", "VOICE", "DIALOGUE_SAMPLE", "VOCABULARY"]):
                    context_parts.append(content)
                    metadata["style_included"] += 1
        
        context_string = "\n".join(f"• {p}" for p in context_parts) if context_parts else ""
        
        return context_string, metadata
    
    def maintenance(self, loop_manager, apply_decay: bool = True, 
                   prune: bool = False) -> Dict:
        """
        Run maintenance tasks:
        - Apply decay
        - Identify/execute pruning
        - Report health
        """
        results = {
            "timestamp": datetime.now().isoformat()
        }
        
        # Apply decay
        if apply_decay:
            results["decay"] = self.degradation.apply_decay(loop_manager)
        
        # Prune (dry run by default)
        results["prune"] = self.degradation.prune_memories(loop_manager, dry_run=not prune)
        
        # Health report
        results["health"] = self.degradation.get_health_report(loop_manager)
        
        # Duplicates
        duplicates = self.degradation.find_duplicates(loop_manager)
        results["duplicates"] = len(duplicates)
        
        return results
    
    def consolidate_memories(self, openrouter_key: str = None) -> Dict:
        """
        🧠 MEMORY CONSOLIDATION - Like human sleep consolidation!
        
        Periodically analyze all memories and generate:
        - [CONTEXT] entries explaining what facts MEAN
        - [RELATIONSHIP] entries showing how things connect
        - [CLARIFICATION] entries preventing misinterpretation
        - [TIMELINE] entries for event sequences
        
        This helps GPT-4o understand the deeper meaning, not just facts.
        
        Returns:
            Dict with consolidation results and new entries created
        """
        api_key = openrouter_key or self.openrouter_key
        if not api_key:
            return {"error": "No OpenRouter API key", "created": 0}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "memories_analyzed": 0,
            "new_entries": [],
            "created": 0,
            "cost": 0.0
        }
        
        # Step 1: Get ALL memories from Mnemo
        headers = {
            "Authorization": f"Bearer {self.hf_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(
                f"{self.mnemo_url}/list",
                headers=headers,
                timeout=30
            )
            if response.status_code != 200:
                return {"error": "Failed to fetch memories", "created": 0}
            
            all_memories = response.json().get("memories", [])
            results["memories_analyzed"] = len(all_memories)
            
            if not all_memories:
                return {"error": "No memories to consolidate", "created": 0}
            
        except Exception as e:
            return {"error": f"Fetch error: {e}", "created": 0}
        
        # Step 2: Separate existing context from facts
        facts = []
        existing_context = []
        existing_clarifications = []
        
        for mem in all_memories:
            content = mem.get("content", "")
            if content.startswith("[CONTEXT]") or content.startswith("[RELATIONSHIP]"):
                existing_context.append(content)
            elif content.startswith("[CLARIFICATION]"):
                existing_clarifications.append(content)
            elif content.startswith("[SESSION]") or content.startswith("[CONVERSATION]"):
                continue  # Skip session metadata
            else:
                facts.append(content)
        
        # Step 3: Build consolidated prompt
        facts_text = "\n".join(facts[:100])  # Limit to avoid token overflow
        existing_context_text = "\n".join(existing_context[:20])
        
        consolidation_prompt = f"""Analyze these story memories and generate DEEP CONTEXT entries.

EXISTING FACTS:
{facts_text}

EXISTING CONTEXT (avoid duplicating):
{existing_context_text if existing_context_text else "None yet"}

YOUR TASK:
1. Identify facts that could be MISINTERPRETED without context
2. Find IMPLICIT RELATIONSHIPS between characters/events
3. Note any TIMELINE or sequence information
4. Create entries that explain WHAT THINGS MEAN, not just what they are

Generate JSON array with NEW entries only (don't duplicate existing):

[
  {{
    "category": "CONTEXT",
    "content": "Explanation of deeper meaning behind a fact"
  }},
  {{
    "category": "RELATIONSHIP", 
    "content": "Character A → Character B: nature of relationship, dynamics"
  }},
  {{
    "category": "CLARIFICATION",
    "content": "When X is mentioned, it means Y, NOT Z. Common misinterpretation to avoid."
  }},
  {{
    "category": "TIMELINE",
    "content": "Sequence: Event A → Event B → Event C (with context)"
  }}
]

RULES:
- Focus on things that could be misunderstood (e.g. metaphorical vs literal, unreliable narration, hidden motives)
- Make relationships explicit (who manipulates whom, who trusts whom)
- Create 5-10 high-value entries, not dozens of low-value ones
- Each entry should help an AI write more accurate scenes
- Skip entries if existing context already covers them

Return ONLY the JSON array, no other text."""

        # Step 4: Call GPT-4o for consolidation
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-2024-11-20",
                    "messages": [
                        {"role": "system", "content": "You are a story analyst creating deep context entries to help AI understand narrative nuance."},
                        {"role": "user", "content": consolidation_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}", "created": 0}
            
            data = response.json()
            
            # Calculate cost
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            results["cost"] = (input_tokens * 2.50 + output_tokens * 15.00) / 1_000_000
            
            # Parse response
            content = data["choices"][0]["message"]["content"]
            
            # Extract JSON
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            new_entries = json.loads(content.strip())
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "created": 0, "raw": content}
        except Exception as e:
            return {"error": f"API error: {e}", "created": 0}
        
        # Step 5: Store new entries in Mnemo
        stored = 0
        for entry in new_entries:
            category = entry.get("category", "CONTEXT").upper()
            content = entry.get("content", "")
            
            if not content:
                continue
            
            # Check if similar entry already exists
            is_duplicate = False
            for existing in existing_context + existing_clarifications:
                # Simple overlap check
                content_words = set(content.lower().split())
                existing_words = set(existing.lower().split())
                overlap = len(content_words & existing_words) / max(len(content_words), 1)
                if overlap > 0.7:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            try:
                response = requests.post(
                    f"{self.mnemo_url}/add",
                    headers=headers,
                    json={
                        "content": f"[{category}] {content}",
                        "metadata": {
                            "category": category,
                            "source": "consolidation",
                            "created": datetime.now().isoformat()
                        }
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    stored += 1
                    results["new_entries"].append({
                        "category": category,
                        "content": content[:100] + "..." if len(content) > 100 else content
                    })
            except Exception:
                continue
        
        results["created"] = stored
        
        return results
    
    def should_consolidate(self, last_consolidation: datetime = None, 
                          message_count: int = 0,
                          new_memories_since: int = 0) -> bool:
        """
        Determine if consolidation should run.
        
        Triggers:
        - 24+ hours since last consolidation
        - 100+ messages since last consolidation
        - 50+ new memories since last consolidation
        """
        # Time-based
        if last_consolidation:
            hours_since = (datetime.now() - last_consolidation).total_seconds() / 3600
            if hours_since >= 24:
                return True
        
        # Message-based
        if message_count >= 100:
            return True
        
        # Memory-based
        if new_memories_since >= 50:
            return True
        
        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def store_with_context(memories: List[Dict], hf_key: str, session_id: str = None,
                       mnemo_url: str = "https://athelaperk-mnemo-mcp.hf.space") -> int:
    """
    Store memories including context entries.
    """
    headers = {
        "Authorization": f"Bearer {hf_key}",
        "Content-Type": "application/json"
    }
    
    stored = 0
    for mem in memories:
        category = mem.get("category", "FACT").upper()
        content = mem.get("content", "")
        
        if not content:
            continue
        
        metadata = {"category": category}
        if session_id:
            metadata["session_id"] = session_id
        
        try:
            response = requests.post(
                f"{mnemo_url}/add",
                headers=headers,
                json={
                    "content": f"[{category}] {content}",
                    "metadata": metadata
                },
                timeout=10
            )
            if response.status_code == 200:
                stored += 1
        except Exception:
            continue
    
    return stored
