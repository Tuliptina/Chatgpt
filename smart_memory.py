"""
Smart Memory System

Reduces latency by detecting which queries need memory lookup.

PROBLEM: Searching memory for EVERY query adds 500ms-2s latency
SOLUTION: Only search memory when the query actually needs it

Query Types:
- SKIP MEMORY: "hi", "thanks", "ok", "continue", etc.
- SKIP MEMORY: Follow-up questions in same context
- USE MEMORY: Questions about stored information
- USE MEMORY: Creative writing requests
- USE MEMORY: Questions with named entities
"""

import os

import re
from typing import Tuple, List
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Types of queries and their memory needs"""
    GREETING = "greeting"           # hi, hello, hey → NO MEMORY
    SIMPLE_RESPONSE = "simple"      # ok, thanks, sure → NO MEMORY
    CONTINUATION = "continuation"   # continue, go on → NO MEMORY (use conversation)
    FOLLOWUP = "followup"           # what about X, and Y? → MAYBE MEMORY
    FACTUAL = "factual"             # who is, what is → USE MEMORY
    CREATIVE = "creative"           # write, describe, create → USE MEMORY
    REFERENCE = "reference"         # my novel, my character → USE MEMORY
    GENERAL = "general"             # default → USE MEMORY


@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    query_type: QueryType
    needs_memory: bool
    confidence: float
    keywords: List[str]
    reason: str


class SmartMemory:
    """
    Intelligent query analyzer that determines if memory search is needed.
    
    This reduces latency by 500ms-2s for simple queries.
    """
    
    # Patterns that DON'T need memory
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|howdy|greetings|good morning|good afternoon|good evening)[\s\!\.\?]*$',
        r'^(what\'s up|whats up|sup|yo)[\s\!\.\?]*$',
    ]
    
    SIMPLE_RESPONSE_PATTERNS = [
        r'^(ok|okay|sure|yes|no|yep|nope|thanks|thank you|thx|ty|great|cool|nice|awesome|perfect|got it|understood|i see|alright|right)[\s\!\.\?]*$',
        r'^(sounds good|makes sense|that works|i agree|exactly|correct|true|false)[\s\!\.\?]*$',
    ]
    
    CONTINUATION_PATTERNS = [
        r'^(continue|go on|keep going|more|next|proceed|carry on)[\s\!\.\?]*$',
        r'^(and then|what happens next|then what|go ahead)[\s\!\.\?]*$',
        r'^(continue writing|keep writing|write more|extend this)[\s\!\.\?]*$',
    ]
    
    # Patterns that NEED memory
    REFERENCE_PATTERNS = [
        r'\b(my|our)\s+(novel|story|book|character|plot|setting|world|outline|project)\b',
        r'\b(the|that)\s+(character|villain|protagonist|antagonist|hero|heroine)\b',
        r'\b(remember|recall|mentioned|told you|we discussed|we talked)\b',
        r'\b(based on what|from what|according to)\b',
    ]
    
    FACTUAL_PATTERNS = [
        r'^(who is|what is|where is|when is|how is|tell me about)\b',
        r'^(describe|explain|summarize)\s+\w+',
        r'\b(what did|who did|where did|when did)\b',
    ]
    
    CREATIVE_PATTERNS = [
        r'^(write|create|compose|draft|generate)\s+',
        r'\b(write me|write a|create a|generate a)\b',
        r'\b(scene|chapter|dialogue|conversation|story|poem|script)\b',
    ]
    
    # Named entity indicators (proper nouns, character names)
    NAME_PATTERN = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    
    def __init__(self):
        # Compile patterns for speed
        self.greeting_re = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self.simple_re = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_RESPONSE_PATTERNS]
        self.continuation_re = [re.compile(p, re.IGNORECASE) for p in self.CONTINUATION_PATTERNS]
        self.reference_re = [re.compile(p, re.IGNORECASE) for p in self.REFERENCE_PATTERNS]
        self.factual_re = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]
        self.creative_re = [re.compile(p, re.IGNORECASE) for p in self.CREATIVE_PATTERNS]
        self.name_re = re.compile(self.NAME_PATTERN)
    
    def analyze(self, query: str, conversation_length: int = 0) -> QueryAnalysis:
        """
        Analyze query to determine if memory search is needed.
        
        Args:
            query: The user's message
            conversation_length: Number of messages in current conversation
            
        Returns:
            QueryAnalysis with decision and reasoning
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()
        
        # Extract potential keywords for memory search
        keywords = self._extract_keywords(query_clean)
        
        # Check patterns in order of priority
        
        # 1. Greetings - NO MEMORY
        for pattern in self.greeting_re:
            if pattern.match(query_clean):
                return QueryAnalysis(
                    query_type=QueryType.GREETING,
                    needs_memory=False,
                    confidence=0.95,
                    keywords=[],
                    reason="Greeting detected - no memory needed"
                )
        
        # 2. Simple responses - NO MEMORY
        for pattern in self.simple_re:
            if pattern.match(query_clean):
                return QueryAnalysis(
                    query_type=QueryType.SIMPLE_RESPONSE,
                    needs_memory=False,
                    confidence=0.95,
                    keywords=[],
                    reason="Simple response - no memory needed"
                )
        
        # 3. Continuation requests - NO MEMORY (use conversation context)
        for pattern in self.continuation_re:
            if pattern.match(query_clean):
                return QueryAnalysis(
                    query_type=QueryType.CONTINUATION,
                    needs_memory=False,
                    confidence=0.90,
                    keywords=[],
                    reason="Continuation request - using conversation context"
                )
        
        # 4. Reference to stored info - USE MEMORY
        for pattern in self.reference_re:
            if pattern.search(query_clean):
                return QueryAnalysis(
                    query_type=QueryType.REFERENCE,
                    needs_memory=True,
                    confidence=0.95,
                    keywords=keywords,
                    reason="Reference to stored information detected"
                )
        
        # 5. Creative writing - USE MEMORY
        for pattern in self.creative_re:
            if pattern.search(query_clean):
                return QueryAnalysis(
                    query_type=QueryType.CREATIVE,
                    needs_memory=True,
                    confidence=0.90,
                    keywords=keywords,
                    reason="Creative writing request - using memory for context"
                )
        
        # 6. Factual questions - USE MEMORY
        for pattern in self.factual_re:
            if pattern.search(query_clean):
                return QueryAnalysis(
                    query_type=QueryType.FACTUAL,
                    needs_memory=True,
                    confidence=0.85,
                    keywords=keywords,
                    reason="Factual question - searching memory"
                )
        
        # 7. Contains named entities (proper nouns) - USE MEMORY
        names = self.name_re.findall(query_clean)
        # Filter out common words that might be capitalized
        common_words = {'I', 'The', 'A', 'An', 'This', 'That', 'What', 'Who', 'Where', 
                       'When', 'How', 'Why', 'Can', 'Could', 'Would', 'Should', 'Do',
                       'Does', 'Is', 'Are', 'Was', 'Were', 'Have', 'Has', 'Had',
                       'Will', 'Would', 'May', 'Might', 'Must', 'Shall'}
        real_names = [n for n in names if n not in common_words]
        
        if real_names:
            return QueryAnalysis(
                query_type=QueryType.REFERENCE,
                needs_memory=True,
                confidence=0.85,
                keywords=real_names + keywords,
                reason=f"Named entities detected: {real_names}"
            )
        
        # 8. Short queries in active conversation - MAYBE NO MEMORY
        if len(query_clean.split()) <= 5 and conversation_length > 2:
            # Short query in ongoing conversation, likely a follow-up
            return QueryAnalysis(
                query_type=QueryType.FOLLOWUP,
                needs_memory=False,
                confidence=0.70,
                keywords=keywords,
                reason="Short follow-up in active conversation"
            )
        
        # 9. Default - USE MEMORY for safety
        return QueryAnalysis(
            query_type=QueryType.GENERAL,
            needs_memory=True,
            confidence=0.60,
            keywords=keywords,
            reason="General query - using memory for context"
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords for memory search"""
        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor', 'so',
            'yet', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
            'also', 'now', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'this', 'that', 'these', 'those', 'what', 'which', 'who',
            'whom', 'whose', 'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
            'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
            'it', 'its', 'they', 'them', 'their', 'theirs', 'write', 'tell',
            'describe', 'create', 'make', 'give', 'show', 'please', 'help'
        }
        
        # Clean and split
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords[:10]  # Max 10 keywords
    
    def should_use_memory(self, query: str, conversation_length: int = 0) -> Tuple[bool, str]:
        """
        Simple method that returns just the decision.
        
        Returns:
            (needs_memory: bool, reason: str)
        """
        analysis = self.analyze(query, conversation_length)
        return analysis.needs_memory, analysis.reason


# =============================================================================
# CONTEXT WINDOW MANAGER (with Loop System Integration)
# =============================================================================

class ContextWindowManager:
    """
    Manages context window with INTEGRATED loop system.
    
    GPT-4o limits:
    - Context window: 128,000 tokens
    - Max output: 16,384 tokens
    - Recommended input: ~100,000 tokens (leave room for output)
    
    Uses metadata loops for efficient memory injection:
    - High relevance (>60%): Full content
    - Medium relevance (30-60%): Metadata only
    - Low relevance (<30%): Skip
    """
    
    # GPT-4o limits
    MAX_CONTEXT = 128_000
    MAX_OUTPUT = 16_384
    SAFE_INPUT_LIMIT = 100_000  # Leave room for output
    
    # Token budgets (configurable)
    BUDGET_SYSTEM_PROMPT = 1_500
    BUDGET_MEMORY_FULL = 3_000      # Full content injection (doubled for rich docs)
    BUDGET_MEMORY_META = 1_000      # Metadata injection (doubled)
    BUDGET_CONVERSATION = 4_000     # Conversation history
    BUDGET_QUERY = 500              # Current query
    TOTAL_BUDGET = 10_000           # Total recommended
    
    # Relevance thresholds (from loop system)
    THRESHOLD_FULL_CONTENT = 0.45   # Inject full content above this (lowered for better recall)
    THRESHOLD_METADATA = 0.20       # Inject metadata above this (lowered)
    
    def __init__(self, loop_manager=None):
        """
        Initialize with optional loop manager for memory integration.
        
        Args:
            loop_manager: LoopManager instance for memory retrieval
        """
        self.loop_manager = loop_manager
    
    def set_loop_manager(self, loop_manager):
        """Set or update the loop manager"""
        self.loop_manager = loop_manager
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        GPT-4o: ~1 token per 4 characters
        """
        if not text:
            return 0
        return len(text) // 4
    
    def estimate_messages_tokens(self, messages: List[dict]) -> int:
        """Estimate tokens for a list of messages"""
        total = 0
        for msg in messages:
            # Add overhead per message (~4 tokens for role, formatting)
            total += 4
            total += self.estimate_tokens(msg.get("content", ""))
        return total
    
    def truncate_to_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if not text:
            return ""
        
        estimated = self.estimate_tokens(text)
        
        if estimated <= max_tokens:
            return text
        
        # Calculate character limit
        char_limit = max_tokens * 4
        
        # Truncate with indicator
        return text[:char_limit - 20] + "\n[...truncated...]"
    
    def build_loop_context(self, query: str) -> Tuple[str, str, dict]:
        """
        Build memory context using the loop system.
        
        Returns:
            (full_content_context, metadata_context, stats)
        """
        if not self.loop_manager:
            return "", "", {"full": 0, "meta": 0, "tokens": 0}
        
        # Update relevance scores in loop manager
        self.loop_manager.update_relevance(query)
        
        full_content_parts = []
        metadata_parts = []
        full_tokens = 0
        meta_tokens = 0
        
        stats = {
            "full_injected": 0,
            "meta_injected": 0,
            "skipped": 0,
            "full_tokens": 0,
            "meta_tokens": 0
        }
        
        # Collect tokens from all loops
        all_tokens = []
        for loop in self.loop_manager.loops.values():
            all_tokens.extend(loop.tokens)
        
        # Sort by relevance
        all_tokens.sort(key=lambda t: t.relevance, reverse=True)
        
        for token in all_tokens:
            # High relevance: inject full content
            if token.relevance >= self.THRESHOLD_FULL_CONTENT:
                if full_tokens < self.BUDGET_MEMORY_FULL:
                    # Get full content
                    full_content = self.loop_manager.content_cache.get(
                        token.full_content_ref, ""
                    )
                    if full_content:
                        content_tokens = self.estimate_tokens(full_content)
                        if full_tokens + content_tokens <= self.BUDGET_MEMORY_FULL:
                            full_content_parts.append(f"• {full_content}")
                            full_tokens += content_tokens
                            stats["full_injected"] += 1
                            token.access_count += 1
            
            # Medium relevance: inject metadata only
            elif token.relevance >= self.THRESHOLD_METADATA:
                if meta_tokens < self.BUDGET_MEMORY_META:
                    meta_str = token.to_context_string()
                    meta_token_count = token.estimate_tokens()
                    if meta_tokens + meta_token_count <= self.BUDGET_MEMORY_META:
                        metadata_parts.append(f"• {meta_str}")
                        meta_tokens += meta_token_count
                        stats["meta_injected"] += 1
            
            # Low relevance: skip
            else:
                stats["skipped"] += 1
        
        stats["full_tokens"] = full_tokens
        stats["meta_tokens"] = meta_tokens
        
        # Build context strings
        full_context = ""
        if full_content_parts:
            full_context = "[RELEVANT MEMORIES]\n" + "\n".join(full_content_parts)
        
        meta_context = ""
        if metadata_parts:
            meta_context = "[RELATED CONTEXT]\n" + "\n".join(metadata_parts)
        
        return full_context, meta_context, stats
    
    def build_optimized_context(
        self,
        system_prompt: str,
        query: str,
        conversation_history: List[dict],
        max_messages: int = 8,
        use_loops: bool = True
    ) -> Tuple[List[dict], dict]:
        """
        Build fully optimized context with loop-based memory.
        
        This is the main method that combines:
        1. System prompt
        2. Loop-based memory injection (full + metadata)
        3. Conversation history (limited to max_messages)
        4. Current query
        
        Args:
            system_prompt: Base system prompt
            query: Current user query
            conversation_history: List of previous messages
            max_messages: Maximum conversation messages to include
            use_loops: Whether to use loop system for memory
            
        Returns:
            (messages_list, stats_dict)
        """
        stats = {
            "system_tokens": 0,
            "memory_full_tokens": 0,
            "memory_meta_tokens": 0,
            "memory_items_full": 0,
            "memory_items_meta": 0,
            "conversation_tokens": 0,
            "conversation_messages": 0,
            "query_tokens": 0,
            "total_tokens": 0,
            "within_budget": True,
            "mode": "loops" if use_loops else "none"
        }
        
        # Start with system prompt
        enhanced_system = system_prompt
        stats["system_tokens"] = self.estimate_tokens(system_prompt)
        
        # Add loop-based memory if enabled
        if use_loops and self.loop_manager:
            full_context, meta_context, loop_stats = self.build_loop_context(query)
            
            if full_context or meta_context:
                memory_section = "\n\n".join(filter(None, [full_context, meta_context]))
                enhanced_system = f"{system_prompt}\n\n{memory_section}"
            
            stats["memory_full_tokens"] = loop_stats["full_tokens"]
            stats["memory_meta_tokens"] = loop_stats["meta_tokens"]
            stats["memory_items_full"] = loop_stats["full_injected"]
            stats["memory_items_meta"] = loop_stats["meta_injected"]
        
        # Build messages array
        messages = [{"role": "system", "content": enhanced_system}]
        
        # Add conversation history (limited)
        recent_history = conversation_history[-max_messages:]
        stats["conversation_messages"] = len(recent_history)
        
        for msg in recent_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        stats["conversation_tokens"] = self.estimate_messages_tokens(recent_history)
        
        # Add current query ONLY if it's not already the last message in history
        # (app.py appends user message to history before calling this)
        if not recent_history or recent_history[-1].get("content") != query or recent_history[-1].get("role") != "user":
            messages.append({"role": "user", "content": query})
            stats["query_tokens"] = self.estimate_tokens(query)
        else:
            stats["query_tokens"] = 0
        
        # Calculate total
        stats["total_tokens"] = (
            self.estimate_tokens(enhanced_system) +
            stats["conversation_tokens"] +
            stats["query_tokens"]
        )
        
        stats["within_budget"] = stats["total_tokens"] <= self.TOTAL_BUDGET
        
        return messages, stats
    
    def get_budget_report(self) -> dict:
        """Get current budget configuration"""
        return {
            "budgets": {
                "system_prompt": self.BUDGET_SYSTEM_PROMPT,
                "memory_full": self.BUDGET_MEMORY_FULL,
                "memory_meta": self.BUDGET_MEMORY_META,
                "conversation": self.BUDGET_CONVERSATION,
                "query": self.BUDGET_QUERY,
                "total": self.TOTAL_BUDGET
            },
            "thresholds": {
                "full_content": self.THRESHOLD_FULL_CONTENT,
                "metadata": self.THRESHOLD_METADATA
            },
            "limits": {
                "max_context": self.MAX_CONTEXT,
                "max_output": self.MAX_OUTPUT,
                "safe_input": self.SAFE_INPUT_LIMIT
            }
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SMART MEMORY + INTEGRATED CONTEXT WINDOW TEST")
    print("=" * 60)
    
    smart = SmartMemory()
    
    test_queries = [
        # Should NOT use memory
        ("hi", False),
        ("Hello!", False),
        ("thanks", False),
        ("ok", False),
        ("continue", False),
        ("go on", False),
        ("sounds good", False),
        
        # Should USE memory
        ("Tell me about the antagonist", True),
        ("Write a scene with the main character", True),
        ("What is my character's backstory?", True),
        ("Describe the setting of my novel", True),
        ("Who is the villain?", True),
        ("Based on what I told you, write a scene", True),
        ("Continue the story with the healer", True),
        
        # Edge cases
        ("What do you think?", False),  # Short, no names
        ("Can you help me?", False),  # Generic
        ("Write something", True),  # Creative request
    ]
    
    print("\n--- Smart Memory Query Analysis ---\n")
    
    correct = 0
    for query, expected in test_queries:
        analysis = smart.analyze(query, conversation_length=3)
        actual = analysis.needs_memory
        status = "✅" if actual == expected else "❌"
        
        if actual == expected:
            correct += 1
        
        print(f"{status} '{query}'")
        print(f"   Expected: {expected}, Got: {actual}")
        print(f"   Reason: {analysis.reason}")
        print()
    
    print(f"Accuracy: {correct}/{len(test_queries)} ({correct/len(test_queries)*100:.0f}%)")
    
    # Test integrated context manager with loop system
    print("\n--- Integrated Context Window Manager ---\n")
    
    # Import and initialize loop manager
    try:
        from metadata_loops import LoopManager
        
        HF_KEY = os.environ.get("HF_KEY", "")
        
        print("Loading loop manager with memories...")
        loop_manager = LoopManager(hf_key=HF_KEY)
        loop_manager.load_from_mnemo()
        
        # Create context manager with loop integration
        cwm = ContextWindowManager(loop_manager=loop_manager)
        
        print(f"✅ Loop manager loaded: {loop_manager.get_stats()['total_items']} memories")
        
        # Test building context
        test_prompt = "Write a scene where the mentor confronts the protagonist"
        system_prompt = "You are a creative writing assistant."
        
        conversation = [
            {"role": "user", "content": "I'm writing a thriller"},
            {"role": "assistant", "content": "Great! Tell me more."},
            {"role": "user", "content": "It's set in a coastal town in the 1990s"},
            {"role": "assistant", "content": "Wonderful setting!"},
        ]
        
        print(f"\nTest query: '{test_prompt}'")
        print(f"Conversation history: {len(conversation)} messages")
        
        # Build context
        messages, stats = cwm.build_optimized_context(
            system_prompt=system_prompt,
            query=test_prompt,
            conversation_history=conversation,
            max_messages=8,
            use_loops=True
        )
        
        print(f"\n📊 Context Stats:")
        print(f"  System tokens: {stats['system_tokens']}")
        print(f"  Memory (full): {stats['memory_full_tokens']} tokens ({stats['memory_items_full']} items)")
        print(f"  Memory (meta): {stats['memory_meta_tokens']} tokens ({stats['memory_items_meta']} items)")
        print(f"  Conversation: {stats['conversation_tokens']} tokens ({stats['conversation_messages']} msgs)")
        print(f"  Query: {stats['query_tokens']} tokens")
        print(f"  TOTAL: {stats['total_tokens']} tokens")
        print(f"  Within budget: {'✅' if stats['within_budget'] else '⚠️'}")
        
        # Show budget report
        print(f"\n📋 Budget Configuration:")
        budget = cwm.get_budget_report()
        print(f"  Memory full budget: {budget['budgets']['memory_full']} tokens")
        print(f"  Memory meta budget: {budget['budgets']['memory_meta']} tokens")
        print(f"  Conversation budget: {budget['budgets']['conversation']} tokens")
        print(f"  Total budget: {budget['budgets']['total']} tokens")
        print(f"  Full content threshold: {budget['thresholds']['full_content']}")
        print(f"  Metadata threshold: {budget['thresholds']['metadata']}")
        
    except Exception as e:
        print(f"❌ Error testing integrated context: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE!")
    print("=" * 60)
