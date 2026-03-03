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

import re
import tiktoken
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

    # Named entity indicators (forces at least TWO capitalized words)
    NAME_PATTERN = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'

    def __init__(self):
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
        keywords = self._extract_keywords(query_clean)

        # 1. Greetings - NO MEMORY
        for pattern in self.greeting_re:
            if pattern.match(query_clean):
                return QueryAnalysis(QueryType.GREETING, False, 0.95, [], "Greeting detected - no memory needed")

        # 2. Simple responses - NO MEMORY
        for pattern in self.simple_re:
            if pattern.match(query_clean):
                return QueryAnalysis(QueryType.SIMPLE_RESPONSE, False, 0.95, [], "Simple response - no memory needed")

        # 3. Continuation requests - NO MEMORY
        for pattern in self.continuation_re:
            if pattern.match(query_clean):
                return QueryAnalysis(QueryType.CONTINUATION, False, 0.90, [], "Continuation request - using conversation context")

        # 4. Reference to stored info - USE MEMORY
        for pattern in self.reference_re:
            if pattern.search(query_clean):
                return QueryAnalysis(QueryType.REFERENCE, True, 0.95, keywords, "Reference to stored information detected")

        # 5. Creative writing - USE MEMORY
        for pattern in self.creative_re:
            if pattern.search(query_clean):
                return QueryAnalysis(QueryType.CREATIVE, True, 0.90, keywords, "Creative writing request - using memory for context")

        # 6. Factual questions - USE MEMORY
        for pattern in self.factual_re:
            if pattern.search(query_clean):
                return QueryAnalysis(QueryType.FACTUAL, True, 0.85, keywords, "Factual question - searching memory")

        # 7. Named entities (proper nouns) - USE MEMORY
        names = self.name_re.findall(query_clean)
        common_words = {
            'I', 'The', 'A', 'An', 'This', 'That', 'What', 'Who', 'Where',
            'When', 'How', 'Why', 'Can', 'Could', 'Would', 'Should', 'Do',
            'Does', 'Is', 'Are', 'Was', 'Were', 'Have', 'Has', 'Had',
            'Will', 'Would', 'May', 'Might', 'Must', 'Shall'
        }
        real_names = [n for n in names if n not in common_words]
        if real_names:
            return QueryAnalysis(QueryType.REFERENCE, True, 0.85, real_names + keywords,
                                 f"Named entities detected: {real_names}")

        # 8. Short queries in active conversation - likely follow-up
        if len(query_clean.split()) <= 5 and conversation_length > 2:
            return QueryAnalysis(QueryType.FOLLOWUP, False, 0.70, keywords, "Short follow-up in active conversation")

        # 9. Default - USE MEMORY for safety
        return QueryAnalysis(QueryType.GENERAL, True, 0.60, keywords, "General query - using memory for context")

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords for memory search"""
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
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stopwords and len(w) > 2][:10]

    def should_use_memory(self, query: str, conversation_length: int = 0) -> Tuple[bool, str]:
        """Simple interface — returns (needs_memory, reason)."""
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

    Delegates memory retrieval to LoopManager.build_context() — no
    duplicated relevance logic here.
    """

    # GPT-4o limits
    MAX_CONTEXT = 128_000
    MAX_OUTPUT = 16_384
    SAFE_INPUT_LIMIT = 100_000

    # Token budgets
    BUDGET_SYSTEM_PROMPT = 1_500
    BUDGET_MEMORY_FULL = 3_000
    BUDGET_MEMORY_META = 1_000
    BUDGET_CONVERSATION = 4_000
    BUDGET_QUERY = 500
    TOTAL_BUDGET = 10_000

    def __init__(self, loop_manager=None):
        self.loop_manager = loop_manager

    def set_loop_manager(self, loop_manager):
        """Set or update the loop manager"""
        self.loop_manager = loop_manager

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken for GPT-4o."""
        if not text:
            return 0
        try:
            encoding = tiktoken.get_encoding("o200k_base")
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4

    def estimate_messages_tokens(self, messages: List[dict]) -> int:
        """Estimate tokens for a list of messages"""
        total = 0
        for msg in messages:
            total += 4  # overhead per message
            total += self.estimate_tokens(msg.get("content", ""))
        return total

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

        Combines:
        1. System prompt
        2. Loop-based memory injection (delegated to LoopManager)
        3. Conversation history (limited to max_messages)
        4. Current query

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

        enhanced_system = system_prompt
        stats["system_tokens"] = self.estimate_tokens(system_prompt)

        # Delegate memory context to LoopManager.build_context()
        if use_loops and self.loop_manager:
            context_string, loop_meta = self.loop_manager.build_context(query)
            if context_string:
                enhanced_system = f"{system_prompt}\n\n{context_string}"

            stats["memory_full_tokens"] = loop_meta.get("full_content_injected", 0) * 50  # estimate
            stats["memory_meta_tokens"] = loop_meta.get("metadata_injected", 0) * 15
            stats["memory_items_full"] = loop_meta.get("full_content_injected", 0)
            stats["memory_items_meta"] = loop_meta.get("metadata_injected", 0)

        # Build messages array
        messages = [{"role": "system", "content": enhanced_system}]

        # Conversation history (limited)
        recent_history = conversation_history[-max_messages:]
        stats["conversation_messages"] = len(recent_history)

        for msg in recent_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        stats["conversation_tokens"] = self.estimate_messages_tokens(recent_history)

        # Add current query only if not already the last message
        if (not recent_history
                or recent_history[-1].get("content") != query
                or recent_history[-1].get("role") != "user"):
            messages.append({"role": "user", "content": query})
            stats["query_tokens"] = self.estimate_tokens(query)
        else:
            stats["query_tokens"] = 0

        stats["total_tokens"] = (
            self.estimate_tokens(enhanced_system)
            + stats["conversation_tokens"]
            + stats["query_tokens"]
        )
        stats["within_budget"] = stats["total_tokens"] <= self.TOTAL_BUDGET

        return messages, stats
