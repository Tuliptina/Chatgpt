"""
Smart Memory System (v5.3)

Reduces latency by detecting which queries need memory lookup.

PROBLEM: Searching memory for EVERY query adds 500ms-2s latency.
SOLUTION: Two-tier injection gate that progressively filters queries.

Two-tier injection gate:
  Tier 1 — SmartMemory.analyze(): Fast local regex/pattern analysis.
      Zero latency. Produces a QueryAnalysis with confidence score.
      - High-confidence SKIP (greetings, acks, continuations): done.
      - High-confidence USE (explicit references, creative writing): done.
      - Moderate-confidence USE (factual, named entities, general): → Tier 2.

  Tier 2 — Mnemo server's /should_inject endpoint: Semantic analysis
      using the server's own knowledge of what's stored. ~200ms.
      Only called for moderate-confidence Tier 1 decisions. Can veto
      injection if the query has no semantic overlap with stored memories.

  Orchestrated by SmartMemory.two_tier_gate():
      analysis = analyze(query)
      if analysis.needs_memory == False:         → skip everything
      elif analysis.confidence >= 0.90:          → retrieve (skip Tier 2)
      elif mnemo_client.should_inject(query):    → retrieve
      else:                                      → skip (Tier 2 vetoed)

  app.py's handle_message() calls two_tier_gate() as the single entry point.

Query Types:
- SKIP MEMORY: "hi", "thanks", "ok", "continue", etc.
- SKIP MEMORY: Follow-up questions in same context
- SKIP MEMORY: Tier 2 veto (no semantic overlap with stored memories)
- USE MEMORY: Explicit references ("my novel", "the character")
- USE MEMORY: Creative writing requests
- USE MEMORY: Questions with named entities (if Tier 2 confirms)
- USE MEMORY: General queries (if Tier 2 confirms)
"""

import re
from typing import Tuple, List
from utils import estimate_tokens
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


@dataclass
class GateDecision:
    """Result of the full two-tier injection gate.

    Returned by SmartMemory.two_tier_gate(). Contains everything
    the caller needs to decide whether to retrieve memory context.
    """
    should_retrieve: bool
    tier_used: int          # 1 = local only, 2 = Mnemo consulted
    reason: str
    confidence: float
    query_type: QueryType
    keywords: List[str]
    mnemo_confidence: float = 0.0  # Tier 2 confidence (0.0 if not consulted)

    @property
    def mode(self) -> str:
        """Return a short label for metadata display."""
        if not self.should_retrieve:
            return "skip"
        if self.tier_used == 1:
            return "t1_direct"   # Tier 1 high-confidence → straight to retrieval
        return "t2_confirmed"    # Tier 2 confirmed retrieval


# Tier 2 is only consulted when Tier 1 confidence falls below this threshold.
# Above this: Tier 1 is confident enough to decide alone (saves ~200ms).
# Below this: Tier 2's semantic check adds real signal.
#
# Current confidence map:
#   REFERENCE=0.95, CREATIVE=0.90  → skip Tier 2 (explicit memory need)
#   FACTUAL=0.85, NAMES=0.85       → consult Tier 2
#   GENERAL=0.60                   → consult Tier 2
TIER_2_THRESHOLD = 0.90


class SmartMemory:
    """
    Two-tier memory injection gate.

    Primary entry point: two_tier_gate(query, mnemo_client, conversation_length)

    Tier 1 (local, ~0ms):
      Regex/pattern analysis. High-confidence decisions (greetings,
      explicit references, creative requests) are returned immediately
      without a network call. Saves ~200ms on obvious cases.

    Tier 2 (remote, ~200ms):
      Only called for moderate-confidence Tier 1 results (FACTUAL,
      named entities, GENERAL queries). Asks Mnemo's /should_inject
      endpoint whether the query has semantic overlap with stored
      memories. Can veto retrieval for queries like "what is
      photosynthesis" that have no stored context.

    Fallback: If mnemo_client is None or unavailable, Tier 1 decision
    is used directly (same behavior as v5.1).

    Backward compat: should_use_memory() still works as a Tier-1-only
    gate for callers that don't have a MnemoClient reference.
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
        """Tier 1 only gate — backward compatibility.

        Returns (needs_memory, reason). Use two_tier_gate() for full
        two-tier analysis when a MnemoClient is available.
        """
        analysis = self.analyze(query, conversation_length)
        return analysis.needs_memory, analysis.reason

    def two_tier_gate(self, query: str, mnemo_client=None,
                      conversation_length: int = 0) -> GateDecision:
        """Full two-tier injection gate.

        This is the primary entry point for app.py's handle_message().

        Flow:
          1. Run Tier 1 (local regex analysis)
          2. If Tier 1 says SKIP → return skip immediately
          3. If Tier 1 says USE with high confidence (≥ 0.90) → retrieve
             (explicit references and creative writing always need context)
          4. If Tier 1 says USE with moderate confidence (< 0.90) →
             consult Tier 2 (Mnemo /should_inject) for semantic validation
          5. If Tier 2 vetoes → return skip
          6. If Tier 2 confirms → return retrieve

        If mnemo_client is None or unavailable, falls back to Tier 1
        decision for moderate-confidence cases (same as v5.1 behavior).

        Args:
            query: The user's message.
            mnemo_client: Optional MnemoClient instance for Tier 2.
            conversation_length: Number of messages in current conversation.

        Returns:
            GateDecision with should_retrieve, tier_used, reason, etc.
        """
        analysis = self.analyze(query, conversation_length)

        # --- Tier 1 SKIP: greetings, acks, continuations, short follow-ups ---
        if not analysis.needs_memory:
            return GateDecision(
                should_retrieve=False,
                tier_used=1,
                reason=f"t1_skip: {analysis.reason}",
                confidence=analysis.confidence,
                query_type=analysis.query_type,
                keywords=analysis.keywords,
            )

        # --- Tier 1 high-confidence USE: explicit references, creative -------
        if analysis.confidence >= TIER_2_THRESHOLD:
            return GateDecision(
                should_retrieve=True,
                tier_used=1,
                reason=f"t1_direct: {analysis.reason}",
                confidence=analysis.confidence,
                query_type=analysis.query_type,
                keywords=analysis.keywords,
            )

        # --- Moderate confidence: consult Tier 2 if available ----------------
        if mnemo_client is None:
            # No client → fall back to Tier 1 decision
            return GateDecision(
                should_retrieve=True,
                tier_used=1,
                reason=f"t1_fallback (no client): {analysis.reason}",
                confidence=analysis.confidence,
                query_type=analysis.query_type,
                keywords=analysis.keywords,
            )

        try:
            should_inject, inject_reason, inject_confidence = mnemo_client.should_inject(query)
        except Exception:
            # Tier 2 failed → fall back to Tier 1
            return GateDecision(
                should_retrieve=True,
                tier_used=1,
                reason=f"t1_fallback (t2 error): {analysis.reason}",
                confidence=analysis.confidence,
                query_type=analysis.query_type,
                keywords=analysis.keywords,
            )

        if should_inject:
            return GateDecision(
                should_retrieve=True,
                tier_used=2,
                reason=f"t2_confirmed: {inject_reason}",
                confidence=analysis.confidence,
                query_type=analysis.query_type,
                keywords=analysis.keywords,
                mnemo_confidence=inject_confidence,
            )
        else:
            return GateDecision(
                should_retrieve=False,
                tier_used=2,
                reason=f"t2_vetoed: {inject_reason}",
                confidence=analysis.confidence,
                query_type=analysis.query_type,
                keywords=analysis.keywords,
                mnemo_confidence=inject_confidence,
            )


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

    v5.1: estimate_tokens() now delegates to utils.estimate_tokens()
    (single source of truth, eliminates duplicate tiktoken setup).
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
        """Estimate token count — delegates to utils.estimate_tokens()."""
        return estimate_tokens(text)

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
