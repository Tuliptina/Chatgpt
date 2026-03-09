"""
Context Engine - Deep Context Generation and Memory Consolidation

Two key features:
1. RICH CONTEXT: Format pre-fetched memories for LLM injection (pure formatter)
2. CONSOLIDATION: K2.5 "sleep" pass that generates deep context entries
   (CONTEXT, RELATIONSHIP, CLARIFICATION, TIMELINE, TONE) from raw facts

v6.0 changes:
- consolidate_memories() now routes to MEMORY_MODEL_ID (K2.5) instead of GPT-4o
  (~77% cost reduction on consolidation runs)
- K2.5 cost rates: $0.60/$3.00 per M tokens (was $2.50/$15.00)
- TONE entries added to consolidation output categories
- build_rich_context() now groups by entity when ConnectionPoints are present
- Consolidation prompt updated to extract TONE directives (scene registers,
  humor types, emotional shifts, do-nots)

v5.1: build_rich_context() is pure formatter, consolidate at priority=1.2.
"""

import json
import requests
from datetime import datetime
from typing import List, Dict, Tuple
from mnemo_client import MnemoClient

# v6.0: Memory model for consolidation (matches app.py config)
MEMORY_MODEL_ID = "moonshotai/kimi-k2.5"
MEMORY_COST_INPUT = 0.60 / 1_000_000
MEMORY_COST_OUTPUT = 3.00 / 1_000_000


# =============================================================================
# UNIFIED CONTEXT ENGINE
# =============================================================================

class ContextEngine:
    """
    Unified engine for deep context formatting and memory consolidation.

    v6.0: Consolidation routes to K2.5 (MEMORY_MODEL_ID) instead of GPT-4o.
    TONE entries are now a first-class output category.

    Used by app.py for:
    - build_rich_context(): Formats pre-fetched memories for prompt injection
    - consolidate_memories(): K2.5 generates deep context from raw facts
    """

    def __init__(self, mnemo_client: MnemoClient, openrouter_key: str = None):
        self.mnemo = mnemo_client
        self.openrouter_key = openrouter_key

    # -----------------------------------------------------------------
    # RICH CONTEXT (for prompt injection)
    # -----------------------------------------------------------------

    def build_rich_context(self, query: str, memories: List[Dict]) -> Tuple[str, Dict]:
        """
        Format pre-fetched memories for LLM injection.

        v5.1: No longer calls self.mnemo.search() — the caller
        (build_memory_context in app.py) does all Mnemo searches so
        there's a single retrieval path. This method is now a pure
        formatter: categorize, deduplicate, and format.

        Args:
            query:    The user's prompt (used only for metadata).
            memories: Pre-fetched list of memory dicts from Mnemo search.
                      Each dict should have at least a "content" key.
        """
        metadata = {
            "facts_included": 0,
            "context_included": 0,
            "style_included": 0,
            "tone_included": 0,
        }
        context_parts = []
        seen = set()  # deduplicate by content prefix

        for r in memories:
            content = r.get("content", "")
            if not content:
                continue

            # Deduplicate — memories from primary + style searches may overlap
            content_key = content[:80].lower()
            if content_key in seen:
                continue
            seen.add(content_key)

            # Categorize — v6.0: added TONE as first-class category
            if any(content.startswith(f"[{tag}]") for tag in
                   ("CONTEXT", "RELATIONSHIP", "CLARIFICATION", "TIMELINE")):
                context_parts.append(content)
                metadata["context_included"] += 1
            elif any(tag in content for tag in
                     ("PROSE_SAMPLE", "VOICE", "DIALOGUE_SAMPLE", "VOCABULARY")):
                context_parts.append(content)
                metadata["style_included"] += 1
            elif content.startswith("[TONE]") or any(tag in content for tag in
                     ("scene_register", "humor_type", "emotional_shift", "do_not", "atmosphere")):
                context_parts.append(content)
                metadata["tone_included"] += 1
            else:
                context_parts.append(content)
                metadata["facts_included"] += 1

        context_string = "\n".join(f"- {p}" for p in context_parts) if context_parts else ""
        return context_string, metadata

    # -----------------------------------------------------------------
    # MEMORY CONSOLIDATION ("sleep" pass)
    # -----------------------------------------------------------------

    def consolidate_memories(self, openrouter_key: str = None) -> Dict:
        """
        Memory consolidation — like human sleep consolidation.

        v6.0: Routes to K2.5 (MEMORY_MODEL_ID) instead of GPT-4o.
        Cost: ~$0.007 per run (was ~$0.03). Now also generates TONE entries.

        Reads all raw facts from Mnemo, sends them to K2.5 to generate
        deep context entries, deduplicates against existing context, and
        stores new entries back.
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

        # Fetch all memories via MnemoClient
        try:
            all_memories = self.mnemo.list_memories()
            if not all_memories:
                return {"error": "No memories to consolidate or API unreachable", "created": 0}

            results["memories_analyzed"] = len(all_memories)

        except Exception as e:
            return {"error": f"Fetch error: {e}", "created": 0}

        # Separate facts from existing context
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
                continue
            else:
                facts.append(content)

        facts_text = "\n".join(facts[:100])
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
5. Extract TONE directives — how scenes should FEEL, not just what happens

Generate ONLY a JSON object containing an array called "entries" with NEW entries only (don't duplicate existing). Example format:
{{
  "entries": [
    {{
      "category": "CONTEXT",
      "content": "Explanation of deeper meaning behind a fact"
    }},
    {{
      "category": "RELATIONSHIP",
      "content": "Character A -> Character B: nature of relationship, dynamics"
    }},
    {{
      "category": "CLARIFICATION",
      "content": "When X is mentioned, it means Y, NOT Z. Common misinterpretation to avoid."
    }},
    {{
      "category": "TIMELINE",
      "content": "Sequence: Event A -> Event B -> Event C (with context)"
    }},
    {{
      "category": "TONE",
      "content": "Character X's scenes should feel [register]. Their humor is [type] — it should make the reader [effect], not [wrong effect]. DO NOT [common mistake]."
    }}
  ]
}}

RULES:
- Focus on things that could be misunderstood
- Make relationships explicit
- Create 5-15 high-value entries
- TONE entries are critical — extract HOW scenes should feel for each character
- Identify humor types (dark, dry, bitter, nervous) and their narrative function
- Note emotional DO NOTs (e.g., "never let Sebastian cry — he dissociates, not emotes")
- Skip entries if existing context already covers them"""

        # v6.0: Call K2.5 (MEMORY_MODEL_ID) for consolidation — ~77% cheaper than GPT-4o
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MEMORY_MODEL_ID,
                    "messages": [
                        {"role": "system", "content": "You are a story analyst creating deep context and tone entries. Return JSON only."},
                        {"role": "user", "content": consolidation_prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"}
                },
                timeout=60
            )

            if response.status_code != 200:
                return {"error": f"API error: {response.status_code}", "created": 0}

            data = response.json()

            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            # v6.0: K2.5 pricing ($0.60/$3.00 per M tokens)
            results["cost"] = (input_tokens * MEMORY_COST_INPUT + output_tokens * MEMORY_COST_OUTPUT)

            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            new_entries = parsed.get("entries", [])

        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "created": 0}
        except Exception as e:
            return {"error": f"API error: {e}", "created": 0}

        # Store new entries, deduplicating against existing context
        stored = 0
        for entry in new_entries:
            category = entry.get("category", "CONTEXT").upper()
            entry_content = entry.get("content", "")

            if not entry_content:
                continue

            # Check for duplicates via word overlap
            is_duplicate = False
            for existing in existing_context + existing_clarifications:
                content_words = set(entry_content.lower().split())
                existing_words = set(existing.lower().split())
                overlap = len(content_words & existing_words) / max(len(content_words), 1)
                if overlap > 0.7:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            try:
                meta = {
                    "category": category,
                    "source": "consolidation",
                    "created": datetime.now().isoformat()
                }
                # v6.0: Consolidation entries get elevated priority (1.2)
                # — these are K2.5-synthesized insights, more valuable than
                # raw auto-extracted facts (0.5) but below deliberate
                # file uploads (1.5)
                mem_id = self.mnemo.add(f"[{category}] {entry_content}", metadata=meta, priority=1.2)
                if mem_id:
                    stored += 1
                    results["new_entries"].append({
                        "category": category,
                        "content": entry_content[:100] + "..." if len(entry_content) > 100 else entry_content
                    })
            except Exception:
                continue

        results["created"] = stored
        return results

    # -----------------------------------------------------------------
    # CONSOLIDATION SCHEDULING
    # -----------------------------------------------------------------

    def should_consolidate(self, last_consolidation=None,
                           message_count: int = 0,
                           new_memories_since: int = 0) -> bool:
        """Check if it's time to run consolidation."""
        if last_consolidation:
            hours_since = (datetime.now() - last_consolidation).total_seconds() / 3600
            if hours_since >= 24:
                return True

        if message_count >= 100:
            return True

        if new_memories_since >= 50:
            return True

        return False
