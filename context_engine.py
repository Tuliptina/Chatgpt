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
MEMORY_MODEL_ID = "moonshotai/kimi-k2"
MEMORY_COST_INPUT = 0.47 / 1_000_000
MEMORY_COST_OUTPUT = 2.00 / 1_000_000


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

        v6.0: Routes to K2 (MEMORY_MODEL_ID) instead of GPT-4o.
        Reads raw facts, generates NEW deep context entries, stores them.
        Never deletes existing memories — purely additive.
        """
        api_key = openrouter_key or self.openrouter_key
        if not api_key:
            return {"error": "No OpenRouter API key", "created": 0}

        results = {
            "timestamp": datetime.now().isoformat(),
            "memories_analyzed": 0,
            "new_entries": [],
            "created": 0,
            "cost": 0.0,
            "k2_returned": 0,
            "dedup_rejected": 0,
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

        for mem in all_memories:
            content = mem.get("content", "")
            if any(content.startswith(f"[{tag}]") for tag in
                   ("CONTEXT", "RELATIONSHIP", "CLARIFICATION", "TIMELINE", "TONE")):
                existing_context.append(content)
            elif content.startswith("[SESSION]") or content.startswith("[CONVERSATION]"):
                continue
            else:
                facts.append(content)

        if not facts:
            return {"error": "No facts to analyze (only context entries exist)", "created": 0}

        facts_text = "\n".join(facts[:100])

        consolidation_prompt = f"""You are analyzing story memories to generate NEW deep context entries.

EXISTING FACTS TO ANALYZE:
{facts_text}

YOUR TASK — generate entries in these categories:

CONTEXT: Explain what a fact MEANS. What could be misunderstood? What's the subtext?
  Example: "When Sebastian stops speaking mid-scene, it's dissociation from trauma — never write it as stoic calm or peaceful acceptance."

RELATIONSHIP: How two characters relate. Power dynamics, emotional undercurrent, history.
  Example: "Alistair → Elijah: estranged brothers. Alistair envies Elijah's peace. Their father disowned Elijah for becoming a Quaker — Alistair secretly resents inheriting the family burden."

CLARIFICATION: Prevent specific misinterpretations.
  Example: "The Red Rose Society is NOT a shadowy cabal of villains. It's a decentralized medical order where members genuinely believe controlled suffering advances humanity. Writing them as evil undermines the story's moral complexity."

TIMELINE: Sequence of events with causation.
  Example: "Captivity arc sequence: Alistair offers 'treatment' → Sebastian arrives at Blackwood Estate → drug protocols begin → Isabella takes over → tattoo scene (branding) → Evelyn launches rescue."

TONE: How specific characters or scenes should FEEL.
  Example: "Isabella's scenes: obsessive tenderness. She genuinely believes she's caring for Sebastian. The horror comes from her sincerity, not cruelty."

Generate 5-15 NEW entries. Be bold — find connections between facts that aren't obvious.

Return ONLY a JSON object:
{{
  "entries": [
    {{"category": "CONTEXT", "content": "your insight"}},
    {{"category": "RELATIONSHIP", "content": "your insight"}},
    {{"category": "CLARIFICATION", "content": "your insight"}},
    {{"category": "TONE", "content": "your insight"}}
  ]
}}"""

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
                        {"role": "system", "content": "You are a story analyst. Generate NEW deep context entries from existing facts. Return ONLY valid JSON with an 'entries' array. No markdown, no explanation — pure JSON only."},
                        {"role": "user", "content": consolidation_prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 3000,
                },
                timeout=90
            )

            if response.status_code != 200:
                return {"error": f"API error: {response.status_code} — {response.text[:200]}", "created": 0}

            data = response.json()

            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            results["cost"] = (input_tokens * MEMORY_COST_INPUT + output_tokens * MEMORY_COST_OUTPUT)

            content = data["choices"][0]["message"]["content"]
            # Strip markdown fences if K2 wraps JSON in ```json ... ```
            clean = content.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            parsed = json.loads(clean)
            new_entries = parsed.get("entries", [])
            results["k2_returned"] = len(new_entries)

        except json.JSONDecodeError as e:
            # Include raw content in error for debugging
            raw_preview = content[:200] if 'content' in dir() else 'no content'
            return {"error": f"JSON parse error: {e} — raw: {raw_preview}", "created": 0}
        except Exception as e:
            return {"error": f"API error: {e}", "created": 0}

        # Handle models returning different key names
        if not new_entries:
            for key in ("results", "memories", "items", "data"):
                new_entries = parsed.get(key, [])
                if new_entries:
                    results["k2_returned"] = len(new_entries)
                    break

        if not new_entries:
            return {**results, "error": f"K2 returned JSON but no entries found. Keys: {list(parsed.keys())}", "created": 0}

        # Store new entries — dedup only against near-exact matches
        stored = 0
        for entry in new_entries:
            category = entry.get("category", "CONTEXT").upper()
            entry_content = entry.get("content", "")

            if not entry_content or len(entry_content) < 20:
                continue

            # Only reject near-exact duplicates (0.85 threshold)
            # Old threshold of 0.7 was too aggressive for same-universe content
            is_duplicate = False
            for existing in existing_context:
                content_words = set(entry_content.lower().split())
                existing_words = set(existing.lower().split())
                if not content_words:
                    break
                overlap = len(content_words & existing_words) / len(content_words)
                if overlap > 0.85:
                    is_duplicate = True
                    results["dedup_rejected"] += 1
                    break

            if is_duplicate:
                continue

            try:
                meta = {
                    "category": category,
                    "source": "consolidation",
                    "created": datetime.now().isoformat()
                }
                mem_id = self.mnemo.add(f"[{category}] {entry_content}", metadata=meta, priority=1.2)
                if mem_id:
                    stored += 1
                    results["new_entries"].append({
                        "category": category,
                        "content": entry_content[:100] + "..." if len(entry_content) > 100 else entry_content
                    })
                    # Also add to existing_context to prevent self-duplication within this batch
                    existing_context.append(f"[{category}] {entry_content}")
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
