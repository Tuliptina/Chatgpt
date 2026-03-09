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

        # Fetch all data — both legacy blobs and ConnectionPoints
        try:
            all_memories = self.mnemo.list_memories() or []
            # v6.1: Also fetch ConnectionPoints
            all_points = []
            if hasattr(self.mnemo, 'list_points'):
                all_points = self.mnemo.list_points(limit=500) or []
            
            total = len(all_memories) + len(all_points)
            if total == 0:
                return {"error": "No memories to consolidate or API unreachable", "created": 0}

            results["memories_analyzed"] = total

        except Exception as e:
            return {"error": f"Fetch error: {e}", "created": 0}

        # Separate facts from existing context
        facts = []
        existing_context = []

        # From legacy blobs
        for mem in all_memories:
            content = mem.get("content", "")
            if any(content.startswith(f"[{tag}]") for tag in
                   ("CONTEXT", "RELATIONSHIP", "CLARIFICATION", "TIMELINE", "TONE")):
                existing_context.append(content)
            elif content.startswith("[SESSION]") or content.startswith("[CONVERSATION]"):
                continue
            else:
                facts.append(content)

        # From ConnectionPoints
        CONTEXT_TYPES = {"context_note", "clarification", "relationship", "tone_directive", "timeline"}
        for cp in all_points:
            entity = cp.get("entity", "")
            value = cp.get("value", "")
            point_type = cp.get("point_type", "")
            conn = cp.get("connects_to", "")
            if point_type in CONTEXT_TYPES:
                line = f"{entity}: {value}" if entity else value
                if conn:
                    line = f"{entity} → {conn}: {value}"
                existing_context.append(line)
            else:
                line = f"{entity}: {value}" if entity else value
                if conn:
                    line = f"{entity} → {conn}: {value}"
                facts.append(line)

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

Generate 5-15 NEW entries. STRICT RULE: ONLY use information that is explicitly stated or directly implied by the facts above. Do NOT invent new details, scenes, or events. If a fact says "Alistair fears dementia," you can infer tone guidance — but do NOT fabricate HOW he discovered this fear.

CRITICAL RULES:
- ONLY derive insights from the facts above. Do NOT fabricate events, dialogue, or details not present in the facts.
- CONTEXT entries should explain what existing facts MEAN, not invent new backstory.
- RELATIONSHIP entries should describe dynamics stated or clearly implied by the facts.
- CLARIFICATION entries should prevent misreading of what the facts actually say.
- TONE entries should describe how to WRITE scenes based on the character/plot facts given.
- If you're unsure whether something is in the facts, leave it out.

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
                        {"role": "system", "content": "You are a story analyst. Your job is to derive CONTEXT, RELATIONSHIP, CLARIFICATION, and TONE entries strictly from the facts provided. Never invent new events or details — only explain, connect, and interpret what is already there. Return ONLY valid JSON with an 'entries' array. No markdown, no explanation — pure JSON only."},
                        {"role": "user", "content": consolidation_prompt}
                    ],
                    "temperature": 0.2,
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
                import re
                # Parse entity from content for RELATIONSHIP entries
                entity = "Story"
                connects_to = ""
                if category == "RELATIONSHIP":
                    arrow_match = re.match(r'(\w+)\s*(?:→|->)\s*(\w+)', entry_content)
                    if arrow_match:
                        entity = arrow_match.group(1)
                        connects_to = arrow_match.group(2)
                else:
                    # Try first proper noun
                    name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', entry_content)
                    if name_match:
                        entity = name_match.group(1)

                # Map category to CP fields
                CONSOL_MAP = {
                    "CONTEXT": ("fact", "context_note"),
                    "RELATIONSHIP": ("relationship", "relationship"),
                    "CLARIFICATION": ("fact", "clarification"),
                    "TIMELINE": ("plot", "timeline"),
                    "TONE": ("tone", "tone_directive"),
                }
                cp_cat, point_type = CONSOL_MAP.get(category, ("fact", "general"))

                cp_id = self.mnemo.add_point(
                    entity=entity, point_type=point_type, value=entry_content,
                    connects_to=connects_to, reason="", weight="0.7",
                    category=cp_cat, session_id="", source="consolidation",
                )
                if cp_id:
                    stored += 1
                    results["new_entries"].append({
                        "category": category,
                        "content": entry_content[:100] + "..." if len(entry_content) > 100 else entry_content
                    })
                    existing_context.append(entry_content)
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
