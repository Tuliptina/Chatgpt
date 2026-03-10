"""
Context Engine - Deep Context Generation and Memory Consolidation

v6.9.1 changes:
- FIX: Updated K2 cost rates to current OpenRouter pricing ($0.55/$2.20 per M)
- FIX: consolidate_memories() now uses robust JSON parsing (brace-depth matching)
  instead of fragile markdown fence stripping that failed on K2 preamble text.

v6.0 changes:
- consolidate_memories() routes to MEMORY_MODEL_ID (K2) instead of GPT-4o
- TONE entries added to consolidation output categories
- build_rich_context() groups by entity when ConnectionPoints are present

v5.1: build_rich_context() is pure formatter, consolidate at priority=1.2.
"""

import json
import re
import requests
from datetime import datetime
from typing import List, Dict, Tuple
from mnemo_client import MnemoClient

# v6.9.1: Updated to current OpenRouter K2 pricing (was $0.47/$2.00)
MEMORY_MODEL_ID = "moonshotai/kimi-k2"
MEMORY_COST_INPUT = 0.55 / 1_000_000
MEMORY_COST_OUTPUT = 2.20 / 1_000_000


# v6.9.1: Robust JSON extraction (same logic as app.py)
def _extract_json(raw: str) -> dict:
    clean = raw.strip()
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    if "```" in clean:
        for pattern in [r'```json\s*\n?(.*?)\n?```', r'```\s*\n?(.*?)\n?```']:
            match = re.search(pattern, clean, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    first_brace = clean.find('{')
    if first_brace >= 0:
        depth = 0
        for i in range(first_brace, len(clean)):
            if clean[i] == '{': depth += 1
            elif clean[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(clean[first_brace:i+1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break
    return {}


class ContextEngine:
    def __init__(self, mnemo_client: MnemoClient, openrouter_key: str = None):
        self.mnemo = mnemo_client
        self.openrouter_key = openrouter_key

    def build_rich_context(self, query: str, memories: List[Dict]) -> Tuple[str, Dict]:
        metadata = {"facts_included": 0, "context_included": 0, "style_included": 0, "tone_included": 0}
        context_parts = []
        seen = set()
        for r in memories:
            content = r.get("content", "")
            if not content:
                continue
            content_key = content[:80].lower()
            if content_key in seen:
                continue
            seen.add(content_key)
            if any(content.startswith(f"[{tag}]") for tag in ("CONTEXT", "RELATIONSHIP", "CLARIFICATION", "TIMELINE")):
                context_parts.append(content); metadata["context_included"] += 1
            elif any(tag in content for tag in ("PROSE_SAMPLE", "VOICE", "DIALOGUE_SAMPLE", "VOCABULARY")):
                context_parts.append(content); metadata["style_included"] += 1
            elif content.startswith("[TONE]") or any(tag in content for tag in ("scene_register", "humor_type", "emotional_shift", "do_not", "atmosphere")):
                context_parts.append(content); metadata["tone_included"] += 1
            else:
                context_parts.append(content); metadata["facts_included"] += 1
        context_string = "\n".join(f"- {p}" for p in context_parts) if context_parts else ""
        return context_string, metadata

    def consolidate_memories(self, openrouter_key: str = None) -> Dict:
        api_key = openrouter_key or self.openrouter_key
        if not api_key:
            return {"error": "No OpenRouter API key", "created": 0}

        results = {
            "timestamp": datetime.now().isoformat(), "memories_analyzed": 0,
            "new_entries": [], "created": 0, "cost": 0.0, "k2_returned": 0, "dedup_rejected": 0,
        }

        try:
            all_memories = self.mnemo.list_memories() or []
            all_points = []
            if hasattr(self.mnemo, 'list_points'):
                all_points = self.mnemo.list_points(limit=500) or []
            total = len(all_memories) + len(all_points)
            if total == 0:
                return {"error": "No memories to consolidate or API unreachable", "created": 0}
            results["memories_analyzed"] = total
        except Exception as e:
            return {"error": f"Fetch error: {e}", "created": 0}

        facts, existing_context = [], []
        for mem in all_memories:
            content = mem.get("content", "")
            if any(content.startswith(f"[{tag}]") for tag in ("CONTEXT", "RELATIONSHIP", "CLARIFICATION", "TIMELINE", "TONE")):
                existing_context.append(content)
            elif content.startswith("[SESSION]") or content.startswith("[CONVERSATION]"):
                continue
            else:
                facts.append(content)

        CONTEXT_TYPES = {"context_note", "clarification", "relationship", "tone_directive", "timeline"}
        for cp in all_points:
            entity, value, point_type, conn = cp.get("entity", ""), cp.get("value", ""), cp.get("point_type", ""), cp.get("connects_to", "")
            line = f"{entity} → {conn}: {value}" if conn else (f"{entity}: {value}" if entity else value)
            if point_type in CONTEXT_TYPES:
                existing_context.append(line)
            else:
                facts.append(line)

        if not facts:
            return {"error": "No facts to analyze (only context entries exist)", "created": 0}

        facts_text = "\n".join(facts[:100])
        consolidation_prompt = f"""You are analyzing story memories to generate NEW deep context entries.

EXISTING FACTS TO ANALYZE:
{facts_text}

YOUR TASK — generate entries in these categories:

CONTEXT: Explain what a fact MEANS. What could be misunderstood? What's the subtext?
RELATIONSHIP: How two characters relate. Power dynamics, emotional undercurrent, history.
CLARIFICATION: Prevent specific misinterpretations.
TIMELINE: Sequence of events with causation.
TONE: How specific characters or scenes should FEEL.

Generate 5-15 NEW entries. ONLY use information explicitly stated or directly implied by the facts above. Do NOT invent new details.

Return ONLY a JSON object:
{{
  "entries": [
    {{"category": "CONTEXT", "content": "your insight"}},
    {{"category": "RELATIONSHIP", "content": "your insight"}}
  ]
}}"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": MEMORY_MODEL_ID,
                    "messages": [
                        {"role": "system", "content": "You are a story analyst. Return ONLY valid JSON with an 'entries' array. No markdown, no explanation — pure JSON only."},
                        {"role": "user", "content": consolidation_prompt}
                    ],
                    "temperature": 0.2, "max_tokens": 3000,
                },
                timeout=90
            )
            if response.status_code != 200:
                return {"error": f"API error: {response.status_code} — {response.text[:200]}", "created": 0}

            data = response.json()
            usage = data.get("usage", {})
            results["cost"] = (usage.get("prompt_tokens", 0) * MEMORY_COST_INPUT + usage.get("completion_tokens", 0) * MEMORY_COST_OUTPUT)

            content = data["choices"][0]["message"]["content"]
            # v6.9.1: Robust JSON extraction
            parsed = _extract_json(content)
            if not parsed:
                return {"error": f"JSON parse error — raw: {content[:200]}", "created": 0}

            new_entries = parsed.get("entries", [])
            results["k2_returned"] = len(new_entries)

        except Exception as e:
            return {"error": f"API error: {e}", "created": 0}

        if not new_entries:
            for key in ("results", "memories", "items", "data"):
                new_entries = parsed.get(key, [])
                if new_entries:
                    results["k2_returned"] = len(new_entries)
                    break
        if not new_entries:
            return {**results, "error": f"K2 returned JSON but no entries found. Keys: {list(parsed.keys())}", "created": 0}

        stored = 0
        for entry in new_entries:
            category = entry.get("category", "CONTEXT").upper()
            entry_content = entry.get("content", "")
            if not entry_content or len(entry_content) < 20:
                continue
            is_duplicate = False
            for existing in existing_context:
                content_words = set(entry_content.lower().split())
                existing_words = set(existing.lower().split())
                if not content_words: break
                overlap = len(content_words & existing_words) / len(content_words)
                if overlap > 0.85:
                    is_duplicate = True; results["dedup_rejected"] += 1; break
            if is_duplicate:
                continue
            try:
                entity, connects_to = "Story", ""
                if category == "RELATIONSHIP":
                    arrow_match = re.match(r'(\w+)\s*(?:→|->)\s*(\w+)', entry_content)
                    if arrow_match: entity, connects_to = arrow_match.group(1), arrow_match.group(2)
                else:
                    name_match = re.match(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', entry_content)
                    if name_match: entity = name_match.group(1)
                CONSOL_MAP = {
                    "CONTEXT": ("fact", "context_note"), "RELATIONSHIP": ("relationship", "relationship"),
                    "CLARIFICATION": ("fact", "clarification"), "TIMELINE": ("plot", "timeline"),
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

    def should_consolidate(self, last_consolidation=None, message_count: int = 0, new_memories_since: int = 0) -> bool:
        if last_consolidation:
            hours_since = (datetime.now() - last_consolidation).total_seconds() / 3600
            if hours_since >= 24: return True
        if message_count >= 100: return True
        if new_memories_since >= 50: return True
        return False
