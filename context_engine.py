"""
Context Engine - Deep Context Generation, Memory Consolidation, and Thread/Knot Synthesis

v7.1 changes:
- NEW: Thread/knot generation moved here from per-message extraction (app.py).
  Threads and knots created during consolidation have session_id="" and
  source="consolidation", making them immune to session deletion.
  This fixes the bug where deleting a session wiped structural narrative elements.
- NEW: _generate_threads_and_knots() method — analyzes all CPs to find
  narrative sequences (threads) and intersection points (knots).

v6.9.1 changes:
- FIX: Updated K2 cost rates to current OpenRouter pricing ($0.55/$2.20 per M)
- FIX: consolidate_memories() now uses robust JSON parsing (brace-depth matching)

v6.0 changes:
- consolidate_memories() routes to MEMORY_MODEL_ID (K2) instead of GPT-4o
- TONE entries added to consolidation output categories
- build_rich_context() groups by entity when ConnectionPoints are present
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
            "threads_created": 0, "knots_created": 0,
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

        # =====================================================================
        # v7.1: THREAD/KNOT STRUCTURAL PASS
        # Moved from per-message extraction to consolidation so threads/knots
        # have session_id="" and are immune to session deletion.
        # =====================================================================
        if len(all_points) >= 5:
            try:
                t_created, k_created, tk_cost = self._generate_threads_and_knots(
                    all_points, api_key)
                results["threads_created"] = t_created
                results["knots_created"] = k_created
                results["cost"] += tk_cost
            except Exception as e:
                print(f"[CONSOLIDATION] Thread/knot pass failed: {e}")

        return results

    def _generate_threads_and_knots(self, all_points: List[Dict],
                                     api_key: str) -> Tuple[int, int, float]:
        """Analyze all CPs to identify narrative threads and knots.

        Creates threads/knots with session_id="" so they survive session deletion.
        Returns (threads_created, knots_created, cost).
        """
        # Build summary of all CPs for the LLM
        summary_lines = []
        for cp in all_points[:150]:  # Cap to avoid token overflow
            cp_id = cp.get("id", "?")
            entity = cp.get("entity", "?")
            cat = cp.get("category", "?")
            value = cp.get("value", "")[:80]
            conn = cp.get("connects_to", "")
            conn_str = f" → {conn}" if conn else ""
            summary_lines.append(f"[{cp_id}] {entity}{conn_str} ({cat}): {value}")

        summary_text = "\n".join(summary_lines)

        # Get existing threads to avoid duplicates
        existing_threads = []
        try:
            existing_threads = self.mnemo.get_active_threads() or []
        except Exception:
            pass
        existing_names = {t.get("name", "").lower() for t in existing_threads}

        prompt = f"""You are analyzing a story's memory database to identify narrative structure.

EXISTING CONNECTION POINTS:
{summary_text}

EXISTING THREADS (do NOT duplicate these):
{', '.join(existing_names) if existing_names else '(none yet)'}

IDENTIFY:
1. THREADS: Narrative sequences where 3+ CPs form a chronological or causal chain.
   Each thread needs a name, the primary entity, a type (plot_line, character_arc,
   thematic_thread, or setting_evolution), and the CP IDs that belong to it.

2. KNOTS: Points where two or more threads collide or intersect.
   Each knot needs a name, the thread names that intersect, and why.

RULES:
- Only create threads from 3+ CPs that form a clear sequence.
- Only create knots where threads genuinely intersect.
- Do NOT recreate threads that already exist.
- If no clear new threads or knots exist, return empty arrays.
- Use the exact CP IDs (e.g., "cp_abc123") from the data above.

Return ONLY a valid JSON object:
{{
  "threads": [
    {{"name": "Captivity Arc", "entity": "Sebastian", "type": "plot_line", "cp_ids": ["cp_abc", "cp_def", "cp_ghi"]}}
  ],
  "knots": [
    {{"name": "Blackwood Arrival", "thread_names": ["Captivity Arc", "Isabella's Obsession"], "reason": "Sebastian enters Isabella's territory"}}
  ]
}}"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": MEMORY_MODEL_ID,
                    "messages": [
                        {"role": "system", "content": "You are a narrative structure analyst. Return ONLY valid JSON. No markdown, no explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2, "max_tokens": 2000,
                },
                timeout=90
            )
            if response.status_code != 200:
                print(f"[THREADS] API error {response.status_code}")
                return 0, 0, 0

            data = response.json()
            usage = data.get("usage", {})
            cost = (usage.get("prompt_tokens", 0) * MEMORY_COST_INPUT +
                    usage.get("completion_tokens", 0) * MEMORY_COST_OUTPUT)

            raw = data["choices"][0]["message"]["content"]
            parsed = _extract_json(raw)
            if not parsed:
                print(f"[THREADS] JSON parse failed. Preview: {raw[:300]}")
                return 0, 0, cost

            threads = parsed.get("threads", [])
            knots = parsed.get("knots", [])
            print(f"[THREADS] K2 returned {len(threads)} threads, {len(knots)} knots")

        except Exception as e:
            print(f"[THREADS] Exception: {e}")
            return 0, 0, 0

        # Store threads (with session_id="" — immune to session deletion)
        threads_created = 0
        thread_name_to_id = {}
        for th in threads:
            t_name = th.get("name", "")
            if not t_name or t_name.lower() in existing_names:
                continue  # Skip duplicates
            t_id = "thread_" + re.sub(r'[^a-z0-9]', '_', t_name.lower())[:20]
            thread_name_to_id[t_name] = t_id
            cp_ids = th.get("cp_ids", [])
            try:
                self.mnemo.add_thread(
                    thread_id=t_id, name=t_name,
                    entity=th.get("entity", ""),
                    thread_type=th.get("type", "plot_line"),
                    session_id="",  # NOT tied to any session
                    point_ids=cp_ids,
                )
                threads_created += 1
            except Exception as e:
                print(f"[THREADS] Failed to create thread '{t_name}': {e}")

        # Store knots (with session_id="" — immune to session deletion)
        knots_created = 0
        for kn in knots:
            k_name = kn.get("name", "")
            if not k_name:
                continue
            k_id = "knot_" + re.sub(r'[^a-z0-9]', '_', k_name.lower())[:20]
            t_ids = [
                thread_name_to_id.get(tn, "thread_" + re.sub(r'[^a-z0-9]', '_', tn.lower())[:20])
                for tn in kn.get("thread_names", [])
            ]
            try:
                self.mnemo.add_knot(
                    knot_id=k_id, name=k_name,
                    thread_ids=t_ids,
                    pivot_type="collision",
                    reason=kn.get("reason", ""),
                    session_id="",  # NOT tied to any session
                )
                knots_created += 1
            except Exception as e:
                print(f"[THREADS] Failed to create knot '{k_name}': {e}")

        return threads_created, knots_created, cost

    def should_consolidate(self, last_consolidation=None, message_count: int = 0, new_memories_since: int = 0) -> bool:
        if last_consolidation:
            hours_since = (datetime.now() - last_consolidation).total_seconds() / 3600
            if hours_since >= 24: return True
        if message_count >= 100: return True
        if new_memories_since >= 50: return True
        return False
