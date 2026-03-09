"""
Memory Schema - Structured Memory Definitions for Streamlit Client (v6.0)

This is the client-side schema layer. Server-side dataclasses live in
mnemo_core.py. This module provides:

1. POINT_TYPES — canonical list of valid point types per domain
2. ConnectionPointData — validated dict wrapper for API communication
3. ThreadData / KnotData — same for threads and knots
4. Validation functions for structured data before API calls
5. K2.5 EXTRACTION PROMPT — the exact template for decomposing text
   into ConnectionPoints (used by app.py extraction functions)
6. K2.5 SIGNAL EXTRACTION PROMPT — template for extracting 4o signals
7. Helpers for building context strings from API response dicts

NOTE: These are NOT dataclasses with numpy embeddings (those live on
the server in mnemo_core.py). These are lightweight dicts/validators
for the REST API boundary.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime


# =============================================================================
# CANONICAL POINT TYPES
# =============================================================================

POINT_TYPES = {
    # Character domain
    "character": [
        "role", "age", "trait", "fear", "secret", "motivation",
        "emotional_tell", "speech_pattern", "physical_detail",
        "backstory", "arc_position",
    ],
    # Relationship domain (requires connects_to)
    "relationship": [
        "brother_of", "sister_of", "rival_of", "married_to",
        "mentor_of", "captor_of", "ally_of", "betrayed_by",
        "fears_for", "envies", "resents", "guilt_over",
        "protects", "manipulates", "loves", "distrusts",
    ],
    # Plot domain
    "plot": [
        "event", "consequence", "setup", "payoff", "unresolved",
        "foreshadowing", "twist", "revelation", "chekhov_gun",
        "conflict", "resolution",
    ],
    # Tone/mood domain
    "tone": [
        "scene_register", "humor_type", "tension_level",
        "emotional_shift", "atmosphere", "tone_directive", "do_not",
    ],
    # Setting domain
    "setting": [
        "location", "time_period", "sensory_detail", "atmosphere",
    ],
    # Style domain
    "style": [
        "voice", "pov", "prose_sample", "dialogue_pattern", "vocabulary",
    ],
    # Signal-generated (from 4o post-writing signals)
    "signal": [
        "new_detail", "new_thread", "needs_payoff",
        "direction_taken", "tone_result", "character_evolution",
    ],
    # General fact
    "fact": [
        "fact", "instruction", "clarification", "context",
    ],
}

# Flat set for validation
ALL_POINT_TYPES = set()
for types in POINT_TYPES.values():
    ALL_POINT_TYPES.update(types)

# Types that require connects_to
RELATIONAL_TYPES = set(POINT_TYPES["relationship"])

# Valid categories
VALID_CATEGORIES = {
    "character", "plot", "setting", "theme", "tone", "style", "fact",
}

# Valid sources
VALID_SOURCES = {
    "auto_extract", "file_upload", "manual", "signal", "consolidation",
}

# Valid thread types
VALID_THREAD_TYPES = {
    "character_arc", "plot_line", "theme_thread", "relationship_arc",
}

# Valid thread statuses
VALID_THREAD_STATUSES = {
    "active", "resolved", "dormant", "setup",
}

# Valid knot pivot types
VALID_PIVOT_TYPES = {
    "collision", "revelation", "betrayal", "convergence",
    "divergence", "escalation",
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_point(data: dict) -> Tuple[bool, str]:
    """Validate a ConnectionPoint dict before sending to API.

    Returns (is_valid, error_message).
    """
    entity = data.get("entity", "").strip()
    if not entity:
        return False, "entity is required"

    point_type = data.get("point_type", "").strip()
    if not point_type:
        return False, "point_type is required"

    # Warn (don't reject) on unknown point types — allows extension
    # if point_type not in ALL_POINT_TYPES:
    #     return False, f"unknown point_type: {point_type}"

    # Relational types should have connects_to
    if point_type in RELATIONAL_TYPES and not data.get("connects_to", "").strip():
        return False, f"point_type '{point_type}' requires connects_to"

    category = data.get("category", "fact").lower()
    if category not in VALID_CATEGORIES:
        return False, f"invalid category: {category}. Valid: {VALID_CATEGORIES}"

    weight = data.get("weight", 0.5)
    if not (0.0 <= weight <= 1.0):
        return False, f"weight must be 0.0-1.0, got {weight}"

    return True, ""


def validate_thread(data: dict) -> Tuple[bool, str]:
    """Validate a Thread dict before sending to API."""
    if not data.get("thread_id", "").strip():
        return False, "thread_id is required"
    if not data.get("name", "").strip():
        return False, "name is required"

    thread_type = data.get("thread_type", "plot_line")
    if thread_type not in VALID_THREAD_TYPES:
        return False, f"invalid thread_type: {thread_type}. Valid: {VALID_THREAD_TYPES}"

    return True, ""


def validate_knot(data: dict) -> Tuple[bool, str]:
    """Validate a Knot dict before sending to API."""
    if not data.get("knot_id", "").strip():
        return False, "knot_id is required"
    if not data.get("name", "").strip():
        return False, "name is required"

    thread_ids = data.get("thread_ids", [])
    if not thread_ids or len(thread_ids) < 2:
        return False, "at least 2 thread_ids required for a knot"

    pivot_type = data.get("pivot_type", "collision")
    if pivot_type not in VALID_PIVOT_TYPES:
        return False, f"invalid pivot_type: {pivot_type}. Valid: {VALID_PIVOT_TYPES}"

    return True, ""


# =============================================================================
# DATA BUILDERS (for API requests)
# =============================================================================

def build_point(entity: str, point_type: str, value: str = "",
                connects_to: str = "", reason: str = "",
                weight: float = 0.5, category: str = "fact",
                session_id: str = "", source: str = "auto_extract",
                thread_id: str = "", position: int = -1) -> dict:
    """Build a validated ConnectionPoint dict for the API."""
    data = {
        "entity": entity.strip(),
        "point_type": point_type.strip(),
        "value": value.strip(),
        "connects_to": connects_to.strip(),
        "reason": reason.strip(),
        "weight": max(0.0, min(1.0, weight)),
        "category": category.lower().strip(),
        "session_id": session_id,
        "source": source,
        "thread_id": thread_id,
        "position": position,
    }
    valid, error = validate_point(data)
    if not valid:
        raise ValueError(f"Invalid ConnectionPoint: {error}")
    return data


def build_thread(thread_id: str, name: str, entity: str = "",
                 thread_type: str = "plot_line", session_id: str = "",
                 point_ids: Optional[List[str]] = None) -> dict:
    """Build a validated Thread dict for the API."""
    data = {
        "thread_id": thread_id.strip(),
        "name": name.strip(),
        "entity": entity.strip(),
        "thread_type": thread_type,
        "session_id": session_id,
        "point_ids": point_ids or [],
    }
    valid, error = validate_thread(data)
    if not valid:
        raise ValueError(f"Invalid Thread: {error}")
    return data


def build_knot(knot_id: str, name: str, thread_ids: List[str],
               pivot_type: str = "collision", reason: str = "",
               session_id: str = "",
               active_points: Optional[Dict[str, List[str]]] = None) -> dict:
    """Build a validated Knot dict for the API."""
    data = {
        "knot_id": knot_id.strip(),
        "name": name.strip(),
        "thread_ids": [t.strip() for t in thread_ids],
        "pivot_type": pivot_type,
        "reason": reason.strip(),
        "session_id": session_id,
        "active_points": active_points or {},
    }
    valid, error = validate_knot(data)
    if not valid:
        raise ValueError(f"Invalid Knot: {error}")
    return data


# =============================================================================
# CONTEXT STRING FORMATTERS (from API response dicts)
# =============================================================================

def point_to_context(p: dict) -> str:
    """Format a ConnectionPoint API response dict for LLM injection."""
    entity = p.get("entity", "")
    pt = p.get("point_type", "")
    connects = p.get("connects_to", "")
    value = p.get("value", "")
    reason = p.get("reason", "")

    if connects:
        s = f"{entity} — {pt} → {connects}: {value}"
    else:
        s = f"{entity} — {pt}: {value}"
    if reason:
        s += f" ({reason})"
    return s


def points_to_grouped_context(points: List[dict]) -> str:
    """Group ConnectionPoints by entity for structured LLM injection.

    Output format:
        [ALISTAIR FITZROY]
        - role: Professor of pharmacology (cover for Red Rose work)
        - fears: developing dementia (drives obsession with legacy)
        - tone: clinical, cold menace (power through precision)

        [SEBASTIAN CARLISLE]
        - ...
    """
    by_entity: Dict[str, List[dict]] = {}
    for p in points:
        entity = p.get("entity", "Unknown")
        if entity not in by_entity:
            by_entity[entity] = []
        by_entity[entity].append(p)

    parts = []
    for entity, entity_points in by_entity.items():
        parts.append(f"[{entity.upper()}]")
        for p in entity_points:
            pt = p.get("point_type", "")
            connects = p.get("connects_to", "")
            value = p.get("value", "")
            reason = p.get("reason", "")

            if connects:
                line = f"- {pt} → {connects}: {value}"
            else:
                line = f"- {pt}: {value}"
            if reason:
                line += f" ({reason})"
            parts.append(line)
        parts.append("")  # blank line between entities

    return "\n".join(parts).strip()


def thread_to_context(thread: dict, traced_points: List[dict] = None) -> str:
    """Format a Thread + traced points for LLM injection.

    Output format:
        [THREAD: Sebastian's Captivity Arc]
        Position: 7/12 | Tension: 0.85 | Status: active
        Tone trajectory: clinical_dread → dark_humor → dissociation

        PRECEDING:
        ← Drug protocols began (Alistair frames as treatment)
        ← Isabella enters (control transfers)
        ← Tattoo scene (branding — trauma marker)
    """
    name = thread.get("name", "Unknown Thread")
    pos = thread.get("current_position", 0)
    total = len(thread.get("points", []))
    tension = thread.get("tension_level", 0.5)
    status = thread.get("status", "active")
    tones = thread.get("tone_trajectory", [])

    parts = [f"[THREAD: {name}]"]
    parts.append(f"Position: {pos}/{total} | Tension: {tension:.2f} | Status: {status}")
    if tones:
        parts.append(f"Tone trajectory: {' → '.join(tones[-5:])}")

    if traced_points:
        parts.append("")
        parts.append("PRECEDING:")
        for p in traced_points:
            value = p.get("value", p.get("content", ""))[:80]
            entity = p.get("entity", "")
            pt = p.get("point_type", "")
            if entity and pt:
                parts.append(f"← {entity}.{pt}: {value}")
            else:
                parts.append(f"← {value}")

    return "\n".join(parts)


def knot_to_context(knot_context: dict) -> str:
    """Format a knot context response for LLM injection.

    Output format:
        [KNOT: Arrival at Blackwood Estate]
        Pivot: collision | Tone shift: dread → horror
        Reason: Sebastian enters Isabella's territory

        Thread: Sebastian's Captivity (tension: 0.85, tone: dark_humor)
          - ...active points...
        Thread: Isabella's Obsession (tension: 0.70, tone: fixation)
          - ...active points...
    """
    name = knot_context.get("name", "Unknown Knot")
    pivot = knot_context.get("pivot_type", "collision")
    tone_shift = knot_context.get("tone_shift", "")
    reason = knot_context.get("reason", "")

    parts = [f"[KNOT: {name}]"]
    header = f"Pivot: {pivot}"
    if tone_shift:
        header += f" | Tone shift: {tone_shift}"
    parts.append(header)
    if reason:
        parts.append(f"Reason: {reason}")

    thread_ctx = knot_context.get("thread_context", {})
    if thread_ctx:
        parts.append("")
        for tid, tdata in thread_ctx.items():
            thread_name = tdata.get("thread_name", tid)
            tension = tdata.get("tension", 0.5)
            tone = tdata.get("tone", "")
            parts.append(f"Thread: {thread_name} (tension: {tension:.2f}, tone: {tone})")
            for ap in tdata.get("active_points", []):
                value = ap.get("value", ap.get("content", ""))[:80]
                parts.append(f"  - {value}")
            for bp in tdata.get("buildup", []):
                value = bp.get("value", bp.get("content", ""))[:60]
                parts.append(f"  ← {value}")

    unresolved = knot_context.get("unresolved", [])
    if unresolved:
        parts.append("")
        parts.append("UNRESOLVED at this knot:")
        for u in unresolved:
            parts.append(f"  ⚡ {u}")

    return "\n".join(parts)


# =============================================================================
# K2.5 EXTRACTION PROMPT TEMPLATE
# =============================================================================

EXTRACTION_PROMPT_TEMPLATE = """Analyze this text and decompose ALL information into atomic connection points.

Each point should be ONE specific fact — never combine multiple facts.

TEXT TO ANALYZE:
{text}

Return ONLY a JSON object:
{{
  "points": [
    {{
      "entity": "character/place/object name",
      "point_type": "attribute type (see list below)",
      "connects_to": "target entity or empty string",
      "value": "the specific atomic fact",
      "reason": "WHY this matters narratively",
      "weight": 0.8,
      "category": "character|plot|setting|theme|tone|style|fact"
    }}
  ],
  "thread_updates": [
    {{
      "thread": "thread_name_suggestion",
      "action": "append|create|activate",
      "point_indices": [0, 1],
      "tension_change": "+0.05",
      "tone_update": "new tone state if changed"
    }}
  ],
  "new_knots": [
    {{
      "name": "descriptive name",
      "threads": ["thread_ids_crossing_here"],
      "pivot_type": "collision|revelation|betrayal|convergence|escalation",
      "reason": "why these threads collide"
    }}
  ]
}}

POINT TYPES:
  Characters: role, age, trait, fear, secret, motivation, emotional_tell,
             speech_pattern, physical_detail, backstory, arc_position
  Relationships: brother_of, rival_of, married_to, mentor_of, captor_of,
                ally_of, betrayed_by, fears_for, envies, resents, guilt_over
  Plot: event, consequence, setup, payoff, unresolved, foreshadowing,
        twist, revelation, chekhov_gun
  Tone: scene_register, humor_type, tension_level, emotional_shift,
        atmosphere, tone_directive, do_not
  Setting: location, time_period, sensory_detail, atmosphere
  Style: voice, pov, prose_sample, dialogue_pattern, vocabulary

RULES:
- ONE fact per point. Split compound facts.
- ALWAYS include reason. Facts without narrative context are incomplete.
- Relationships get TWO points (one from each entity's perspective).
- Tone notes are critical. Extract HOW scenes should feel.
- connects_to is empty string for standalone traits/facts.
- Weight: 0.3 minor detail, 0.5 standard, 0.7 important, 0.9 critical, 1.0 foundational.
- thread_updates and new_knots are optional — only include if the text
  clearly establishes narrative sequences or thread intersections."""


def get_extraction_prompt(text: str, max_chars: int = 8000) -> str:
    """Build the K2.5 extraction prompt for a given text chunk."""
    truncated = text[:max_chars]
    return EXTRACTION_PROMPT_TEMPLATE.format(text=truncated)


# =============================================================================
# K2.5 SIGNAL EXTRACTION PROMPT (for 4o post-writing signals)
# =============================================================================

SIGNAL_INSTRUCTION = """After writing the scene, append a signal block wrapped in <signals> tags.
This is for the story engine, not the user. Include ALL that apply:

<signals>
{{
  "direction_taken": "how you interpreted the creative brief",
  "new_threads": ["any new details you introduced that need tracking"],
  "tone_result": "the emotional register you actually landed on",
  "needs_from_next": ["what the next scene should address"],
  "character_evolution": "any character state changes for continuity",
  "unresolved_setup": ["anything you set up that needs future payoff"]
}}
</signals>"""


def parse_signals_from_response(response: str) -> Tuple[str, Optional[dict]]:
    """Extract and strip <signals> block from 4o response.

    Returns (clean_response, signals_dict_or_None).
    """
    import re
    match = re.search(r'<signals>\s*(\{.*?\})\s*</signals>', response, re.DOTALL)
    if not match:
        return response, None

    clean = response[:match.start()].rstrip() + response[match.end():]
    try:
        import json
        signals = json.loads(match.group(1))
        return clean.strip(), signals
    except (json.JSONDecodeError, ValueError):
        return clean.strip(), None


def signals_to_points(signals: dict, session_id: str = "") -> List[dict]:
    """Convert 4o signals into ConnectionPoint dicts for storage.

    Maps signal fields to structured points:
      new_threads → new_detail points
      tone_result → tone_result point
      unresolved_setup → chekhov_gun points
      character_evolution → arc_position point
      needs_from_next → needs_payoff points
    """
    points = []

    # new_threads → new_detail
    for detail in signals.get("new_threads", []):
        if detail and detail.strip():
            points.append({
                "entity": "Story",
                "point_type": "new_detail",
                "value": detail.strip(),
                "reason": "Introduced by 4o during writing — needs tracking",
                "weight": 0.7,
                "category": "plot",
                "session_id": session_id,
                "source": "signal",
            })

    # tone_result
    tone = signals.get("tone_result", "")
    if tone and tone.strip():
        points.append({
            "entity": "Scene",
            "point_type": "tone_result",
            "value": tone.strip(),
            "reason": "4o's actual tonal register for this scene",
            "weight": 0.6,
            "category": "tone",
            "session_id": session_id,
            "source": "signal",
        })

    # unresolved_setup → chekhov_gun
    for setup in signals.get("unresolved_setup", []):
        if setup and setup.strip():
            points.append({
                "entity": "Story",
                "point_type": "chekhov_gun",
                "value": setup.strip(),
                "reason": "Set up by 4o but not paid off — track for future",
                "weight": 0.8,
                "category": "plot",
                "session_id": session_id,
                "source": "signal",
            })

    # character_evolution
    evolution = signals.get("character_evolution", "")
    if evolution and evolution.strip():
        points.append({
            "entity": "Character",
            "point_type": "arc_position",
            "value": evolution.strip(),
            "reason": "Character state change from this scene",
            "weight": 0.7,
            "category": "character",
            "session_id": session_id,
            "source": "signal",
        })

    # needs_from_next → needs_payoff
    for need in signals.get("needs_from_next", []):
        if need and need.strip():
            points.append({
                "entity": "Story",
                "point_type": "needs_payoff",
                "value": need.strip(),
                "reason": "4o flagged this as needed in the next scene",
                "weight": 0.8,
                "category": "plot",
                "session_id": session_id,
                "source": "signal",
            })

    return points
