"""
Signal Protocol - Inter-Model Communication Layer (v6.0)

Orchestrates the signal exchange between GPT-4o (writing) and K2.5 (memory):

  4o writes scene → emits <signals> block → SignalProcessor converts to
  ConnectionPoints + thread updates + narrative state changes → stored
  in Mnemo → K2.5 uses updated state to build next creative brief.

Two main components:

1. SignalProcessor — Processes 4o's post-writing signals:
   - Converts signal fields to ConnectionPoints (via memory_schema)
   - Updates thread positions, tension levels, tone trajectories
   - Tracks narrative state (active threads, unresolved setups)
   - Maintains signal history for prefetch engine predictions

2. BriefBuilder — Builds K2.5→4o creative briefs:
   - Assembles narrative position from active threads
   - Pulls relevant ConnectionPoints via graph search
   - Traces thread context (preceding + upcoming)
   - Expands knot context for multi-thread scenes
   - Injects tone directives
   - Formats everything into structured sections for 4o

Both are used by app.py's handle_message() and the prefetch engine.
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from memory_schema import (
    parse_signals_from_response,
    signals_to_points,
    points_to_grouped_context,
    thread_to_context,
    knot_to_context,
    SIGNAL_INSTRUCTION,
)


# =============================================================================
# NARRATIVE STATE — tracks where the story is right now
# =============================================================================

@dataclass
class NarrativeState:
    """Snapshot of the current story state across all threads.

    Updated after each scene by SignalProcessor. Used by BriefBuilder
    to construct creative briefs and by PrefetchEngine for predictions.
    """
    active_thread_ids: List[str] = field(default_factory=list)
    thread_positions: Dict[str, int] = field(default_factory=dict)
    thread_tensions: Dict[str, float] = field(default_factory=dict)
    thread_tones: Dict[str, str] = field(default_factory=dict)  # last tone per thread
    unresolved_setups: List[str] = field(default_factory=list)
    needs_payoff: List[str] = field(default_factory=list)
    recent_entities: List[str] = field(default_factory=list)  # entities active in last 3 scenes
    last_tone_result: str = ""
    last_direction_taken: str = ""
    scenes_written: int = 0
    last_updated: float = field(default_factory=time.time)

    def fingerprint(self) -> str:
        """Generate a narrative fingerprint for prefetch cache keys.

        Two prompts that map to the same fingerprint should get the
        same cached creative brief — keyed on story state, not wording.
        """
        components = []
        for tid in sorted(self.active_thread_ids):
            pos = self.thread_positions.get(tid, 0)
            components.append(f"{tid}:{pos}")
        # Include last 3 unresolved setups
        for setup in self.unresolved_setups[-3:]:
            components.append(f"u:{setup[:30]}")
        # Include scene count for temporal uniqueness
        components.append(f"s:{self.scenes_written}")
        raw = "|".join(components)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "active_thread_ids": list(self.active_thread_ids),
            "thread_positions": dict(self.thread_positions),
            "thread_tensions": {k: round(v, 3) for k, v in self.thread_tensions.items()},
            "thread_tones": dict(self.thread_tones),
            "unresolved_setups": list(self.unresolved_setups),
            "needs_payoff": list(self.needs_payoff),
            "recent_entities": list(self.recent_entities),
            "last_tone_result": self.last_tone_result,
            "scenes_written": self.scenes_written,
            "fingerprint": self.fingerprint(),
        }


# =============================================================================
# SIGNAL HISTORY — rolling window for predictions
# =============================================================================

@dataclass
class SignalRecord:
    """One processed signal from 4o, with timestamp."""
    timestamp: float
    signals: dict               # Raw signal dict from 4o
    points_created: List[str]   # CP IDs created from this signal
    thread_updates: List[str]   # Thread IDs updated
    scene_summary: str = ""     # Brief description of what was written

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "signals": self.signals,
            "points_created": self.points_created,
            "thread_updates": self.thread_updates,
            "scene_summary": self.scene_summary,
        }


class SignalHistory:
    """Rolling window of recent signal records for prediction."""

    def __init__(self, max_size: int = 20):
        self._records: List[SignalRecord] = []
        self._max_size = max_size

    def add(self, record: SignalRecord):
        self._records.append(record)
        if len(self._records) > self._max_size:
            self._records = self._records[-self._max_size:]

    def recent(self, n: int = 5) -> List[SignalRecord]:
        return self._records[-n:]

    def last(self) -> Optional[SignalRecord]:
        return self._records[-1] if self._records else None

    def all_unresolved(self) -> List[str]:
        """Collect all unresolved setups across recent signals."""
        setups = []
        for r in self._records:
            for s in r.signals.get("unresolved_setup", []):
                if s and s not in setups:
                    setups.append(s)
        return setups

    def all_needs(self) -> List[str]:
        """Collect all needs_from_next across recent signals."""
        needs = []
        for r in self._records:
            for n in r.signals.get("needs_from_next", []):
                if n and n not in needs:
                    needs.append(n)
        return needs

    def recent_entities(self, n: int = 3) -> List[str]:
        """Extract entity names mentioned in recent signals."""
        import re
        entities = []
        for r in self._records[-n:]:
            # Extract capitalized names from signal text
            for field_val in r.signals.values():
                if isinstance(field_val, str):
                    names = re.findall(r'\b[A-Z][a-z]{2,}\b', field_val)
                    entities.extend(names)
                elif isinstance(field_val, list):
                    for item in field_val:
                        if isinstance(item, str):
                            names = re.findall(r'\b[A-Z][a-z]{2,}\b', item)
                            entities.extend(names)
        # Deduplicate preserving order
        seen = set()
        unique = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique[:10]

    def to_list(self) -> List[dict]:
        return [r.to_dict() for r in self._records]

    def __len__(self) -> int:
        return len(self._records)


# =============================================================================
# SIGNAL PROCESSOR — handles 4o's post-writing signals
# =============================================================================

class SignalProcessor:
    """Processes 4o's post-writing signals into memory and state updates.

    Called by app.py after each creative writing response. Flow:
      1. parse_signals_from_response() strips <signals> from 4o output
      2. process() converts signals → ConnectionPoints, thread updates
      3. Updates NarrativeState for the prefetch engine
      4. Stores signal in SignalHistory for prediction patterns

    Usage in app.py:
        processor = st.session_state.signal_processor
        clean_response, signals = parse_signals_from_response(raw_response)
        if signals:
            result = processor.process(signals, mnemo_client, session_id)
    """

    def __init__(self):
        self.state = NarrativeState()
        self.history = SignalHistory(max_size=20)

    def process(self, signals: dict, mnemo_client,
                session_id: str = "", user_prompt: str = "") -> dict:
        """Process a 4o signal dict into memory operations.

        Args:
            signals: Parsed dict from parse_signals_from_response()
            mnemo_client: MnemoClient for storing ConnectionPoints
            session_id: Current session ID
            user_prompt: The user's original prompt (for scene summary)

        Returns:
            dict with counts: points_created, threads_updated, etc.
        """
        result = {
            "points_created": 0,
            "cp_ids": [],
            "threads_updated": [],
            "narrative_state_changed": False,
        }

        # 1. Convert signals to ConnectionPoints
        points = signals_to_points(signals, session_id=session_id)
        stored_ids = []
        for p in points:
            try:
                cp_id = mnemo_client.add_point(**p)
                if cp_id:
                    stored_ids.append(cp_id)
                    result["points_created"] += 1
            except Exception:
                continue
        result["cp_ids"] = stored_ids

        # 2. Update narrative state from signals
        self._update_state(signals)
        result["narrative_state_changed"] = True

        # 3. Record in signal history
        record = SignalRecord(
            timestamp=time.time(),
            signals=signals,
            points_created=stored_ids,
            thread_updates=result["threads_updated"],
            scene_summary=user_prompt[:100] if user_prompt else "",
        )
        self.history.add(record)

        # 4. Update state with history-derived data
        self.state.recent_entities = self.history.recent_entities(n=3)
        self.state.unresolved_setups = self.history.all_unresolved()
        self.state.needs_payoff = self.history.all_needs()

        return result

    def _update_state(self, signals: dict):
        """Update NarrativeState from signal fields."""
        # Tone result
        tone = signals.get("tone_result", "")
        if tone:
            self.state.last_tone_result = tone

        # Direction taken
        direction = signals.get("direction_taken", "")
        if direction:
            self.state.last_direction_taken = direction

        # Scene counter
        self.state.scenes_written += 1
        self.state.last_updated = time.time()

    def update_threads_from_server(self, mnemo_client):
        """Refresh narrative state from server's thread data.

        Call this on startup or after loading a session to sync
        the local NarrativeState with server-side thread positions.
        """
        try:
            active = mnemo_client.get_active_threads()
            self.state.active_thread_ids = [t["id"] for t in active]
            for t in active:
                tid = t["id"]
                self.state.thread_positions[tid] = t.get("current_position", 0)
                self.state.thread_tensions[tid] = t.get("tension_level", 0.5)
                tones = t.get("tone_trajectory", [])
                if tones:
                    self.state.thread_tones[tid] = tones[-1]
        except Exception:
            pass

    def get_state(self) -> NarrativeState:
        return self.state

    def get_fingerprint(self) -> str:
        return self.state.fingerprint()


# =============================================================================
# BRIEF BUILDER — constructs K2.5→4o creative briefs
# =============================================================================

class BriefBuilder:
    """Builds structured creative briefs for 4o from memory + narrative state.

    The brief is what K2.5 gives to 4o before writing a scene. It contains:
    - Narrative position (which thread, where on it, tension, tone)
    - Relevant character/plot ConnectionPoints
    - Thread trace (what happened before, what's coming)
    - Knot context (if near a thread intersection)
    - Tone directives (how the scene should feel)
    - Story opportunities (callbacks, foreshadowing, unresolved guns)
    - Enhancement suggestions (specific creative additions)

    Usage:
        builder = BriefBuilder()
        brief = builder.build(
            query="Write the scene where Sebastian escapes",
            mnemo_client=client,
            narrative_state=processor.get_state(),
            signal_history=processor.history,
        )
        # brief is a string to prepend to 4o's system prompt
    """

    # Maximum brief size (tokens, approximate)
    MAX_BRIEF_TOKENS = 2000

    def build(self, query: str, mnemo_client,
              narrative_state: NarrativeState,
              signal_history: Optional[SignalHistory] = None) -> str:
        """Build a complete creative brief for 4o.

        Args:
            query: User's prompt
            mnemo_client: For graph_search, trace_thread, knot_context
            narrative_state: Current story state
            signal_history: Recent signal records

        Returns:
            Formatted brief string for system prompt injection
        """
        sections = []

        # 1. NARRATIVE POSITION — where are we in the story?
        position_section = self._build_position_section(mnemo_client, narrative_state)
        if position_section:
            sections.append(position_section)

        # 2. RELEVANT CHARACTERS & FACTS — graph search
        facts_section = self._build_facts_section(query, mnemo_client)
        if facts_section:
            sections.append(facts_section)

        # 3. THREAD TRACE — what happened before, what's coming
        trace_section = self._build_trace_section(mnemo_client, narrative_state)
        if trace_section:
            sections.append(trace_section)

        # 4. KNOT CONTEXT — if near a thread intersection
        knot_section = self._build_knot_section(mnemo_client, narrative_state)
        if knot_section:
            sections.append(knot_section)

        # 5. TONE DIRECTIVE — how should this scene feel?
        tone_section = self._build_tone_section(query, mnemo_client, narrative_state)
        if tone_section:
            sections.append(tone_section)

        # 6. STORY OPPORTUNITIES — callbacks, foreshadowing, Chekhov's guns
        opportunities = self._build_opportunities_section(
            narrative_state, signal_history
        )
        if opportunities:
            sections.append(opportunities)

        if not sections:
            return ""

        brief = "\n\n".join(sections)
        return f"\n\n[CREATIVE BRIEF — use this context to write the scene]\n\n{brief}\n\n[END CREATIVE BRIEF]"

    def _build_position_section(self, mnemo_client,
                                state: NarrativeState) -> str:
        """Build [NARRATIVE POSITION] section."""
        if not state.active_thread_ids:
            return ""

        lines = ["[NARRATIVE POSITION]"]
        for tid in state.active_thread_ids[:3]:  # Top 3 active threads
            try:
                thread = mnemo_client.get_thread(tid)
                if not thread:
                    continue
                name = thread.get("name", tid)
                pos = thread.get("current_position", 0)
                total = len(thread.get("points", []))
                tension = thread.get("tension_level", 0.5)
                status = thread.get("status", "active")
                tones = thread.get("tone_trajectory", [])

                line = f"  Thread: {name} — position {pos}/{total}"
                line += f" | tension: {tension:.2f} | status: {status}"
                if tones:
                    line += f"\n    Tone: {' → '.join(tones[-3:])}"
                lines.append(line)
            except Exception:
                continue

        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_facts_section(self, query: str, mnemo_client) -> str:
        """Build [RELEVANT CONTEXT] section via graph search."""
        try:
            results = mnemo_client.graph_search(query, top_k=12)
            if not results:
                return ""
            formatted = points_to_grouped_context(results)
            if formatted:
                return f"[RELEVANT CONTEXT]\n{formatted}"
        except Exception:
            pass
        return ""

    def _build_trace_section(self, mnemo_client,
                             state: NarrativeState) -> str:
        """Build [PRECEDING CONTEXT] and [UPCOMING] sections."""
        if not state.active_thread_ids:
            return ""

        lines = []
        # Use the first active thread (most likely the current scene's thread)
        tid = state.active_thread_ids[0]
        try:
            # Trace back
            back_points = mnemo_client.trace_thread(tid, direction="back", steps=3)
            if back_points:
                lines.append("[PRECEDING CONTEXT]")
                for p in back_points:
                    entity = p.get("entity", "")
                    pt = p.get("point_type", "")
                    value = p.get("value", "")[:80]
                    lines.append(f"  ← {entity}.{pt}: {value}")

            # Trace forward (what's planned next — for foreshadowing)
            fwd_points = mnemo_client.trace_thread(tid, direction="forward", steps=2)
            if fwd_points:
                lines.append("[UPCOMING]")
                for p in fwd_points:
                    entity = p.get("entity", "")
                    pt = p.get("point_type", "")
                    value = p.get("value", "")[:80]
                    lines.append(f"  → {entity}.{pt}: {value}")
        except Exception:
            pass

        return "\n".join(lines) if lines else ""

    def _build_knot_section(self, mnemo_client,
                            state: NarrativeState) -> str:
        """Build [KNOT CONTEXT] if near a thread intersection."""
        if not state.active_thread_ids:
            return ""

        # Check if any active thread has an upcoming knot
        for tid in state.active_thread_ids[:2]:
            try:
                thread = mnemo_client.get_thread(tid)
                if not thread:
                    continue
                knot_ids = thread.get("knots", [])
                pos = thread.get("current_position", 0)

                # Find the next knot after current position
                # (knots are stored as IDs, we check if any are "near")
                for kid in knot_ids:
                    ctx = mnemo_client.get_knot_context(kid)
                    if ctx:
                        return f"[KNOT CONTEXT]\n{knot_to_context(ctx)}"
            except Exception:
                continue
        return ""

    def _build_tone_section(self, query: str, mnemo_client,
                            state: NarrativeState) -> str:
        """Build [TONE DIRECTIVE] section from tone CPs + state."""
        lines = ["[TONE DIRECTIVE]"]

        # Get tone CPs from graph search
        try:
            # Search for tone-related CPs about entities in the query
            import re
            entities = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
            for entity in entities[:3]:
                tone_results = mnemo_client.entity_lookup(entity)
                for t in tone_results:
                    if t.get("point_type", "") in ("scene_register", "humor_type",
                            "tone_directive", "do_not", "emotional_shift", "atmosphere"):
                        value = t.get("value", "")
                        reason = t.get("reason", "")
                        line = f"  {t['point_type']}: {value}"
                        if reason:
                            line += f" ({reason})"
                        lines.append(line)
        except Exception:
            pass

        # Add current thread tone
        if state.last_tone_result:
            lines.append(f"  Last scene tone: {state.last_tone_result}")

        # Add thread tone trajectories
        for tid in state.active_thread_ids[:2]:
            tone = state.thread_tones.get(tid, "")
            if tone:
                lines.append(f"  Thread {tid} tone: {tone}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_opportunities_section(self, state: NarrativeState,
                                     history: Optional[SignalHistory]) -> str:
        """Build [STORY OPPORTUNITIES] section."""
        lines = []

        # Unresolved setups (Chekhov's guns)
        if state.unresolved_setups:
            lines.append("[STORY OPPORTUNITIES]")
            lines.append("  Unresolved setups (consider paying off):")
            for setup in state.unresolved_setups[-5:]:
                lines.append(f"    ⚡ {setup}")

        # Things 4o flagged as needed in the next scene
        if state.needs_payoff:
            if not lines:
                lines.append("[STORY OPPORTUNITIES]")
            lines.append("  Flagged for this scene:")
            for need in state.needs_payoff[-3:]:
                lines.append(f"    → {need}")

        # Direction from last scene (for continuity)
        if state.last_direction_taken:
            if not lines:
                lines.append("[STORY OPPORTUNITIES]")
            lines.append(f"  Last scene direction: {state.last_direction_taken}")

        return "\n".join(lines) if lines else ""


# =============================================================================
# CONVENIENCE: Get the signal instruction to append to 4o system prompt
# =============================================================================

def get_signal_instruction() -> str:
    """Return the instruction block that tells 4o to emit signals.

    Append this to the system prompt ONLY for creative writing queries.
    For recall/conversation queries, omit it.
    """
    return SIGNAL_INSTRUCTION
