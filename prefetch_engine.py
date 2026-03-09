"""
Prefetch Engine - Predictive Creative Brief Caching (v6.0)

Exploits the dead time between user messages (30s-5min while the user
reads what 4o wrote) to pre-build the next creative brief via K2.5.

Cache architecture:
  NarrativeFingerprint → CachedBrief
  
  Fingerprint = hash(active_thread_ids + positions + recent_signal_hashes)
  Two different prompts that map to the same narrative state get the
  same cached brief — keyed on story state, not user wording.

Prediction sources (in order of confidence):
  1. Thread continuation (0.85-0.95) — next point on active thread
  2. 4o's needs_from_next signal (0.80-0.90) — 4o flagged what's next
  3. Upcoming knot (0.70-0.80) — thread intersection approaching
  4. Most recent entity (0.60-0.70) — likely about same character

Cache outcomes on next prompt:
  - HIT (fingerprint matches):  0ms curation latency, use cached brief
  - NEAR-MISS (partial match):  ~500ms delta update on cached brief
  - MISS (no prediction):       2-3s full K2.5 curation

Estimated hit rate: ~85% for serial creative writing sessions.

Lifecycle:
  1. After 4o writes → SignalProcessor updates NarrativeState
  2. PrefetchEngine.trigger() starts background prediction
  3. K2.5 builds brief for predicted next state → cached
  4. User sends next prompt → check_cache() returns brief or None
  5. If hit: inject cached brief (0ms). If miss: build live (2-3s).

Thread-safe: all cache access is locked. Background prediction runs
in a daemon thread so it never blocks the Streamlit event loop.
"""

import time
import hashlib
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from signal_protocol import NarrativeState, SignalHistory, BriefBuilder


# =============================================================================
# CACHE STRUCTURES
# =============================================================================

@dataclass
class CachedBrief:
    """A pre-built creative brief with its prediction metadata."""
    fingerprint: str            # NarrativeState fingerprint that produced this
    brief: str                  # The formatted brief string
    created_at: float           # time.time() when built
    prediction_source: str      # "thread_continuation", "signal_needs", "knot_upcoming", "entity_recent"
    confidence: float           # 0.0-1.0 prediction confidence
    state_snapshot: dict        # NarrativeState.to_dict() at build time
    build_time_ms: float = 0.0  # How long K2.5 took to build this
    hit_count: int = 0          # How many times this cache entry was used

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def is_stale(self) -> bool:
        """Briefs older than 10 minutes are considered stale."""
        return self.age_seconds > 600

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "brief_length": len(self.brief),
            "prediction_source": self.prediction_source,
            "confidence": round(self.confidence, 3),
            "age_seconds": round(self.age_seconds, 1),
            "build_time_ms": round(self.build_time_ms, 1),
            "hit_count": self.hit_count,
            "is_stale": self.is_stale,
        }


class BriefCache:
    """LRU cache for pre-built creative briefs, keyed by fingerprint.

    Thread-safe. Max 10 entries (each brief is ~2KB, total ~20KB).
    Entries auto-expire after 10 minutes (story state changes fast).
    """

    def __init__(self, max_size: int = 10):
        self._cache: Dict[str, CachedBrief] = {}
        self._max_size = max_size
        self._lock = threading.Lock()

        # Stats
        self.hits = 0
        self.misses = 0
        self.near_misses = 0

    def get(self, fingerprint: str) -> Optional[CachedBrief]:
        """Look up a cached brief by fingerprint."""
        with self._lock:
            entry = self._cache.get(fingerprint)
            if entry and not entry.is_stale:
                entry.hit_count += 1
                self.hits += 1
                return entry
            elif entry and entry.is_stale:
                del self._cache[fingerprint]
                self.misses += 1
                return None
            else:
                self.misses += 1
                return None

    def put(self, entry: CachedBrief):
        """Store a cached brief. Evicts oldest if at capacity."""
        with self._lock:
            # Evict stale entries first
            stale_keys = [k for k, v in self._cache.items() if v.is_stale]
            for k in stale_keys:
                del self._cache[k]

            # Evict oldest if still at capacity
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
                del self._cache[oldest_key]

            self._cache[entry.fingerprint] = entry

    def find_nearest(self, fingerprint: str,
                     state: NarrativeState) -> Optional[CachedBrief]:
        """Find a near-miss: cached brief with overlapping thread state.

        A near-miss has the same active threads but different positions.
        The brief can be delta-updated (~500ms) instead of rebuilt (~3s).
        """
        with self._lock:
            current_threads = set(state.active_thread_ids)
            if not current_threads:
                return None

            best = None
            best_overlap = 0.0

            for entry in self._cache.values():
                if entry.is_stale:
                    continue
                cached_threads = set(
                    entry.state_snapshot.get("active_thread_ids", [])
                )
                if not cached_threads:
                    continue

                # Jaccard similarity on thread sets
                overlap = len(current_threads & cached_threads) / len(current_threads | cached_threads)
                if overlap > best_overlap and overlap >= 0.5:
                    best = entry
                    best_overlap = overlap

            if best:
                self.near_misses += 1
                return best
            return None

    def clear(self):
        """Clear all cached briefs."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        with self._lock:
            total_requests = self.hits + self.misses
            return {
                "cached_entries": len(self._cache),
                "max_size": self._max_size,
                "hits": self.hits,
                "misses": self.misses,
                "near_misses": self.near_misses,
                "hit_rate": round(self.hits / max(total_requests, 1), 3),
                "entries": [v.to_dict() for v in self._cache.values()],
            }


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

class PredictionEngine:
    """Predicts the most likely next narrative state for prefetching.

    Uses multiple signals to guess what the user will ask for next:
      1. Thread continuation — advance position on active threads
      2. Signal needs — 4o said "next scene should address X"
      3. Knot approach — an upcoming thread intersection
      4. Entity recency — the user is probably still writing about
         the same character

    Returns predicted NarrativeState + confidence + source label.
    """

    def predict(self, current_state: NarrativeState,
                signal_history: Optional[SignalHistory] = None) -> List[Tuple[NarrativeState, float, str]]:
        """Generate 1-3 predicted next states, ranked by confidence.

        Returns list of (predicted_state, confidence, source_label).
        """
        predictions = []

        # 1. Thread continuation (highest confidence for serial writing)
        thread_pred = self._predict_thread_continuation(current_state)
        if thread_pred:
            predictions.append(thread_pred)

        # 2. Signal needs (4o told us what's needed next)
        if signal_history:
            signal_pred = self._predict_from_signals(current_state, signal_history)
            if signal_pred:
                predictions.append(signal_pred)

        # 3. Entity recency (same character, next scene)
        entity_pred = self._predict_entity_continuation(current_state)
        if entity_pred:
            predictions.append(entity_pred)

        # Sort by confidence, take top 3
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:3]

    def _predict_thread_continuation(self, state: NarrativeState
                                     ) -> Optional[Tuple[NarrativeState, float, str]]:
        """Predict: user continues the current active thread."""
        if not state.active_thread_ids:
            return None

        predicted = NarrativeState(
            active_thread_ids=list(state.active_thread_ids),
            thread_positions={
                tid: pos + 1
                for tid, pos in state.thread_positions.items()
            },
            thread_tensions=dict(state.thread_tensions),
            thread_tones=dict(state.thread_tones),
            unresolved_setups=list(state.unresolved_setups),
            needs_payoff=list(state.needs_payoff),
            recent_entities=list(state.recent_entities),
            last_tone_result=state.last_tone_result,
            scenes_written=state.scenes_written + 1,
        )

        # Confidence: high for single active thread, lower for multiple
        n_threads = len(state.active_thread_ids)
        confidence = 0.95 if n_threads == 1 else 0.85

        return predicted, confidence, "thread_continuation"

    def _predict_from_signals(self, state: NarrativeState,
                              history: SignalHistory
                              ) -> Optional[Tuple[NarrativeState, float, str]]:
        """Predict: use 4o's needs_from_next to anticipate next prompt."""
        last = history.last()
        if not last:
            return None

        needs = last.signals.get("needs_from_next", [])
        if not needs:
            return None

        predicted = NarrativeState(
            active_thread_ids=list(state.active_thread_ids),
            thread_positions=dict(state.thread_positions),
            thread_tensions=dict(state.thread_tensions),
            thread_tones=dict(state.thread_tones),
            unresolved_setups=list(state.unresolved_setups),
            needs_payoff=list(needs),  # Override with 4o's explicit needs
            recent_entities=history.recent_entities(n=2),
            last_tone_result=state.last_tone_result,
            scenes_written=state.scenes_written + 1,
        )

        confidence = 0.85
        return predicted, confidence, "signal_needs"

    def _predict_entity_continuation(self, state: NarrativeState
                                     ) -> Optional[Tuple[NarrativeState, float, str]]:
        """Predict: user continues writing about the same entities."""
        if not state.recent_entities:
            return None

        predicted = NarrativeState(
            active_thread_ids=list(state.active_thread_ids),
            thread_positions=dict(state.thread_positions),
            thread_tensions=dict(state.thread_tensions),
            thread_tones=dict(state.thread_tones),
            unresolved_setups=list(state.unresolved_setups),
            needs_payoff=list(state.needs_payoff),
            recent_entities=list(state.recent_entities),
            last_tone_result=state.last_tone_result,
            scenes_written=state.scenes_written + 1,
        )

        confidence = 0.65
        return predicted, confidence, "entity_recent"


# =============================================================================
# PREFETCH ENGINE — main orchestrator
# =============================================================================

class PrefetchEngine:
    """Pre-builds creative briefs during user read time.

    After each scene, trigger() starts a background thread that:
      1. Predicts the most likely next narrative state
      2. Builds a creative brief for that state via BriefBuilder
      3. Caches the brief keyed by NarrativeFingerprint

    On next user prompt:
      check_cache() returns the cached brief (0ms) or None (2-3s build).

    Thread-safe. Background work never blocks the Streamlit event loop.
    Single writer: only one prediction runs at a time (newest wins).

    Usage in app.py:
        # After each creative writing response:
        prefetch = st.session_state.prefetch_engine
        prefetch.trigger(processor.get_state(), processor.history, mnemo_client)

        # Before building context for next prompt:
        cached = prefetch.check_cache(current_state)
        if cached:
            brief = cached.brief  # 0ms
        else:
            brief = builder.build(...)  # 2-3s
    """

    def __init__(self):
        self.cache = BriefCache(max_size=10)
        self.predictor = PredictionEngine()
        self.builder = BriefBuilder()

        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_flag = threading.Event()
        self._lock = threading.Lock()

        # Stats
        self.predictions_made = 0
        self.briefs_built = 0
        self.build_errors = 0

    def trigger(self, state: NarrativeState,
                history: Optional[SignalHistory],
                mnemo_client) -> None:
        """Start background prefetch after a scene is written.

        Cancels any in-progress prediction (newest state wins).
        Non-blocking — returns immediately.
        """
        # Cancel any in-progress work
        self._cancel_flag.set()

        # Wait for previous thread to finish (brief — it checks cancel flag)
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=0.5)

        # Reset cancel flag and start new prediction
        self._cancel_flag.clear()

        self._worker_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(state, history, mnemo_client),
            daemon=True,
            name="prefetch-worker",
        )
        self._worker_thread.start()

    def check_cache(self, state: NarrativeState) -> Optional[CachedBrief]:
        """Check if a pre-built brief exists for the current state.

        Returns:
            CachedBrief if hit, None if miss.
            On near-miss, returns the closest entry (caller should
            delta-update if desired).
        """
        fingerprint = state.fingerprint()

        # Exact match
        exact = self.cache.get(fingerprint)
        if exact:
            return exact

        # Near-miss (same threads, different position)
        near = self.cache.find_nearest(fingerprint, state)
        if near:
            return near

        return None

    def invalidate(self):
        """Clear all cached briefs. Call when session changes."""
        self.cache.clear()

    def stats(self) -> dict:
        """Return prefetch engine statistics."""
        return {
            "predictions_made": self.predictions_made,
            "briefs_built": self.briefs_built,
            "build_errors": self.build_errors,
            "cache": self.cache.stats(),
        }

    # =========================================================================
    # BACKGROUND WORKER
    # =========================================================================

    def _prefetch_worker(self, state: NarrativeState,
                         history: Optional[SignalHistory],
                         mnemo_client) -> None:
        """Background thread: predict → build → cache.

        Runs in a daemon thread. Checks _cancel_flag periodically
        so it can be interrupted when a newer trigger() arrives.
        """
        try:
            # Step 1: Predict next states
            predictions = self.predictor.predict(state, history)
            self.predictions_made += 1

            if not predictions or self._cancel_flag.is_set():
                return

            # Step 2: Build brief for the top prediction
            predicted_state, confidence, source = predictions[0]

            if self._cancel_flag.is_set():
                return

            start = time.time()
            brief = self.builder.build(
                query="",  # No query yet — building from state alone
                mnemo_client=mnemo_client,
                narrative_state=predicted_state,
                signal_history=history,
            )
            build_ms = (time.time() - start) * 1000

            if self._cancel_flag.is_set():
                return

            if brief:
                # Step 3: Cache the built brief
                entry = CachedBrief(
                    fingerprint=predicted_state.fingerprint(),
                    brief=brief,
                    created_at=time.time(),
                    prediction_source=source,
                    confidence=confidence,
                    state_snapshot=predicted_state.to_dict(),
                    build_time_ms=build_ms,
                )
                self.cache.put(entry)
                self.briefs_built += 1

            # Optionally build the #2 prediction if time allows
            if (len(predictions) > 1
                    and not self._cancel_flag.is_set()
                    and build_ms < 2000):
                self._build_secondary(predictions[1], history, mnemo_client)

        except Exception:
            self.build_errors += 1

    def _build_secondary(self, prediction: Tuple[NarrativeState, float, str],
                         history: Optional[SignalHistory],
                         mnemo_client) -> None:
        """Build brief for the second-most-likely prediction."""
        predicted_state, confidence, source = prediction

        if self._cancel_flag.is_set():
            return

        try:
            start = time.time()
            brief = self.builder.build(
                query="",
                mnemo_client=mnemo_client,
                narrative_state=predicted_state,
                signal_history=history,
            )
            build_ms = (time.time() - start) * 1000

            if brief and not self._cancel_flag.is_set():
                entry = CachedBrief(
                    fingerprint=predicted_state.fingerprint(),
                    brief=brief,
                    created_at=time.time(),
                    prediction_source=source,
                    confidence=confidence,
                    state_snapshot=predicted_state.to_dict(),
                    build_time_ms=build_ms,
                )
                self.cache.put(entry)
                self.briefs_built += 1
        except Exception:
            self.build_errors += 1
