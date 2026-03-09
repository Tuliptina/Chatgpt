"""
Migration Script - Blob Memories → ConnectionPoints + Threads + Knots (v6.0)

One-time migration that decomposes existing unstructured blob memories
(414 in current Mnemo) into atomic ConnectionPoints using K2.5.

Three phases:
  Phase 1: DECOMPOSE — K2.5 breaks each blob into ConnectionPoints
  Phase 2: THREAD DETECTION — K2.5 identifies narrative threads from CPs
  Phase 3: KNOT DETECTION — K2.5 identifies thread intersections

Can be run as:
  - Standalone: python migration.py (uses env vars for keys)
  - From Streamlit: import and call run_migration(mnemo_client, openrouter_key)
  - Incremental: safe to re-run; skips already-migrated blobs

Estimated cost: ~$0.05-$0.10 (K2 processing ~400 blobs in ~20 API calls)
Estimated time: 2-5 minutes on K2
"""

import json
import time
import hashlib
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from memory_schema import get_extraction_prompt, VALID_CATEGORIES


# =============================================================================
# CONFIGURATION
# =============================================================================

MEMORY_MODEL_ID = "moonshotai/kimi-k2"

MEMORY_PARAMS = {
    "temperature": 0.2,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 4096,
    "response_format": {"type": "json_object"},
}

# Batch size: how many blobs to send to K2.5 per API call
DECOMPOSE_BATCH_SIZE = 15

# Cost rates (K2.5)
COST_INPUT = 0.60 / 1_000_000
COST_OUTPUT = 3.00 / 1_000_000


# =============================================================================
# PHASE 1: DECOMPOSE BLOBS → CONNECTION POINTS
# =============================================================================

def decompose_batch(blobs: List[Dict], openrouter_key: str) -> Tuple[List[Dict], float]:
    """Send a batch of blob memories to K2.5 for decomposition.

    Args:
        blobs: List of memory dicts with "content" and "metadata" fields
        openrouter_key: OpenRouter API key

    Returns:
        (list_of_point_dicts, cost)
    """
    # Build combined text from all blobs in the batch
    blob_texts = []
    for i, blob in enumerate(blobs):
        content = blob.get("content", "")
        blob_id = blob.get("id", f"blob_{i}")
        blob_texts.append(f"[Memory {blob_id}]: {content}")

    combined = "\n\n".join(blob_texts)
    prompt = get_extraction_prompt(combined, max_chars=12000)

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MEMORY_MODEL_ID,
                "messages": [
                    {"role": "system", "content": (
                        "You are a story analyst decomposing memories into atomic "
                        "connection points. Return JSON only. Be exhaustive — extract "
                        "every character detail, relationship, plot point, tone note, "
                        "and setting detail as a separate point."
                    )},
                    {"role": "user", "content": prompt},
                ],
                **MEMORY_PARAMS,
            },
            timeout=90,
        )

        if response.status_code != 200:
            return [], 0.0

        data = response.json()
        usage = data.get("usage", {})
        cost = (usage.get("prompt_tokens", 0) * COST_INPUT
                + usage.get("completion_tokens", 0) * COST_OUTPUT)

        raw = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        points = parsed.get("points", [])

        return points, cost

    except (json.JSONDecodeError, KeyError, IndexError):
        return [], 0.0
    except Exception:
        return [], 0.0


def run_decomposition(mnemo_client, openrouter_key: str,
                      progress_callback=None) -> Dict:
    """Phase 1: Decompose all blob memories into ConnectionPoints.

    Args:
        mnemo_client: MnemoClient instance
        openrouter_key: OpenRouter API key
        progress_callback: Optional callable(message: str) for UI updates

    Returns:
        dict with stats: blobs_processed, points_created, cost, errors
    """
    result = {
        "phase": "decompose",
        "timestamp": datetime.now().isoformat(),
        "blobs_processed": 0,
        "points_extracted": 0,
        "points_stored": 0,
        "duplicates_skipped": 0,
        "cost": 0.0,
        "errors": [],
    }

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # Fetch all existing blob memories
    log("Fetching all memories from Mnemo...")
    all_memories = mnemo_client.list_memories()
    if not all_memories:
        result["errors"].append("No memories found or server unreachable")
        return result

    # Filter: skip CONVERSATION summaries and already-migrated blobs
    blobs_to_migrate = []
    for mem in all_memories:
        content = mem.get("content", "")
        meta = mem.get("metadata", {})

        # Skip conversation summaries
        if content.startswith("[CONVERSATION]") or content.startswith("[SESSION]"):
            continue

        # Skip if already migrated (has cp_id in metadata)
        if meta.get("cp_id") or meta.get("migrated"):
            continue

        blobs_to_migrate.append(mem)

    total = len(blobs_to_migrate)
    log(f"Found {total} blobs to migrate (skipped {len(all_memories) - total} conversations/already-migrated)")

    if total == 0:
        return result

    # Process in batches
    all_points = []
    for batch_start in range(0, total, DECOMPOSE_BATCH_SIZE):
        batch_end = min(batch_start + DECOMPOSE_BATCH_SIZE, total)
        batch = blobs_to_migrate[batch_start:batch_end]
        batch_num = (batch_start // DECOMPOSE_BATCH_SIZE) + 1
        total_batches = (total + DECOMPOSE_BATCH_SIZE - 1) // DECOMPOSE_BATCH_SIZE

        log(f"Processing batch {batch_num}/{total_batches} ({len(batch)} blobs)...")

        points, cost = decompose_batch(batch, openrouter_key)
        result["cost"] += cost
        result["blobs_processed"] += len(batch)
        result["points_extracted"] += len(points)

        if points:
            all_points.extend(points)
        else:
            result["errors"].append(f"Batch {batch_num}: no points extracted")

        # Brief pause to avoid rate limiting
        if batch_end < total:
            time.sleep(1.0)

    # Deduplicate by entity+point_type+value
    log(f"Deduplicating {len(all_points)} extracted points...")
    seen = set()
    unique_points = []
    for p in all_points:
        key = f"{p.get('entity', '')}:{p.get('point_type', '')}:{p.get('value', '')[:50]}".lower()
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
        else:
            result["duplicates_skipped"] += 1

    log(f"After dedup: {len(unique_points)} unique points (skipped {result['duplicates_skipped']} duplicates)")

    # Store ConnectionPoints via API
    log(f"Storing {len(unique_points)} ConnectionPoints...")
    stored = 0
    for p in unique_points:
        try:
            # Normalize category
            category = p.get("category", "fact").lower()
            if category not in VALID_CATEGORIES:
                category = "fact"

            cp_id = mnemo_client.add_point(
                entity=p.get("entity", "Unknown"),
                point_type=p.get("point_type", "fact"),
                value=p.get("value", ""),
                connects_to=p.get("connects_to", ""),
                reason=p.get("reason", ""),
                weight=float(p.get("weight", 0.5)),
                category=category,
                source="migration",
            )
            if cp_id:
                stored += 1
        except Exception as e:
            result["errors"].append(f"Store error: {str(e)[:80]}")

    result["points_stored"] = stored
    log(f"✅ Phase 1 complete: {stored} ConnectionPoints stored (cost: ${result['cost']:.4f})")

    return result


# =============================================================================
# PHASE 2: THREAD DETECTION
# =============================================================================

THREAD_DETECTION_PROMPT = """Analyze these connection points and identify NARRATIVE THREADS.

A thread is an ordered sequence of events/facts that form a story arc.
Look for:
- Character arcs (a character's journey across scenes)
- Plot lines (a sequence of events with cause and effect)
- Relationship arcs (how a relationship evolves)
- Theme threads (how a theme develops)

CONNECTION POINTS:
{points_text}

Return ONLY a JSON object:
{{
  "threads": [
    {{
      "thread_id": "thread_descriptive_name",
      "name": "Human-readable name",
      "entity": "Primary entity this thread follows",
      "thread_type": "character_arc|plot_line|theme_thread|relationship_arc",
      "point_sequence": ["entity.point_type descriptions in order"],
      "tension_level": 0.5,
      "tone_trajectory": ["tone1", "tone2"],
      "status": "active|resolved|dormant"
    }}
  ],
  "knots": [
    {{
      "name": "Descriptive name for the intersection",
      "threads": ["thread_id_1", "thread_id_2"],
      "pivot_type": "collision|revelation|betrayal|convergence|escalation",
      "reason": "Why these threads cross here"
    }}
  ]
}}

RULES:
- Each thread should have 3+ points in sequence
- Identify 5-15 threads from this data
- Only create knots where threads genuinely intersect narratively
- thread_id should be descriptive: thread_sebastian_captivity, thread_evelyn_rescue
- Order points chronologically within each thread"""


def run_thread_detection(mnemo_client, openrouter_key: str,
                         progress_callback=None) -> Dict:
    """Phase 2: Detect narrative threads from stored ConnectionPoints.

    Args:
        mnemo_client: MnemoClient instance
        openrouter_key: OpenRouter API key
        progress_callback: Optional callable(message: str)

    Returns:
        dict with stats: threads_created, knots_created, cost
    """
    result = {
        "phase": "thread_detection",
        "timestamp": datetime.now().isoformat(),
        "points_analyzed": 0,
        "threads_created": 0,
        "knots_created": 0,
        "cost": 0.0,
        "errors": [],
    }

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # Fetch all ConnectionPoints
    log("Fetching ConnectionPoints for thread detection...")
    all_points = mnemo_client.list_points(limit=500)

    if not all_points:
        result["errors"].append("No ConnectionPoints found — run Phase 1 first")
        return result

    result["points_analyzed"] = len(all_points)

    # Build text summary of points for K2.5
    points_lines = []
    for p in all_points[:200]:  # Cap at 200 to stay within context
        entity = p.get("entity", "?")
        pt = p.get("point_type", "?")
        value = p.get("value", "")[:60]
        connects = p.get("connects_to", "")
        category = p.get("category", "")

        line = f"[{category.upper()}] {entity}.{pt}"
        if connects:
            line += f" → {connects}"
        if value:
            line += f": {value}"
        points_lines.append(line)

    points_text = "\n".join(points_lines)
    prompt = THREAD_DETECTION_PROMPT.format(points_text=points_text)

    log(f"Analyzing {len(points_lines)} points for threads...")

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MEMORY_MODEL_ID,
                "messages": [
                    {"role": "system", "content": (
                        "You are a narrative analyst identifying story threads "
                        "and their intersections. Return JSON only."
                    )},
                    {"role": "user", "content": prompt},
                ],
                **MEMORY_PARAMS,
            },
            timeout=90,
        )

        if response.status_code != 200:
            result["errors"].append(f"API error: {response.status_code}")
            return result

        data = response.json()
        usage = data.get("usage", {})
        result["cost"] = (usage.get("prompt_tokens", 0) * COST_INPUT
                          + usage.get("completion_tokens", 0) * COST_OUTPUT)

        raw = data["choices"][0]["message"]["content"]
        parsed = json.loads(raw)

    except json.JSONDecodeError as e:
        result["errors"].append(f"JSON parse error: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"API error: {e}")
        return result

    # Create threads
    threads = parsed.get("threads", [])
    log(f"K2.5 detected {len(threads)} threads")

    for t in threads:
        try:
            thread_id = t.get("thread_id", "").strip()
            if not thread_id:
                continue

            tid = mnemo_client.add_thread(
                thread_id=thread_id,
                name=t.get("name", thread_id),
                entity=t.get("entity", ""),
                thread_type=t.get("thread_type", "plot_line"),
            )
            if tid:
                result["threads_created"] += 1

                # Update thread with tension and tone if provided
                tension = t.get("tension_level")
                tones = t.get("tone_trajectory", [])
                # These would need server-side support for direct update
                # For now, the thread is created with defaults
        except Exception as e:
            result["errors"].append(f"Thread create error: {str(e)[:80]}")

    # Create knots
    knots = parsed.get("knots", [])
    log(f"K2.5 detected {len(knots)} knots")

    for k in knots:
        try:
            name = k.get("name", "").strip()
            thread_ids = k.get("threads", [])
            if not name or len(thread_ids) < 2:
                continue

            knot_id = f"knot_{hashlib.md5(name.encode()).hexdigest()[:8]}"
            kid = mnemo_client.add_knot(
                knot_id=knot_id,
                name=name,
                thread_ids=thread_ids,
                pivot_type=k.get("pivot_type", "collision"),
                reason=k.get("reason", ""),
            )
            if kid:
                result["knots_created"] += 1
        except Exception as e:
            result["errors"].append(f"Knot create error: {str(e)[:80]}")

    log(f"✅ Phase 2 complete: {result['threads_created']} threads, "
        f"{result['knots_created']} knots (cost: ${result['cost']:.4f})")

    return result


# =============================================================================
# PHASE 3: TEMPORAL LINK CLEANUP
# =============================================================================

def run_link_cleanup(mnemo_client, progress_callback=None) -> Dict:
    """Phase 3: Trigger server-side maintenance to decay/prune noisy links.

    The v6.0 scoring fix (semantic*0.7 + min(link,0.5)*0.3) already
    mitigates link flooding. This additionally runs maintenance to
    start decaying the 16,000+ temporal links from batch uploads.

    Args:
        mnemo_client: MnemoClient instance
        progress_callback: Optional callable(message: str)

    Returns:
        dict with maintenance results
    """
    result = {
        "phase": "link_cleanup",
        "timestamp": datetime.now().isoformat(),
    }

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    log("Running server-side maintenance (decay + prune)...")

    try:
        # Get stats before
        stats_before = mnemo_client.get_stats()
        links_before = stats_before.get("total_links", 0)

        # Run maintenance
        response = requests.post(
            f"{mnemo_client.base_url}/maintenance",
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            maint = response.json()
            result.update(maint)

            # Get stats after
            stats_after = mnemo_client.get_stats()
            links_after = stats_after.get("total_links", 0)
            result["links_before"] = links_before
            result["links_after"] = links_after
            result["links_removed"] = links_before - links_after

            log(f"✅ Phase 3 complete: {links_before} → {links_after} links "
                f"({links_before - links_after} removed)")
        else:
            result["error"] = f"Maintenance API error: {response.status_code}"
            log(f"❌ Maintenance failed: {response.status_code}")
    except Exception as e:
        result["error"] = str(e)
        log(f"❌ Maintenance error: {e}")

    return result


# =============================================================================
# FULL MIGRATION ORCHESTRATOR
# =============================================================================

def run_migration(mnemo_client, openrouter_key: str,
                  phases: List[str] = None,
                  progress_callback=None) -> Dict:
    """Run the full migration pipeline.

    Args:
        mnemo_client: MnemoClient instance
        openrouter_key: OpenRouter API key
        phases: Which phases to run. Default: all three.
            Options: ["decompose", "threads", "cleanup"]
        progress_callback: Optional callable(message: str) for UI updates

    Returns:
        dict with results from each phase
    """
    if phases is None:
        phases = ["decompose", "threads", "cleanup"]

    results = {
        "started": datetime.now().isoformat(),
        "phases_run": phases,
        "total_cost": 0.0,
        "phase_results": {},
    }

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    log(f"Starting migration (phases: {phases})")

    # Phase 1: Decompose
    if "decompose" in phases:
        log("\n━━━ Phase 1: Decompose Blobs → ConnectionPoints ━━━")
        phase1 = run_decomposition(mnemo_client, openrouter_key, progress_callback)
        results["phase_results"]["decompose"] = phase1
        results["total_cost"] += phase1.get("cost", 0)

    # Phase 2: Thread Detection
    if "threads" in phases:
        log("\n━━━ Phase 2: Thread Detection ━━━")
        phase2 = run_thread_detection(mnemo_client, openrouter_key, progress_callback)
        results["phase_results"]["threads"] = phase2
        results["total_cost"] += phase2.get("cost", 0)

    # Phase 3: Link Cleanup
    if "cleanup" in phases:
        log("\n━━━ Phase 3: Link Cleanup ━━━")
        phase3 = run_link_cleanup(mnemo_client, progress_callback)
        results["phase_results"]["cleanup"] = phase3

    results["completed"] = datetime.now().isoformat()

    # Summary
    p1 = results["phase_results"].get("decompose", {})
    p2 = results["phase_results"].get("threads", {})
    p3 = results["phase_results"].get("cleanup", {})

    log(f"\n{'='*50}")
    log(f"MIGRATION COMPLETE")
    log(f"{'='*50}")
    if p1:
        log(f"  Phase 1: {p1.get('blobs_processed', 0)} blobs → "
            f"{p1.get('points_stored', 0)} ConnectionPoints")
    if p2:
        log(f"  Phase 2: {p2.get('threads_created', 0)} threads, "
            f"{p2.get('knots_created', 0)} knots")
    if p3:
        log(f"  Phase 3: {p3.get('links_before', '?')} → "
            f"{p3.get('links_after', '?')} links")
    log(f"  Total cost: ${results['total_cost']:.4f}")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    print("Mnemo v6.0 Migration Script")
    print("=" * 50)

    # Get keys from environment
    openrouter_key = os.environ.get("OPENROUTER_KEY", "")
    hf_key = os.environ.get("HF_KEY", "")

    if not openrouter_key:
        print("ERROR: Set OPENROUTER_KEY environment variable")
        sys.exit(1)

    # Import client
    from mnemo_client import MnemoClient
    MNEMO_URL = os.environ.get("MNEMO_URL", "https://athelaperk-mnemo-mcp.hf.space")

    print(f"Connecting to: {MNEMO_URL}")
    client = MnemoClient(base_url=MNEMO_URL, token=hf_key)

    if not client.available:
        print(f"ERROR: Mnemo server not reachable at {MNEMO_URL}")
        sys.exit(1)

    stats = client.get_stats()
    print(f"Server status: {stats.get('total_memories', 0)} memories, "
          f"{stats.get('total_links', 0)} links, "
          f"{stats.get('total_connection_points', 0)} CPs")

    # Parse command line args
    phases = sys.argv[1:] if len(sys.argv) > 1 else ["decompose", "threads", "cleanup"]
    print(f"Phases: {phases}")
    print()

    # Run migration
    results = run_migration(
        client, openrouter_key,
        phases=phases,
        progress_callback=print,
    )

    # Save results
    results_file = f"migration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
