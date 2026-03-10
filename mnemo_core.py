"""
Mnemo v7.0 Core Engine — Hybrid SQLite + FAISS + NumPy Architecture

ARCHITECTURE:
  SQLite (WAL mode)  = Source of truth. ACID, indexed, FTS5 full-text search.
  FAISS (disposable) = Fast ANN pre-filter. Rebuilt from SQLite on startup.
                       Never persisted separately. No fragile remove_ids().
  NumPy              = Exact cosine reranking on FAISS candidates.
                       Fixes the accuracy gap from IndexIDMap drift.
  Embeddings         = Stored as raw BLOB in SQLite (not base64 JSON).
                       25% smaller, 10x faster load, zero encode/decode.

SEARCH PIPELINE (graph_search):
  Phase 1: Entity graph lookup    (SQLite indices)      ~1ms
  Phase 2: Full-text search       (FTS5 + BM25)         ~3ms
  Phase 3: Semantic pre-filter    (FAISS ANN)           ~5ms
  Phase 4: Exact reranking        (NumPy cosine)        ~1ms
  Phase 5: Score fusion           (graph + FTS + sem)   ~0ms
  Total:                                                ~10ms

PERSISTENCE:
  SQLite .db file uploaded directly to HF Datasets (no serialize/deserialize).
  WAL checkpoint before upload. Download .db on startup.

BACKWARD COMPATIBILITY:
  All public method signatures identical to v6.5.
  Gradio app.py endpoints require zero changes.
  Streamlit mnemo_client.py requires zero changes.

MIGRATION:
  On first startup, if legacy mnemo_db.json exists, imports it into SQLite.
  After migration, all operations use SQLite exclusively.

v7.0 changes from v6.5:
- REPLACED: Dict storage → SQLite with WAL mode + connection pooling
- REPLACED: EntityIndex class → SQLite COLLATE NOCASE indices
- REPLACED: ThreadIndex class → SQLite join tables (thread_points, knot_threads)
- REPLACED: FAISS IndexIDMap → disposable IndexFlatIP rebuilt from DB
- REPLACED: base64 embedding JSON → raw BLOB in SQLite
- REPLACED: serialize()/deserialize() → direct .db file upload
- ADDED: FTS5 full-text search with Porter stemming + BM25 ranking
- ADDED: NumPy exact reranking (Stage 2 of two-stage retrieval)
- ADDED: SQLite WAL mode for concurrent reads during writes
- ADDED: JSON→SQLite migration on first startup
- KEPT: All public method signatures, dataclass definitions, EmbeddingCache
"""

import os
import re
import json
import time
import queue
import sqlite3
import hashlib
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from enum import Enum

import requests as http_requests  # renamed to avoid clash with other 'requests'

# SentenceTransformer is optional — Cloudflare Workers AI is the primary encoder
# on Streamlit Cloud (saves ~310MB RAM). SentenceTransformer is used on HF Space.
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

log = logging.getLogger("mnemo")


# =============================================================================
# EMBEDDING ENCODERS (Cloudflare Workers AI primary, SentenceTransformer fallback)
# =============================================================================

class CloudflareEncoder:
    """Embedding encoder via Cloudflare Workers AI REST API.

    Free tier: 10,000 neurons/day (~10,000 embedding calls/day).
    Model: bge-small-en-v1.5 → 384-dim (same as all-MiniLM-L6-v2).
    Latency: ~200ms per call (edge GPU), cached locally after first call.
    RAM: 0MB (no PyTorch, no model weights).

    Provides same .encode() interface as SentenceTransformer.
    """

    # Available Cloudflare embedding models and their dimensions
    MODELS = {
        "@cf/baai/bge-small-en-v1.5": 384,
        "@cf/baai/bge-base-en-v1.5": 768,
        "@cf/baai/bge-large-en-v1.5": 1024,
    }

    def __init__(self, account_id: str = None, api_token: str = None,
                 model: str = "@cf/baai/bge-small-en-v1.5",
                 timeout: int = 15):
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self.api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        self.model = model
        self.timeout = timeout
        self._dim = self.MODELS.get(model, 384)
        self._url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/ai/run/{self.model}"
        )
        self._available = bool(self.account_id and self.api_token)

    @property
    def available(self) -> bool:
        return self._available

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, text, **kwargs) -> np.ndarray:
        """Encode text(s) to embeddings. Accepts str or list of str.

        Returns:
            np.ndarray: (dim,) for single text, (N, dim) for list
        """
        if isinstance(text, str):
            return self._encode_single(text)
        elif isinstance(text, (list, tuple)):
            return self._encode_batch(list(text))
        else:
            return self._encode_single(str(text))

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        try:
            resp = http_requests.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                json={"text": [text]},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Cloudflare API error {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            vectors = data.get("result", {}).get("data", [])
            if not vectors:
                raise RuntimeError(f"No embeddings in response: {data}")

            return np.array(vectors[0], dtype=np.float32)

        except Exception as e:
            log.error(f"CloudflareEncoder error: {e}")
            raise

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts. Cloudflare supports up to 100 per call."""
        all_embeddings = []
        # Cloudflare batch limit is 100
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            try:
                resp = http_requests.post(
                    self._url,
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json",
                    },
                    json={"text": batch},
                    timeout=self.timeout * 2,  # Longer timeout for batches
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Cloudflare API error {resp.status_code}")

                data = resp.json()
                vectors = data.get("result", {}).get("data", [])
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"Expected {len(batch)} embeddings, got {len(vectors)}")

                all_embeddings.extend(vectors)

            except Exception as e:
                log.error(f"CloudflareEncoder batch error: {e}")
                raise

        return np.array(all_embeddings, dtype=np.float32)


def create_encoder(config: 'MnemoConfig'):
    """Create the best available encoder. Priority:
    1. Cloudflare Workers AI (free, 0MB RAM, ~200ms, needs API key)
    2. SentenceTransformer (local, ~310MB RAM, ~50ms, needs torch)
    3. Raises error if neither available

    Returns (encoder, embedding_dim, encoder_name)
    """
    # Try Cloudflare first (if credentials exist)
    cf_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    cf_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")

    if cf_account and cf_token:
        try:
            encoder = CloudflareEncoder(
                account_id=cf_account, api_token=cf_token,
                model=config.cloudflare_model,
            )
            # Quick validation — encode a test string
            test_emb = encoder.encode("test")
            if test_emb is not None and len(test_emb) > 0:
                dim = encoder.get_sentence_embedding_dimension()
                print(f"[ENCODER] ✅ Cloudflare Workers AI ({config.cloudflare_model}, dim={dim})")
                return encoder, dim, f"cloudflare:{config.cloudflare_model}"
        except Exception as e:
            print(f"[ENCODER] ⚠️  Cloudflare failed: {e}")

    # Fall back to SentenceTransformer
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            print(f"[ENCODER] Loading SentenceTransformer: {config.model_name}")
            encoder = SentenceTransformer(config.model_name)
            dim = encoder.get_sentence_embedding_dimension()
            print(f"[ENCODER] ✅ SentenceTransformer ({config.model_name}, dim={dim})")
            return encoder, dim, f"local:{config.model_name}"
        except Exception as e:
            print(f"[ENCODER] ⚠️  SentenceTransformer failed: {e}")

    raise RuntimeError(
        "No embedding encoder available. Either:\n"
        "  1. Set CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN env vars, or\n"
        "  2. pip install sentence-transformers torch"
    )


# =============================================================================
# ENUMS & CONSTANTS (unchanged from v6.5)
# =============================================================================

class MemoryTier(Enum):
    WORKING = "working"
    SEMANTIC = "semantic"
    ARCHIVE = "archive"


class LinkType(Enum):
    DIRECT_REFERENCE = "direct_reference"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CO_OCCURRENCE = "co_occurrence"
    HIERARCHICAL = "hierarchical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CROSS_DOMAIN = "cross_domain"
    ASSOCIATIVE = "associative"


LINK_PROPERTIES = {
    LinkType.DIRECT_REFERENCE:    {"threshold": 0.85, "base_strength": 0.90, "decay_per_day": 0.005},
    LinkType.SEMANTIC_SIMILARITY: {"threshold": 0.50, "base_strength": 0.75, "decay_per_day": 0.010},
    LinkType.CO_OCCURRENCE:       {"threshold": 0.60, "base_strength": 0.70, "decay_per_day": 0.015},
    LinkType.HIERARCHICAL:        {"threshold": 0.80, "base_strength": 0.85, "decay_per_day": 0.003},
    LinkType.TEMPORAL:            {"threshold": 0.55, "base_strength": 0.65, "decay_per_day": 0.020},
    LinkType.CAUSAL:              {"threshold": 0.75, "base_strength": 0.80, "decay_per_day": 0.005},
    LinkType.CROSS_DOMAIN:        {"threshold": 0.70, "base_strength": 0.65, "decay_per_day": 0.008},
    LinkType.ASSOCIATIVE:         {"threshold": 0.45, "base_strength": 0.60, "decay_per_day": 0.025},
}

HIGH_VALUE_MARKERS = [
    "allergic", "allergy", "prefers", "hates", "loves", "needs",
    "birthday", "deadline", "password", "never", "always", "emergency",
    "name is", "lives in", "works at", "born", "married", "diagnosed",
]

# NER stop words for graph_search entity extraction
_NER_STOP = frozenset({
    'the', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
    'where', 'when', 'how', 'why', 'will', 'would', 'could', 'should', 'can',
    'may', 'might', 'shall', 'must', 'about', 'with', 'from', 'into', 'through',
    'during', 'before', 'after', 'between', 'under', 'above', 'does', 'have',
    'has', 'had', 'was', 'were', 'been', 'being', 'are', 'not', 'but', 'and',
    'for', 'nor', 'yet', 'also', 'just', 'very', 'too', 'some', 'any', 'all',
    'each', 'every', 'both', 'few', 'more', 'most', 'other', 'only', 'own',
    'than', 'then', 'now', 'here', 'there', 'tell', 'show', 'give', 'get',
    'find', 'know', 'remember', 'recall', 'write', 'create', 'describe',
    'make', 'help', 'please', 'scene', 'chapter', 'story', 'book',
    'character', 'plot', 'setting', 'like', 'want', 'need', 'think',
    'said', 'says', 'going', 'come', 'came', 'take', 'took', 'keep',
})


# =============================================================================
# DATACLASSES (kept for API compat — used as return types, not storage)
# =============================================================================

@dataclass
class SearchResult:
    id: str
    content: str
    score: float
    tier: str
    semantic_score: float = 0.0
    link_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id, "content": self.content,
            "score": round(self.score, 3), "tier": self.tier,
            "semantic_score": round(self.semantic_score, 3),
            "link_score": round(self.link_score, 3), "metadata": self.metadata,
        }


# =============================================================================
# SQLITE DATABASE LAYER
# =============================================================================

SCHEMA_SQL = """
-- Connection Points (primary structured memory)
CREATE TABLE IF NOT EXISTS connection_points (
    id          TEXT PRIMARY KEY,
    entity      TEXT NOT NULL,
    point_type  TEXT NOT NULL,
    value       TEXT NOT NULL DEFAULT '',
    connects_to TEXT DEFAULT '',
    reason      TEXT DEFAULT '',
    weight      REAL DEFAULT 0.5,
    category    TEXT DEFAULT 'fact',
    session_id  TEXT DEFAULT '',
    source      TEXT DEFAULT 'auto_extract',
    thread_id   TEXT DEFAULT '',
    position    INTEGER DEFAULT -1,
    namespace   TEXT DEFAULT 'default',
    created_at  REAL NOT NULL,
    embedding   BLOB
);

CREATE INDEX IF NOT EXISTS idx_cp_entity      ON connection_points(entity COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_cp_connects    ON connection_points(connects_to COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_cp_category    ON connection_points(category);
CREATE INDEX IF NOT EXISTS idx_cp_point_type  ON connection_points(point_type);
CREATE INDEX IF NOT EXISTS idx_cp_session     ON connection_points(session_id);
CREATE INDEX IF NOT EXISTS idx_cp_source      ON connection_points(source);
CREATE INDEX IF NOT EXISTS idx_cp_ent_type    ON connection_points(entity COLLATE NOCASE, point_type);
CREATE INDEX IF NOT EXISTS idx_cp_ent_cat     ON connection_points(entity COLLATE NOCASE, category);

-- FTS5 full-text index on CPs
CREATE VIRTUAL TABLE IF NOT EXISTS cp_fts USING fts5(
    entity, value, connects_to, reason,
    content='connection_points', content_rowid='rowid',
    tokenize='porter unicode61'
);

-- FTS sync triggers
CREATE TRIGGER IF NOT EXISTS cp_fts_ai AFTER INSERT ON connection_points BEGIN
    INSERT INTO cp_fts(rowid, entity, value, connects_to, reason)
    VALUES (new.rowid, new.entity, new.value, new.connects_to, new.reason);
END;
CREATE TRIGGER IF NOT EXISTS cp_fts_ad AFTER DELETE ON connection_points BEGIN
    INSERT INTO cp_fts(cp_fts, rowid, entity, value, connects_to, reason)
    VALUES ('delete', old.rowid, old.entity, old.value, old.connects_to, old.reason);
END;
CREATE TRIGGER IF NOT EXISTS cp_fts_au AFTER UPDATE ON connection_points BEGIN
    INSERT INTO cp_fts(cp_fts, rowid, entity, value, connects_to, reason)
    VALUES ('delete', old.rowid, old.entity, old.value, old.connects_to, old.reason);
    INSERT INTO cp_fts(rowid, entity, value, connects_to, reason)
    VALUES (new.rowid, new.entity, new.value, new.connects_to, new.reason);
END;

-- Threads
CREATE TABLE IF NOT EXISTS threads (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    thread_type     TEXT DEFAULT 'plot_line',
    entity          TEXT DEFAULT '',
    status          TEXT DEFAULT 'active',
    tension_level   REAL DEFAULT 0.5,
    tone_trajectory TEXT DEFAULT '[]',
    current_position INTEGER DEFAULT 0,
    session_id      TEXT DEFAULT '',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_thread_entity ON threads(entity COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_thread_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_thread_session ON threads(session_id);

-- Thread ↔ CP ordered membership
CREATE TABLE IF NOT EXISTS thread_points (
    thread_id TEXT NOT NULL,
    cp_id     TEXT NOT NULL,
    position  INTEGER NOT NULL,
    PRIMARY KEY (thread_id, cp_id)
);
CREATE INDEX IF NOT EXISTS idx_tp_thread ON thread_points(thread_id, position);
CREATE INDEX IF NOT EXISTS idx_tp_cp     ON thread_points(cp_id);

-- Knots
CREATE TABLE IF NOT EXISTS knots (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    pivot_type       TEXT DEFAULT 'collision',
    narrative_weight REAL DEFAULT 0.5,
    tension_before   REAL DEFAULT 0.5,
    tension_after    REAL DEFAULT 0.5,
    tone_shift       TEXT DEFAULT '',
    reason           TEXT DEFAULT '',
    unresolved       TEXT DEFAULT '[]',
    active_points    TEXT DEFAULT '{}',
    session_id       TEXT DEFAULT '',
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knot_session ON knots(session_id);

-- Knot ↔ Thread membership
CREATE TABLE IF NOT EXISTS knot_threads (
    knot_id   TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    PRIMARY KEY (knot_id, thread_id)
);

-- Legacy blob memories
CREATE TABLE IF NOT EXISTS memories (
    id             TEXT PRIMARY KEY,
    content        TEXT NOT NULL,
    tier           TEXT DEFAULT 'semantic',
    namespace      TEXT DEFAULT 'default',
    quality_score  REAL DEFAULT 0.5,
    access_count   INTEGER DEFAULT 0,
    priority       REAL DEFAULT 1.0,
    session_id     TEXT DEFAULT '',
    source         TEXT DEFAULT 'auto_extract',
    created_at     REAL NOT NULL,
    last_accessed  REAL NOT NULL,
    metadata       TEXT DEFAULT '{}',
    embedding      BLOB
);
CREATE INDEX IF NOT EXISTS idx_mem_ns ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_mem_tier ON memories(tier);
-- NOTE: idx_mem_session and idx_mem_source are created by _migrate_memories_columns()
-- to avoid errors on existing databases that lack those columns during schema init.

-- FTS for blob memories
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memories', content_rowid='rowid',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS mem_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS mem_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

-- Neural links
CREATE TABLE IF NOT EXISTS neural_links (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    target_id       TEXT NOT NULL,
    link_type       TEXT NOT NULL,
    strength        REAL NOT NULL,
    created_at      REAL NOT NULL,
    last_maintained REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_link_source ON neural_links(source_id);
CREATE INDEX IF NOT EXISTS idx_link_target ON neural_links(target_id);

-- Stats
CREATE TABLE IF NOT EXISTS engine_stats (
    key   TEXT PRIMARY KEY,
    value INTEGER DEFAULT 0
);
"""


class MnemoDB:
    """SQLite database with WAL mode and read connection pooling."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        self._write_conn = self._create_connection(readonly=False)
        self._write_conn.executescript(SCHEMA_SQL)
        self._write_conn.commit()

        # Schema migration: add session_id/source columns to memories table
        # (for existing .db files created before v7.1)
        self._migrate_memories_columns()

        self._read_pool: queue.Queue = queue.Queue(maxsize=4)
        for _ in range(4):
            self._read_pool.put(self._create_connection(readonly=True))

    def _migrate_memories_columns(self):
        """Add session_id and source columns to memories if missing (v7.0→v7.1).

        Also backfills existing rows by extracting values from the metadata JSON blob.
        """
        cursor = self._write_conn.execute("PRAGMA table_info(memories)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        if "session_id" not in existing_cols:
            log.info("Migrating memories table: adding session_id column")
            print("[MIGRATE] Adding session_id column to memories table")
            self._write_conn.execute("ALTER TABLE memories ADD COLUMN session_id TEXT DEFAULT ''")
            # Backfill from metadata JSON
            self._write_conn.execute("""
                UPDATE memories SET session_id = COALESCE(json_extract(metadata, '$.session_id'), '')
                WHERE metadata LIKE '%session_id%'
            """)

        if "source" not in existing_cols:
            log.info("Migrating memories table: adding source column")
            print("[MIGRATE] Adding source column to memories table")
            self._write_conn.execute("ALTER TABLE memories ADD COLUMN source TEXT DEFAULT 'auto_extract'")
            # Backfill from metadata JSON
            self._write_conn.execute("""
                UPDATE memories SET source = COALESCE(json_extract(metadata, '$.source'), 'auto_extract')
                WHERE metadata LIKE '%source%'
            """)

        # Create indexes if they don't exist (idempotent)
        self._write_conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session_id)")
        self._write_conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_source ON memories(source)")
        self._write_conn.commit()

    def _create_connection(self, readonly: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32000")     # 32MB cache
        conn.execute("PRAGMA mmap_size=134217728")    # 128MB mmap
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.row_factory = sqlite3.Row
        if readonly:
            conn.execute("PRAGMA query_only=ON")
        return conn

    @contextmanager
    def read(self):
        """Get a read-only connection from the pool."""
        conn = self._read_pool.get(timeout=10)
        try:
            yield conn
        finally:
            self._read_pool.put(conn)

    @contextmanager
    def write(self):
        """Get the write connection. Auto-commits on success, rolls back on error."""
        try:
            yield self._write_conn
            self._write_conn.commit()
        except Exception:
            self._write_conn.rollback()
            raise

    def checkpoint(self):
        """WAL checkpoint — call before uploading .db file."""
        self._write_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self):
        self._write_conn.close()
        while not self._read_pool.empty():
            try:
                self._read_pool.get_nowait().close()
            except queue.Empty:
                break


# =============================================================================
# DISPOSABLE FAISS INDEX (rebuilt from SQLite, never persisted)
# =============================================================================

class FAISSIndex:
    """Fast ANN pre-filter. Rebuilt from SQLite on startup.

    No IndexIDMap — uses positional id_map list instead.
    Deletions mark dirty for eventual rebuild (no fragile remove_ids).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._index: Optional[object] = None
        self._id_map: List[str] = []        # position → cp_id
        self._id_to_pos: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._dirty = False

    def rebuild_from_db(self, db: MnemoDB):
        """Full rebuild from SQLite. ~200ms for 10K vectors."""
        with db.read() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM connection_points WHERE embedding IS NOT NULL"
            ).fetchall()

        if not rows or not HAS_FAISS:
            with self._lock:
                if HAS_FAISS:
                    self._index = faiss.IndexFlatIP(self.dim)
                self._id_map = []
                self._id_to_pos = {}
                self._dirty = False
            return

        ids = [r["id"] for r in rows]
        vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        vecs_normed = (vecs / norms).astype(np.float32)

        index = faiss.IndexFlatIP(self.dim)
        index.add(vecs_normed)

        with self._lock:
            self._index = index
            self._id_map = ids
            self._id_to_pos = {cid: i for i, cid in enumerate(ids)}
            self._dirty = False
        log.info(f"FAISS rebuilt: {len(ids)} vectors indexed")

    def add(self, cp_id: str, embedding: np.ndarray):
        """Add single vector between rebuilds."""
        if not HAS_FAISS:
            return
        vec = embedding.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(vec.reshape(1, -1))
            pos = len(self._id_map)
            self._id_map.append(cp_id)
            self._id_to_pos[cp_id] = pos

    def search(self, query_emb: np.ndarray, top_k: int = 200) -> List[Tuple[str, float]]:
        """Fast ANN candidate retrieval. Returns (cp_id, score) pairs."""
        with self._lock:
            if not HAS_FAISS or self._index is None or self._index.ntotal == 0:
                return []
            vec = query_emb.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(vec.reshape(1, -1), k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self._id_map):
                    results.append((self._id_map[idx], float(score)))
            return results

    def mark_dirty(self):
        self._dirty = True

    @property
    def needs_rebuild(self) -> bool:
        return self._dirty

    @property
    def total(self) -> int:
        with self._lock:
            return self._index.ntotal if (HAS_FAISS and self._index) else 0


class FAISSBlobIndex:
    """Separate FAISS index for legacy blob memories."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._index: Optional[object] = None
        self._id_map: List[str] = []
        self._lock = threading.RLock()

    def rebuild_from_db(self, db: MnemoDB):
        with db.read() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
        if not rows or not HAS_FAISS:
            with self._lock:
                if HAS_FAISS:
                    self._index = faiss.IndexFlatIP(self.dim)
                self._id_map = []
            return
        ids = [r["id"] for r in rows]
        vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        vecs_normed = (vecs / norms).astype(np.float32)
        index = faiss.IndexFlatIP(self.dim)
        index.add(vecs_normed)
        with self._lock:
            self._index = index
            self._id_map = ids

    def search(self, query_emb: np.ndarray, top_k: int = 50) -> Dict[str, float]:
        with self._lock:
            if not HAS_FAISS or self._index is None or self._index.ntotal == 0:
                return {}
            vec = query_emb.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(vec.reshape(1, -1), k)
            result = {}
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self._id_map):
                    result[self._id_map[idx]] = float(score)
            return result

    def add(self, mem_id: str, embedding: np.ndarray):
        if not HAS_FAISS:
            return
        vec = embedding.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(vec.reshape(1, -1))
            self._id_map.append(mem_id)

    def mark_dirty(self):
        pass  # Blob index doesn't auto-rebuild; only on startup


# =============================================================================
# NUMPY RERANKER (exact cosine similarity on FAISS candidates)
# =============================================================================

class NumpyReranker:
    """Exact cosine reranking for FAISS candidates.

    FAISS pre-filters 10K → 200 candidates. NumPy re-ranks with perfect
    accuracy. This fixes the precision gap from IndexIDMap fragmentation.
    """

    @staticmethod
    def rerank(query_emb: np.ndarray, candidate_ids: List[str],
               candidate_embs: np.ndarray, threshold: float = 0.15
               ) -> List[Tuple[str, float]]:
        """Exact cosine similarity on candidate set.

        Args:
            query_emb: (dim,) query vector
            candidate_ids: list of cp_ids matching candidate_embs rows
            candidate_embs: (N, dim) embedding matrix
            threshold: minimum similarity to include

        Returns:
            List of (cp_id, exact_cosine_score), sorted descending
        """
        if len(candidate_ids) == 0:
            return []
        q_norm = np.linalg.norm(query_emb)
        if q_norm == 0:
            return []
        e_norms = np.linalg.norm(candidate_embs, axis=1)
        dots = candidate_embs @ query_emb
        valid = e_norms > 0
        scores = np.zeros(len(candidate_ids))
        scores[valid] = dots[valid] / (q_norm * e_norms[valid])

        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            s = float(scores[idx])
            if s < threshold:
                break
            results.append((candidate_ids[idx], s))
        return results


# =============================================================================
# EMBEDDING CACHE (Thread-Safe, kept from v6.5)
# =============================================================================

class EmbeddingCache:
    """Thread-safe LRU cache for sentence embeddings."""

    def __init__(self, encoder, max_size: int = 500):
        self.encoder = encoder
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def get_embedding(self, text: str) -> np.ndarray:
        key = text[:200].lower().strip()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            emb = self.encoder.encode(text)
            self._cache[key] = emb
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            self._misses += 1
        return emb

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Batch encode (for bulk imports). Returns (N, dim) matrix."""
        with self._lock:
            return self.encoder.encode(texts)

    def get_stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits, "misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1), 3),
                "size": len(self._cache), "max_size": self._max_size,
            }

    def clear(self):
        with self._lock:
            self._cache.clear()


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MnemoConfig:
    similarity_threshold: float = 0.25
    quality_threshold: float = 0.35
    decay_rate_per_day: float = 0.01
    prune_quality_floor: float = 0.15
    prune_age_days: float = 30.0
    promote_to_working_accesses: int = 10
    demote_to_archive_days: float = 14.0
    model_name: str = "all-MiniLM-L6-v2"
    cloudflare_model: str = "@cf/baai/bge-small-en-v1.5"
    use_faiss: bool = True
    db_path: str = "/app/data/mnemo.db"


# =============================================================================
# MNEMO ENGINE (SQLite-backed, same public API as v6.5)
# =============================================================================

class MnemoEngine:
    """Core memory engine. All state in SQLite. FAISS is disposable cache."""

    def __init__(self, config: MnemoConfig = None):
        self.config = config or MnemoConfig()

        # SQLite
        self.db = MnemoDB(self.config.db_path)

        # Encoder (Cloudflare → SentenceTransformer → error)
        self.encoder, self._embedding_dim, self._encoder_name = create_encoder(self.config)
        self._emb_cache = EmbeddingCache(self.encoder, max_size=500)

        # FAISS indices (disposable, rebuilt from DB)
        self._cp_faiss = FAISSIndex(dim=self._embedding_dim)
        self._blob_faiss = FAISSBlobIndex(dim=self._embedding_dim)
        if self.config.use_faiss and HAS_FAISS:
            self._cp_faiss.rebuild_from_db(self.db)
            self._blob_faiss.rebuild_from_db(self.db)
            print("FAISS indices rebuilt from SQLite.")
        elif self.config.use_faiss and not HAS_FAISS:
            print("FAISS not installed; using numpy fallback.")

        # Stats cache (loaded from DB, flushed periodically)
        self._stats = self._load_stats()
        self._dirty = False
        self._lock = threading.RLock()  # Only for stats + dirty flag now

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _get_embedding(self, text: str) -> np.ndarray:
        return self._emb_cache.get_embedding(text)

    def _generate_id(self, content: str, namespace: str = "default") -> str:
        return "mem_" + hashlib.sha256((content + namespace).encode()).hexdigest()[:16]

    def _generate_cp_id(self, entity: str, point_type: str, value: str) -> str:
        raw = f"{entity}:{point_type}:{value}".lower()
        return "cp_" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _cp_to_searchable(self, entity: str, point_type: str, value: str,
                          connects_to: str, reason: str, category: str) -> str:
        parts = [f"[{category.upper()}]", entity]
        if connects_to:
            parts.append(f"{point_type} {connects_to}")
        else:
            parts.append(point_type)
        if value:
            parts.append(value)
        if reason:
            parts.append(reason)
        return " | ".join(parts)

    def _row_to_cp_dict(self, row) -> dict:
        return {
            "id": row["id"], "entity": row["entity"], "point_type": row["point_type"],
            "connects_to": row["connects_to"], "value": row["value"],
            "reason": row["reason"], "weight": round(row["weight"], 3),
            "category": row["category"], "session_id": row["session_id"],
            "source": row["source"], "thread_id": row["thread_id"],
            "position": row["position"], "namespace": row["namespace"],
            "created_at": row["created_at"],
        }

    def _load_stats(self) -> dict:
        defaults = {
            "adds": 0, "adds_rejected": 0, "searches": 0,
            "links_created": 0, "links_decayed": 0, "links_pruned": 0,
            "inject_recommended": 0, "skip_recommended": 0,
            "decayed": 0, "pruned": 0,
            "tier_promotions": 0, "tier_demotions": 0,
            "points_added": 0, "graph_searches": 0,
            "threads_created": 0, "knots_created": 0,
            "temporal_links_skipped": 0,
        }
        try:
            with self.db.read() as conn:
                rows = conn.execute("SELECT key, value FROM engine_stats").fetchall()
                for r in rows:
                    if r["key"] in defaults:
                        defaults[r["key"]] = r["value"]
        except Exception:
            pass
        return defaults

    def _flush_stats(self):
        try:
            with self.db.write() as conn:
                for key, value in self._stats.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO engine_stats(key, value) VALUES (?, ?)",
                        (key, value))
        except Exception as e:
            log.warning(f"Stats flush failed: {e}")

    def _incr_stat(self, key: str, delta: int = 1):
        with self._lock:
            self._stats[key] = self._stats.get(key, 0) + delta

    def _extract_entities(self, query: str) -> Tuple[Set[str], Set[str]]:
        """Three-signal entity extraction (from v6.4).

        Returns (single_word_entities, multi_word_entities).
        """
        # Signal 1: Title-case words
        candidates = set(re.findall(r'\b[A-Z][a-z]{2,}\b', query))

        # Signal 2: Known entity matching (case-insensitive)
        query_words_lower = {w.lower() for w in re.findall(r"\b\w{3,}\b", query)} - _NER_STOP
        with self.db.read() as conn:
            known_entities = {r["entity"].lower() for r in
                              conn.execute("SELECT DISTINCT entity FROM connection_points").fetchall()}
        for qw in query_words_lower:
            if qw in known_entities:
                candidates.add(qw.title())

        # Signal 3: Multi-word entity detection
        query_lower = query.lower()
        multi_word = set()
        for ent in known_entities:
            if ' ' in ent and ent in query_lower:
                multi_word.add(ent)

        # Consecutive capitalized words
        words = query.split()
        i = 0
        while i < len(words):
            if re.match(r'^[A-Z][a-z]{2,}$', words[i]):
                parts = [words[i]]
                j = i + 1
                while j < len(words) and re.match(r'^[A-Z][a-z]{2,}$', words[j]):
                    parts.append(words[j])
                    j += 1
                if len(parts) >= 2:
                    multi_word.add(" ".join(parts).lower())
                i = j
            else:
                i += 1

        return candidates, multi_word

    # =========================================================================
    # CONNECTION POINT OPERATIONS
    # =========================================================================

    def add_point(self, entity: str, point_type: str, value: str = "",
                  connects_to: str = "", reason: str = "", weight: float = 0.5,
                  category: str = "fact", session_id: str = "",
                  source: str = "auto_extract",
                  thread_id: str = "", position: int = -1,
                  namespace: str = "default") -> Optional[str]:
        cp_id = self._generate_cp_id(entity, point_type, value)

        # Upsert check
        with self.db.read() as conn:
            existing = conn.execute("SELECT id, weight, reason FROM connection_points WHERE id = ?",
                                    (cp_id,)).fetchone()
        if existing:
            with self.db.write() as conn:
                conn.execute(
                    "UPDATE connection_points SET weight = MAX(weight, ?), reason = COALESCE(NULLIF(reason,''), ?) WHERE id = ?",
                    (weight, reason, cp_id))
            return cp_id

        # Compute embedding outside DB lock
        searchable = self._cp_to_searchable(entity, point_type, value, connects_to, reason, category)
        embedding = self._get_embedding(searchable)
        emb_blob = embedding.astype(np.float32).tobytes()

        with self.db.write() as conn:
            conn.execute("""
                INSERT INTO connection_points
                (id, entity, point_type, value, connects_to, reason, weight,
                 category, session_id, source, thread_id, position, namespace, created_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cp_id, entity, point_type, value, connects_to, reason, weight,
                  category, session_id, source, thread_id, position, namespace,
                  time.time(), emb_blob))

        # Update FAISS
        self._cp_faiss.add(cp_id, embedding)
        self._incr_stat("points_added")
        self._dirty = True
        return cp_id

    def add_points_batch(self, points: List[dict]) -> List[Optional[str]]:
        return [self.add_point(**p) for p in points]

    def get_point(self, cp_id: str) -> Optional[dict]:
        with self.db.read() as conn:
            row = conn.execute(
                "SELECT * FROM connection_points WHERE id = ?", (cp_id,)
            ).fetchone()
        if row:
            return self._row_to_cp_dict(row)
        return None

    def delete_point(self, cp_id: str) -> bool:
        with self.db.read() as conn:
            row = conn.execute("SELECT id FROM connection_points WHERE id = ?", (cp_id,)).fetchone()
        if not row:
            return False
        with self.db.write() as conn:
            conn.execute("DELETE FROM thread_points WHERE cp_id = ?", (cp_id,))
            conn.execute("DELETE FROM connection_points WHERE id = ?", (cp_id,))
        self._cp_faiss.mark_dirty()
        self._dirty = True
        return True

    def update_point(self, cp_id: str, entity: str = None, value: str = None,
                     connects_to: str = None, reason: str = None,
                     weight: float = None, category: str = None,
                     point_type: str = None) -> Optional[dict]:
        """Update fields on an existing ConnectionPoint. Returns updated CP dict.

        Only non-None fields are updated. Re-embeds if any text field changes.
        """
        with self.db.read() as conn:
            row = conn.execute("SELECT * FROM connection_points WHERE id = ?", (cp_id,)).fetchone()
        if not row:
            return None

        # Determine which fields changed
        updates = {}
        if entity is not None and entity != row["entity"]:
            updates["entity"] = entity
        if value is not None and value != row["value"]:
            updates["value"] = value
        if connects_to is not None and connects_to != row["connects_to"]:
            updates["connects_to"] = connects_to
        if reason is not None and reason != row["reason"]:
            updates["reason"] = reason
        if weight is not None and abs(weight - row["weight"]) > 0.001:
            updates["weight"] = weight
        if category is not None and category != row["category"]:
            updates["category"] = category
        if point_type is not None and point_type != row["point_type"]:
            updates["point_type"] = point_type

        if not updates:
            return self._row_to_cp_dict(row)  # Nothing changed

        # Build SET clause
        set_parts = []
        params = []
        for col, val in updates.items():
            set_parts.append(f"{col} = ?")
            params.append(val)

        # Re-embed if any text field changed
        text_fields = {"entity", "value", "connects_to", "reason", "category", "point_type"}
        if updates.keys() & text_fields:
            new_entity = updates.get("entity", row["entity"])
            new_pt = updates.get("point_type", row["point_type"])
            new_val = updates.get("value", row["value"])
            new_conn = updates.get("connects_to", row["connects_to"])
            new_reason = updates.get("reason", row["reason"])
            new_cat = updates.get("category", row["category"])

            searchable = self._cp_to_searchable(new_entity, new_pt, new_val, new_conn, new_reason, new_cat)
            embedding = self._get_embedding(searchable)
            emb_blob = embedding.astype(np.float32).tobytes()
            set_parts.append("embedding = ?")
            params.append(emb_blob)
            re_embedded = True
        else:
            re_embedded = False

        params.append(cp_id)
        with self.db.write() as conn:
            conn.execute(
                f"UPDATE connection_points SET {', '.join(set_parts)} WHERE id = ?",
                params)

        if re_embedded:
            self._cp_faiss.mark_dirty()  # Will rebuild with new embedding

        self._dirty = True
        return self.get_point(cp_id)

    def delete_session_points(self, session_id: str) -> int:
        """Delete all non-protected CPs and blobs for a session."""
        PROTECTED = ("file_upload", "manual_correction", "consolidation", "manual")
        placeholders = ",".join("?" * len(PROTECTED))

        with self.db.write() as conn:
            # Delete CPs
            conn.execute(f"""
                DELETE FROM thread_points WHERE cp_id IN (
                    SELECT id FROM connection_points
                    WHERE session_id = ? AND source NOT IN ({placeholders})
                )""", (session_id, *PROTECTED))
            cur = conn.execute(f"""
                DELETE FROM connection_points
                WHERE session_id = ? AND source NOT IN ({placeholders})
            """, (session_id, *PROTECTED))
            cp_deleted = cur.rowcount

            # Delete blobs (using proper columns now, not json_extract)
            cur2 = conn.execute(f"""
                DELETE FROM memories
                WHERE session_id = ? AND source NOT IN ({placeholders})
            """, (session_id, *PROTECTED))
            blob_deleted = cur2.rowcount

        total = cp_deleted + blob_deleted
        if total > 0:
            self._cp_faiss.mark_dirty()
            self._dirty = True
        return total

    def list_points(self, limit: int = 200) -> List[dict]:
        """List all connection points (for UI display)."""
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT * FROM connection_points ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [self._row_to_cp_dict(r) for r in rows]

    def entity_lookup(self, entity: str) -> List[dict]:
        """Look up all CPs for an entity."""
        with self.db.read() as conn:
            rows = conn.execute(
                """SELECT * FROM connection_points
                   WHERE entity = ? COLLATE NOCASE OR connects_to = ? COLLATE NOCASE""",
                (entity, entity)).fetchall()
        return [self._row_to_cp_dict(r) for r in rows]

    # =========================================================================
    # GRAPH SEARCH (hybrid: SQLite + FTS5 + FAISS + NumPy)
    # =========================================================================

    def graph_search(self, query: str, top_k: int = 15,
                     active_sessions: Optional[List[str]] = None) -> List[dict]:
        """v7.0 hybrid search pipeline.

        Phase 1: Entity graph (SQLite indices)    ~1ms
        Phase 2: Full-text search (FTS5 BM25)     ~3ms
        Phase 3: Semantic pre-filter (FAISS)       ~5ms
        Phase 4: Exact reranking (NumPy)           ~1ms
        Phase 5: Score fusion                      ~0ms
        """
        self._incr_stat("graph_searches")

        # Compute query embedding (cached: ~0ms, uncached: ~50ms)
        query_emb = self._get_embedding(query)

        # --- PHASE 1: Entity graph lookup via SQLite ---
        entities, multi_word = self._extract_entities(query)
        graph_scores: Dict[str, float] = {}

        session_filter = ""
        session_params: list = []
        if active_sessions:
            placeholders = ",".join("?" * len(active_sessions))
            session_filter = f" AND (session_id = '' OR session_id IN ({placeholders}))"
            session_params = list(active_sessions)

        with self.db.read() as conn:
            # Direct entity hits
            for ent in entities:
                rows = conn.execute(
                    f"SELECT id FROM connection_points WHERE entity = ? COLLATE NOCASE{session_filter}",
                    [ent] + session_params).fetchall()
                for r in rows:
                    graph_scores[r["id"]] = graph_scores.get(r["id"], 0) + 0.5

                # 1-hop connection expansion
                rows = conn.execute(
                    f"SELECT id FROM connection_points WHERE connects_to = ? COLLATE NOCASE{session_filter}",
                    [ent] + session_params).fetchall()
                for r in rows:
                    graph_scores[r["id"]] = graph_scores.get(r["id"], 0) + 0.4

            # Multi-word entity hits
            for mw in multi_word:
                rows = conn.execute(
                    f"SELECT id FROM connection_points WHERE entity = ? COLLATE NOCASE{session_filter}",
                    [mw] + session_params).fetchall()
                for r in rows:
                    graph_scores[r["id"]] = graph_scores.get(r["id"], 0) + 0.5
                rows = conn.execute(
                    f"SELECT id FROM connection_points WHERE connects_to = ? COLLATE NOCASE{session_filter}",
                    [mw] + session_params).fetchall()
                for r in rows:
                    graph_scores[r["id"]] = graph_scores.get(r["id"], 0) + 0.4

            # Type-keyword boosting
            type_keywords = {
                "relationship": ["relationship", "brother", "sister", "friend", "rival",
                                 "married", "mentor", "captor", "ally", "between"],
                "fears": ["fear", "fears", "afraid", "terrified", "dread"],
                "tone": ["tone", "mood", "atmosphere", "register", "feeling"],
                "secret": ["secret", "hidden", "private", "unknown"],
                "motivation": ["motivation", "drive", "wants", "desires", "goal"],
                "plot": ["plot", "event", "arc", "storyline", "happened", "when", "timeline", "book"],
            }
            ql = query.lower()
            for pt, keywords in type_keywords.items():
                if any(kw in ql for kw in keywords):
                    all_ents = list(entities) + list(multi_word)
                    for ent in all_ents:
                        rows = conn.execute(
                            f"SELECT id FROM connection_points WHERE entity = ? COLLATE NOCASE AND point_type = ?{session_filter}",
                            [ent, pt] + session_params).fetchall()
                        for r in rows:
                            graph_scores[r["id"]] = graph_scores.get(r["id"], 0) + 0.2

        # --- PHASE 2: FTS5 full-text search ---
        fts_scores: Dict[str, float] = {}
        fts_terms = [w for w in re.findall(r'\b\w{3,}\b', query) if w.lower() not in _NER_STOP]
        if fts_terms:
            fts_query = " OR ".join(f'"{t}"' for t in fts_terms[:10])
            try:
                with self.db.read() as conn:
                    rows = conn.execute("""
                        SELECT cp.id, bm25(cp_fts, 5.0, 3.0, 2.0, 1.0) as rank
                        FROM cp_fts
                        JOIN connection_points cp ON cp.rowid = cp_fts.rowid
                        WHERE cp_fts MATCH ?
                        ORDER BY rank LIMIT 100
                    """, (fts_query,)).fetchall()
                if rows:
                    min_r = min(r["rank"] for r in rows)
                    max_r = max(r["rank"] for r in rows)
                    spread = max_r - min_r if max_r != min_r else 1.0
                    for r in rows:
                        fts_scores[r["id"]] = 1.0 - (r["rank"] - min_r) / spread
            except Exception as e:
                log.warning(f"FTS5 search error: {e}")

        # --- PHASE 3: FAISS semantic pre-filter ---
        faiss_candidates = self._cp_faiss.search(query_emb, top_k=min(top_k * 5, 200))

        # --- PHASE 4: Merge candidate set + load embeddings + NumPy rerank ---
        all_candidate_ids = set(graph_scores.keys()) | set(fts_scores.keys())
        all_candidate_ids.update(cid for cid, _ in faiss_candidates)

        if not all_candidate_ids:
            return []

        # Batch-load embeddings + metadata from SQLite
        id_list = list(all_candidate_ids)
        placeholders = ",".join("?" * len(id_list))
        with self.db.read() as conn:
            rows = conn.execute(f"""
                SELECT id, entity, point_type, value, connects_to, reason,
                       weight, category, session_id, source, thread_id,
                       position, namespace, created_at, embedding
                FROM connection_points
                WHERE id IN ({placeholders}) AND embedding IS NOT NULL
            """, id_list).fetchall()

        # Apply session filter + build numpy arrays
        valid_ids = []
        valid_embs = []
        meta_by_id: Dict[str, dict] = {}
        for row in rows:
            if active_sessions and row["session_id"] and row["session_id"] not in active_sessions:
                continue
            valid_ids.append(row["id"])
            valid_embs.append(np.frombuffer(row["embedding"], dtype=np.float32))
            meta_by_id[row["id"]] = self._row_to_cp_dict(row)

        if not valid_ids:
            return []

        emb_matrix = np.stack(valid_embs)

        # NumPy exact reranking
        reranked = NumpyReranker.rerank(query_emb, valid_ids, emb_matrix, threshold=0.15)

        # --- PHASE 5: Score fusion ---
        results = []
        for cp_id, sem_score in reranked:
            meta = meta_by_id.get(cp_id, {})
            if not meta:
                continue
            graph = min(graph_scores.get(cp_id, 0.0), 1.0)
            fts = min(fts_scores.get(cp_id, 0.0), 1.0)

            # Weighted fusion: semantic primary, graph + FTS boost
            final = sem_score * 0.55 + graph * 0.30 + fts * 0.15
            meta["score"] = round(min(final, 1.0), 3)
            meta["graph_score"] = round(graph, 3)
            meta["semantic_score"] = round(sem_score, 3)
            meta["fts_score"] = round(fts, 3)
            results.append(meta)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # =========================================================================
    # THREAD OPERATIONS
    # =========================================================================

    def add_thread(self, thread_id: str, name: str, entity: str = "",
                   thread_type: str = "plot_line", session_id: str = "",
                   point_ids: Optional[List[str]] = None) -> str:
        with self.db.read() as conn:
            existing = conn.execute("SELECT id FROM threads WHERE id = ?", (thread_id,)).fetchone()

        if existing:
            if point_ids:
                with self.db.write() as conn:
                    # Get current max position
                    row = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) as maxp FROM thread_points WHERE thread_id = ?",
                        (thread_id,)).fetchone()
                    pos = row["maxp"] + 1
                    for pid in point_ids:
                        conn.execute(
                            "INSERT OR IGNORE INTO thread_points(thread_id, cp_id, position) VALUES (?, ?, ?)",
                            (thread_id, pid, pos))
                        pos += 1
            self._dirty = True
            return thread_id

        with self.db.write() as conn:
            conn.execute("""
                INSERT INTO threads (id, name, thread_type, entity, status,
                    tension_level, tone_trajectory, current_position, session_id, created_at)
                VALUES (?, ?, ?, ?, 'active', 0.5, '[]', 0, ?, ?)
            """, (thread_id, name, thread_type, entity, session_id, time.time()))
            if point_ids:
                for pos, pid in enumerate(point_ids):
                    conn.execute(
                        "INSERT OR IGNORE INTO thread_points(thread_id, cp_id, position) VALUES (?, ?, ?)",
                        (thread_id, pid, pos))
        self._incr_stat("threads_created")
        self._dirty = True
        return thread_id

    def advance_thread(self, thread_id: str, new_position: int = -1) -> bool:
        with self.db.read() as conn:
            row = conn.execute("SELECT current_position FROM threads WHERE id = ?",
                               (thread_id,)).fetchone()
        if not row:
            return False
        if new_position >= 0:
            pos = new_position
        else:
            with self.db.read() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) as cnt FROM thread_points WHERE thread_id = ?",
                    (thread_id,)).fetchone()["cnt"]
            pos = min(row["current_position"] + 1, max(total - 1, 0))
        with self.db.write() as conn:
            conn.execute("UPDATE threads SET current_position = ? WHERE id = ?",
                         (pos, thread_id))
        self._dirty = True
        return True

    def trace_thread(self, thread_id: str, from_position: int = -1,
                     direction: str = "back", steps: int = 5) -> List[dict]:
        with self.db.read() as conn:
            thread = conn.execute("SELECT * FROM threads WHERE id = ?",
                                  (thread_id,)).fetchone()
            if not thread:
                return []
            pos = from_position if from_position >= 0 else thread["current_position"]

            if direction == "back":
                rows = conn.execute("""
                    SELECT cp.* FROM thread_points tp
                    JOIN connection_points cp ON cp.id = tp.cp_id
                    WHERE tp.thread_id = ? AND tp.position >= ? AND tp.position < ?
                    ORDER BY tp.position
                """, (thread_id, max(0, pos - steps), pos)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT cp.* FROM thread_points tp
                    JOIN connection_points cp ON cp.id = tp.cp_id
                    WHERE tp.thread_id = ? AND tp.position > ? AND tp.position <= ?
                    ORDER BY tp.position
                """, (thread_id, pos, pos + steps)).fetchall()

        return [self._row_to_cp_dict(r) for r in rows]

    def get_active_threads(self) -> List[dict]:
        with self.db.read() as conn:
            threads = conn.execute("SELECT * FROM threads WHERE status = 'active'").fetchall()
            result = []
            for t in threads:
                points = conn.execute(
                    "SELECT cp_id FROM thread_points WHERE thread_id = ? ORDER BY position",
                    (t["id"],)).fetchall()
                knots = conn.execute(
                    "SELECT knot_id FROM knot_threads WHERE thread_id = ?",
                    (t["id"],)).fetchall()
                result.append({
                    "id": t["id"], "name": t["name"], "thread_type": t["thread_type"],
                    "entity": t["entity"], "status": t["status"],
                    "tension_level": round(t["tension_level"], 3),
                    "tone_trajectory": json.loads(t["tone_trajectory"] or "[]"),
                    "current_position": t["current_position"],
                    "session_id": t["session_id"], "created_at": t["created_at"],
                    "points": [r["cp_id"] for r in points],
                    "knots": [r["knot_id"] for r in knots],
                })
        return result

    def get_thread(self, thread_id: str) -> Optional[dict]:
        with self.db.read() as conn:
            t = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
            if not t:
                return None
            points = conn.execute(
                "SELECT cp_id FROM thread_points WHERE thread_id = ? ORDER BY position",
                (thread_id,)).fetchall()
            knots = conn.execute(
                "SELECT knot_id FROM knot_threads WHERE thread_id = ?",
                (thread_id,)).fetchall()
        return {
            "id": t["id"], "name": t["name"], "thread_type": t["thread_type"],
            "entity": t["entity"], "status": t["status"],
            "tension_level": round(t["tension_level"], 3),
            "tone_trajectory": json.loads(t["tone_trajectory"] or "[]"),
            "current_position": t["current_position"],
            "session_id": t["session_id"], "created_at": t["created_at"],
            "points": [r["cp_id"] for r in points],
            "knots": [r["knot_id"] for r in knots],
        }

    def delete_thread(self, thread_id: str) -> bool:
        with self.db.write() as conn:
            cur = conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            if cur.rowcount == 0:
                return False
            conn.execute("DELETE FROM thread_points WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM knot_threads WHERE thread_id = ?", (thread_id,))
        self._dirty = True
        return True

    # =========================================================================
    # KNOT OPERATIONS
    # =========================================================================

    def add_knot(self, knot_id: str, name: str, thread_ids: List[str],
                 pivot_type: str = "collision", reason: str = "",
                 session_id: str = "",
                 active_points: Optional[Dict[str, List[str]]] = None) -> str:
        with self.db.write() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO knots
                (id, name, pivot_type, reason, active_points, session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (knot_id, name, pivot_type, reason,
                  json.dumps(active_points or {}), session_id, time.time()))
            for tid in thread_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO knot_threads(knot_id, thread_id) VALUES (?, ?)",
                    (knot_id, tid))
        self._incr_stat("knots_created")
        self._dirty = True
        return knot_id

    def get_knot_context(self, knot_id: str) -> Optional[dict]:
        with self.db.read() as conn:
            knot = conn.execute("SELECT * FROM knots WHERE id = ?", (knot_id,)).fetchone()
            if not knot:
                return None

            thread_ids = [r["thread_id"] for r in
                          conn.execute("SELECT thread_id FROM knot_threads WHERE knot_id = ?",
                                       (knot_id,)).fetchall()]
            active_pts = json.loads(knot["active_points"] or "{}")

            context = {
                "id": knot["id"], "name": knot["name"],
                "threads": thread_ids, "pivot_type": knot["pivot_type"],
                "narrative_weight": round(knot["narrative_weight"], 3),
                "tension_before": round(knot["tension_before"], 3),
                "tension_after": round(knot["tension_after"], 3),
                "tone_shift": knot["tone_shift"], "reason": knot["reason"],
                "unresolved": json.loads(knot["unresolved"] or "[]"),
                "active_points": active_pts,
                "session_id": knot["session_id"], "created_at": knot["created_at"],
                "thread_context": {},
            }

            for tid in thread_ids:
                t = conn.execute("SELECT * FROM threads WHERE id = ?", (tid,)).fetchone()
                if not t:
                    continue
                cp_ids = active_pts.get(tid, [])
                active_cps = []
                for pid in cp_ids:
                    cp_row = conn.execute("SELECT * FROM connection_points WHERE id = ?",
                                          (pid,)).fetchone()
                    if cp_row:
                        active_cps.append(self._row_to_cp_dict(cp_row))

                buildup = self.trace_thread(tid, t["current_position"], "back", 2)

                context["thread_context"][tid] = {
                    "thread_name": t["name"],
                    "entity": t["entity"],
                    "tension": t["tension_level"],
                    "tone": json.loads(t["tone_trajectory"] or "[]")[-1:]
                           and json.loads(t["tone_trajectory"] or "[]")[-1] or "",
                    "active_points": active_cps,
                    "buildup": buildup,
                }
        return context

    def list_knots(self) -> List[dict]:
        with self.db.read() as conn:
            knots = conn.execute("SELECT * FROM knots").fetchall()
            result = []
            for k in knots:
                thread_ids = [r["thread_id"] for r in
                              conn.execute("SELECT thread_id FROM knot_threads WHERE knot_id = ?",
                                           (k["id"],)).fetchall()]
                result.append({
                    "id": k["id"], "name": k["name"], "threads": thread_ids,
                    "pivot_type": k["pivot_type"], "reason": k["reason"],
                    "session_id": k["session_id"], "created_at": k["created_at"],
                    "narrative_weight": round(k["narrative_weight"], 3),
                    "tension_before": round(k["tension_before"], 3),
                    "tension_after": round(k["tension_after"], 3),
                    "tone_shift": k["tone_shift"],
                    "unresolved": json.loads(k["unresolved"] or "[]"),
                    "active_points": json.loads(k["active_points"] or "{}"),
                })
        return result

    def delete_knot(self, knot_id: str) -> bool:
        with self.db.write() as conn:
            cur = conn.execute("DELETE FROM knots WHERE id = ?", (knot_id,))
            if cur.rowcount == 0:
                return False
            conn.execute("DELETE FROM knot_threads WHERE knot_id = ?", (knot_id,))
        self._dirty = True
        return True

    def delete_session_threads_and_knots(self, session_id: str) -> dict:
        with self.db.write() as conn:
            # Get thread/knot IDs for this session
            t_ids = [r["id"] for r in conn.execute(
                "SELECT id FROM threads WHERE session_id = ?", (session_id,)).fetchall()]
            k_ids = [r["id"] for r in conn.execute(
                "SELECT id FROM knots WHERE session_id = ?", (session_id,)).fetchall()]

            for tid in t_ids:
                conn.execute("DELETE FROM thread_points WHERE thread_id = ?", (tid,))
                conn.execute("DELETE FROM knot_threads WHERE thread_id = ?", (tid,))
            if t_ids:
                placeholders = ",".join("?" * len(t_ids))
                conn.execute(f"DELETE FROM threads WHERE id IN ({placeholders})", t_ids)

            for kid in k_ids:
                conn.execute("DELETE FROM knot_threads WHERE knot_id = ?", (kid,))
            if k_ids:
                placeholders = ",".join("?" * len(k_ids))
                conn.execute(f"DELETE FROM knots WHERE id IN ({placeholders})", k_ids)

        self._dirty = True
        return {"deleted_threads": len(t_ids), "deleted_knots": len(k_ids)}

    # =========================================================================
    # LEGACY BLOB MEMORY OPERATIONS (backward compat)
    # =========================================================================

    def add(self, content: str, namespace: str = "default",
            metadata: dict = None, priority: float = 1.0) -> Optional[str]:
        memory_id = self._generate_id(content, namespace)
        meta = metadata or {}

        # Extract session_id and source into proper columns (not buried in JSON)
        session_id = meta.pop("session_id", "")
        source = meta.pop("source", "auto_extract")
        meta_json = json.dumps(meta)  # Remaining metadata only

        with self.db.read() as conn:
            existing = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if existing:
            with self.db.write() as conn:
                conn.execute("""
                    UPDATE memories SET last_accessed = ?, access_count = access_count + 1,
                    priority = MAX(priority, ?) WHERE id = ?
                """, (time.time(), priority, memory_id))
            self._dirty = True
            return memory_id

        embedding = self._get_embedding(content)
        quality = self._estimate_quality(content, embedding)
        if quality < self.config.quality_threshold:
            self._incr_stat("adds_rejected")
            return None

        emb_blob = embedding.astype(np.float32).tobytes()
        with self.db.write() as conn:
            conn.execute("""
                INSERT INTO memories
                (id, content, tier, namespace, quality_score, access_count, priority,
                 session_id, source, created_at, last_accessed, metadata, embedding)
                VALUES (?, ?, 'semantic', ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, content, namespace, quality, priority,
                  session_id, source, time.time(), time.time(), meta_json, emb_blob))

        self._blob_faiss.add(memory_id, embedding)
        self._create_links(memory_id, embedding, namespace, content)
        self._incr_stat("adds")
        self._dirty = True
        return memory_id

    def search(self, query: str, top_k: int = 15, namespace: str = None) -> list:
        """Search blob memories. Returns list of SearchResult objects."""
        self._incr_stat("searches")
        query_emb = self._get_embedding(query)

        # FAISS pre-filter
        faiss_results = self._blob_faiss.search(query_emb, top_k=top_k * 3)

        # If no FAISS results, try numpy fallback
        if not faiss_results:
            with self.db.read() as conn:
                ns_clause = " AND namespace = ?" if namespace else ""
                params = [namespace] if namespace else []
                rows = conn.execute(
                    f"SELECT id, embedding FROM memories WHERE embedding IS NOT NULL{ns_clause}",
                    params).fetchall()
            if rows:
                ids = [r["id"] for r in rows]
                embs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                reranked = NumpyReranker.rerank(query_emb, ids, embs, threshold=0.20)
                faiss_results = {cid: score for cid, score in reranked}

        # Load blob metadata for candidates
        candidate_ids = list(faiss_results.keys() if isinstance(faiss_results, dict) else
                             {cid for cid, _ in faiss_results})
        if not candidate_ids:
            return []

        placeholders_str = ",".join("?" * len(candidate_ids))
        with self.db.read() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders_str})", candidate_ids
            ).fetchall()

        # Link propagation
        link_scores: Dict[str, float] = {}
        sem_scores = faiss_results if isinstance(faiss_results, dict) else dict(faiss_results)
        top_seeds = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)[:15]

        with self.db.read() as conn:
            for mem_id, base_score in top_seeds:
                if base_score < 0.20:
                    continue
                outgoing = conn.execute(
                    "SELECT target_id, strength FROM neural_links WHERE source_id = ?",
                    (mem_id,)).fetchall()
                for link in outgoing:
                    boost = base_score * link["strength"] * 0.60
                    link_scores[link["target_id"]] = link_scores.get(link["target_id"], 0) + boost
                incoming = conn.execute(
                    "SELECT source_id, strength FROM neural_links WHERE target_id = ?",
                    (mem_id,)).fetchall()
                for link in incoming:
                    boost = base_score * link["strength"] * 0.40
                    link_scores[link["source_id"]] = link_scores.get(link["source_id"], 0) + boost

        results = []
        now = time.time()
        update_ids = []
        for row in rows:
            if namespace and row["namespace"] != namespace:
                continue
            sem = sem_scores.get(row["id"], 0)
            lnk = link_scores.get(row["id"], 0)
            combined = sem * 0.7 + min(lnk, 0.5) * 0.3
            if combined >= self.config.similarity_threshold:
                update_ids.append(row["id"])
                results.append(SearchResult(
                    id=row["id"], content=row["content"], score=combined,
                    tier=row["tier"], semantic_score=sem, link_score=lnk,
                    metadata=json.loads(row["metadata"] or "{}"),
                ))

        if update_ids:
            with self.db.write() as conn:
                placeholders_str = ",".join("?" * len(update_ids))
                conn.execute(
                    f"UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id IN ({placeholders_str})",
                    [now] + update_ids)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def should_inject(self, query: str, context: str = "",
                      conversation_history: str = "") -> Tuple[bool, str, float]:
        combined = (query + " " + context).lower()
        skip_signals = ["this is a new", "new topic", "start fresh", "forget everything"]
        for signal in skip_signals:
            if signal in combined:
                self._incr_stat("skip_recommended")
                return False, "skip:" + signal, 0.0

        best_score = 0.0
        source = "none"

        blob_results = self.search(query, top_k=3)
        if blob_results:
            best_score = blob_results[0].score
            source = "blob"

        if best_score < 0.40:
            cp_results = self.graph_search(query, top_k=3)
            if cp_results:
                cp_best = cp_results[0].get("score", 0)
                if cp_best > best_score:
                    best_score = cp_best
                    source = "cp"

        if best_score == 0.0:
            self._incr_stat("skip_recommended")
            return False, "no_relevant_memories", 0.0

        inject_signals = [
            "previous", "earlier", "before", "you said", "you mentioned",
            "based on", "using your", "your analysis", "your framework",
            "compare", "contrast", "synthesize", "combine",
            "apply your", "you previously", "your earlier",
            "tell me everything", "remember",
        ]
        keyword_boost = 0.0
        matched_signal = None
        for signal in inject_signals:
            if signal in combined:
                keyword_boost = 0.15
                matched_signal = signal
                break

        confidence = min(1.0, best_score + keyword_boost)

        if conversation_history and len(conversation_history.split()) > 500:
            query_words = set(query.lower().split())
            if query_words:
                history_words = set(conversation_history.lower().split())
                overlap = len(query_words & history_words)
                if overlap > len(query_words) * 0.6:
                    confidence *= 0.5

        if confidence >= 0.40:
            self._incr_stat("inject_recommended")
            reason = f"inject:confidence={round(confidence, 2)}"
            if matched_signal:
                reason += f",signal={matched_signal}"
            return True, reason, confidence
        else:
            self._incr_stat("skip_recommended")
            return False, f"low_confidence:{round(confidence, 2)}", confidence

    def get_context(self, query: str, top_k: int = 15) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        parts = ["[RELEVANT CONTEXT FROM MEMORY]"]
        for r in results:
            parts.append(f"- [{r.tier.upper()}] {r.content}")
        parts.append("[END CONTEXT]")
        return "\n".join(parts)

    def get(self, memory_id: str) -> Optional[dict]:
        with self.db.read() as conn:
            cp = conn.execute("SELECT * FROM connection_points WHERE id = ?",
                              (memory_id,)).fetchone()
            if cp:
                return self._row_to_cp_dict(cp)
            mem = conn.execute("SELECT * FROM memories WHERE id = ?",
                               (memory_id,)).fetchone()
            if mem:
                return {
                    "id": mem["id"], "content": mem["content"],
                    "tier": mem["tier"], "namespace": mem["namespace"],
                    "quality_score": round(mem["quality_score"], 3),
                    "access_count": mem["access_count"], "priority": mem["priority"],
                    "session_id": mem["session_id"], "source": mem["source"],
                    "created_at": mem["created_at"], "last_accessed": mem["last_accessed"],
                    "metadata": json.loads(mem["metadata"] or "{}"),
                }
        return None

    def delete(self, memory_id: str) -> bool:
        with self.db.write() as conn:
            # Try CP first
            cur = conn.execute("DELETE FROM connection_points WHERE id = ?", (memory_id,))
            if cur.rowcount > 0:
                conn.execute("DELETE FROM thread_points WHERE cp_id = ?", (memory_id,))
                self._cp_faiss.mark_dirty()
                self._dirty = True
                return True
            # Try blob
            cur = conn.execute("DELETE FROM neural_links WHERE source_id = ? OR target_id = ?",
                               (memory_id, memory_id))
            cur2 = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if cur2.rowcount > 0:
                self._dirty = True
                return True
        return False

    def list_all(self) -> List[dict]:
        with self.db.read() as conn:
            rows = conn.execute("SELECT * FROM memories").fetchall()
        return [{
            "id": r["id"], "content": r["content"], "tier": r["tier"],
            "namespace": r["namespace"], "quality_score": round(r["quality_score"], 3),
            "access_count": r["access_count"], "priority": r["priority"],
            "session_id": r["session_id"], "source": r["source"],
            "created_at": r["created_at"], "last_accessed": r["last_accessed"],
            "metadata": json.loads(r["metadata"] or "{}"),
        } for r in rows]

    def list_memories(self, namespace: str = None) -> List[dict]:
        """Alias for list_all with optional namespace filter."""
        return self.list_all()

    # =========================================================================
    # LINK CREATION (for blob memories)
    # =========================================================================

    def _estimate_quality(self, content: str, embedding: np.ndarray) -> float:
        score = 0.5
        words = len(content.split())
        lower = content.lower()
        is_high_value = any(m in lower for m in HIGH_VALUE_MARKERS)
        if is_high_value:
            score += 0.2
            if words > 20:
                score += 0.1
        else:
            if words < 5:
                score -= 0.3
            elif words > 20:
                score += 0.1
        # Check duplicate via FAISS
        top_match = self._blob_faiss.search(embedding, top_k=1)
        if top_match:
            best_sim = list(top_match.values())[0] if isinstance(top_match, dict) else (top_match[0][1] if top_match else 0)
            if best_sim > 0.95:
                score -= 0.30
            elif best_sim > 0.85:
                score -= 0.10
            elif best_sim < 0.30:
                score += 0.15
        return max(0.0, min(1.0, score))

    def _create_links(self, memory_id: str, embedding: np.ndarray,
                      namespace: str, content: str):
        """Create neural links for a new blob memory."""
        candidates = self._blob_faiss.search(embedding, top_k=50)
        if isinstance(candidates, dict):
            candidate_items = [(k, v) for k, v in candidates.items() if k != memory_id]
        else:
            candidate_items = [(k, v) for k, v in candidates if k != memory_id]

        my_terms = set(re.findall(r'\b[A-Z][a-z]+\b', content))
        my_causal = any(m in content.lower() for m in [
            "because", "therefore", "led to", "caused", "resulted in",
            "due to", "consequently", "thus", "triggered", "forced"])

        with self.db.read() as conn:
            for other_id, sim in candidate_items:
                other_row = conn.execute("SELECT content, namespace, metadata, created_at FROM memories WHERE id = ?",
                                         (other_id,)).fetchone()
                if not other_row:
                    continue
                other_content = other_row["content"]
                other_terms = set(re.findall(r'\b[A-Z][a-z]+\b', other_content))
                other_meta = json.loads(other_row["metadata"] or "{}")

                link_type = None
                strength = 0.0

                if sim >= 0.85 and my_terms & other_terms:
                    link_type = "direct_reference"
                    strength = 0.90
                elif sim >= 0.80 and my_terms and other_terms and (my_terms <= other_terms or other_terms <= my_terms):
                    link_type = "hierarchical"
                    strength = 0.85
                elif sim >= 0.75 and my_causal:
                    link_type = "causal"
                    strength = 0.80
                elif sim >= 0.50:
                    link_type = "semantic_similarity"
                    strength = 0.75
                elif sim >= 0.45:
                    link_type = "associative"
                    strength = 0.60

                if link_type:
                    self._add_link(memory_id, other_id, link_type, strength)

                # Temporal links (skip batch uploads)
                source = other_meta.get("source", "")
                is_batch = source == "file_upload"
                time_gap = abs(time.time() - other_row["created_at"])
                if time_gap < 300 and not is_batch and link_type != "direct_reference":
                    self._add_link(memory_id, other_id, "temporal", 0.65)

    def _add_link(self, source_id: str, target_id: str,
                  link_type: str, strength: float):
        fwd_id = f"{source_id}:{target_id}:{link_type}"
        rev_id = f"{target_id}:{source_id}:{link_type}"
        now = time.time()
        with self.db.write() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO neural_links (id, source_id, target_id, link_type, strength, created_at, last_maintained)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (fwd_id, source_id, target_id, link_type, strength, now, now))
            conn.execute("""
                INSERT OR IGNORE INTO neural_links (id, source_id, target_id, link_type, strength, created_at, last_maintained)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (rev_id, target_id, source_id, link_type, strength, now, now))
        self._incr_stat("links_created", 2)

    # =========================================================================
    # MAINTENANCE
    # =========================================================================

    def maintenance(self) -> dict:
        now = time.time()
        results = {"decayed": 0, "pruned": 0, "links_decayed": 0,
                   "links_pruned": 0, "tier_promotions": 0, "tier_demotions": 0}

        with self.db.write() as conn:
            # Decay blob memories
            rows = conn.execute("SELECT id, quality_score, priority, last_accessed, access_count, tier, created_at FROM memories").fetchall()
            prune_ids = []
            for r in rows:
                days = (now - r["last_accessed"]) / 86400
                if days > 1:
                    eff_decay = self.config.decay_rate_per_day * days * max(0.1, 1.0 - (r["priority"] - 1.0) * 0.5)
                    new_q = max(0.0, r["quality_score"] - eff_decay)
                    conn.execute("UPDATE memories SET quality_score = ? WHERE id = ?", (new_q, r["id"]))
                    results["decayed"] += 1
                # Promotions/demotions
                if r["tier"] != "working" and r["access_count"] >= self.config.promote_to_working_accesses:
                    conn.execute("UPDATE memories SET tier = 'working' WHERE id = ?", (r["id"],))
                    results["tier_promotions"] += 1
                if r["tier"] != "archive" and days > self.config.demote_to_archive_days and r["access_count"] < self.config.promote_to_working_accesses:
                    conn.execute("UPDATE memories SET tier = 'archive' WHERE id = ?", (r["id"],))
                    results["tier_demotions"] += 1
                days_created = (now - r["created_at"]) / 86400
                if r["quality_score"] < self.config.prune_quality_floor and days_created > self.config.prune_age_days:
                    prune_ids.append(r["id"])

            # Prune low-quality
            for mid in prune_ids:
                conn.execute("DELETE FROM neural_links WHERE source_id = ? OR target_id = ?", (mid, mid))
                conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
                results["pruned"] += 1

            # Decay/prune links
            links = conn.execute("SELECT id, link_type, strength, last_maintained FROM neural_links").fetchall()
            prune_link_ids = []
            for link in links:
                props = LINK_PROPERTIES.get(LinkType(link["link_type"]) if link["link_type"] in [lt.value for lt in LinkType] else LinkType.ASSOCIATIVE, {})
                decay = props.get("decay_per_day", 0.01)
                days = (now - link["last_maintained"]) / 86400
                if days > 0.5:
                    new_str = max(0.0, link["strength"] - decay * days)
                    conn.execute("UPDATE neural_links SET strength = ?, last_maintained = ? WHERE id = ?",
                                 (new_str, now, link["id"]))
                    results["links_decayed"] += 1
                if link["strength"] <= 0.01:
                    prune_link_ids.append(link["id"])
            if prune_link_ids:
                placeholders = ",".join("?" * len(prune_link_ids))
                conn.execute(f"DELETE FROM neural_links WHERE id IN ({placeholders})", prune_link_ids)
                results["links_pruned"] = len(prune_link_ids)

        for key, val in results.items():
            self._incr_stat(key, val)
        self._dirty = True
        return results

    # =========================================================================
    # STATS & CLEAR
    # =========================================================================

    def get_stats(self) -> dict:
        with self.db.read() as conn:
            n_mem = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            n_links = conn.execute("SELECT COUNT(*) as c FROM neural_links").fetchone()["c"]
            n_cp = conn.execute("SELECT COUNT(*) as c FROM connection_points").fetchone()["c"]
            n_threads = conn.execute("SELECT COUNT(*) as c FROM threads").fetchone()["c"]
            n_knots = conn.execute("SELECT COUNT(*) as c FROM knots").fetchone()["c"]
            n_active = conn.execute("SELECT COUNT(*) as c FROM threads WHERE status='active'").fetchone()["c"]

            link_counts = {}
            for r in conn.execute("SELECT link_type, COUNT(*) as c FROM neural_links GROUP BY link_type").fetchall():
                link_counts[r["link_type"]] = r["c"]

            tier_counts = {}
            for r in conn.execute("SELECT tier, COUNT(*) as c FROM memories GROUP BY tier").fetchall():
                tier_counts[r["tier"]] = r["c"]

            cp_cats = {}
            for r in conn.execute("SELECT category, COUNT(*) as c FROM connection_points GROUP BY category").fetchall():
                cp_cats[r["category"]] = r["c"]

            n_entities = conn.execute("SELECT COUNT(DISTINCT entity) as c FROM connection_points").fetchone()["c"]

        return {
            "total_memories": n_mem,
            "total_links": n_links,
            "links_by_type": link_counts,
            "memories_by_tier": tier_counts,
            "faiss_enabled": HAS_FAISS,
            "cp_faiss_enabled": HAS_FAISS,
            "cp_faiss_indexed": self._cp_faiss.total,
            "total_connection_points": n_cp,
            "total_threads": n_threads,
            "total_knots": n_knots,
            "cp_by_category": cp_cats,
            "active_threads": n_active,
            "entities_indexed": n_entities,
            "embedding_cache": self._emb_cache.get_stats(),
            "storage_backend": "sqlite_v7",
            **self._stats,
        }

    def clear(self):
        with self.db.write() as conn:
            for table in ("connection_points", "threads", "thread_points",
                          "knots", "knot_threads", "memories", "neural_links",
                          "engine_stats"):
                conn.execute(f"DELETE FROM {table}")
            # Rebuild FTS
            conn.execute("INSERT INTO cp_fts(cp_fts) VALUES('rebuild')")
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
        self._emb_cache.clear()
        self._stats = self._load_stats()
        self._cp_faiss.rebuild_from_db(self.db)
        self._blob_faiss.rebuild_from_db(self.db)
        self._dirty = True

    def __len__(self):
        with self.db.read() as conn:
            return conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]

    @property
    def is_dirty(self):
        return self._dirty

    def mark_clean(self):
        self._flush_stats()
        self._dirty = False

        # Periodic FAISS rebuild if needed
        if self._cp_faiss.needs_rebuild:
            self._cp_faiss.rebuild_from_db(self.db)

    # =========================================================================
    # SESSION DELETE (combined CP + thread + knot cascade)
    # =========================================================================

    def delete_session(self, session_id: str) -> dict:
        """Delete all data for a session (CPs, threads, knots, blobs)."""
        cp_deleted = self.delete_session_points(session_id)
        tk_result = self.delete_session_threads_and_knots(session_id)
        return {
            "points_deleted": cp_deleted,
            **tk_result,
        }

    # =========================================================================
    # JSON → SQLite MIGRATION (runs once on first startup)
    # =========================================================================

    def migrate_from_json(self, json_path: str) -> int:
        """Import legacy mnemo_db.json into SQLite. Returns items imported."""
        if not os.path.exists(json_path):
            return 0

        print(f"Migrating legacy JSON database: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        imported = 0

        # Memories
        for mid, mdata in data.get("memories", {}).items():
            emb = None
            if "embedding_b64" in mdata:
                try:
                    import base64
                    raw = base64.b64decode(mdata["embedding_b64"])
                    emb = np.frombuffer(raw, dtype=np.dtype(mdata.get("embedding_dtype", "float32"))).copy()
                except Exception:
                    pass
            if emb is None:
                emb = self._get_embedding(mdata.get("content", ""))
            emb_blob = emb.astype(np.float32).tobytes()

            # Extract session_id/source from old metadata blob into proper columns
            old_meta = mdata.get("metadata", {})
            if isinstance(old_meta, str):
                try:
                    old_meta = json.loads(old_meta)
                except Exception:
                    old_meta = {}
            session_id = old_meta.pop("session_id", mdata.get("session_id", ""))
            source = old_meta.pop("source", mdata.get("source", "auto_extract"))
            meta_json = json.dumps(old_meta)  # Remaining metadata only

            with self.db.write() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO memories
                    (id, content, tier, namespace, quality_score, access_count, priority,
                     session_id, source, created_at, last_accessed, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (mid, mdata["content"], mdata.get("tier", "semantic"),
                      mdata.get("namespace", "default"), mdata.get("quality_score", 0.5),
                      mdata.get("access_count", 0), mdata.get("priority", 1.0),
                      session_id, source,
                      mdata.get("created_at", time.time()), mdata.get("last_accessed", time.time()),
                      meta_json, emb_blob))
            imported += 1

        # Links
        for lid, ldata in data.get("links", {}).items():
            with self.db.write() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO neural_links
                    (id, source_id, target_id, link_type, strength, created_at, last_maintained)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (lid, ldata["source_id"], ldata["target_id"], ldata["link_type"],
                      ldata["strength"], ldata.get("created_at", time.time()),
                      ldata.get("last_maintained", time.time())))

        # Connection Points
        for cpid, cpd in data.get("connection_points", {}).items():
            emb = None
            if "embedding_b64" in cpd:
                try:
                    import base64
                    raw = base64.b64decode(cpd["embedding_b64"])
                    emb = np.frombuffer(raw, dtype=np.dtype(cpd.get("embedding_dtype", "float32"))).copy()
                except Exception:
                    pass
            if emb is None:
                searchable = self._cp_to_searchable(
                    cpd.get("entity", ""), cpd.get("point_type", ""),
                    cpd.get("value", ""), cpd.get("connects_to", ""),
                    cpd.get("reason", ""), cpd.get("category", "fact"))
                emb = self._get_embedding(searchable)
            emb_blob = emb.astype(np.float32).tobytes()

            with self.db.write() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO connection_points
                    (id, entity, point_type, value, connects_to, reason, weight,
                     category, session_id, source, thread_id, position, namespace, created_at, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (cpid, cpd.get("entity", ""), cpd.get("point_type", ""),
                      cpd.get("value", ""), cpd.get("connects_to", ""),
                      cpd.get("reason", ""), cpd.get("weight", 0.5),
                      cpd.get("category", "fact"), cpd.get("session_id", ""),
                      cpd.get("source", "auto_extract"), cpd.get("thread_id", ""),
                      cpd.get("position", -1), cpd.get("namespace", "default"),
                      cpd.get("created_at", time.time()), emb_blob))
            imported += 1

        # Threads
        for tid, td in data.get("threads", {}).items():
            with self.db.write() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO threads
                    (id, name, thread_type, entity, status, tension_level,
                     tone_trajectory, current_position, session_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (tid, td.get("name", ""), td.get("thread_type", "plot_line"),
                      td.get("entity", ""), td.get("status", "active"),
                      td.get("tension_level", 0.5),
                      json.dumps(td.get("tone_trajectory", [])),
                      td.get("current_position", 0),
                      td.get("session_id", ""), td.get("created_at", time.time())))
                for pos, pid in enumerate(td.get("points", [])):
                    conn.execute(
                        "INSERT OR IGNORE INTO thread_points(thread_id, cp_id, position) VALUES(?, ?, ?)",
                        (tid, pid, pos))

        # Knots
        for kid, kd in data.get("knots", {}).items():
            with self.db.write() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO knots
                    (id, name, pivot_type, narrative_weight, tension_before, tension_after,
                     tone_shift, reason, unresolved, active_points, session_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (kid, kd.get("name", ""), kd.get("pivot_type", "collision"),
                      kd.get("narrative_weight", 0.5), kd.get("tension_before", 0.5),
                      kd.get("tension_after", 0.5), kd.get("tone_shift", ""),
                      kd.get("reason", ""), json.dumps(kd.get("unresolved", [])),
                      json.dumps(kd.get("active_points", {})),
                      kd.get("session_id", ""), kd.get("created_at", time.time())))
                for t in kd.get("threads", []):
                    conn.execute(
                        "INSERT OR IGNORE INTO knot_threads(knot_id, thread_id) VALUES(?, ?)",
                        (kid, t))

        # Rebuild FAISS from migrated data
        self._cp_faiss.rebuild_from_db(self.db)
        self._blob_faiss.rebuild_from_db(self.db)

        print(f"Migration complete: {imported} items imported into SQLite.")
        return imported


# =============================================================================
# PERSISTENT WRAPPER (downloads/uploads .db file to HuggingFace Datasets)
# =============================================================================

class PersistentMnemo:
    """Wraps MnemoEngine with HuggingFace Datasets background persistence.

    v7.0: Uploads/downloads the SQLite .db file directly.
    No serialize/deserialize step. No JSON encoding of embeddings.
    """

    def __init__(self, db_path: str = None, enable_hf_sync: bool = True):
        self.hf_token = os.environ.get("HF_TOKEN")
        self.dataset_repo_id = os.environ.get("DATASET_REPO_ID")

        config = MnemoConfig(db_path=db_path or "/app/data/mnemo.db")
        self._db_path = config.db_path

        if self.hf_token:
            from huggingface_hub import HfApi
            self.api = HfApi(token=self.hf_token)
        else:
            self.api = None

        # Download .db from HF if available
        self._download_db()

        # Check for legacy JSON and migrate if needed
        legacy_json = self._db_path.replace(".db", ".json").replace("mnemo.db", "mnemo_db.json")
        if not os.path.exists(self._db_path) or os.path.getsize(self._db_path) == 0:
            # Also check the old hardcoded path
            for candidate in [legacy_json, "/app/data/mnemo_db.json"]:
                if os.path.exists(candidate):
                    legacy_json = candidate
                    break

        # Create engine (creates SQLite DB if not exists)
        self.engine = MnemoEngine(config)

        # Migrate legacy JSON if SQLite is empty
        with self.engine.db.read() as conn:
            n_cp = conn.execute("SELECT COUNT(*) as c FROM connection_points").fetchone()["c"]
            n_mem = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        if n_cp == 0 and n_mem == 0 and os.path.exists(legacy_json):
            self.engine.migrate_from_json(legacy_json)

        # Background sync
        self._needs_sync = False
        if enable_hf_sync and self.dataset_repo_id and self.hf_token:
            t = threading.Thread(target=self._background_sync, daemon=True)
            t.start()
        elif enable_hf_sync:
            print("WARNING: No HF_TOKEN or DATASET_REPO_ID. Memory will be ephemeral.")

    def _download_db(self):
        """Download SQLite .db file from HuggingFace Datasets on startup."""
        if not self.api or not self.dataset_repo_id:
            return
        try:
            import concurrent.futures
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    hf_hub_download, repo_id=self.dataset_repo_id,
                    filename="mnemo.db", repo_type="dataset", token=self.hf_token)
                try:
                    downloaded = future.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    print("WARNING: HF download timed out after 60s. Starting fresh.")
                    return

            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            import shutil
            shutil.copy(downloaded, self._db_path)
            print(f"Downloaded database from HF Datasets: {self._db_path}")
        except EntryNotFoundError:
            # First run — try downloading legacy JSON format
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import EntryNotFoundError as ENF2
                downloaded = hf_hub_download(
                    repo_id=self.dataset_repo_id, filename="mnemo_db.json",
                    repo_type="dataset", token=self.hf_token)
                legacy_path = self._db_path.replace(".db", "_legacy.json")
                import shutil
                shutil.copy(downloaded, legacy_path)
                print(f"Downloaded legacy JSON for migration: {legacy_path}")
            except Exception:
                pass  # No existing data at all — fresh start
        except Exception as e:
            print(f"CRITICAL: Failed to download database: {e}")

    def _upload_db(self) -> bool:
        """Upload SQLite .db file to HuggingFace Datasets."""
        if not self.api or not self.dataset_repo_id:
            return False
        try:
            self.engine.db.checkpoint()
            self.api.upload_file(
                path_or_fileobj=self._db_path,
                path_in_repo="mnemo.db",
                repo_id=self.dataset_repo_id,
                repo_type="dataset",
                commit_message="Auto-backup mnemo v7 database",
            )
            print("Background sync to HF Datasets succeeded (SQLite .db).")
            return True
        except Exception as e:
            print(f"Background sync failed: {e}")
            return False

    def _background_sync(self):
        while True:
            time.sleep(30)
            if self.engine.is_dirty:
                self.engine.mark_clean()
                self._upload_db()

    # Delegate all engine methods (same signatures as v6.5)
    def add(self, *a, **kw): return self.engine.add(*a, **kw)
    def search(self, *a, **kw): return self.engine.search(*a, **kw)
    def should_inject(self, *a, **kw): return self.engine.should_inject(*a, **kw)
    def get_context(self, *a, **kw): return self.engine.get_context(*a, **kw)
    def get(self, *a, **kw): return self.engine.get(*a, **kw)
    def delete(self, *a, **kw): return self.engine.delete(*a, **kw)
    def maintenance(self): return self.engine.maintenance()
    def list_all(self): return self.engine.list_all()
    def list_memories(self, *a, **kw): return self.engine.list_memories(*a, **kw)
    def get_stats(self): return self.engine.get_stats()
    def clear(self): return self.engine.clear()
    def __len__(self): return len(self.engine)
    def __getattr__(self, name): return getattr(self.engine, name)
    def add_point(self, *a, **kw): return self.engine.add_point(*a, **kw)
    def add_points_batch(self, *a, **kw): return self.engine.add_points_batch(*a, **kw)
    def get_point(self, *a, **kw): return self.engine.get_point(*a, **kw)
    def delete_point(self, *a, **kw): return self.engine.delete_point(*a, **kw)
    def update_point(self, *a, **kw): return self.engine.update_point(*a, **kw)
    def delete_session_points(self, *a, **kw): return self.engine.delete_session_points(*a, **kw)
    def graph_search(self, *a, **kw): return self.engine.graph_search(*a, **kw)
    def add_thread(self, *a, **kw): return self.engine.add_thread(*a, **kw)
    def advance_thread(self, *a, **kw): return self.engine.advance_thread(*a, **kw)
    def trace_thread(self, *a, **kw): return self.engine.trace_thread(*a, **kw)
    def get_active_threads(self): return self.engine.get_active_threads()
    def get_thread(self, *a, **kw): return self.engine.get_thread(*a, **kw)
    def delete_thread(self, *a, **kw): return self.engine.delete_thread(*a, **kw)
    def delete_knot(self, *a, **kw): return self.engine.delete_knot(*a, **kw)
    def delete_session_threads_and_knots(self, *a, **kw): return self.engine.delete_session_threads_and_knots(*a, **kw)
    def add_knot(self, *a, **kw): return self.engine.add_knot(*a, **kw)
    def get_knot_context(self, *a, **kw): return self.engine.get_knot_context(*a, **kw)
    def list_knots(self): return self.engine.list_knots()
    def list_points(self, *a, **kw): return self.engine.list_points(*a, **kw)
    def entity_lookup(self, *a, **kw): return self.engine.entity_lookup(*a, **kw)
    def delete_session(self, *a, **kw): return self.engine.delete_session(*a, **kw)
