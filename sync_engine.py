"""
Sync Engine — SQLite ↔ Cloudflare R2 / HuggingFace Dataset Sync (v7.1)

PRIMARY:  Cloudflare R2 (S3-compatible, zero egress, 10GB free)
FALLBACK: HuggingFace Datasets (original v7.0 behavior)

R2 advantages over HF Datasets:
  - Faster uploads/downloads (S3 API, global CDN vs Git LFS)
  - No HF rate limits or intermittent 503s
  - Zero egress fees (data out is free)
  - 10GB free storage (our .db is ~10-50MB)
  - S3 API = industry standard, battle-tested boto3 client

Architecture:

  Streamlit App (local SQLite)  ←──sync──→  R2 Bucket (backup .db)
                                            HF Space (reads same .db from R2)

Lifecycle:
  1. On startup: download mnemo.db from R2 → local path
  2. If no .db exists, check for legacy mnemo_db.json (HF) and flag migration
  3. Background thread: every SYNC_INTERVAL seconds, if dirty:
     a. WAL checkpoint (flush pending writes to main .db)
     b. Upload .db to R2
  4. On demand: force_sync() for immediate upload

Conflict resolution: last-write-wins (single user, safe for Tina's setup).

Required Streamlit secrets (for R2 primary):
  R2_ACCOUNT_ID       = "your-cloudflare-account-id"
  R2_ACCESS_KEY_ID    = "your-r2-access-key"
  R2_SECRET_ACCESS_KEY = "your-r2-secret-key"
  R2_BUCKET_NAME      = "mnemo"        # optional, defaults to "mnemo"

Fallback HF secrets (existing):
  HF_TOKEN            = "hf_..."
  DATASET_REPO_ID     = "AthelaPerk/Private"

Thread-safe. Non-blocking. All sync operations run in a daemon thread.
"""

import os
import time
import shutil
import sqlite3
import logging
import threading
from typing import Optional
from pathlib import Path

log = logging.getLogger("mnemo.sync")

# Sync interval in seconds
SYNC_INTERVAL = 30

# R2 config defaults
DEFAULT_R2_BUCKET = "mnemo"
DB_KEY_IN_R2 = "mnemo.db"

# HF Dataset config (fallback)
DEFAULT_DATASET_REPO = "AthelaPerk/Private"
DB_FILENAME_IN_REPO = "mnemo.db"
LEGACY_JSON_FILENAME = "mnemo_db.json"


# =============================================================================
# R2 CLIENT (lazy, lightweight S3-compatible)
# =============================================================================

class R2Client:
    """Minimal S3-compatible client for Cloudflare R2.

    Uses boto3 under the hood — already in most Python environments.
    Falls back gracefully if credentials are missing.
    """

    def __init__(self, account_id: str = None, access_key_id: str = None,
                 secret_access_key: str = None, bucket_name: str = None):
        self.account_id = account_id or os.environ.get("R2_ACCOUNT_ID", "")
        self.access_key_id = access_key_id or os.environ.get("R2_ACCESS_KEY_ID", "")
        self.secret_access_key = secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY", "")
        self.bucket_name = bucket_name or os.environ.get("R2_BUCKET_NAME", DEFAULT_R2_BUCKET)
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.account_id and self.access_key_id and self.secret_access_key)

    @property
    def client(self):
        """Lazy-init boto3 S3 client pointing at R2 endpoint."""
        if self._client is None and self.available:
            import boto3
            self._client = boto3.client(
                "s3",
                endpoint_url=f"https://{self.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name="auto",
            )
        return self._client

    def download(self, key: str, local_path: str) -> bool:
        """Download object from R2 to local file."""
        if not self.available:
            return False
        try:
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            self.client.download_file(self.bucket_name, key, local_path)
            return True
        except Exception as e:
            error_code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False  # Object doesn't exist yet — normal on first run
            log.warning(f"R2 download failed ({key}): {type(e).__name__}: {e}")
            return False

    def upload(self, local_path: str, key: str) -> bool:
        """Upload local file to R2."""
        if not self.available:
            return False
        try:
            self.client.upload_file(local_path, self.bucket_name, key)
            return True
        except Exception as e:
            log.error(f"R2 upload failed ({key}): {type(e).__name__}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if object exists in R2."""
        if not self.available:
            return False
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception:
            return False


# =============================================================================
# SYNC ENGINE (R2 primary, HF fallback)
# =============================================================================

class SyncEngine:
    """Bidirectional sync between local SQLite and cloud storage.

    Priority: R2 (if credentials set) → HF Datasets (fallback) → disabled.

    Usage:
        sync = SyncEngine(
            db_path="/home/user/.mnemo/mnemo.db",
            hf_token="hf_...",
            dataset_repo_id="AthelaPerk/Private",
        )
        sync.download()          # On startup — get latest .db
        sync.start_background()  # Start daemon sync thread

        # ... app runs, writes to SQLite ...

        sync.mark_dirty()        # After writes, signal upload needed
        sync.force_sync()        # Immediate upload (e.g., before shutdown)
        sync.stop()              # Clean shutdown
    """

    def __init__(self, db_path: str, hf_token: str = None,
                 dataset_repo_id: str = None,
                 sync_interval: int = SYNC_INTERVAL):
        self.db_path = db_path
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.dataset_repo_id = dataset_repo_id or os.environ.get(
            "DATASET_REPO_ID", DEFAULT_DATASET_REPO)
        self.sync_interval = sync_interval

        self._dirty = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_sync: float = 0
        self._sync_count: int = 0
        self._sync_errors: int = 0

        # R2 client (primary)
        self._r2 = R2Client()

        # HF API (fallback, lazy init)
        self._hf_api_instance = None

        # Determine backend
        if self._r2.available:
            self._backend = "r2"
            log.info("[SYNC] ✅ Using Cloudflare R2 (primary)")
            print("[SYNC] ✅ Using Cloudflare R2")
        elif self.hf_token:
            self._backend = "hf"
            log.info("[SYNC] Using HuggingFace Datasets (fallback)")
            print("[SYNC] Using HuggingFace Datasets (fallback)")
        else:
            self._backend = "none"
            log.warning("[SYNC] No sync credentials — running offline")
            print("[SYNC] ⚠️  No sync credentials — running offline")

    @property
    def backend(self) -> str:
        """Current sync backend: 'r2', 'hf', or 'none'."""
        return self._backend

    @property
    def _hf_api(self):
        """Lazy-init HfApi for fallback."""
        if self._hf_api_instance is None and self.hf_token:
            from huggingface_hub import HfApi
            self._hf_api_instance = HfApi(token=self.hf_token)
        return self._hf_api_instance

    @property
    def has_credentials(self) -> bool:
        return self._backend != "none"

    # =========================================================================
    # DOWNLOAD (startup)
    # =========================================================================

    def download(self) -> bool:
        """Download .db from cloud storage. Returns True if downloaded.

        Tries R2 first, falls back to HF Datasets.
        If neither has a .db, checks HF for legacy JSON for migration.
        """
        if not self.has_credentials:
            log.warning("No sync credentials — skipping download.")
            return False

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        # === R2 PRIMARY ===
        if self._backend == "r2":
            if self._r2.download(DB_KEY_IN_R2, self.db_path):
                size_mb = os.path.getsize(self.db_path) / 1_048_576
                log.info(f"Downloaded {DB_KEY_IN_R2} from R2 ({size_mb:.1f} MB)")
                print(f"[SYNC] Downloaded mnemo.db from R2 ({size_mb:.1f} MB)")
                return True
            log.info("No .db in R2 — checking HF for legacy data...")
            # Fall through to HF for legacy migration check

        # === HF DATASETS (primary if no R2, or fallback for legacy) ===
        if self.hf_token:
            # Try .db from HF
            if self._backend == "hf":
                if self._download_hf_file(DB_FILENAME_IN_REPO, self.db_path):
                    log.info(f"Downloaded {DB_FILENAME_IN_REPO} from HF Datasets.")
                    return True

            # Try legacy JSON for migration
            legacy_path = self._legacy_json_path()
            if self._download_hf_file(LEGACY_JSON_FILENAME, legacy_path):
                log.info(f"Downloaded legacy {LEGACY_JSON_FILENAME} for migration.")
                return False  # Signal caller to run migration

        log.info("No existing data found — starting fresh.")
        return False

    def _download_hf_file(self, filename: str, local_path: str) -> bool:
        """Download a single file from HF Datasets repo."""
        try:
            import concurrent.futures
            from huggingface_hub import hf_hub_download

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    hf_hub_download,
                    repo_id=self.dataset_repo_id,
                    filename=filename,
                    repo_type="dataset",
                    token=self.hf_token,
                    force_download=True,
                )
                try:
                    downloaded = future.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    log.warning(f"HF download timed out: {filename}")
                    return False

            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            shutil.copy2(downloaded, local_path)
            return True

        except Exception as e:
            if "EntryNotFoundError" in type(e).__name__ or "404" in str(e):
                return False
            log.warning(f"HF download failed ({filename}): {type(e).__name__}: {e}")
            return False

    def _legacy_json_path(self) -> str:
        return os.path.join(
            os.path.dirname(self.db_path) or ".",
            "mnemo_legacy.json"
        )

    def get_legacy_json_path(self) -> Optional[str]:
        path = self._legacy_json_path()
        return path if os.path.exists(path) else None

    # =========================================================================
    # UPLOAD (background sync)
    # =========================================================================

    def upload(self) -> bool:
        """Upload local .db to cloud storage.

        Performs WAL checkpoint first to ensure .db is self-contained.
        Uses R2 if available, falls back to HF Datasets.
        """
        if not self.has_credentials:
            return False

        if not os.path.exists(self.db_path):
            log.warning("No local .db file to upload.")
            return False

        try:
            # WAL checkpoint — merge journal into main .db
            self._wal_checkpoint()

            success = False

            # === R2 PRIMARY ===
            if self._backend == "r2":
                success = self._r2.upload(self.db_path, DB_KEY_IN_R2)
                if success:
                    size_mb = os.path.getsize(self.db_path) / 1_048_576
                    log.info(f"Uploaded to R2 ({size_mb:.1f} MB)")

            # === HF FALLBACK ===
            elif self._backend == "hf":
                self._hf_api.upload_file(
                    path_or_fileobj=self.db_path,
                    path_in_repo=DB_FILENAME_IN_REPO,
                    repo_id=self.dataset_repo_id,
                    repo_type="dataset",
                    commit_message="Auto-backup mnemo v7 database",
                )
                success = True

            if success:
                with self._lock:
                    self._dirty = False
                    self._last_sync = time.time()
                    self._sync_count += 1
                log.info(f"Sync #{self._sync_count} complete ({self._backend}).")
                return True
            else:
                raise RuntimeError("Upload returned False")

        except Exception as e:
            with self._lock:
                self._sync_errors += 1
            log.error(f"Upload failed ({self._backend}): {type(e).__name__}: {e}")
            return False

    def _wal_checkpoint(self):
        """Flush WAL journal into the main .db file."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except Exception as e:
            log.warning(f"WAL checkpoint failed: {e}")

    # =========================================================================
    # DIRTY TRACKING
    # =========================================================================

    def mark_dirty(self):
        """Signal that local .db has changed and needs uploading."""
        with self._lock:
            self._dirty = True

    @property
    def is_dirty(self) -> bool:
        with self._lock:
            return self._dirty

    # =========================================================================
    # BACKGROUND SYNC THREAD
    # =========================================================================

    def start_background(self):
        """Start background sync daemon thread."""
        if not self.has_credentials:
            log.warning("No credentials — background sync disabled.")
            return

        if self._worker and self._worker.is_alive():
            return

        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="mnemo-sync",
        )
        self._worker.start()
        log.info(f"Background sync started (every {self.sync_interval}s, backend={self._backend}).")

    def stop(self):
        """Stop background sync. Does a final upload if dirty."""
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5)

        if self.is_dirty:
            self.upload()

    def force_sync(self):
        """Force an immediate sync (blocking)."""
        if self.is_dirty:
            self.upload()

    def _sync_loop(self):
        """Background sync loop. Runs in daemon thread."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.sync_interval)
            if self._stop_event.is_set():
                break
            if self.is_dirty:
                self.upload()

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "backend": self._backend,
                "db_path": self.db_path,
                "r2_bucket": self._r2.bucket_name if self._r2.available else None,
                "dataset_repo": self.dataset_repo_id if self._backend == "hf" else None,
                "has_credentials": self.has_credentials,
                "is_dirty": self._dirty,
                "last_sync": self._last_sync,
                "last_sync_ago": round(time.time() - self._last_sync, 1) if self._last_sync else None,
                "sync_count": self._sync_count,
                "sync_errors": self._sync_errors,
                "sync_interval": self.sync_interval,
                "background_running": self._worker.is_alive() if self._worker else False,
                "db_exists": os.path.exists(self.db_path),
                "db_size_mb": round(os.path.getsize(self.db_path) / 1_048_576, 2)
                              if os.path.exists(self.db_path) else 0,
            }
