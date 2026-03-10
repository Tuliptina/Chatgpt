"""
Sync Engine — SQLite ↔ HuggingFace Dataset Bidirectional Sync (v7.0)

Handles the persistence layer for Option D (embedded local engine):

  Streamlit App (local SQLite)  ←──sync──→  HF Datasets (backup .db)
                                            HF Space (reads same .db)

Lifecycle:
  1. On startup: download mnemo.db from HF Datasets → local path
  2. If no .db exists, check for legacy mnemo_db.json and flag for migration
  3. Background thread: every SYNC_INTERVAL seconds, if dirty:
     a. WAL checkpoint (flush pending writes to main .db)
     b. Upload .db to HF Datasets
  4. On demand: force_sync() for immediate upload

Conflict resolution: last-write-wins (single user, safe for Tina's setup).
The HF Space also reads the same .db file independently — it downloads on
its own startup, so both sides eventually converge.

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

# HF Dataset config
DEFAULT_DATASET_REPO = "AthelaPerk/Private"
DB_FILENAME_IN_REPO = "mnemo.db"
LEGACY_JSON_FILENAME = "mnemo_db.json"


class SyncEngine:
    """Bidirectional sync between local SQLite and HuggingFace Datasets.

    Usage:
        sync = SyncEngine(
            db_path="/home/user/.mnemo/mnemo.db",
            hf_token="hf_...",
            dataset_repo_id="AthelaPerk/Private",
        )
        sync.download()          # On startup — get latest .db from HF
        sync.start_background()  # Start daemon sync thread

        # ... app runs, writes to SQLite ...

        sync.mark_dirty()        # After writes, signal that upload is needed
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

        # HF API (lazy init)
        self._api = None

    @property
    def _hf_api(self):
        """Lazy-init HfApi to avoid import cost if sync is disabled."""
        if self._api is None and self.hf_token:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.hf_token)
        return self._api

    @property
    def has_credentials(self) -> bool:
        return bool(self.hf_token and self.dataset_repo_id)

    # =========================================================================
    # DOWNLOAD (startup)
    # =========================================================================

    def download(self) -> bool:
        """Download .db from HF Datasets. Returns True if downloaded.

        Falls back to downloading legacy JSON if .db doesn't exist yet.
        Caller should check for legacy JSON and run migration if needed.
        """
        if not self.has_credentials:
            log.warning("No HF credentials — skipping download.")
            return False

        # Ensure local directory exists
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        # Try downloading SQLite .db
        if self._download_file(DB_FILENAME_IN_REPO, self.db_path):
            log.info(f"Downloaded {DB_FILENAME_IN_REPO} from HF Datasets.")
            return True

        # No .db found — try legacy JSON for migration
        legacy_path = self._legacy_json_path()
        if self._download_file(LEGACY_JSON_FILENAME, legacy_path):
            log.info(f"Downloaded legacy {LEGACY_JSON_FILENAME} for migration.")
            return False  # Signal caller to run migration

        log.info("No existing data in HF Datasets — starting fresh.")
        return False

    def _download_file(self, filename: str, local_path: str) -> bool:
        """Download a single file from HF Datasets repo."""
        try:
            import concurrent.futures
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError

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
                    log.warning(f"Download timed out after 60s: {filename}")
                    return False

            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            shutil.copy2(downloaded, local_path)
            return True

        except Exception as e:
            # EntryNotFoundError is expected on first run
            if "EntryNotFoundError" in type(e).__name__ or "404" in str(e):
                return False
            log.warning(f"Download failed ({filename}): {type(e).__name__}: {e}")
            return False

    def _legacy_json_path(self) -> str:
        """Path for downloaded legacy JSON (next to the .db file)."""
        return os.path.join(
            os.path.dirname(self.db_path) or ".",
            "mnemo_legacy.json"
        )

    def get_legacy_json_path(self) -> Optional[str]:
        """Return path to legacy JSON if it exists, else None."""
        path = self._legacy_json_path()
        return path if os.path.exists(path) else None

    # =========================================================================
    # UPLOAD (background sync)
    # =========================================================================

    def upload(self) -> bool:
        """Upload local .db to HF Datasets (immediate).

        Performs WAL checkpoint first to ensure the .db file is self-contained.
        """
        if not self.has_credentials:
            return False

        if not os.path.exists(self.db_path):
            log.warning("No local .db file to upload.")
            return False

        try:
            # WAL checkpoint — merge WAL journal into main .db
            self._wal_checkpoint()

            self._hf_api.upload_file(
                path_or_fileobj=self.db_path,
                path_in_repo=DB_FILENAME_IN_REPO,
                repo_id=self.dataset_repo_id,
                repo_type="dataset",
                commit_message="Auto-backup mnemo v7 database",
            )

            with self._lock:
                self._dirty = False
                self._last_sync = time.time()
                self._sync_count += 1

            log.info(f"Synced to HF Datasets (sync #{self._sync_count}).")
            return True

        except Exception as e:
            with self._lock:
                self._sync_errors += 1
            log.error(f"Upload failed: {type(e).__name__}: {e}")
            return False

    def _wal_checkpoint(self):
        """Flush WAL journal into the main .db file.

        Required before uploading — WAL journal is a separate file that
        won't be included in the upload otherwise.
        """
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
            log.warning("No HF credentials — background sync disabled.")
            return

        if self._worker and self._worker.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="mnemo-sync",
        )
        self._worker.start()
        log.info(f"Background sync started (every {self.sync_interval}s).")

    def stop(self):
        """Stop background sync. Does a final upload if dirty."""
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=5)

        # Final sync
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
                "db_path": self.db_path,
                "dataset_repo": self.dataset_repo_id,
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
