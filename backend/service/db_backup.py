"""Keep the SQLite database on hosts that have no persistent disk.

Free hosting (e.g. Render Free) gives the container an ephemeral filesystem:
every deploy, restart or spin-down starts from an empty disk. This module keeps
the database in S3-compatible object storage (e.g. Backblaze B2's free tier):

1. Startup: if there is no local database file, download the latest snapshot.
   Then write an "owner" marker naming this process.
2. While running: every BACKUP_INTERVAL_SECONDS, if the database files
   changed, take an online, consistent snapshot with SQLite's backup API and
   upload it. The snapshot stays SQLCipher-encrypted with DATABASE_KEY, so the
   storage provider never sees plaintext.
3. Shutdown (SIGTERM): one final snapshot.

Only the newest process uploads: before each upload the owner marker is
checked, so an old instance that is still shutting down during a deploy can
never overwrite the new instance's data.

Disabled unless BACKUP_S3_BUCKET is set (local and Docker setups keep using a
plain file or volume).
"""

import json
import logging
import os
import tempfile
import threading
import time
import uuid

import sqlcipher3

from backend.service import db

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 60


class S3Store:
    """Tiny wrapper over the S3 API (B2, R2, MinIO, AWS ... all speak it)."""

    def __init__(self, *, bucket: str, endpoint: str | None, key_id: str,
                 secret: str, region: str | None):
        import boto3
        from botocore.config import Config

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name=region or None,
            config=Config(
                # B2 and other non-AWS stores reject the newer default checksums.
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                retries={"max_attempts": 5, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            ),
        )

    def get_bytes(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            return self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def download_file(self, key: str, path: str) -> bool:
        data = self.get_bytes(key)
        if data is None:
            return False
        with open(path, "wb") as file:
            file.write(data)
        return True

    def upload_file(self, path: str, key: str) -> None:
        self._client.upload_file(path, self._bucket, key)


class DatabaseBackup:
    def __init__(self, store, *, prefix: str = "",
                 interval: float = DEFAULT_INTERVAL_SECONDS):
        self._store = store
        self._db_key = f"{prefix}app.db.enc"
        self._owner_key = f"{prefix}owner.json"
        self._interval = interval
        self._instance_id = uuid.uuid4().hex
        self._lock = threading.Lock()  # one snapshot at a time
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_signature = None
        self._superseded = False

    # ---------- startup ----------
    def restore_and_claim(self) -> None:
        """Download the snapshot if there is no local DB, then claim ownership.

        Raises if a snapshot exists but cannot be restored: starting with an
        empty database would later overwrite the good backup.
        """
        path = db.DB_PATH
        if os.path.exists(path):
            logger.info("Local database %s exists; not restoring a backup.", path)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            temporary = f"{path}.restore"
            try:
                if self._store.download_file(self._db_key, temporary):
                    _check_opens(temporary)
                    os.replace(temporary, path)
                    logger.info("Restored database from backup (%d bytes).",
                                os.path.getsize(path))
                else:
                    logger.info("No database backup found; starting fresh.")
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        self._store.put_bytes(self._owner_key, json.dumps({
            "instance": self._instance_id, "claimed_at": time.time(),
        }).encode())
        # A restored file counts as already backed up.
        self._last_signature = _signature(path)

    # ---------- running ----------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="db-backup-worker", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker and take a final snapshot."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        try:
            self.backup_now()
        except Exception:
            logger.exception("Final database backup failed.")

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                self.backup_now()
            except Exception:
                logger.exception("Database backup failed; will retry.")

    def backup_now(self) -> bool:
        """Upload a snapshot if the database changed. Returns True if uploaded."""
        with self._lock:
            if self._superseded:
                return False
            signature = _signature(db.DB_PATH)
            if signature is None or signature == self._last_signature:
                return False
            if not self._still_owner():
                self._superseded = True
                logger.warning(
                    "A newer instance owns the database backup; this instance "
                    "stops uploading."
                )
                return False

            descriptor, snapshot = tempfile.mkstemp(
                prefix=".snapshot.", suffix=".db",
                dir=os.path.dirname(os.path.abspath(db.DB_PATH)),
            )
            os.close(descriptor)
            try:
                _snapshot(snapshot)
                self._store.upload_file(snapshot, self._db_key)
            finally:
                os.unlink(snapshot)
            self._last_signature = signature
            logger.info("Database backed up.")
            return True

    def _still_owner(self) -> bool:
        raw = self._store.get_bytes(self._owner_key)
        if raw is None:
            return True
        try:
            return json.loads(raw).get("instance") == self._instance_id
        except ValueError:
            return True


def _signature(path: str):
    """Cheap change detector: size + mtime of the DB and its WAL file."""
    parts = []
    for name in (path, f"{path}-wal"):
        try:
            stat = os.stat(name)
        except FileNotFoundError:
            continue
        parts.append((name, stat.st_size, stat.st_mtime_ns))
    return tuple(parts) or None


def _snapshot(target: str) -> None:
    """Consistent copy of the live database, encrypted with the same key."""
    source = db.get_connection()
    try:
        destination = sqlcipher3.connect(target)
        try:
            db._apply_database_key(destination, db._database_key())
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()


def _check_opens(path: str) -> None:
    conn = sqlcipher3.connect(path)
    try:
        db._apply_database_key(conn, db._database_key())
    except sqlcipher3.DatabaseError as exc:
        raise RuntimeError(
            "The database backup cannot be opened with DATABASE_KEY."
        ) from exc
    finally:
        conn.close()


def from_env() -> DatabaseBackup | None:
    bucket = os.getenv("BACKUP_S3_BUCKET")
    if not bucket:
        return None
    key_id = os.getenv("BACKUP_S3_KEY_ID")
    secret = os.getenv("BACKUP_S3_SECRET")
    if not key_id or not secret:
        raise RuntimeError(
            "BACKUP_S3_BUCKET is set; also set BACKUP_S3_KEY_ID and BACKUP_S3_SECRET."
        )
    store = S3Store(
        bucket=bucket,
        endpoint=os.getenv("BACKUP_S3_ENDPOINT"),
        key_id=key_id,
        secret=secret,
        region=os.getenv("BACKUP_S3_REGION"),
    )
    return DatabaseBackup(
        store,
        prefix=os.getenv("BACKUP_S3_PREFIX", ""),
        interval=float(os.getenv("BACKUP_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS)),
    )
