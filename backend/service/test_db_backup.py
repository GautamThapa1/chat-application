"""Database backup/restore for hosts without a persistent disk."""

import json
import os

import pytest

import backend.service.db as db
from backend.service import db_backup


class FakeStore:
    """In-memory stand-in for S3Store."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads = 0

    def get_bytes(self, key):
        return self.objects.get(key)

    def put_bytes(self, key, data):
        self.objects[key] = data

    def download_file(self, key, path):
        if key not in self.objects:
            return False
        with open(path, "wb") as file:
            file.write(self.objects[key])
        return True

    def upload_file(self, path, key):
        self.uploads += 1
        with open(path, "rb") as file:
            self.objects[key] = file.read()


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    path = tmp_path / "data" / "app.db"
    path.parent.mkdir()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    monkeypatch.setenv("DATABASE_KEY", "test-database-key")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("QDRANT_COLLECTION_PREFIX", f"test_{tmp_path.name.replace('-', '_')}")
    return path


def wipe_local(path):
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(f"{path}{suffix}"):
            os.unlink(f"{path}{suffix}")


def test_data_survives_a_wiped_disk(fresh_db):
    store = FakeStore()
    first = db_backup.DatabaseBackup(store)
    first.restore_and_claim()  # nothing to restore yet
    db.init_db()
    user_id = db.register_user("alice", "correct-horse")
    assert first.backup_now() is True
    assert store.objects["app.db.enc"][:16] != b"SQLite format 3\x00", "stays encrypted"

    wipe_local(fresh_db)  # the platform restarts with an empty filesystem

    second = db_backup.DatabaseBackup(store)
    second.restore_and_claim()
    db.init_db()
    assert db.authenticate("alice", "correct-horse") == user_id


def test_unchanged_database_is_not_uploaded_again(fresh_db):
    store = FakeStore()
    backup = db_backup.DatabaseBackup(store)
    backup.restore_and_claim()
    db.init_db()
    assert backup.backup_now() is True
    assert backup.backup_now() is False
    assert store.uploads == 1


def test_superseded_instance_never_overwrites_backup(fresh_db):
    store = FakeStore()
    old = db_backup.DatabaseBackup(store)
    old.restore_and_claim()
    db.init_db()
    db.register_user("alice", "correct-horse")

    # A new deploy starts and claims the backup.
    new = db_backup.DatabaseBackup(store)
    new.restore_and_claim()
    assert json.loads(store.objects["owner.json"])["instance"] != old._instance_id

    db.register_user("bob", "another-horse")
    assert old.backup_now() is False
    assert store.uploads == 0


def test_backup_with_wrong_key_refuses_to_start(fresh_db, monkeypatch):
    store = FakeStore()
    backup = db_backup.DatabaseBackup(store)
    backup.restore_and_claim()
    db.init_db()
    backup.backup_now()
    wipe_local(fresh_db)

    monkeypatch.setenv("DATABASE_KEY", "a-different-key")
    with pytest.raises(RuntimeError, match="DATABASE_KEY"):
        db_backup.DatabaseBackup(store).restore_and_claim()
    assert not fresh_db.exists(), "must not start with an empty database"


def test_existing_local_database_is_kept(fresh_db):
    store = FakeStore()
    store.objects["app.db.enc"] = b"not a database"
    db.init_db()
    db.register_user("alice", "correct-horse")
    db_backup.DatabaseBackup(store).restore_and_claim()
    assert db.authenticate("alice", "correct-horse")


def test_disabled_without_bucket(monkeypatch):
    monkeypatch.delenv("BACKUP_S3_BUCKET", raising=False)
    assert db_backup.from_env() is None


def test_bucket_without_credentials_fails_loudly(monkeypatch):
    monkeypatch.setenv("BACKUP_S3_BUCKET", "bucket")
    monkeypatch.delenv("BACKUP_S3_KEY_ID", raising=False)
    monkeypatch.delenv("BACKUP_S3_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="BACKUP_S3_KEY_ID"):
        db_backup.from_env()


def test_s3_store_talks_s3(tmp_path):
    """S3Store against a stubbed S3 client: 404 -> None, bytes round-trip."""
    from botocore.stub import Stubber

    store = db_backup.S3Store(
        bucket="b", endpoint="https://s3.us-west-004.backblazeb2.com",
        key_id="id", secret="secret", region="us-west-004",
    )
    with Stubber(store._client) as stub:
        stub.add_client_error("get_object", service_error_code="NoSuchKey",
                              http_status_code=404,
                              expected_params={"Bucket": "b", "Key": "owner.json"})
        assert store.get_bytes("owner.json") is None

        stub.add_response("put_object", {}, {"Bucket": "b", "Key": "owner.json", "Body": b"x"})
        store.put_bytes("owner.json", b"x")

        from botocore.response import StreamingBody
        import io
        stub.add_response("get_object", {"Body": StreamingBody(io.BytesIO(b"snap"), 4)},
                          {"Bucket": "b", "Key": "app.db.enc"})
        target = tmp_path / "restored"
        assert store.download_file("app.db.enc", str(target)) is True
        assert target.read_bytes() == b"snap"

        stub.add_client_error("get_object", service_error_code="AccessDenied",
                              http_status_code=403)
        with pytest.raises(Exception):
            store.get_bytes("app.db.enc")  # real errors must not look like "no backup"
