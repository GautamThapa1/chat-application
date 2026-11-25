from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from pywebpush import WebPushException

from backend.service import task_reminder_worker


def _job(due_at_utc: int) -> dict:
    return {
        "task_id": 7,
        "user_id": 3,
        "title": "Submit the project",
        "due_at_utc": due_at_utc,
        "timezone": "America/Los_Angeles",
    }


def test_worker_sends_task_specific_reminder_with_deadline_ttl(monkeypatch):
    worker = task_reminder_worker.TaskReminderWorker()
    worker._lease_owner = "lease"
    deadline = int(
        (datetime.now(timezone.utc) + timedelta(minutes=20)).timestamp()
    )
    sent = {}
    marked = []
    monkeypatch.setattr(
        task_reminder_worker.db,
        "get_push_subscriptions",
        lambda user_id: [
            {
                "endpoint": "https://fcm.googleapis.com/push/token",
                "p256dh": "key",
                "auth": "auth",
            }
        ],
    )
    monkeypatch.setattr(
        task_reminder_worker.db, "task_reminder_is_sendable", lambda *_: True
    )

    def fake_webpush(**kwargs):
        sent.update(kwargs)

    monkeypatch.setattr(task_reminder_worker, "webpush", fake_webpush)
    monkeypatch.setattr(
        task_reminder_worker.db,
        "mark_task_reminder_sent",
        lambda task_id, lease_owner: marked.append((task_id, lease_owner)) or True,
    )

    worker._deliver(_job(deadline), "private-key", "mailto:test@example.com")

    assert 0 < sent["ttl"] <= 20 * 60
    assert sent["subscription_info"]["endpoint"].startswith("https://")
    assert "Submit the project" in sent["data"]
    assert '"taskId": 7' in sent["data"]
    assert marked == [(7, "lease")]


def test_worker_removes_expired_subscription_and_retries(monkeypatch):
    worker = task_reminder_worker.TaskReminderWorker()
    worker._lease_owner = "lease"
    deadline = int(
        (datetime.now(timezone.utc) + timedelta(minutes=20)).timestamp()
    )
    endpoint = "https://fcm.googleapis.com/push/token"
    removed = []
    failures = []
    monkeypatch.setattr(
        task_reminder_worker.db,
        "get_push_subscriptions",
        lambda user_id: [
            {"endpoint": endpoint, "p256dh": "key", "auth": "auth"}
        ],
    )
    monkeypatch.setattr(
        task_reminder_worker.db, "task_reminder_is_sendable", lambda *_: True
    )

    def fail_webpush(**kwargs):
        raise WebPushException(
            "subscription expired",
            response=SimpleNamespace(status_code=410),
        )

    monkeypatch.setattr(task_reminder_worker, "webpush", fail_webpush)
    monkeypatch.setattr(
        task_reminder_worker.db,
        "delete_push_subscription_by_endpoint",
        lambda user_id, value: removed.append((user_id, value)) or True,
    )
    monkeypatch.setattr(
        task_reminder_worker.db,
        "record_task_reminder_failure",
        lambda task_id, lease_owner, error: failures.append(
            (task_id, lease_owner, error)
        ),
    )

    worker._deliver(_job(deadline), "private-key", "mailto:test@example.com")

    assert removed == [(3, endpoint)]
    assert failures
    assert failures[0][:2] == (7, "lease")
