"""Deliver due task reminders to registered Web Push subscriptions."""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from pywebpush import WebPushException, webpush

from backend.service import db

logger = logging.getLogger(__name__)


class TaskReminderWorker:
    def __init__(self, *, batch_size: int = 20, poll_interval: float = 5):
        self._batch_size = batch_size
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lease_owner = uuid.uuid4().hex

    def start(self) -> None:
        if self._thread is not None:
            return
        private_key = os.getenv("VAPID_PRIVATE_KEY")
        subject = os.getenv("VAPID_SUBJECT")
        if not private_key or not subject:
            logger.warning(
                "Task reminder worker is disabled; set VAPID_PRIVATE_KEY "
                "and VAPID_SUBJECT."
            )
            return
        self._thread = threading.Thread(
            target=self._run,
            args=(private_key, subject),
            name="task-reminder-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self, private_key: str, subject: str) -> None:
        while not self._stop_event.is_set():
            jobs = []
            try:
                jobs = db.claim_due_task_reminders(
                    self._batch_size,
                    lease_owner=self._lease_owner,
                )
                for job in jobs:
                    if self._stop_event.is_set():
                        break
                    self._deliver(job, private_key, subject)
            except Exception:
                logger.exception("Could not process task reminders.")
            if not jobs:
                self._stop_event.wait(self._poll_interval)

    def _deliver(self, job: dict, private_key: str, subject: str) -> None:
        now = int(datetime.now(timezone.utc).timestamp())
        deadline = job["due_at_utc"]
        if deadline <= now:
            db.record_task_reminder_failure(
                job["task_id"],
                self._lease_owner,
                "Task deadline has passed.",
            )
            return
        if not db.task_reminder_is_sendable(
            job["task_id"], self._lease_owner
        ):
            return

        zone = ZoneInfo(job["timezone"] or "UTC")
        due_local = datetime.fromtimestamp(deadline, timezone.utc).astimezone(zone)
        due_time = due_local.strftime("%I:%M %p").lstrip("0")
        payload = json.dumps(
            {
                "title": "Task reminder",
                "body": f"Complete {job['title'][:500]} before {due_time}.",
                "taskId": job["task_id"],
            },
            ensure_ascii=False,
        )
        subscriptions = db.get_push_subscriptions(job["user_id"])
        if not subscriptions:
            db.record_task_reminder_failure(
                job["task_id"],
                self._lease_owner,
                "No active push subscriptions.",
            )
            return

        delivered = False
        errors: list[str] = []
        for subscription in subscriptions:
            remaining = deadline - int(datetime.now(timezone.utc).timestamp())
            if remaining <= 0:
                break
            try:
                webpush(
                    subscription_info={
                        "endpoint": subscription["endpoint"],
                        "keys": {
                            "p256dh": subscription["p256dh"],
                            "auth": subscription["auth"],
                        },
                    },
                    data=payload,
                    vapid_private_key=private_key,
                    vapid_claims={"sub": subject},
                    ttl=remaining,
                    timeout=10,
                )
                delivered = True
            except WebPushException as exc:
                status_code = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )
                if status_code in (404, 410):
                    db.delete_push_subscription_by_endpoint(
                        job["user_id"], subscription["endpoint"]
                    )
                else:
                    errors.append(str(exc))
                    logger.warning(
                        "Web Push failed for task %s (status %s).",
                        job["task_id"],
                        status_code,
                        exc_info=True,
                    )
            except Exception as exc:
                errors.append(str(exc))
                logger.warning(
                    "Web Push failed for task %s.",
                    job["task_id"],
                    exc_info=True,
                )

        if delivered:
            db.mark_task_reminder_sent(job["task_id"], self._lease_owner)
        else:
            db.record_task_reminder_failure(
                job["task_id"],
                self._lease_owner,
                "; ".join(errors) or "No push subscription accepted the reminder.",
            )
