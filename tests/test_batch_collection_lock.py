"""Tests for app._collection_in_progress — the lock that stops a second
"Check status" click from restarting an in-flight pause_turn continuation
from scratch (see the redundant-continuation bug: repeated clicks caused the
same paused batch item to be re-digitized several times over)."""
from datetime import datetime, timedelta, timezone

import app


def test_no_lock_when_field_absent():
    assert app._collection_in_progress({}) is False


def test_locked_when_recently_started():
    payload = {"collection_started_at": datetime.now(timezone.utc).isoformat()}
    assert app._collection_in_progress(payload) is True


def test_lock_self_heals_after_staleness_window():
    stale = datetime.now(timezone.utc) - app._COLLECTION_LOCK_STALE_AFTER - timedelta(minutes=1)
    payload = {"collection_started_at": stale.isoformat()}
    assert app._collection_in_progress(payload) is False
