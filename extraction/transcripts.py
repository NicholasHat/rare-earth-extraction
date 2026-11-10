"""Every paid model turn, kept on disk under data/transcripts/.

A run's transcript is the chain of API responses it was billed for: the batch
turn (or first synchronous turn) plus each pause_turn continuation, written
after every completed turn. Two jobs:

  - **Nothing paid is lost.** The raw output of a run that later fails to
    parse, or fails QA so badly it is rejected, is still on disk for
    inspection — the Batches API keeps results only 29 days, and a
    synchronous run's output never lived anywhere else.
  - **A continuation resumes, it doesn't restart.** A paused batch item is
    finished off with synchronous continuation turns that can die on a laptop
    sleep, a dropped connection or an exhausted credit balance. Each completed
    turn is saved before the next is sent, so the next collection replays the
    saved chain and sends only the turns still missing — instead of re-paying
    every turn from the batch result onwards (~150k cache-write tokens per
    death, three times, before this existed).

One gzipped JSON file per (run, paper): `<run_id>/<custom_id>.json.gz`, where
`run_id` is the batch id or `sync_<utc time>`. Messages are stored with
`model_dump(mode="json", exclude_unset=True)` and rebuilt with
`BetaMessage.construct`, which round-trips to the identical request body
the SDK would have sent for the original objects — required, since a
continuation's transcript must be append-only (a changed thinking block is a
400 on Sonnet 5.5). Writes are atomic (temp file + os.replace).
"""
from __future__ import annotations

import gzip
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from anthropic.types.beta import BetaMessage

import config


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _dump(message) -> dict:
    return message.model_dump(mode="json", exclude_unset=True)


class TranscriptStore:
    """The saved turns of one run (a batch, or one synchronous extraction),
    keyed by custom_id (the paper's sha256). Implements the chain-store
    protocol anthropic_client's collection and continuation code take:
    load / save / save_error."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dir = config.TRANSCRIPTS_DIR / run_id

    @classmethod
    def for_sync_run(cls) -> "TranscriptStore":
        return cls("sync_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))

    def path(self, custom_id: str) -> Path:
        return self.dir / f"{custom_id}.json.gz"

    def relpath(self, custom_id: str) -> str:
        """The transcript's path relative to the data directory (for logs)."""
        try:
            return str(self.path(custom_id).relative_to(config.DATA_DIR))
        except ValueError:
            return str(self.path(custom_id))

    def load(self, custom_id: str) -> list | str | None:
        """The saved chain of completed turns (BetaMessage objects), the error
        string a batch item came back with, or None when nothing is saved."""
        path = self.path(custom_id)
        if not path.exists():
            return None
        payload = json.loads(gzip.decompress(path.read_bytes()))
        if payload.get("error") is not None:
            return str(payload["error"])
        # construct, not model_validate: it rebuilds the typed blocks the way the
        # SDK builds an API response — without validating — so a field the API
        # sends in a shape this SDK version's types don't list still loads.
        # (2026-11-09: every batch result's usage.iterations entries lacked the
        # `model` the types require; validating refused all ten saved results.)
        return [BetaMessage.construct(**m) for m in payload["messages"]]

    def save(self, custom_id: str, chain: list) -> None:
        """Save a chain of *completed* turns (each with a stop_reason)."""
        self._write(custom_id, {"messages": [_dump(m) for m in chain]})

    def save_error(self, custom_id: str, error: str) -> None:
        self._write(custom_id, {"error": error})

    def _write(self, custom_id: str, body: dict) -> None:
        payload = {"run_id": self.run_id, "custom_id": custom_id,
                   "saved_at": datetime.now(timezone.utc).isoformat(), **body}
        _write_atomic(self.path(custom_id), gzip.compress(json.dumps(payload).encode("utf-8")))


def final_text(store: TranscriptStore, custom_id: str) -> str | None:
    """The last saved turn's text — what a parse failure was looking at."""
    chain = store.load(custom_id)
    if not isinstance(chain, list) or not chain:
        return None
    return "\n".join(b.text for b in chain[-1].content if getattr(b, "type", None) == "text")
