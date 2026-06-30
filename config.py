"""Central configuration and filesystem paths for the REE Extraction Dashboard.

Loads `.env` once and exposes the handful of settings the rest of the app reads.
Keeping paths here (rather than scattered string literals) means the `data/`
layout is defined in exactly one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # read .env from the project root if present

# --- Filesystem layout (see README §4) -------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INCOMING_DIR = DATA_DIR / "incoming"   # persisted source PDFs, named <sha256>.pdf
STAGING_DIR = DATA_DIR / "staging"     # per-paper extracted XLSX awaiting review
EXPORTS_DIR = DATA_DIR / "exports"     # approved per-paper XLSX exports
DB_PATH = DATA_DIR / "master.db"       # the SQLite master database

PROMPTS_DIR = ROOT / "prompts"

# --- Settings (with sensible local-demo defaults) --------------------------
REQUIRE_PASSWORD = os.getenv("REQUIRE_PASSWORD", "false").strip().lower() == "true"
WRITE_PASSWORD = os.getenv("WRITE_PASSWORD", "")
EXTRACTION_PROMPT_VERSION = os.getenv("EXTRACTION_PROMPT_VERSION", "extraction_v6")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "claude-opus-4-8")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "claude-haiku-4-5-20251001")


def ensure_dirs() -> None:
    """Create the runtime data directories if they don't exist."""
    for d in (DATA_DIR, INCOMING_DIR, STAGING_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
