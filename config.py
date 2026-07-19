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
EXTRACTION_PROMPT_VERSION = os.getenv("EXTRACTION_PROMPT_VERSION", "extraction_v7")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "claude-opus-4-8")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "claude-haiku-4-5-20251001")

# Loose backstop on total tokens (thinking + tool use) an extraction call may
# spend across its whole code-execution loop — self-moderated by the model,
# not an enforced ceiling. Guards against the runaway narrated-clustering /
# repeated-re-render failure mode (prompts/CHANGELOG.md, extraction_v6/v7).
#
# 100_000 was too tight: a real multi-element paper (13 REE elements, each a
# separate digitized curve) hit this ceiling and the model gave up partway
# through, silently regressing to ~2 points/element (the text endpoints only)
# instead of fully digitizing each curve. 500_000 is a deliberately generous
# floor so this only catches genuinely pathological runaway loops, not
# legitimate rich extractions — tune down once more real-run telemetry exists.
EXTRACTION_TASK_BUDGET_TOKENS = int(os.getenv("EXTRACTION_TASK_BUDGET_TOKENS", "500000"))


def ensure_dirs() -> None:
    """Create the runtime data directories if they don't exist."""
    for d in (DATA_DIR, INCOMING_DIR, STAGING_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
