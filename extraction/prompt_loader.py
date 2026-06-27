"""Load a pinned, versioned extraction prompt (README §6).

The prompt is the black box. Nothing else in `extraction/` hardcodes its text —
the runner asks here for a `PromptBundle` and records the version + sha on every
run. Improving the prompt = drop in `prompts/extraction_vN.md` and bump the
EXTRACTION_PROMPT_VERSION setting. No pipeline code changes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import config

# Sentinel left in the shipped placeholder prompts; refuse to run until replaced.
_PLACEHOLDER_MARKER = "<<< PASTE YOUR extraction_v5 PROMPT BODY HERE >>>"


@dataclass(frozen=True)
class PromptBundle:
    version: str
    text: str
    sha256: str


class PromptNotReadyError(RuntimeError):
    """Raised when the pinned prompt file still contains the paste placeholder."""


def load_prompt(version: str | None = None) -> PromptBundle:
    """Load prompts/<version>.md. Raises if missing or still a placeholder.

    We never silently fall back to another version — reproducibility depends on
    knowing exactly which prompt produced each row.
    """
    version = version or config.EXTRACTION_PROMPT_VERSION
    path = config.PROMPTS_DIR / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    text = path.read_text()
    if _PLACEHOLDER_MARKER in text:
        raise PromptNotReadyError(
            f"{path.name} still contains the paste placeholder — paste the real "
            "extraction_v5 prompt body before running extractions."
        )
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptBundle(version=version, text=text, sha256=sha)
