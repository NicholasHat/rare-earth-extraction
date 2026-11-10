"""Known-value vocabularies for soft (flag-only) drift checks (see README §9).

These are *not* DB CHECK constraints — a value not in the list is flagged amber
("new value — confirm"), never rejected, so a legitimately new extractant or
mixing method can still be entered. The lists grow over time as the corpus does;
seed them from the first batch of approved papers.
"""
from __future__ import annotations

import re

# Field -> set of canonical values seen so far. Lowercased for comparison.
# Seed values are illustrative; extend as real papers are approved.
KNOWN_VALUES: dict[str, set[str]] = {
    "Extractant type": {
        # The chemical classes the extraction prompt prescribes (extraction_v14+
        # Step 8). Missing until 2026-11, so every paper of the 2026-10 trial
        # drew an amber "new value" flag for following the prompt.
        "phosphoric acid based",
        "phosphonic acid based",
        "phosphinic acid based",
        "carboxylic acid based",
        "amine based",
        "ionic liquid",
        "solvating (neutral organophosphorus)",
        "acidic",
        "basic",
        "neutral",
        "solvating",
        "chelating",
        "cation exchanger",
        "anion exchanger",
    },
    "mixing method": {
        "shaking",
        "magnetic stirring",
        "mechanical stirring",
        "vortex",
        "orbital shaker",
        "stirring",
        "shaker",
        "mechanical shaking",
        "wrist action shaker",
        "mechanical shaker",
        "impeller",
    },
}


def unknown_values(field: str, values) -> list[str]:
    """Return the distinct values for `field` not in the known list.

    Comparison is case-insensitive and whitespace-trimmed, and detail after
    the method's name is ignored — a parenthesis, or anything from the first
    comma or digit on ("shaking (250 rpm)", "stirring, 450 rpm" and
    "shaking 2000 rpm" are "shaking"/"stirring"; every paper of the 2026-11
    trial drew a "new value" flag for stating its speed). Fields with no known
    list configured are never flagged.
    """
    known = KNOWN_VALUES.get(field)
    if known is None:
        return []
    seen: list[str] = []
    for v in values:
        if v is None:
            continue
        norm = str(v).strip().lower()
        if not norm:
            continue
        if norm not in known and _name(norm) not in known and str(v) not in seen:
            seen.append(str(v))
    return seen


_PAREN = re.compile(r"\s*\([^)]*\)")
_DETAIL = re.compile(r"\s*[,;\d].*$")


def _name(value: str) -> str:
    """The method's name without its stated detail."""
    return _DETAIL.sub("", _PAREN.sub("", value)).strip()
