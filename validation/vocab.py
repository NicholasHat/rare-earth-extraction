"""Known-value vocabularies for soft (flag-only) drift checks (see README §9).

These are *not* DB CHECK constraints — a value not in the list is flagged amber
("new value — confirm"), never rejected, so a legitimately new extractant or
mixing method can still be entered. The lists grow over time as the corpus does;
seed them from the first batch of approved papers.
"""
from __future__ import annotations

# Field -> set of canonical values seen so far. Lowercased for comparison.
# Seed values are illustrative; extend as real papers are approved.
KNOWN_VALUES: dict[str, set[str]] = {
    "Extractant type": {
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
    },
}


def unknown_values(field: str, values) -> list[str]:
    """Return the distinct values for `field` not in the known list.

    Comparison is case-insensitive and whitespace-trimmed. Fields with no known
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
        if norm not in known and str(v) not in seen:
            seen.append(str(v))
    return seen
