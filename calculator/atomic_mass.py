"""Atomic mass table (g/mol) — single source of truth (README §7).

These are the same values the extraction prompt (prompts/extraction_v5.md,
Step 8) uses to derive "RRE composition (mM)" from ppm during extraction;
this module makes that table explicit and unit-testable instead of leaving it
to live only inside prompt text.
"""
from __future__ import annotations

ATOMIC_MASS_G_PER_MOL: dict[str, float] = {
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.04,
    "Lu": 174.97,
    "Y": 88.91,
    "Sc": 44.96,
    "U": 238.03,
    "Th": 232.04,
    "Co": 58.93,
    "Ni": 58.69,
    "Cu": 63.55,
    "Zn": 65.38,
    "Li": 6.94,
}

# The 14 lanthanides plus Y/Sc — the REY set the extraction prompt targets.
REE_ELEMENTS: list[str] = [
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc",
]


def atomic_mass(element: str) -> float:
    """Look up g/mol for an element symbol. Raises ValueError if unknown."""
    try:
        return ATOMIC_MASS_G_PER_MOL[element]
    except KeyError:
        raise ValueError(f"unknown element {element!r}; not in the atomic mass table") from None
