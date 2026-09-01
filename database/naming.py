"""Naming conventions shared by the export writer and the tracking sheet.

The approved per-paper export is named `<Short_citation>_<n_rows>.csv`
(e.g. `Swain_&_Otu_322.csv`) to match the external "REE Paper Tracking"
spreadsheet's `Output file` column. Both the file written to data/exports/
and the tracking sheet's `Output file` cell go through `export_filename`,
so the two can never disagree.
"""
from __future__ import annotations

import re

# Path separators, characters Windows/macOS forbid in filenames, and control
# chars. Everything else (letters, digits, '&', '+', '-', '.') survives.
_HOSTILE = re.compile(r'[/\\:*?"<>|\x00-\x1f]')


def export_filename(short_citation: str | None, paper_id: int, n_rows: int) -> str:
    """`Swain & Otu` + 322 rows -> `Swain_&_Otu_322.csv`.

    Falls back to `paper_<id>_<n_rows>.csv` when no short citation was entered.
    """
    stem = _HOSTILE.sub("", (short_citation or "").strip())
    stem = re.sub(r"\s+", "_", stem)
    if not stem:
        stem = f"paper_{paper_id}"
    return f"{stem}_{n_rows}.csv"


def doi_link(doi: str | None) -> str | None:
    """Canonical DB DOI ('10.1016/...') -> clickable 'https://doi.org/10.1016/...'."""
    if not doi:
        return None
    if doi.startswith(("http://", "https://")):
        return doi
    return f"https://doi.org/{doi}"
