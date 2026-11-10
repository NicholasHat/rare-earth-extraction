"""Read the open-access paper manifest (data/open_access/manifest.csv).

Another agent curates it: one row per openly-licensed PDF, keyed by the PDF's
content sha256, with the PDF itself at data/open_access/pdfs/<sha256>.pdf.
Column contract (this module reads them by name; extra columns are ignored):

  sha256, doi, title, year, venue, license, license_evidence, oa_status,
  source_url, pdf_path, figure_format, extractants, elements, facets,
  n_data_figures, n_data_tables, quality_score, tier, first_batch, notes

`license` is a lower-case id (`cc-by`, `cc-by-nc-nd`, `cc0`, `public-domain`);
`tier` is A / B / reject / unverified; `first_batch` marks the papers chosen
for the first reviewed batch. Read-only: nothing here writes to the manifest
or its directory.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import config

CONTRACT = [
    "sha256", "doi", "title", "year", "venue", "license", "license_evidence", "oa_status",
    "source_url", "pdf_path", "figure_format", "extractants", "elements", "facets",
    "n_data_figures", "n_data_tables", "quality_score", "tier", "first_batch", "notes",
]
REQUIRED = ("sha256", "license")
_TRUE = {"1", "true", "yes", "y", "x"}


class ManifestError(ValueError):
    """The manifest is missing or doesn't have the columns this code needs."""


@dataclass(frozen=True)
class ManifestEntry:
    sha256: str
    license: str
    doi: str | None
    title: str | None
    year: str | None
    license_evidence: str | None
    source_url: str | None
    pdf: Path                       # resolved path of the PDF
    tier: str | None                # 'A' | 'B' | 'reject' | 'unverified'
    first_batch: bool
    quality_score: float | None
    row: dict = field(default_factory=dict, compare=False, repr=False)   # every column as read


def _blank_to_none(value) -> str | None:
    text = (value or "").strip()
    return text or None


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_pdf(raw: str | None, sha: str, manifest: Path) -> Path:
    """The manifest's pdf_path, read relative to the repo root or to the
    manifest's own directory; the conventional pdfs/<sha256>.pdf otherwise."""
    if raw:
        p = Path(raw).expanduser()
        candidates = [p] if p.is_absolute() else [config.ROOT / p, manifest.parent / p]
        for c in candidates:
            if c.exists():
                return c
    return manifest.parent / "pdfs" / f"{sha}.pdf"


def read(path: Path | None = None) -> list[ManifestEntry]:
    """Every row of the manifest, in file order."""
    path = Path(path or config.OPEN_ACCESS_MANIFEST)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            raise ManifestError(f"{path} lacks required column(s) {missing}; header is {header}")
        entries = []
        for raw in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            sha = row.get("sha256", "").lower()
            if not sha:
                continue
            entries.append(ManifestEntry(
                sha256=sha,
                license=row.get("license", "").lower(),
                doi=_blank_to_none(row.get("doi")),
                title=_blank_to_none(row.get("title")),
                year=_blank_to_none(row.get("year")),
                license_evidence=_blank_to_none(row.get("license_evidence")),
                source_url=_blank_to_none(row.get("source_url")),
                pdf=_resolve_pdf(_blank_to_none(row.get("pdf_path")), sha, path),
                tier=_blank_to_none(row.get("tier")),
                first_batch=row.get("first_batch", "").lower() in _TRUE,
                quality_score=_float(row.get("quality_score")),
                row=row,
            ))
    return entries


def by_sha(path: Path | None = None) -> dict[str, ManifestEntry]:
    """The manifest keyed by sha256 (empty when there is no manifest)."""
    try:
        return {e.sha256: e for e in read(path)}
    except ManifestError:
        return {}
