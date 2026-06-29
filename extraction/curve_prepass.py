"""Deterministic curve pre-pass — run BEFORE the model call (plan §6, option B).

Scans a PDF's pages with the deterministic curve extractor and aggregates the
result into (a) a prompt block injected as guidance and (b) a list of per-series
marker counts the QA layer cross-checks the model's output against. Pure / no API.

IMPORTANT — validated scope. The extractor is only trustworthy on a *clean
single-panel colour figure* (it recovers uniform counts there, e.g. Swain & Otu
Fig. 2 = 9 series × 19). On multi-panel pages one colour spans several panels and
its markers merge, and title-page logos / legend swatches produce spurious small
"series". So this pre-pass only treats a page as **authoritative** when its series
counts are internally consistent (uniform within tolerance); every other detected
figure page is reported as "verify visually", and raster pages are flagged for the
model to digitise as usual. This keeps us from ever injecting a wrong count as
ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pdfplumber

from .curve_extractor import extract_curves
from .curve_extractor.detect import _BytesIO

# A page must have at least this many series and markers to count as a data
# figure (filters title-page logos, CrossMark badges, single stray graphics).
_MIN_SERIES = 2
_MIN_MARKERS = 24
# Series counts on a page are "uniform" (=> one clean single-panel figure we
# trust) when their spread is within this fraction of the median.
_UNIFORM_SPREAD_FRAC = 0.15


@dataclass
class FigurePage:
    page_index: int
    series_counts: list[int]
    confident: bool          # uniform counts => trusted as authoritative


@dataclass
class CurvePrepass:
    confident_pages: list[FigurePage] = field(default_factory=list)
    unverified_pages: list[FigurePage] = field(default_factory=list)
    raster_pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def authoritative_counts(self) -> list[int]:
        """Sorted-descending marker counts from confident pages only (QA anchor)."""
        out: list[int] = []
        for fp in self.confident_pages:
            out.extend(fp.series_counts)
        return sorted(out, reverse=True)

    def to_prompt_block(self) -> str:
        if not (self.confident_pages or self.unverified_pages or self.raster_pages):
            return ""
        lines = ["## DETERMINISTIC CURVE ANALYSIS (computed from the PDF's own vector geometry)"]
        for fp in self.confident_pages:
            n = len(fp.series_counts)
            if len(set(fp.series_counts)) == 1:
                howmany = f"each with {fp.series_counts[0]} digitised markers"
            else:
                howmany = (f"with {min(fp.series_counts)}–{max(fp.series_counts)} markers each "
                           f"(counts {fp.series_counts})")
            lines.append(
                f"- **Page {fp.page_index} (authoritative):** {n} distinct data series, "
                f"{howmany}. This is ground truth from the figure's drawing commands — "
                f"every series must yield this many rows. A lower row count means you "
                f"under-digitised (likely missed points in a dense transition zone); go "
                f"back and capture them. Use the legend to map series → element."
            )
        for fp in self.unverified_pages:
            lines.append(
                f"- **Page {fp.page_index} (estimate, verify visually):** ~{len(fp.series_counts)} "
                f"vector series detected with varied counts {fp.series_counts} — likely a "
                f"multi-panel figure the deterministic pass can't cleanly separate. Digitise "
                f"it fully yourself; treat these counts only as a floor."
            )
        if self.raster_pages:
            lines.append(
                f"- **Page(s) {self.raster_pages} (raster images):** not deterministically "
                f"counted — digitise visually as usual."
            )
        return "\n".join(lines)


def _is_uniform(counts: list[int]) -> bool:
    if len(counts) < _MIN_SERIES:
        return False
    med = sorted(counts)[len(counts) // 2]
    if med == 0:
        return False
    return (max(counts) - min(counts)) <= _UNIFORM_SPREAD_FRAC * med


def analyze(pdf_bytes: bytes) -> CurvePrepass:
    result = CurvePrepass()
    with pdfplumber.open(_BytesIO(pdf_bytes)) as pdf:
        n_pages = len(pdf.pages)

    for idx in range(n_pages):
        try:
            res = extract_curves(pdf_bytes, idx)
        except Exception as e:
            result.warnings.append(f"page {idx}: pre-pass failed: {e}")
            continue

        if res.is_vector and res.per_group_counts:
            counts = sorted(res.per_group_counts.values(), reverse=True)
            if len(counts) < _MIN_SERIES or sum(counts) < _MIN_MARKERS:
                continue  # not a data figure (logo / stray graphic)
            fp = FigurePage(idx, counts, confident=_is_uniform(counts))
            (result.confident_pages if fp.confident else result.unverified_pages).append(fp)
            result.warnings.extend(f"page {idx}: {w}" for w in res.warnings)
        elif res.source == "raster" and res.figure_bbox is not None:
            result.raster_pages.append(idx)

    return result
