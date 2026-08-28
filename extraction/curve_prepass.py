"""Deterministic curve pre-pass — run BEFORE the model call (plan §6, option B).

Scans a PDF's pages with the deterministic curve extractor and aggregates the
result into (a) a prompt block injected as guidance and (b) a list of per-series
marker counts the QA layer cross-checks the model's output against. Pure / no API.

IMPORTANT — validated scope. The extractor is only trustworthy on a *clean
single-panel colour figure* (it recovers uniform counts there, e.g. Swain & Otu
Fig. 2 = 9 series × 19). On multi-panel pages one colour spans several panels and
its markers merge, and title-page logos / legend swatches produce spurious small
"series". So this pre-pass only treats a page as **authoritative** when its series
counts are internally consistent (uniform within tolerance) AND no colour group's
own marker positions betray a panel merge (`_looks_panel_merged` — a same-colour
curve repeated across side-by-side panels splits into two balanced x-clusters
separated by the inter-panel gutter, which count uniformity alone cannot catch
when every series is inflated by the same factor, as on Swain & Otu Fig. 6);
every other detected figure page is reported as "verify visually", and raster
pages are flagged for the model to digitise as usual. This keeps us from ever
injecting a wrong count as ground truth.

Authoritative pages also carry pre-calibrated (x, y) marker coordinates
(`FigurePage.markers`), since `curve_extractor.extractor._extract_vector`
already runs axis calibration for every vector page — this pre-pass previously
discarded those coordinates and kept only counts. `to_prompt_block()` now
injects them as a JSON block so the model can skip its own calibration/
clustering work for those series entirely (docs/curve_extractor_plan.md §6),
which is the main token-cost driver on dense multi-figure papers. Monochrome
(stroked-marker) series are deliberately NOT included here even though
`markers.assemble_stroked` now exists — it has no oracle-paper validation
(plan §5.2's "second paper" case) the way the filled/colour path has, so its
coordinates aren't trusted as ground truth yet. Only filled-marker coordinates
are injected, same set the authoritative gate below already trusts.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import pdfplumber

from .curve_extractor import extract_curves
from .curve_extractor.detect import _BytesIO
from .curve_extractor.types import MarkerRecord

# A page must have at least this many series and markers to count as a data
# figure (filters title-page logos, CrossMark badges, single stray graphics).
_MIN_SERIES = 2
_MIN_MARKERS = 24
# Series counts on a page are "uniform" (=> one clean single-panel figure we
# trust) when their spread is within this fraction of the median.
_UNIFORM_SPREAD_FRAC = 0.15
# Panel-merge gate (checked on each colour group's raw pixel_x positions; the
# y-axis is unusable here — a sigmoid isotherm's two plateaus form a natural
# balanced y-gap that looks exactly like a stacked-panel gutter). A group is
# panel-suspect when its sorted x positions split at one dominant gap into two
# BALANCED halves: a genuine sparse data region also has a big gap but splits
# unbalanced (e.g. 17|2 on Swain Fig. 2), while the same curve drawn once per
# side-by-side panel splits ~evenly (~15|15 on Swain Fig. 6). Stacked panels
# with identical x-ranges aren't caught by this (their x values coincide
# instead of offsetting) — the count-uniformity gate stays the defense there.
_PANEL_GAP_FACTOR = 3.0     # dominant x-gap must exceed this multiple of the median gap
_PANEL_BALANCE_FRAC = 0.35  # ...and the smaller side must hold this fraction of markers
_PANEL_MIN_GROUP = 6        # smaller groups give meaningless split statistics
_PANEL_MIN_SUSPECTS = 2     # one odd group can be noise; a merge inflates several at once


@dataclass
class FigurePage:
    page_index: int
    series_counts: list[int]
    confident: bool          # uniform counts => trusted as authoritative
    # Pre-calibrated (data_x, data_y) markers, grouped by legend colour —
    # populated only for confident/authoritative pages (see module docstring).
    markers: list[MarkerRecord] | None = None


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
            points_json = _markers_json_block(fp.markers)
            if points_json:
                lines.append(
                    f"  **DIGITIZED CURVE DATA for page {fp.page_index}** (already calibrated "
                    f"from this figure's own axis ticks — do not recalibrate or re-cluster "
                    f"these series yourself): `{points_json}`. Map each `group_key` (legend "
                    f"colour) to its element via the legend, apply this paper's stated units/"
                    f"conditions, and use these (x, y) values directly as your digitised "
                    f"points for this page."
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


def _group_spans_panels(xs: list[float]) -> bool:
    """True when one colour group's x positions split at a dominant gap into two
    balanced halves — the signature of the same curve drawn once per panel."""
    xs = sorted(xs)
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    k = max(range(len(gaps)), key=gaps.__getitem__)
    others = gaps[:k] + gaps[k + 1:]
    med = sorted(others)[len(others) // 2] if others else 0.0
    if med <= 0:
        return False
    balance = min(k + 1, len(xs) - (k + 1)) / len(xs)
    return gaps[k] >= _PANEL_GAP_FACTOR * med and balance >= _PANEL_BALANCE_FRAC


def _looks_panel_merged(markers: list[MarkerRecord]) -> bool:
    """Page-level panel-merge verdict over all colour groups (see the constants
    above for the per-group rule and its rationale)."""
    by_group: dict[str, list[float]] = defaultdict(list)
    for m in markers:
        by_group[m.group_key].append(m.pixel_x)
    qualifying = suspect = 0
    for xs in by_group.values():
        if len(xs) < _PANEL_MIN_GROUP:
            continue
        qualifying += 1
        suspect += _group_spans_panels(xs)
    return suspect >= _PANEL_MIN_SUSPECTS and suspect * 2 >= qualifying


def _round(x: float) -> float:
    """4 significant figures — plenty for experimental data, keeps the
    injected coordinate block compact."""
    return float(f"{x:.4g}")


def _markers_json_block(markers: list[MarkerRecord] | None) -> str:
    """Serialize calibrated filled-marker points as `{group_key: [[x, y], ...]}`,
    keyed by legend colour. Only "filled" markers are included — stroked/
    monochrome markers exist on `res.markers` too now, but aren't validated as
    authoritative yet (module docstring), so this filters them out here rather
    than assuming callers only ever pass filled markers in."""
    if not markers:
        return ""
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for m in markers:
        if m.marker_type != "filled" or m.data_x is None or m.data_y is None:
            continue
        grouped[m.group_key].append([_round(m.data_x), _round(m.data_y)])
    if not grouped:
        return ""
    return json.dumps(dict(grouped), separators=(",", ":"))


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

        if res.is_vector:
            # This gate is only validated against filled/colour markers; a
            # monochrome stroked group (markers.assemble_stroked) has no
            # oracle-paper count yet (plan §5.2), so it must not join the
            # uniformity/panel-merge/authoritative logic below and silently
            # demote or pollute an otherwise clean colour figure's confidence.
            filled = [m for m in res.markers if m.marker_type == "filled"]
            per_group: dict[str, int] = defaultdict(int)
            for m in filled:
                per_group[m.group_key] += 1
            if not per_group:
                continue
            counts = sorted(per_group.values(), reverse=True)
            if len(counts) < _MIN_SERIES or sum(counts) < _MIN_MARKERS:
                continue  # not a data figure (logo / stray graphic)
            # Two gates: uniform counts AND no colour group spatially split
            # across panels — a two-panel figure inflates every series by the
            # same factor, so uniformity alone passes it (Swain & Otu Fig. 6).
            confident = _is_uniform(counts) and not _looks_panel_merged(filled)
            # Coordinates are only carried for authoritative pages AND when
            # both axes actually calibrated within tolerance — extractor.py's
            # _apply_calibration() writes data_x/data_y even when an axis's
            # residual is too high to trust (AxisCalibration.ok=False), it
            # only appends a warning, so page-level count-uniformity alone
            # isn't enough to trust the coordinates it produced.
            calibrated = (
                res.x_calibration is not None and res.x_calibration.ok
                and res.y_calibration is not None and res.y_calibration.ok
            )
            markers = filled if (confident and calibrated) else None
            fp = FigurePage(idx, counts, confident=confident, markers=markers)
            (result.confident_pages if fp.confident else result.unverified_pages).append(fp)
            result.warnings.extend(f"page {idx}: {w}" for w in res.warnings)
        elif res.source == "raster" and res.figure_bbox is not None:
            result.raster_pages.append(idx)

    return result
