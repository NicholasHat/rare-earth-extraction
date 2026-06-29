"""Orchestrator — route a figure page to the vector or raster path (plan §2, §6).

`extract_curves` inspects the page: if it carries filled vector paths it uses the
deterministic vector path (validated to recover full, uniform marker counts on
Swain & Otu 2011); otherwise the figure is an embedded raster image and it uses
the best-effort raster CV path (Quinn et al. 2015). Either way it returns the
same `CurveExtractionResult`, with `source`/`is_vector` recording which ran and
`warnings` carrying any low-confidence flags for the downstream review step.
"""
from __future__ import annotations

from . import calibrate, detect, markers, raster
from .types import CurveExtractionResult, MarkerRecord


def _largest_image_bbox(page):
    if not page.images:
        return None
    im = max(page.images, key=lambda i: i["width"] * i["height"])
    return (im["x0"], im["top"], im["x1"], im["bottom"])


def _calibrate(page, frame):
    """Try to auto-calibrate both axes; return (x_cal, y_cal, warnings)."""
    warns: list[str] = []
    x_cal = y_cal = None
    for axis, setter in (("x", "x"), ("y", "y")):
        ticks = calibrate.auto_ticks(page, frame, axis)
        if ticks is None:
            warns.append(f"{axis}-axis: could not auto-read ticks; supply tick values to calibrate.")
            continue
        cal = calibrate.fit_axis(axis, ticks[0], ticks[1])
        if axis == "x":
            x_cal = cal
        else:
            y_cal = cal
        if not cal.ok:
            warns.append(f"{axis}-axis: calibration residual high ({cal.residual_rms:.3g}); review.")
    return x_cal, y_cal, warns


def _apply_calibration(recs, x_cal, y_cal) -> list[MarkerRecord]:
    if not (x_cal and y_cal):
        return recs
    return [
        MarkerRecord(
            group_key=r.group_key, marker_type=r.marker_type,
            pixel_x=r.pixel_x, pixel_y=r.pixel_y,
            data_x=x_cal.pixel_to_data(r.pixel_x),
            data_y=y_cal.pixel_to_data(r.pixel_y),
        )
        for r in recs
    ]


def extract_curves(pdf_bytes: bytes, page_index: int, *, legend_bbox=None) -> CurveExtractionResult:
    pdf, page = detect.open_page(pdf_bytes, page_index)
    try:
        if detect.page_is_vector(page):
            return _extract_vector(page, page_index, legend_bbox)
        return _extract_raster(page, page_index)
    finally:
        pdf.close()


def _extract_vector(page, page_index, legend_bbox) -> CurveExtractionResult:
    frame = detect.find_plot_frame(page)
    filled = detect.collect_filled_markers(page, legend_bbox=legend_bbox)
    groups = markers.group_filled_by_colour(filled)

    recs: list[MarkerRecord] = []
    per_group: dict[str, int] = {}
    for key, objs in groups.items():
        assembled = markers.assemble_filled(key, objs)
        if len(assembled) >= markers.MIN_MARKERS_PER_GROUP:
            per_group[key] = len(assembled)
            recs += assembled

    warns = markers.detect_merge_warnings(recs, per_group)
    x_cal = y_cal = None
    if frame:
        x_cal, y_cal, cal_warns = _calibrate(page, frame)
        warns += cal_warns
        recs = _apply_calibration(recs, x_cal, y_cal)

    return CurveExtractionResult(
        source="vector", is_vector=True, markers=recs,
        x_calibration=x_cal, y_calibration=y_cal, per_group_counts=per_group,
        page_index=page_index, figure_bbox=frame, warnings=warns,
    )


def _extract_raster(page, page_index) -> CurveExtractionResult:
    bbox = _largest_image_bbox(page)
    if bbox is None:
        return CurveExtractionResult(
            source="raster", is_vector=False, markers=[], x_calibration=None,
            y_calibration=None, per_group_counts={}, page_index=page_index,
            figure_bbox=None, warnings=["raster: no image found on page"],
        )
    recs, warns = raster.detect_markers(page, bbox)
    per_group: dict[str, int] = {}
    for r in recs:
        per_group[r.group_key] = per_group.get(r.group_key, 0) + 1
    return CurveExtractionResult(
        source="raster", is_vector=False, markers=recs, x_calibration=None,
        y_calibration=None, per_group_counts=per_group, page_index=page_index,
        figure_bbox=bbox, warnings=warns,
    )
