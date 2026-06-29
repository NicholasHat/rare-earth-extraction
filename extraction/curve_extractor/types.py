"""Shared result types for the deterministic curve extractor (docs/curve_extractor_plan.md §3).

These are produced by BOTH the vector path (extraction from pdfplumber geometry)
and the raster path (classical CV on a rendered figure image), so the downstream
LLM schema-mapping step sees one stable structure regardless of how the figure
was encoded in the PDF.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class MarkerRecord:
    group_key: str           # "#ff0000" (colour hex) | "filled_circle" | "cross" ...
    marker_type: str         # "filled" | "stroked"
    pixel_x: float
    pixel_y: float
    data_x: float | None = None   # None until axis calibrated
    data_y: float | None = None


@dataclass
class AxisCalibration:
    axis: str                # "x" | "y"
    model: str               # "linear" | "log10"
    slope: float
    intercept: float
    residual_rms: float      # RMS of (fit - tick_value), in DATA units
    r_squared: float
    n_ticks: int
    tick_values: list[float]
    ok: bool                 # residual below threshold

    def pixel_to_data(self, pixel: float) -> float:
        v = self.slope * pixel + self.intercept
        return 10.0**v if self.model == "log10" else v


@dataclass
class CurveExtractionResult:
    source: str              # "vector" | "raster"
    is_vector: bool
    markers: list[MarkerRecord]
    x_calibration: AxisCalibration | None
    y_calibration: AxisCalibration | None
    per_group_counts: dict[str, int]
    page_index: int
    figure_bbox: tuple[float, float, float, float] | None
    warnings: list[str] = field(default_factory=list)

    def to_json(self, **kw) -> str:
        return json.dumps(asdict(self), **kw)
