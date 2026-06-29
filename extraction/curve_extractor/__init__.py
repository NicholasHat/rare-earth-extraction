"""Deterministic PDF curve extractor (docs/curve_extractor_plan.md).

Routes a figure page to a deterministic vector path (pdfplumber geometry) or a
best-effort raster CV path (rendered image), returning structured marker records
for the downstream LLM schema-mapping step instead of LLM-guessed clustering.
"""
from .extractor import extract_curves
from .types import AxisCalibration, CurveExtractionResult, MarkerRecord

__all__ = ["extract_curves", "CurveExtractionResult", "MarkerRecord", "AxisCalibration"]
