"""Solve-for-the-blank orchestration over the pure conversions (README §7).

The calculator form lets the user fill in what they know and leave the rest
blank. This module fills in what's missing and, when the user supplies *both*
molar ratio and extractant concentration, flags a disagreement between them
instead of silently picking one (README: "show the discrepancy").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import conversions
from .atomic_mass import atomic_mass

# Relative disagreement between a supplied extractant conc. and the conc.
# implied by the supplied molar ratio, above which we warn the user.
_RATIO_CONSISTENCY_TOL = 0.05


@dataclass
class CalculatorInputs:
    element: str
    feed_value: float | None = None
    feed_unit: Literal["ppm", "mM"] = "ppm"
    target_molar_ratio: float | None = None
    target_extractant_conc_mM: float | None = None
    volume_mL: float | None = None


@dataclass
class CalculatorResult:
    ree_ppm: float | None = None
    ree_mM: float | None = None
    extractant_conc_mM: float | None = None
    molar_ratio: float | None = None
    ree_mass_mg: float | None = None
    ree_mmol_total: float | None = None
    extractant_mmol_total: float | None = None
    warnings: list[str] = field(default_factory=list)


def solve(inputs: CalculatorInputs) -> CalculatorResult:
    mass = atomic_mass(inputs.element)
    result = CalculatorResult()

    if inputs.feed_value is not None:
        if inputs.feed_unit == "ppm":
            result.ree_ppm = inputs.feed_value
            result.ree_mM = conversions.ppm_to_mM(inputs.feed_value, mass)
        else:
            result.ree_mM = inputs.feed_value
            result.ree_ppm = conversions.mM_to_ppm(inputs.feed_value, mass)

    have_ratio = inputs.target_molar_ratio is not None
    have_conc = inputs.target_extractant_conc_mM is not None

    if have_ratio and have_conc:
        result.molar_ratio = inputs.target_molar_ratio
        result.extractant_conc_mM = inputs.target_extractant_conc_mM
        if result.ree_mM:
            implied_conc = conversions.extractant_conc_from_ratio(
                result.ree_mM, inputs.target_molar_ratio
            )
            if implied_conc and abs(implied_conc - inputs.target_extractant_conc_mM) / implied_conc > _RATIO_CONSISTENCY_TOL:
                result.warnings.append(
                    f"Inconsistent inputs: molar ratio {inputs.target_molar_ratio:g} implies "
                    f"~{implied_conc:.3g} mM extractant at this feed concentration, but "
                    f"{inputs.target_extractant_conc_mM:g} mM was entered directly. Double-check."
                )
    elif have_ratio and result.ree_mM is not None:
        result.molar_ratio = inputs.target_molar_ratio
        result.extractant_conc_mM = conversions.extractant_conc_from_ratio(
            result.ree_mM, inputs.target_molar_ratio
        )
    elif have_conc and result.ree_mM is not None:
        result.extractant_conc_mM = inputs.target_extractant_conc_mM
        result.molar_ratio = conversions.molar_ratio_from_conc(
            inputs.target_extractant_conc_mM, result.ree_mM
        )
    elif have_ratio or have_conc:
        result.warnings.append(
            "Feed concentration is required to resolve molar ratio / extractant concentration."
        )

    if inputs.volume_mL is not None:
        if result.ree_mM is not None:
            result.ree_mmol_total = conversions.mmol_in_volume(result.ree_mM, inputs.volume_mL)
            result.ree_mass_mg = conversions.mass_mg_in_volume(
                result.ree_mM, inputs.volume_mL, mass
            )
        if result.extractant_conc_mM is not None:
            result.extractant_mmol_total = conversions.mmol_in_volume(
                result.extractant_conc_mM, inputs.volume_mL
            )

    return result
