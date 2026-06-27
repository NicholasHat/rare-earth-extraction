"""Pure conversion math for the extractant calculator (README §7).

No I/O, no DB — these are the same formulas the extraction pipeline applies
implicitly when computing "RRE composition (mM)" and "Molar ratio of EX/REE",
made explicit and exhaustively testable here so Pillar C can also call them.
"""
from __future__ import annotations


def ppm_to_mM(ppm: float, atomic_mass_g_per_mol: float) -> float:
    """ppm = mg/L; mM = mmol/L.  mM = ppm / atomic_mass."""
    return ppm / atomic_mass_g_per_mol


def mM_to_ppm(mM: float, atomic_mass_g_per_mol: float) -> float:
    return mM * atomic_mass_g_per_mol


def extractant_conc_from_ratio(ree_mM: float, molar_ratio_ex_per_ree: float) -> float:
    """Extractant Conc. (mM) = [REE](mM) * (EX/REE molar ratio)."""
    return ree_mM * molar_ratio_ex_per_ree


def molar_ratio_from_conc(extractant_mM: float, ree_mM: float) -> float:
    return extractant_mM / ree_mM


def volume_for_target_moles(target_mmol: float, conc_mM: float) -> float:
    """Volume (L) = amount (mmol) / concentration (mM)."""
    return target_mmol / conc_mM


def mmol_in_volume(conc_mM: float, volume_mL: float) -> float:
    """Absolute amount (mmol) present in a given volume at a given concentration."""
    return conc_mM * (volume_mL / 1000.0)


def mass_mg_in_volume(conc_mM: float, volume_mL: float, atomic_mass_g_per_mol: float) -> float:
    """Absolute mass (mg) of solute present in a given volume.

    g/mol is numerically equal to mg/mmol, so mmol * atomic_mass gives mg directly.
    """
    return mmol_in_volume(conc_mM, volume_mL) * atomic_mass_g_per_mol
