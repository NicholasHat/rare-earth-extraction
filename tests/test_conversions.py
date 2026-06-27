"""Exhaustive, fast tests for the pure calculator math (README §7)."""
import pytest

from calculator import conversions
from calculator.atomic_mass import atomic_mass

LA_MASS = 138.91


def test_atomic_mass_known():
    assert atomic_mass("La") == 138.91
    assert atomic_mass("Lu") == 174.97


def test_atomic_mass_unknown_raises():
    with pytest.raises(ValueError):
        atomic_mass("Xx")


def test_ppm_to_mM():
    assert conversions.ppm_to_mM(100, LA_MASS) == pytest.approx(100 / 138.91)


def test_mM_to_ppm():
    assert conversions.mM_to_ppm(0.72, LA_MASS) == pytest.approx(0.72 * 138.91)


def test_ppm_mM_roundtrip():
    ppm = 250.0
    mM = conversions.ppm_to_mM(ppm, LA_MASS)
    assert conversions.mM_to_ppm(mM, LA_MASS) == pytest.approx(ppm)


def test_extractant_conc_from_ratio():
    assert conversions.extractant_conc_from_ratio(0.72, 694.4) == pytest.approx(500.0, rel=1e-3)


def test_molar_ratio_from_conc():
    assert conversions.molar_ratio_from_conc(500.0, 0.72) == pytest.approx(694.44, rel=1e-3)


def test_ratio_and_conc_are_inverses():
    ree_mM = 1.234
    ratio = 50.0
    conc = conversions.extractant_conc_from_ratio(ree_mM, ratio)
    assert conversions.molar_ratio_from_conc(conc, ree_mM) == pytest.approx(ratio)


def test_volume_for_target_moles():
    assert conversions.volume_for_target_moles(5.0, 500.0) == pytest.approx(0.01)


def test_mmol_in_volume():
    assert conversions.mmol_in_volume(500.0, 10.0) == pytest.approx(5.0)


def test_mass_mg_in_volume():
    assert conversions.mass_mg_in_volume(0.72, 1000.0, LA_MASS) == pytest.approx(0.72 * 138.91)


def test_volume_and_mmol_in_volume_are_inverses():
    conc = 250.0
    vol_mL = 33.0
    mmol = conversions.mmol_in_volume(conc, vol_mL)
    vol_L = conversions.volume_for_target_moles(mmol, conc)
    assert vol_L * 1000 == pytest.approx(vol_mL)
