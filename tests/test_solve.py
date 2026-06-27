"""Tests for calculator.solve — the solve-for-the-blank form logic (README §7)."""
import pytest

from calculator.solve import CalculatorInputs, solve


def test_solve_ppm_feed_computes_mM():
    result = solve(CalculatorInputs(element="La", feed_value=100.0, feed_unit="ppm"))
    assert result.ree_ppm == 100.0
    assert result.ree_mM == pytest.approx(100 / 138.91)
    assert result.extractant_conc_mM is None
    assert result.molar_ratio is None
    assert result.warnings == []


def test_solve_mM_feed_computes_ppm():
    result = solve(CalculatorInputs(element="Lu", feed_value=0.72, feed_unit="mM"))
    assert result.ree_mM == 0.72
    assert result.ree_ppm == pytest.approx(0.72 * 174.97)


def test_solve_ratio_given_computes_conc():
    result = solve(
        CalculatorInputs(element="La", feed_value=100.0, feed_unit="ppm", target_molar_ratio=694.4)
    )
    assert result.extractant_conc_mM == pytest.approx(result.ree_mM * 694.4)
    assert result.warnings == []


def test_solve_conc_given_computes_ratio():
    result = solve(
        CalculatorInputs(element="La", feed_value=100.0, feed_unit="ppm", target_extractant_conc_mM=500.0)
    )
    assert result.molar_ratio == pytest.approx(500.0 / result.ree_mM)


def test_solve_consistent_ratio_and_conc_no_warning():
    ree_mM = 100.0 / 138.91
    conc = ree_mM * 694.4
    result = solve(
        CalculatorInputs(
            element="La",
            feed_value=100.0,
            feed_unit="ppm",
            target_molar_ratio=694.4,
            target_extractant_conc_mM=conc,
        )
    )
    assert result.warnings == []


def test_solve_inconsistent_ratio_and_conc_warns():
    result = solve(
        CalculatorInputs(
            element="La",
            feed_value=100.0,
            feed_unit="ppm",
            target_molar_ratio=694.4,
            target_extractant_conc_mM=10.0,
        )
    )
    assert len(result.warnings) == 1
    assert "Inconsistent" in result.warnings[0]


def test_solve_ratio_without_feed_warns():
    result = solve(CalculatorInputs(element="La", target_molar_ratio=500.0))
    assert result.extractant_conc_mM is None
    assert len(result.warnings) == 1


def test_solve_volume_computes_absolute_amounts():
    result = solve(
        CalculatorInputs(
            element="La",
            feed_value=100.0,
            feed_unit="ppm",
            target_extractant_conc_mM=500.0,
            volume_mL=1000.0,
        )
    )
    assert result.ree_mmol_total == pytest.approx(result.ree_mM * 1.0)
    assert result.ree_mass_mg == pytest.approx(result.ree_mmol_total * 138.91)
    assert result.extractant_mmol_total == pytest.approx(500.0 * 1.0)


def test_solve_unknown_element_raises():
    with pytest.raises(ValueError):
        solve(CalculatorInputs(element="Xx", feed_value=1.0))
