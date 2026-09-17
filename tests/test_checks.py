"""validation.checks: each known failure mode trips the right flag (README §9)."""
import pandas as pd

from validation import checks, schema
from validation.report import Severity

EL = schema.ELEMENT_COLUMN


def _df(rows: list[dict]) -> pd.DataFrame:
    return schema.coerce_schema(pd.DataFrame(rows))


def _good_curve(element="La", n=12, start=10.0, step=7.0):
    """A clean, monotonic, in-bounds %E-vs-pH series with >= 8 points."""
    return [
        {EL: element, "pH": 1.0 + 0.3 * i, "Extract%": min(99.0, start + step * i)}
        for i in range(n)
    ]


def test_clean_extraction_passes_green():
    report = checks.run(_df(_good_curve()), [], figure_is_curve=True)
    assert report.verdict is Severity.GREEN
    assert report.passed


def test_sparse_result_is_red():
    rows = [{EL: "Yb", "pH": 1.0, "Extract%": 20.0}, {EL: "Yb", "pH": 5.0, "Extract%": 90.0}]
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not report.passed
    assert any(f.check == "row_count_sanity" for f in report.reds)


def test_sparse_result_not_flagged_when_not_a_curve():
    rows = [{EL: "Yb", "pH": 1.0, "Extract%": 20.0}, {EL: "Yb", "pH": 5.0, "Extract%": 90.0}]
    report = checks.run(_df(rows), [], figure_is_curve=False)
    assert not any(f.check == "row_count_sanity" for f in report.flags)


def test_axis_out_of_bounds_is_red():
    rows = _good_curve()
    rows[0]["Extract%"] = 150.0  # impossible %E
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert any(f.check == "axis_bounds" for f in report.reds)


def test_garbled_numeric_is_red():
    report = checks.run(_df(_good_curve()), [], figure_is_curve=True, coercion_failures=3)
    assert any(f.check == "schema_conformance" for f in report.reds)


def test_text_endpoint_mismatch_is_red():
    # Curve only reaches ~50% at pH 3, but the paper claims 95% there.
    rows = [
        {EL: "Nd", "pH": 1.0 + 0.25 * i, "Extract%": min(55.0, 10.0 + 4.5 * i)}
        for i in range(12)
    ]
    endpoints = [
        {"element": "Nd", "x_value": 3.0, "x_basis": "pH", "y_value": 95.0, "y_metric": "Extract%"}
    ]
    report = checks.run(_df(rows), endpoints, figure_is_curve=True)
    assert any(f.check == "text_endpoint_cross_check" for f in report.reds)


def test_text_endpoint_match_passes():
    rows = [
        {EL: "Nd", "pH": 1.0 + 0.25 * i, "Extract%": min(96.0, 10.0 + 8.0 * i)}
        for i in range(12)
    ]
    # At pH ~3.0 (i=8) Extract% is ~74; claim 74 should match within tolerance.
    endpoints = [
        {"element": "Nd", "x_value": 3.0, "x_basis": "pH", "y_value": 74.0, "y_metric": "Extract%"}
    ]
    report = checks.run(_df(rows), endpoints, figure_is_curve=True)
    assert not any(f.check == "text_endpoint_cross_check" for f in report.reds)


def test_vocabulary_drift_is_amber_not_blocking():
    rows = _good_curve()
    for r in rows:
        r["Extractant type"] = "totally-new-type"
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert report.passed  # amber doesn't block
    assert any(f.check == "vocabulary" and f.severity is Severity.AMBER for f in report.flags)


def test_empty_extraction_is_red():
    report = checks.run(_df([]), [], figure_is_curve=True)
    assert not report.passed
    assert any(f.check == "schema_conformance" for f in report.reds)


def _two_experiment_rows(element="Nd"):
    """A pH-sweep experiment (Extractant Conc. fixed at 500) plus a second,
    separate concentration-sweep experiment for the same element (pH fixed
    at 1.75) — the OUTPUT CONTRACT combines every experiment into one flat
    `rows` list, so both legitimately share the element column."""
    ph_sweep = [
        {EL: element, "pH": 0.87 + 0.15 * i, "Extract%": min(99.0, 2.0 + 6.0 * i),
         "Extractant Conc. (mM)": 500.0}
        for i in range(14)
    ]
    conc_sweep = [
        {EL: element, "pH": 1.75, "Extract%": pct, "Extractant Conc. (mM)": conc}
        for conc, pct in [(50, 0.24), (100, 25.66), (250, 7.38), (500, 14.1), (1000, 26.92)]
    ]
    return ph_sweep + conc_sweep


def test_monotonicity_does_not_pool_a_second_experiment():
    # The conc-sweep block alone (all at pH=1.75, %E jumping 0.24->26.92) would
    # look wildly non-monotonic if pooled with the pH-sweep by element alone.
    report = checks.run(_df(_two_experiment_rows()), [], figure_is_curve=True)
    assert not any(f.check == "monotonicity" for f in report.flags)


def test_text_endpoint_cross_check_picks_matching_experiment_not_nearest_tie():
    # Five conc-sweep rows all share pH=1.75 exactly (a tie on x-distance).
    # The paper's stated point (Extract% 14.02 at pH 1.75) matches the
    # Extractant Conc.=500 row (14.1) — not the Conc.=50 row (0.24), which an
    # arbitrary "first nearest x" tie-break would previously have picked.
    endpoints = [
        {"element": "Nd", "x_value": 1.75, "x_basis": "pH", "y_value": 14.02, "y_metric": "Extract%"}
    ]
    report = checks.run(_df(_two_experiment_rows()), endpoints, figure_is_curve=True)
    assert not any(f.check == "text_endpoint_cross_check" for f in report.reds)


def test_text_endpoint_cross_check_still_flags_a_real_mismatch():
    # None of the candidate rows at pH=1.75 comes anywhere near a claimed 60%.
    endpoints = [
        {"element": "Nd", "x_value": 1.75, "x_basis": "pH", "y_value": 60.0, "y_metric": "Extract%"}
    ]
    report = checks.run(_df(_two_experiment_rows()), endpoints, figure_is_curve=True)
    assert any(f.check == "text_endpoint_cross_check" for f in report.reds)


def _conc_endpoint(x_value, y_value, element="Nd"):
    return {"element": element, "x_value": x_value, "x_basis": "extractant_conc_mM",
            "y_value": y_value, "y_metric": "Extract%"}


def test_text_endpoint_conc_far_from_any_row_is_amber_not_red():
    # The paper says "0.05 to 1 M" and the model captured the endpoint as
    # x=1.0 but labelled it mM. The rows run 50..1000 mM, so nothing is within
    # a factor of 2 of 1.0 — comparing against the nearest row (50 mM, 0.24%)
    # would be comparing against the wrong point. Warn, don't gate.
    report = checks.run(_df(_two_experiment_rows()), [_conc_endpoint(1.0, 98.76)],
                        figure_is_curve=True)
    flags = [f for f in report.flags if f.check == "text_endpoint_cross_check"]
    assert len(flags) == 1
    assert flags[0].severity is Severity.AMBER
    assert "units" in flags[0].message
    assert not report.reds


def test_text_endpoint_conc_exact_row_still_flags_a_real_mismatch():
    # 1000 mM row reads 26.92%; a claimed 98.76% there is a genuine mismatch.
    report = checks.run(_df(_two_experiment_rows()), [_conc_endpoint(1000.0, 98.76)],
                        figure_is_curve=True)
    assert any(f.check == "text_endpoint_cross_check" for f in report.reds)


def test_text_endpoint_conc_exact_row_match_passes():
    report = checks.run(_df(_two_experiment_rows()), [_conc_endpoint(1000.0, 26.5)],
                        figure_is_curve=True)
    assert not any(f.check == "text_endpoint_cross_check" for f in report.flags)


def test_text_endpoint_ph_beyond_curve_is_still_red():
    # pH branch is unchanged: a stated pH the digitized curve never reaches is
    # a truncated-curve RED, not a units warning.
    rows = _good_curve("Nd", n=8)  # pH 1.0 .. 3.1
    endpoints = [{"element": "Nd", "x_value": 5.0, "x_basis": "pH",
                  "y_value": 99.0, "y_metric": "Extract%"}]
    report = checks.run(_df(rows), endpoints, figure_is_curve=True)
    reds = [f for f in report.reds if f.check == "text_endpoint_cross_check"]
    assert reds and "truncated" in reds[0].message


# --------------------------------------------------------------------------- #
# off_curve: points far off the straight line (in log D) through the majority
# of their own curve — in-plot text digitised as markers, misassigned markers.
# --------------------------------------------------------------------------- #
def _logistic_curve(element="Lu", extractant="EHEHPA", conc=1000.0, xs=None, slope=3.0, x_half=1.5):
    """A clean cation-exchange curve: log D = slope * (pH - pH½), i.e. %E logistic in pH."""
    xs = xs if xs is not None else [0.6 + 0.15 * i for i in range(12)]
    return [
        {EL: element, "Extractant": extractant, "Extractant Conc. (mM)": conc, "pH": x,
         "Extract%": 100.0 / (1.0 + 10 ** (-slope * (x - x_half)))}
        for x in xs
    ]


def _off_curve_rows(report):
    import re
    rows = []
    for f in report.flags:
        if f.check == "off_curve":
            rows += [int(r) for r in re.search(r"row\(s\) ([\d, ]+)", f.message).group(1).split(",")]
    return sorted(rows)


def test_off_curve_clean_logistic_curve_is_silent():
    report = checks.run(_df(_logistic_curve()), [], figure_is_curve=True)
    assert not any(f.check == "off_curve" for f in report.flags)


def test_off_curve_names_a_panel_title_run_including_adjacent_and_leading_points():
    # Quinn 2015 shape: a low-lying curve plus the panel title digitised as
    # three adjacent ~84 %E markers, and one more at the very start of the
    # series — the cases a neighbour-based test cannot see.
    rows = _logistic_curve(xs=[-0.70, -0.66, -0.61, -0.21, -0.05], x_half=0.0, slope=2.5)
    labels = [dict(rows[0], pH=x, **{"Extract%": e}) for x, e in
              [(-0.77, 84.2), (-0.55, 85.4), (-0.45, 84.2), (-0.30, 82.6)]]
    table = labels[:1] + rows[:3] + labels[1:] + rows[3:]
    report = checks.run(_df(table), [], figure_is_curve=True)
    assert _off_curve_rows(report) == [1, 5, 6, 7]
    assert all(f.severity is Severity.AMBER for f in report.flags if f.check == "off_curve")


def test_off_curve_flags_a_neighbouring_series_marker_assigned_to_this_one():
    rows = _logistic_curve()
    rows[6]["Extract%"] = 97.0   # a marker from a stronger extractant's curve, mid-series
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert _off_curve_rows(report) == [7]


def test_off_curve_is_silent_when_no_line_explains_the_curve():
    # Non-linearity is not evidence of artifacts: when no line runs through
    # half the band, the check says nothing rather than guess.
    import numpy as np
    x = np.linspace(0.0, 2 * np.pi, 12)
    pct = 100.0 / (1.0 + 10.0 ** (-2.5 * np.sin(x)))   # log D swings ±2.5, nowhere straight
    assert checks._off_curve_points(x, pct) == ([], 0)


def test_off_curve_tolerates_a_saturation_plateau_and_low_end_scatter():
    # Swain & Otu shape: a rising limb, then 98–99 % for two pH units, plus a
    # few-percent wobble at the low end. All real; none of it is off-curve.
    pct = [5.0, 5.6, 3.4, 4.8, 7.0, 8.4, 9.8, 12.0, 15.2, 26.8, 61.0, 65.6,
           98.1, 96.9, 98.9, 99.3, 99.3]
    xs = [0.87, 1.01, 1.17, 1.24, 1.34, 1.40, 1.47, 1.51, 1.56, 1.76, 1.99, 2.04,
          2.22, 2.34, 2.48, 2.89, 4.05]
    rows = [{EL: "Eu", "pH": x, "Extract%": e} for x, e in zip(xs, pct)]
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not any(f.check == "off_curve" for f in report.flags)


def test_off_curve_needs_at_least_five_points():
    rows = _logistic_curve(xs=[1.0, 1.2, 1.4, 1.6])
    rows[1]["Extract%"] = 95.0
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not any(f.check == "off_curve" for f in report.flags)


def test_curves_are_keyed_by_extractant_too():
    # Same element, two extractants, both clean but offset: pooled by element
    # alone they interleave into a zig-zag; keyed by extractant they are two
    # clean curves and neither monotonicity nor off_curve has anything to say.
    rows = _logistic_curve(extractant="EHEHPA", x_half=1.2) + _logistic_curve(extractant="Cyanex 272", x_half=2.2)
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not any(f.check in ("monotonicity", "off_curve") for f in report.flags)


def test_monotonicity_message_names_the_rows_where_the_curve_falls():
    rows = _logistic_curve()
    rows[3]["Extract%"] = 60.0
    rows[8]["Extract%"] = 20.0
    report = checks.run(_df(rows), [], figure_is_curve=True)
    (flag,) = [f for f in report.flags if f.check == "monotonicity"]
    assert "row(s) 5, 9" in flag.message and "Lu (EHEHPA, 1000 mM)" in flag.message


def test_off_curve_ignores_censored_points_at_the_axis_frame():
    # A weak extractant's curve digitised from 0 % up: the leading 0.0 %E
    # points are censored (no log D), not evidence of anything.
    rows = _logistic_curve(x_half=2.4)
    for r in rows[:4]:
        r["Extract%"] = 0.0
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not any(f.check == "off_curve" for f in report.flags)


def test_off_curve_flag_names_its_rows_structurally():
    rows = _logistic_curve()
    rows[6]["Extract%"] = 97.0
    report = checks.run(_df(rows), [], figure_is_curve=True)
    (flag,) = [f for f in report.flags if f.check == "off_curve"]
    assert flag.rows == (7,) and report.flagged_rows == [7]


def test_duplicate_rows_names_same_curve_repeats_but_not_cross_series_copies():
    rows = _logistic_curve()
    rows.append(dict(rows[4]))                                  # digitising loop: exact repeat, same curve
    rows.append(dict(rows[2], Extractant="Cyanex 272"))         # same triple under another extractant
    report = checks.run(_df(rows), [], figure_is_curve=True)
    (flag,) = [f for f in report.flags if f.check == "duplicate_rows"]
    assert flag.rows == (13,)          # only the same-curve repeat is droppable
    assert "4 row(s)" in flag.message  # both pairs are still reported
