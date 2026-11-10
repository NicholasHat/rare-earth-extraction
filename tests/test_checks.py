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
    rows = [dict(r, **{"Figure/Table": "Fig. 2"}) for r in _good_curve()]
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert report.verdict is Severity.GREEN
    assert report.passed


def test_sparse_result_is_red():
    rows = [{EL: "Yb", "pH": 1.0, "Extract%": 20.0}, {EL: "Yb", "pH": 5.0, "Extract%": 90.0}]
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not report.passed
    assert any(f.check == "row_count_sanity" for f in report.reds)


def test_a_few_real_markers_are_not_flagged():
    # Quinn et al. 2015 series really have 4-7 markers, as did most curves of
    # the 2026-11 trial; flagging that would push a re-extraction to pad them.
    rows = [{EL: "Tb", "pH": 1.0 + 0.37 * i, "Extract%": 10.0 + 12.0 * i, "Figure/Table": "Fig. 3"}
            for i in range(6)]
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert report.verdict is Severity.GREEN


def _line_series(element, n, *, step=0.1, noise=None):
    """%E points generated from one log D line (slope 3), optionally jittered."""
    out = []
    for i in range(n):
        x = 0.5 + step * i
        log_d = -2.0 + 3.0 * (x - 0.5) + (noise[i] if noise else 0.0)
        out.append({EL: element, "pH": round(x, 3), "Extract%": 100.0 / (1.0 + 10 ** -log_d)})
    return out


def test_points_sampled_off_a_fitted_line_are_red():
    report = checks.run(_df(_line_series("Ce", 13)), [], figure_is_curve=True)
    flags = [f for f in report.reds if f.check == "sampled_from_line"]
    assert len(flags) == 1 and len(flags[0].rows) == 13


def test_evenly_spaced_real_data_is_not_called_sampled():
    # Even pH setpoints are a normal design; real readings scatter off the line.
    jitter = [0.05, -0.07, 0.04, -0.03, 0.08, -0.06, 0.02, -0.05, 0.07, -0.04, 0.03, -0.08, 0.05]
    report = checks.run(_df(_line_series("Ce", 13, noise=jitter)), [], figure_is_curve=True)
    assert not any(f.check == "sampled_from_line" for f in report.flags)


def test_unevenly_spaced_points_exactly_on_a_line_are_not_called_sampled():
    rows = _line_series("Ce", 8)
    for r, x in zip(rows, [0.5, 0.62, 0.81, 0.9, 1.07, 1.2, 1.24, 1.4]):
        log_d = -2.0 + 3.0 * (x - 0.5)
        r["pH"], r["Extract%"] = x, 100.0 / (1.0 + 10 ** -log_d)
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not any(f.check == "sampled_from_line" for f in report.flags)


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
    assert flag.rows == (13,)          # only the same-curve repeat is flagged, and droppable
    assert "1 row(s)" in flag.message


def test_equal_pct_across_a_concentration_sweep_or_two_figures_is_not_a_duplicate():
    """The 2026-10 trial's false positives: a concentration sweep below
    detection (0.02 %E at every concentration, one pH), and one measurement
    plotted in both a pH sweep (Fig. 1) and a concentration sweep (Fig. 2)."""
    below_detection = [{EL: "La", "pH": 4.94, "Extract%": 0.02, "Extractant Conc. (mM)": c,
                        "Figure/Table": "Fig. 5c"} for c in (10, 30, 50, 100, 300)]
    shared = [{EL: "Y", "pH": 4.0, "Extract%": 33.1, "Extractant Conc. (mM)": 500.0, "Figure/Table": f}
              for f in ("Fig. 1", "Fig. 2")]
    kinetics = [{EL: "Nd", "pH": 2.5, "Extract%": 20.7, "Contact time (minute)": t, "Figure/Table": "Fig. 7"}
                for t in (15, 30, 60)]
    report = checks.run(_df(below_detection + shared + kinetics), [], figure_is_curve=True)
    assert not any(f.check == "duplicate_rows" for f in report.flags)


def test_monotonicity_does_not_pool_a_kinetics_figure_into_the_ph_sweep():
    """49558d1d on the trial: Eu's pH sweep (Fig. 6, 60 min) and its kinetics
    run (Fig. 7, pH 2.5) share the 60-minute contact time; pooled, the
    kinetics points made the pH sweep 'non-monotonic'."""
    sweep = [{EL: "Eu", "pH": ph, "Extract%": pct, "Contact time (minute)": 60, "Figure/Table": "Fig. 6"}
             for ph, pct in [(1.95, 18.9), (2.45, 61.0), (2.72, 60.3), (2.78, 58.3), (2.84, 59.8), (2.88, 52.6), (2.91, 55.7)]]
    kinetics = [{EL: "Eu", "pH": 2.5, "Extract%": pct, "Contact time (minute)": t, "Figure/Table": "Fig. 7"}
                for t, pct in [(15, 34.6), (30, 44.1), (60, 49.7), (90, 55.7)]]
    assert not any(f.check == "monotonicity" for f in checks.run(_df(sweep + kinetics), []).flags)
    pooled = [dict(r, **{"Figure/Table": None}) for r in sweep + kinetics]
    assert any(f.check == "monotonicity" for f in checks.run(_df(pooled), []).flags)


def _d_row(d, oa, pct, element="Tb"):
    return {EL: element, "pH": 2.0, "Extract%": pct, "Distribution ratio (D)": d,
            "Phase ratio (O/A)": oa, "Figure/Table": "Table 2"}


def test_pct_e_must_match_d_at_the_rows_phase_ratio():
    ok = [_d_row(1.0, 1.0, 50.0), _d_row(24, 0.01, 19.4), _d_row(500, 1, 100), _d_row(0.02, 1, 2.0),
          _d_row(3.0, 2.0, 85.7)]
    assert not any(f.check == "arithmetic" for f in checks.run(_df(ok), [], figure_is_curve=False).flags)
    # A resin's D (mL/g) converted with R = 1 instead of V/m = 100 (the v13 Wang 1998 error),
    # and a log D (1.3) stored as D.
    bad = [_d_row(24, 0.01, 96.0), _d_row(1.3, 1.0, 95.2)]
    (flag,) = [f for f in checks.run(_df(ok + bad), [], figure_is_curve=False).reds if f.check == "arithmetic"]
    assert flag.rows == (6, 7)


def test_negative_d_or_nonpositive_phase_ratio_is_out_of_bounds():
    report = checks.run(_df([_d_row(-1.3, 1.0, None), _d_row(None, 0.0, 40.0)]), [], figure_is_curve=False)
    msgs = " ".join(f.message for f in report.reds if f.check == "axis_bounds")
    assert "Distribution ratio (D)" in msgs and "Phase ratio (O/A)" in msgs


def test_ph_type_vocabulary():
    rows = [dict(r, **{"pH type": t, "Figure/Table": "Fig. 1"})
            for r, t in zip(_good_curve(n=3), ("equilibrium", "Initial", "final"))]
    (flag,) = [f for f in checks.run(_df(rows), []).flags if f.check == "ph_type"]
    assert "'final'" in flag.message and "Initial" not in flag.message


def test_rows_without_a_source_are_a_provenance_warning():
    rows = _good_curve()
    rows[0]["Figure/Table"] = "Fig. 2"
    (flag,) = [f for f in checks.run(_df(rows), []).flags if f.check == "provenance"]
    assert flag.severity is Severity.AMBER and "11 of 12" in flag.message


def test_text_endpoint_for_one_of_several_loadings_is_a_warning_not_a_gate():
    """557600c4 on the trial: 'about 70 % for Cyanex 272 at pH 4.5–4.75' was
    compared with another resin loading's 96 %. With several curves of the
    element along pH, the statement can't be pinned to one of them."""
    def curve(conc, top):
        return [{EL: "Nd", "pH": 1.0 + 0.5 * i, "Extract%": min(top, 10.0 + 12.0 * i),
                 "Extractant": "Cyanex 272", "Extractant Conc. (mM)": conc, "Figure/Table": "Fig. 3"}
                for i in range(9)]
    ep = [{"element": "Nd", "x_value": 4.6, "x_basis": "pH", "y_value": 70.0, "y_metric": "Extract%",
           "extractant": "Cyanex272"}]
    report = checks.run(_df(curve(7.84, 96.3) + curve(4.0, 99.0)), ep)
    flags = [f for f in report.flags if f.check == "text_endpoint_cross_check"]
    assert flags and all(f.severity is Severity.AMBER for f in flags)
    # One curve: the same mismatch gates the merge.
    report = checks.run(_df(curve(7.84, 96.3)), ep)
    assert any(f.check == "text_endpoint_cross_check" for f in report.reds)


def test_text_endpoint_element_match_is_exact_not_a_substring():
    """'Y' must not match 'Yb' rows."""
    rows = [{EL: "Yb", "pH": 1.0 + 0.3 * i, "Extract%": 10.0 + 7.0 * i, "Figure/Table": "Fig. 1"} for i in range(10)]
    ep = [{"element": "Y", "x_value": 2.2, "x_basis": "pH", "y_value": 95.0, "y_metric": "Extract%"}]
    assert not any(f.check == "text_endpoint_cross_check" for f in checks.run(_df(rows), ep).flags)


def _feed_rows(element, ppm, mm, conc=500.0, ratio=None, n=8):
    return [{EL: element, "pH": 1.0 + 0.37 * i, "Extract%": 10.0 + 10.0 * i,
             "RRE composition (ppm)": ppm, "RRE composition (mM)": mm,
             "Extractant Conc. (mM)": conc, "Molar ratio of EX/REE": ratio if ratio is not None else conc / mm}
            for i in range(n)]


def test_consistent_derived_columns_pass():
    rows = _feed_rows("La", 100.0, 100.0 / 138.91) + _feed_rows("Lu", 100.0, 100.0 / 174.97)
    report = checks.run(_df(rows), [], figure_is_curve=True)
    assert not any(f.check in ("arithmetic", "feed_total") for f in report.flags)


def test_a_molar_ratio_divided_by_the_feed_total_is_red():
    rows = _feed_rows("La", 100.0, 100.0 / 138.91, ratio=500.0 / 7.0)   # ÷ total, not this element's mM
    flags = [f for f in checks.run(_df(rows), [], figure_is_curve=True).reds if f.check == "arithmetic"]
    assert len(flags) == 1 and len(flags[0].rows) == 8 and "Molar ratio" in flags[0].message


def test_mm_that_does_not_follow_from_ppm_is_red():
    rows = _feed_rows("La", 100.0, 1.0)                                   # 100 ppm La is 0.72 mM
    assert any(f.check == "arithmetic" and "ppm" in f.message
               for f in checks.run(_df(rows), [], figure_is_curve=True).reds)


def test_a_total_feed_copied_onto_each_element_is_flagged():
    rows = [r for el in ("Ce", "Pr", "Nd", "Sm") for r in _feed_rows(el, None, 1.9, conc=1000.0)]
    flags = [f for f in checks.run(_df(rows), [], figure_is_curve=True).ambers if f.check == "feed_total"]
    assert len(flags) == 1 and "0.475" in flags[0].message


def test_a_non_rare_earth_element_is_red():
    rows = _feed_rows("Pb", 10.0, 10.0 / 207.2, conc=7.0) + _feed_rows("La", 100.0, 100.0 / 138.91)
    flags = [f for f in checks.run(_df(rows), [], figure_is_curve=True).reds if f.check == "element_scope"]
    assert len(flags) == 1 and "Pb" in flags[0].message and len(flags[0].rows) == 8


def test_a_ratio_rounded_to_its_printed_precision_is_not_an_arithmetic_error():
    mm = 781 / 138.91                                            # 5.622 mM La
    rows = _feed_rows("La", 781.0, round(mm, 3), conc=1.0, ratio=0.2)   # 1 / 5.622 = 0.178, printed 0.2
    assert not any(f.check == "arithmetic" for f in checks.run(_df(rows), [], figure_is_curve=True).flags)
    rows = _feed_rows("La", 781.0, round(mm, 3), conc=1.0, ratio=0.25)  # 0.25 is not 0.178 rounded
    assert any(f.check == "arithmetic" for f in checks.run(_df(rows), [], figure_is_curve=True).reds)


def test_one_element_at_two_feed_levels_is_two_curves():
    rows = _feed_rows("La", 10.0, 10.0 / 138.91) + [
        {**r, "Extract%": 95.0 - r["Extract%"]} for r in _feed_rows("La", 25.0, 25.0 / 138.91)]
    flags = checks.run(_df(rows), [], figure_is_curve=True).flags
    assert not any(f.check in ("monotonicity", "duplicate_rows") for f in flags)


# --------------------------------------------------------------------------- #
# The 2026-11 trial (extraction_v15): a curve is element + figure + every
# fixed condition, along the figure's own x axis (validation/curves.py).
# --------------------------------------------------------------------------- #
def _flags(rows, endpoints=(), check=None, **kw):
    flags = checks.run(_df(rows), list(endpoints), **kw).flags
    return [f for f in flags if check is None or f.check == check]


def _kinetics(series: dict, fig="Fig. 4(b)", *, fixed):
    """{series value: [%E at each time]} -> rows of one kinetics figure."""
    times = (1, 3, 5, 10, 20, 50)
    return [{EL: "Dy", "Contact time (minute)": t, "Extract%": e, "pH": 5.8, "Figure/Table": fig,
             **fixed(v)} for v, pcts in series.items() for t, e in zip(times, pcts)]


def test_series_differing_only_in_a_descriptor_are_separate_curves():
    """aee49f5e Fig. 4(b): rpm series pooled into one zig-zag; c48d4843:
    saponified vs not, told apart only in Material Process."""
    rpm = _kinetics({100: [5, 10, 15, 20, 25, 28], 300: [20, 40, 60, 75, 90, 95], 400: [20, 41, 60, 76, 90, 95]},
                    fixed=lambda v: {"mixing method": f"stirring {v} rpm"})
    assert not _flags(rpm, check="monotonicity") and not _flags(rpm, check="duplicate_rows")
    sapon = [{EL: "Gd", "pH": ph, "Extract%": e, "Material Process": mp, "Figure/Table": "Fig. 13"}
             for mp, pcts in (("adjust pH", (30.8, 38.5, 38.9, 41.0)), ("saponification 30%", (49.7, 50.4, 52.5, 55.0)))
             for ph, e in zip((2.5, 3.0, 3.5, 4.0), pcts)]
    assert not _flags(sapon, check="monotonicity")


def test_a_figure_x_axis_is_found_not_assumed():
    from validation import curves
    grid = _kinetics({ph: [ph * 2 + t for t in (1, 3, 5, 10, 20, 50)] for ph in (1, 2, 3, 4, 5, 5.8)},
                     fixed=lambda v: {"pH": v})
    assert curves.figure_x(_df(grid)) == "Contact time (minute)"     # a time × pH grid is kinetics
    # An acid sweep that also records each point's pH: along the acid, not the pH.
    acid = [{EL: "Nd", "Acid Solution conc. (M)": c, "pH": round(1 - c, 2), "Extract%": 10 * c} for c in (0.1, 0.5, 1, 2)]
    assert curves.figure_x(_df(acid)) == "Acid Solution conc. (M)"
    # One bar per extractant: nothing varies within a series.
    bars = [{EL: el, "Extractant": ex, "Extractant Conc. (mM)": c, "pH": 4.4, "Extract%": 50.0}
            for el in ("La", "Ce") for ex, c in (("D2EHPA", 1000), ("EHEHPA", 900), ("Cyanex 572", 1050))]
    assert curves.figure_x(_df(bars)) is None


def test_monotonicity_judges_the_curve_along_its_own_x():
    zigzag = _kinetics({1: [10, 40, 15, 50, 20, 60]}, fixed=lambda v: {})
    (flag,) = _flags(zigzag, check="monotonicity")
    assert "Contact time (minute)" in flag.message
    # A single maximum is chemistry (aee49f5e Fig. 8(a), 65 °C: the emulsion breaks), not a misread.
    hump = _kinetics({1: [15, 30, 70, 77, 50, 8]}, fixed=lambda v: {})
    assert not _flags(hump, check="monotonicity")


def test_off_curve_skips_initial_ph_and_curves_that_level_off():
    """78ebe569 Fig. 12: initial-pH curves levelling off at ~35 % (rows 58,
    59, 62 were called off-curve)."""
    level = [{EL: "Nd", "pH": x, "Extract%": e, "Figure/Table": "Fig. 12"} for x, e in
             [(0.88, 1.3), (1.36, 3.05), (1.72, 8.84), (1.89, 12.4), (2.06, 19.0), (2.26, 22.5),
              (2.66, 33.4), (2.75, 35.1), (3.20, 35.4), (4.63, 35.9), (6.54, 36.0)]]
    assert not _flags(level, check="off_curve")
    rows = _logistic_curve()
    rows[6]["Extract%"] = 97.0
    assert _flags(rows, check="off_curve")                                        # equilibrium / unrecorded
    assert not _flags([dict(r, **{"pH type": "initial"}) for r in rows], check="off_curve")


def test_row_count_is_per_curve_and_a_bar_chart_is_not_a_short_curve():
    """868fbcf3 Fig. 7: one bar per extractant drew 11 'short curve' flags."""
    bars = [{EL: el, "Extractant": ex, "pH": 4.4, "Extract%": 50.0, "Figure/Table": "Fig. 7"}
            for el in ("La", "Ce", "Nd") for ex in ("D2EHPA", "EHEHPA", "Cyanex 572")]
    assert not _flags(bars, check="row_count_sanity")
    two = [{EL: "Nd", "pH": x, "Extract%": e, "Figure/Table": "Fig. 2"} for x, e in ((1.0, 10), (4.0, 95))]
    full = [{EL: "La", "pH": 1.0 + 0.3 * i, "Extract%": 10 + 7 * i, "Figure/Table": "Fig. 2"} for i in range(10)]
    (flag,) = _flags(two + full, check="row_count_sanity")
    assert flag.severity is Severity.RED and "Nd" in flag.message and "La" not in flag.message


def test_a_series_with_under_half_its_siblings_points_is_amber():
    """3b88a3f8 Fig. 5: Ce had 1 of 5 kinetics markers."""
    rows = [{EL: el, "Contact time (minute)": t, "Extract%": 50 + t, "Figure/Table": "Fig. 5"}
            for el in ("La", "Pr", "Nd") for t in (5, 10, 15, 20, 25)]
    rows += [{EL: "Ce", "Contact time (minute)": 15, "Extract%": 81.5, "Figure/Table": "Fig. 5"}]
    (flag,) = _flags(rows, check="sparse_sibling")
    assert flag.severity is Severity.AMBER and "Ce" in flag.message and "La" not in flag.message
    assert not _flags(rows[:-1], check="sparse_sibling")


def test_rows_with_equal_conditions_and_different_values_are_red():
    """00fbdc33 Fig. 1: a Cyanex 923 wt% sweep — no column for the x — became
    six rows per element with identical conditions."""
    rows = [{EL: el, "Extractant": "Cyanex 923", "pH": 2.5, "Extract%": e, "Figure/Table": "Fig. 1"}
            for el, pcts in (("Nd", (7.7, 14.0, 17.8, 22.1)), ("Dy", (32.5, 54.4, 64.4, 69.5))) for e in pcts]
    (flag,) = _flags(rows, check="repeated_x")
    assert flag.severity is Severity.RED and "8 row(s)" in flag.message
    # The same rows inside a sweep whose other rows do carry the x.
    sweep = [{EL: "Nd", "Extractant": "Cyanex 923", "pH": 2.5, "Extract%": 10.0 * c, "Extractant Conc. (mM)": 100 * c,
              "Figure/Table": "Fig. 1"} for c in (1, 2, 3, 4)]
    assert _flags(sweep + rows[:4], check="repeated_x") and not _flags(sweep, check="repeated_x")
    # Replicates within noise are not a lost axis.
    assert not _flags([dict(r, **{"Extract%": 20.0 + i}) for i, r in enumerate(rows[:4])], check="repeated_x")


def test_back_extraction_percentages_belong_in_recovery():
    """f5038b04: 70 stripping rows in Extract%."""
    strip = [{EL: "Er", "Treatment": "Stripping", "Acid Solution conc. (M)": c, "Extract%": e, "Figure/Table": "Table 3"}
             for c, e in ((0.5, 40), (1, 70), (2, 95))]
    (flag,) = _flags(strip, check="treatment_y_column")
    assert flag.severity is Severity.RED and "1, 2, 3" in flag.message and not flag.rows
    ok = [dict(r, **{"Extract%": None, "Recovery %": r["Extract%"]}) for r in strip]
    ok[0]["Treatment"] = "Scrubbing"
    assert not _flags(ok, check="treatment_y_column")
    extraction_as_recovery = [dict(r, Treatment="Extraction only") for r in ok]
    assert _flags(extraction_as_recovery, check="treatment_y_column")
    # A combined process names extraction first: its Extract% rows are extraction rows.
    assert not _flags([dict(r, Treatment="Extraction + Stripping") for r in strip], check="treatment_y_column")


def test_d_vs_pct_arithmetic_ignores_stripping_rows():
    rows = [dict(_d_row(2.0, 1.0, 90.0), Treatment="Stripping")]      # a stripping D is not this %E's D
    assert not _flags(rows, check="arithmetic")
    assert _flags([dict(rows[0], Treatment="Extraction only")], check="arithmetic")


def test_text_endpoints_are_checked_on_every_x_column():
    rows = [{EL: "La", "Contact time (minute)": t, "Extract%": e, "Figure/Table": "Fig. 5"}
            for t, e in ((5, 55.3), (10, 61.8), (15, 72.1), (20, 80.5), (25, 82.0))]
    ep = {"element": "La", "x_basis": "Contact time (minute)", "x_value": 20, "y_metric": "Extract%"}
    assert not _flags(rows, [dict(ep, y_value=80.7)], check="text_endpoint_cross_check")
    (flag,) = _flags(rows, [dict(ep, y_value=95.0)], check="text_endpoint_cross_check")
    assert flag.severity is Severity.RED


def test_text_endpoints_on_no_schema_column_or_with_x_equal_to_y_are_amber():
    rows = _good_curve("Pr")
    eps = [{"element": "Pr", "x_basis": "Cyanex 923 wt%", "x_value": 40, "y_metric": "Extract%", "y_value": 60},
           {"element": "Pr", "x_basis": "pH", "x_value": 0.49, "y_metric": "D", "y_value": 0.49},
           {"element": "Pr", "x_basis": "whatever", "x_value": None, "y_metric": "Extract%", "y_value": 32.9}]
    flags = _flags(rows, eps, check="text_endpoint_cross_check")
    assert {f.severity for f in flags} == {Severity.AMBER} and len(flags) == 2
    assert any("'Cyanex 923 wt%'" in f.message for f in flags) and any("equals" in f.message for f in flags)


def _sf_pair(co, er, y, sf, fig="Fig. 7", **extra):
    return [{EL: el, "Co-extractant Conc. (mM)": co, "Extract%": e, "Separation factor (SF%)": sf,
             "Figure/Table": fig, **extra} for el, e in (("Er", er), ("Y", y))]


def test_a_separation_factor_that_fits_only_with_the_elements_swapped_is_amber():
    """0346dd7e Fig. 7, rows 62/72: Y 98.2 / Er 97.0 under SF 1.537 (Er/Y
    1.47–1.84 at every other point), i.e. the two values exchanged."""
    rows = (_sf_pair(120, 87.7, 82.83, 1.460) + _sf_pair(160, 90.3, 85.42, 1.577)
            + _sf_pair(200, 93.41, 88.43, 1.836) + _sf_pair(240, 97.03, 98.20, 1.537))
    (flag,) = _flags(rows, check="separation_factor")
    assert "rows 7, 8" in flag.message and "0.599" in flag.message
    # An SF fitting neither direction is the paper's own inconsistency (0346dd7e Fig. 3).
    odd = rows[:6] + _sf_pair(240, 95.33, 88.93, 1.59)
    assert not _flags(odd, check="separation_factor")
    # With the pair stated, one point is enough.
    stated = _sf_pair(240, 97.03, 98.20, 1.537, **{"Material Process": "SF Er/Y"})
    assert _flags(stated, check="separation_factor")
    assert not _flags(_sf_pair(240, 97.03, 98.20, 1.537), check="separation_factor")   # no direction known


def test_mixing_method_detail_is_not_vocabulary_drift():
    from validation import vocab
    assert vocab.unknown_values("mixing method", ["stirring, 450 rpm", "shaking 2000 rpm",
                                                  "mechanical shaker (300 rpm)", "impeller", "Stirring"]) == []
    assert vocab.unknown_values("mixing method", ["stiring", "sonication 5 min"]) == ["stiring", "sonication 5 min"]


def test_element_groups_are_out_of_scope():
    rows = [{EL: g, "Extractant Conc. (mM)": c, "Extract%": 50.0} for g in ("LREE", "HREE") for c in (100, 200, 300)]
    (flag,) = _flags(rows, check="element_scope")
    assert flag.severity is Severity.RED and "element group" in flag.message
