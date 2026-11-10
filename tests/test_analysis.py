"""analysis/: curve assembly, derived quantities, fits and chart specs (pure)."""
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from analysis import charts, curves, data, derived, series, slopes
from analysis.curves import EXTRACTION, FIGURE
from database import merge
from ingestion.licensing import License
from validation import schema

EL = schema.ELEMENT_COLUMN


def frame(rows, paper_id=1, paper="Paper A"):
    """Schema-shaped rows plus the paper columns analysis.data.load adds."""
    df = schema.coerce_schema(pd.DataFrame(rows))
    df.insert(0, "paper_id", paper_id)
    df.insert(1, data.PAPER, paper)
    return df


def ph_sweep(element, phs, pcts, **fixed):
    return [{EL: element, "pH": p, "Extract%": e, "Extractant": "Cyanex 272",
             "Extractant Conc. (mM)": 500.0, **fixed} for p, e in zip(phs, pcts)]


def pct_from_logd(logd, oa=1.0):
    """%E for a given log D at phase ratio O/A (inverse of the D formula)."""
    d = 10.0 ** np.asarray(logd, float)
    return 100.0 * d / (d + 1.0 / oa)


# --------------------------------------------------------------------------- #
# Curve key and assembly
# --------------------------------------------------------------------------- #

def test_key_is_the_qa_curve_key_plus_the_paper():
    from validation.curves import curve_key
    key = curves.key_columns("pH")
    assert key == ["paper_id"] + curve_key("pH")
    assert "pH" not in key and EL in key
    assert {"Sources", "Treatment", "Material Process", "Acid Solution", "Diluent", "mixing method"} <= set(key)


def test_the_page_offers_every_x_axis_qa_knows_and_no_other():
    from validation.curves import X_COLUMNS
    assert {a.column for a in curves.X_AXES} == set(X_COLUMNS)


def test_key_holds_ph_fixed_for_other_axes_except_acid_concentration():
    assert "pH" in curves.key_columns("Extractant Conc. (mM)")
    assert "Extractant Conc. (mM)" not in curves.key_columns("Extractant Conc. (mM)")
    assert "pH" not in curves.key_columns("Acid Solution conc. (M)")      # pH follows the acid
    feed = curves.key_columns("Total REE feed (mM)")
    assert "RRE composition (ppm)" not in feed and "RRE composition (mM)" not in feed


def test_two_figures_of_one_element_are_two_curves():
    df = frame(ph_sweep("Nd", [1, 2, 3], [10, 50, 90], **{FIGURE: "Fig. 2"})
               + ph_sweep("Nd", [1, 2, 3], [12, 48, 91], **{FIGURE: "Fig. 5"}))
    cs = curves.assemble(df, "pH", "Extract%")
    assert len(cs.curves) == 2
    assert sorted(cs.curves["label"]) == ["Fig. 2", "Fig. 5"]       # only what differs
    assert "Nd" in cs.shared and "Cyanex 272" in cs.shared
    assert not cs.notes


def test_rows_without_figure_group_by_conditions_and_say_so():
    df = frame(ph_sweep("La", [1, 2, 3], [10, 50, 90]) + ph_sweep("Ce", [1, 2, 3], [20, 60, 95]))
    cs = curves.assemble(df, "pH", "Extract%")
    assert list(cs.curves[EL]) == ["La", "Ce"]
    assert any("Source figure not recorded" in n for n in cs.notes)


def test_experiments_told_apart_only_by_sources_are_not_pooled():
    rows = [{EL: "La", "Acid Solution conc. (M)": c, "Extract%": e, "Sources": src}
            for src in ("water", "ethylene glycol") for c, e in zip([0, 1, 2], [3, 5, 8])]
    cs = curves.assemble(frame(rows), "Acid Solution conc. (M)", "Extract%")
    assert len(cs.curves) == 2 and not cs.curves["repeated_x"].any()


def test_single_points_along_the_axis_are_left_out_with_a_note():
    # a concentration sweep seen along pH: one point per concentration
    rows = ph_sweep("La", [1, 2, 3], [10, 50, 90], **{FIGURE: "Fig. 1"})
    rows += [{EL: "La", "pH": 1.75, "Extract%": e, "Extractant": "Cyanex 272", "Extractant Conc. (mM)": c,
              FIGURE: "Fig. 2"} for c, e in zip([50, 100, 250], [1, 5, 20])]
    cs = curves.assemble(frame(rows), "pH", "Extract%")
    assert len(cs.curves) == 1
    assert any("3 point(s) measured at a single pH" in n for n in cs.notes)


def test_figureless_point_shared_with_another_sweep_is_named_not_reassigned():
    # Swain & Otu: the concentration sweep at pH 1.75 has a 500 mM point that
    # matches the pH sweep's fixed conditions.
    rows = ph_sweep("Eu", [1.5, 1.7, 1.76, 2.0], [12, 16, 27, 61])
    rows += [{EL: "Eu", "pH": 1.75, "Extract%": e, "Extractant": "Cyanex 272", "Extractant Conc. (mM)": c}
             for c, e in zip([100, 250, 500], [5, 20, 57])]
    cs = curves.assemble(frame(rows), "pH", "Extract%")
    curve = cs.points[cs.points["curve"] == 0]
    assert len(curve) == 5
    shared = curve[curve["also_on"] != ""]
    assert list(shared["x"]) == [1.75] and "Extractant conc. (mM) sweep" in shared["also_on"].iloc[0]
    assert any("also sit on a sweep along another axis" in n for n in cs.notes)


def test_thin_and_ambiguous_curves_are_drawn_as_points():
    df = frame(ph_sweep("La", [1, 2], [10, 50], **{FIGURE: "a"})
               + ph_sweep("Ce", [1, 2, 2, 3], [10, 30, 70, 90], **{FIGURE: "b"})
               + ph_sweep("Pr", [1, 2, 3], [10, 50, 90], **{FIGURE: "c"}))
    cs = curves.assemble(df, "pH", "Extract%")
    draw = dict(zip(cs.curves[EL], cs.curves["draw"]))
    assert draw == {"La": "points", "Ce": "points", "Pr": "line"}
    assert cs.curves.set_index(EL).loc["Ce", "repeated_x"]
    assert any("more than one point at the same pH" in n for n in cs.notes)


def test_curves_sort_along_the_series_not_the_alphabet():
    df = frame(ph_sweep("Lu", [1, 2, 3], [1, 2, 3]) + ph_sweep("La", [1, 2, 3], [1, 2, 3])
               + ph_sweep("Y", [1, 2, 3], [1, 2, 3]) + ph_sweep("Ce", [1, 2, 3], [1, 2, 3]))
    assert list(curves.assemble(df, "pH", "Extract%").curves[EL]) == ["La", "Ce", "Lu", "Y"]


def test_numeric_conditions_sort_as_numbers():
    df = frame(ph_sweep("La", [1, 2, 3], [1, 2, 3], **{"Extractant Conc. (mM)": 1000.0})
               + ph_sweep("La", [1, 2, 3], [1, 2, 3], **{"Extractant Conc. (mM)": 50.0}))
    assert list(curves.assemble(df, "pH", "Extract%").curves["Extractant Conc. (mM)"]) == [50.0, 1000.0]


def test_paper_label_leads_when_papers_differ():
    df = pd.concat([frame(ph_sweep("Nd", [1, 2, 3], [1, 2, 3]), 1, "Smith 2020"),
                    frame(ph_sweep("Nd", [1, 2, 3], [1, 2, 3], Diluent="kerosene"), 2, "Lee 2021")])
    labels = list(curves.assemble(df, "pH", "Extract%").curves["label"])
    assert labels == ["Lee 2021, in kerosene", "Smith 2020"]


def test_varying_axes_offers_only_axes_the_data_varies_along():
    df = frame(ph_sweep("La", [1, 2, 3], [10, 50, 90]))
    assert [a.column for a in curves.varying_axes(df, "Extract%")] == ["pH"]
    assert curves.varying_axes(df, "Recovery %") == []
    assert curves.varying_axes(df, "D") == []                 # no D, no phase ratio


def test_ph_types_are_split_never_shared_on_one_axis():
    df = frame(ph_sweep("La", [1, 2, 3], [1, 2, 3], **{"pH type": "equilibrium"})
               + ph_sweep("La", [1, 2, 3], [1, 2, 3], **{"pH type": "initial"})
               + ph_sweep("La", [1, 2, 3], [1, 2, 3]))
    groups = curves.ph_type_groups(curves.assemble(df, "pH", "Extract%"))
    assert [t for t, _ in groups] == ["Equilibrium pH", "Initial pH", "pH type not recorded"]
    one = frame(ph_sweep("La", [1, 2, 3], [1, 2, 3], **{"pH type": "equilibrium"}))
    assert curves.ph_type_groups(curves.assemble(one, "pH", "Extract%"))[0][0] is None


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #

def test_distribution_ratio_prefers_reported_and_derives_with_phase_ratio_only():
    df = frame([
        {EL: "La", "Extract%": 50.0, "Distribution ratio (D)": 2.0, "Phase ratio (O/A)": 1.0},
        {EL: "La", "Extract%": 80.0, "Phase ratio (O/A)": 2.0},     # D = 4 / 2
        {EL: "La", "Extract%": 80.0},                               # no phase ratio
        {EL: "La", "Extract%": 100.0, "Phase ratio (O/A)": 1.0},    # plateau: undefined
    ])
    d, src = derived.distribution_ratio(df)
    assert d[0] == 2.0 and src[0] == derived.D_REPORTED
    assert d[1] == pytest.approx(2.0) and src[1] == derived.D_DERIVED
    assert math.isnan(d[2]) and pd.isna(src[2])
    assert math.isnan(d[3])


def test_d_measure_marks_derived_points_and_notes_missing_phase_ratio():
    df = frame([{EL: "La", "pH": p, "Extract%": e, "Phase ratio (O/A)": 1.0, FIGURE: "F"} for p, e in [(1, 20), (2, 50)]]
               + [{EL: "La", "pH": 3, "Extract%": 80.0, FIGURE: "F"}])
    cs = curves.assemble(df, "pH", "D")
    assert list(cs.points["y"]) == pytest.approx([0.25, 1.0])
    assert set(cs.points["y_source"]) == {derived.D_DERIVED}
    assert any("no recorded phase ratio" in n for n in cs.notes)


def test_x_at_half_interpolates_in_log_ratio_between_brackets():
    # Swain & Otu La: pH 2.48 -> 29.64 %, 2.89 -> 71.68 %
    h = derived.x_at_half([2.0, 2.48, 2.89, 4.05], [12.0, 29.64, 71.68, 98.98])
    l0, l1 = math.log10(29.64 / 70.36), math.log10(71.68 / 28.32)
    assert h.x == pytest.approx(2.48 + (-l0) / (l1 - l0) * 0.41)
    assert h.bracket == ((2.48, 29.64), (2.89, 71.68))


def test_x_at_half_never_extrapolates():
    assert derived.x_at_half([1, 2, 3], [5, 20, 40]).x is None
    assert "below 50 %" in derived.x_at_half([1, 2, 3], [5, 20, 40]).note
    assert "above 50 %" in derived.x_at_half([1, 2, 3], [60, 80, 95]).note


def test_x_at_half_refuses_a_curve_that_crosses_twice():
    h = derived.x_at_half([1, 2, 3, 4], [20, 60, 30, 80])
    assert h.x is None and "more than once" in h.note


def test_x_at_half_edge_cases():
    assert derived.x_at_half([1, 2, 3], [20, 50, 80]).x == 2
    assert derived.x_at_half([1, 2], [20, 80]).x is None                 # too few points
    assert derived.x_at_half([1, 2, 2, 3], [20, 40, 60, 90]).x == 2      # repeats averaged
    h = derived.x_at_half([1, 2, 3], [0, 20, 100])                       # bracket at 100 %: linear in %E
    assert h.x == pytest.approx(2 + 30 / 80) and "linearly" in h.note


def test_interpolate_inside_never_extrapolates():
    assert derived.interpolate_inside([1, 2], [0, 10], 1.5) == (5.0, False)
    assert derived.interpolate_inside([1, 2], [0, 10], 2.0) == (10.0, True)
    assert derived.interpolate_inside([1, 2], [0, 10], 2.5) == (None, False)
    v, _ = derived.interpolate_inside([10, 1000], [0, 2], 100, log_x=True)
    assert v == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Across the series
# --------------------------------------------------------------------------- #

def _series_frame(intercepts, phs=(1.0, 1.5, 2.0, 2.5, 3.0), slope=3.0, **fixed):
    """Curves with log D = slope·pH + c per element (O/A 1), in one figure."""
    rows = []
    for el, c in intercepts.items():
        pct = pct_from_logd(slope * np.asarray(phs) + c)
        rows += ph_sweep(el, phs, pct, **{FIGURE: "Fig. 1", **fixed})
    return frame(rows)


def test_half_extraction_per_element_in_series_order():
    df = _series_frame({"Nd": -6.0, "La": -7.5})
    table = series.half_extraction(curves.assemble(df, "pH", "Extract%"))
    assert list(table[EL]) == ["La", "Nd"]
    assert table["pH_half"].tolist() == pytest.approx([2.5, 2.0], abs=0.02)


def test_separation_factor_from_percent_extraction_with_unknown_phase_ratio():
    df = _series_frame({"La": -7.0, "Ce": -6.6, "Pr": -6.3})
    pts, summary, unavailable = series.separation_factors(curves.assemble(df, "pH", EXTRACTION))
    assert list(summary["pair"]) == ["Ce/La", "Pr/Ce"]
    assert summary["beta_median"].tolist() == pytest.approx([10 ** 0.4, 10 ** 0.3], rel=1e-6)
    assert set(summary["basis"]) == {"%E (phase ratio cancels)"}
    assert not unavailable
    # only points inside 5-95 % take part: log D within ±1.28, i.e. pH 2.0 and 2.5 of each curve
    assert summary["n"].tolist() == [2, 2]
    assert sorted(set(pts["x"])) == [2.0, 2.5]


def test_separation_factor_interpolates_the_lighter_curve_inside_its_range_only():
    rows = ph_sweep("La", [1.0, 2.0, 3.0], pct_from_logd([-1.0, 0.0, 1.0]), **{FIGURE: "F"})
    rows += ph_sweep("Ce", [1.5, 2.5, 3.5], pct_from_logd([0.0, 1.0, 2.0]) , **{FIGURE: "F"})
    pts, summary, _ = series.separation_factors(curves.assemble(frame(rows), "pH", EXTRACTION))
    # Ce at pH 3.5 lies outside La's measured range: not used. 1.5 and 2.5 are interpolated.
    assert sorted(pts["x"]) == [1.5, 2.5]
    assert set(pts["matched"]) == {"La interpolated"}
    assert pts["beta"].tolist() == pytest.approx([10 ** 0.5, 10 ** 0.5], rel=1e-3)


def test_separation_factor_uses_reported_d_when_both_curves_report_it():
    rows = [{EL: el, "pH": p, "Distribution ratio (D)": 10 ** (p + c), FIGURE: "F"}
            for el, c in (("Nd", 0.0), ("Sm", 0.7)) for p in (1, 2, 3)]
    _, summary, _ = series.separation_factors(curves.assemble(frame(rows), "pH", EXTRACTION))
    assert summary["basis"].iloc[0] == "reported D"
    assert summary["beta_median"].iloc[0] == pytest.approx(10 ** 0.7)


def test_separation_factor_unavailable_reasons():
    mixed = [{EL: "Nd", "pH": p, "Distribution ratio (D)": 10 ** p, FIGURE: "F", "Extractant": "Cyanex 272",
              "Extractant Conc. (mM)": 500.0} for p in (1, 2, 3)]
    mixed += ph_sweep("Pr", [1, 2, 3], [20, 50, 80], **{FIGURE: "F"})
    _, summary, why = series.separation_factors(curves.assemble(frame(mixed), "pH", EXTRACTION))
    assert summary.empty and "phase ratio" in why[0]

    apart = ph_sweep("La", [1, 2], [20, 40], **{FIGURE: "F"}) + ph_sweep("Ce", [3, 4], [30, 60], **{FIGURE: "F"})
    _, summary, why = series.separation_factors(curves.assemble(frame(apart), "pH", EXTRACTION))
    assert summary.empty and "no measured pH in common" in why[0]

    twice = (ph_sweep("La", [1, 2, 3], [20, 40, 60], **{FIGURE: "F", "RRE composition (ppm)": 10.0})
             + ph_sweep("La", [1, 2, 3], [25, 45, 65], **{FIGURE: "F", "RRE composition (ppm)": 100.0})
             + ph_sweep("Ce", [1, 2, 3], [30, 50, 70], **{FIGURE: "F", "RRE composition (ppm)": 10.0}))
    _, summary, why = series.separation_factors(curves.assemble(frame(twice), "pH", EXTRACTION))
    assert summary.empty and "La has 2 curves" in why[0]


def test_adjacent_pairs_skip_y_and_name_gaps():
    df = _series_frame({"La": -7.0, "Nd": -6.0, "Y": -5.0}, phs=(1.5, 2.0, 2.5))
    _, adjacent, _ = series.separation_factors(curves.assemble(df, "pH", EXTRACTION))
    assert list(adjacent["pair"]) == ["Nd/La"] and adjacent["skips"].iloc[0] == "Ce, Pr"
    _, every, _ = series.separation_factors(curves.assemble(df, "pH", EXTRACTION), pairs="all")
    assert set(every["pair"]) == {"Nd/La", "Y/La", "Y/Nd"}


# --------------------------------------------------------------------------- #
# Slopes
# --------------------------------------------------------------------------- #

def test_slope_vs_ph_recovers_three_and_drops_the_plateau():
    phs = [1.0, 1.6, 1.8, 2.0, 2.2, 2.4, 3.0]
    df = frame(ph_sweep("Nd", phs, pct_from_logd(3 * np.asarray(phs) - 6), **{FIGURE: "F"}))
    pts, fits = slopes.analyse(curves.assemble(df, "pH", EXTRACTION), "pH")
    f = fits.iloc[0]
    assert f["slope"] == pytest.approx(3.0) and f["r2"] == pytest.approx(1.0)
    assert f["basis"] == slopes.BASIS_RATIO
    # log D -3 at pH 1.0 (0.1 %) and +3 at pH 3.0 (99.9 %) are left out
    assert f["n_fit"] == 5 and f["n_left_out"] == 2
    assert set(pts.loc[~pts["used"], "why"]) == {"outside 5–95 % extraction"}


def test_slope_uses_log_d_when_the_phase_ratio_is_known():
    phs = [1.0, 1.25, 1.5, 1.75, 2.0]
    pct = pct_from_logd(3 * np.asarray(phs) - 4.5, oa=2.0)
    df = frame(ph_sweep("Nd", phs, pct, **{FIGURE: "F", "Phase ratio (O/A)": 2.0}))
    _, fits = slopes.analyse(curves.assemble(df, "pH", EXTRACTION), "pH")
    assert fits["basis"].iloc[0] == slopes.BASIS_D
    assert fits["intercept"].iloc[0] == pytest.approx(-4.5)     # true log D, not shifted by log(O/A)


def test_no_fit_below_four_points_and_initial_ph_is_flagged():
    df = frame(ph_sweep("Nd", [1.0, 1.5, 2.0], [10, 40, 80], **{FIGURE: "F", "pH type": "initial"}))
    _, fits = slopes.analyse(curves.assemble(df, "pH", EXTRACTION), "pH")
    assert fits["slope"].isna().all()
    assert "a fit needs 4" in fits["note"].iloc[0] and "initial pH" in fits["note"].iloc[0]


def test_slope_vs_log_extractant_skips_zero_and_flags_low_excess():
    concs = [0.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    logd = [np.nan] + [2 * math.log10(c) - 3 for c in concs[1:]]
    rows = [{EL: "Pr", "pH": 3.0, "Extractant Conc. (mM)": c, "Extract%": (float(pct_from_logd(l)) if c else 0.0),
             "Molar ratio of EX/REE": c / 2, FIGURE: "F"} for c, l in zip(concs, logd)]
    pts, fits = slopes.analyse(curves.assemble(frame(rows), "Extractant Conc. (mM)", EXTRACTION), "extractant")
    f = fits.iloc[0]
    assert f["slope"] == pytest.approx(2.0)
    assert "concentration 0 has no logarithm" in set(pts["why"])
    assert "extractant/metal below 10" in f["note"]


def test_fit_line_standard_error():
    fit = slopes.fit_line([0, 1, 2, 3], [0, 1.1, 1.9, 3.0])
    b, a = np.polyfit([0, 1, 2, 3], [0, 1.1, 1.9, 3.0], 1)
    assert fit.slope == pytest.approx(b) and 0 < fit.slope_se < 0.1
    assert slopes.fit_line([0, 1, 1, 2], [0, 1, 1, 2]) is None              # 3 distinct x


# --------------------------------------------------------------------------- #
# Loading from the DB (read-only, v_current_best)
# --------------------------------------------------------------------------- #

def _commit(conn, sha, rows, citation=None):
    merge.commit_extraction(
        conn, license=License("cc-by", origin="manifest"), content_sha256=sha,
        pdf_path=f"data/incoming/{sha}.pdf", df=schema.coerce_schema(pd.DataFrame(rows)), text_endpoints=[],
        prompt_version="extraction_v15", prompt_sha256="p", model="m", qa_passed=True,
        qa_report_json="[]", raw_response="{}", doi=f"10.1/{sha}", title=f"Title of {sha}",
        tracking={"short_citation": citation, "pub_year": "2020" if citation else None},
    )


def test_load_reads_current_best_labels_papers_and_leaves_out_non_ree(conn):
    _commit(conn, "a", ph_sweep("La", [1, 2], [10, 20]) + [{EL: "Pb", "pH": 1, "Extract%": 5}], citation="Smith")
    _commit(conn, "b", ph_sweep("Nd", [1, 2], [10, 20]))
    rows = data.load(conn)
    assert rows.excluded == {"Pb": 1}
    assert set(rows.df[data.PAPER]) == {"Smith 2020", "Title of b"}
    assert list(rows.df.columns[2:]) == schema.COLUMNS


def test_load_tolerates_a_db_without_the_added_columns():
    c = sqlite3.connect(":memory:")
    cols = ", ".join(f'"{col}"' for col in schema.LAB_COLUMNS)
    c.executescript(f"""
        CREATE TABLE papers (paper_id INTEGER, short_citation TEXT, pub_year TEXT, title TEXT);
        CREATE TABLE ex (paper_id INTEGER, {cols});
        CREATE VIEW v_current_best AS SELECT * FROM ex;
        INSERT INTO papers VALUES (1, 'Swain & Otu', '2011', NULL);
        INSERT INTO ex (paper_id, "{EL}", pH, "Extract%") VALUES (1, 'La', 1.0, 5.0);
    """)
    rows = data.load(c)
    assert rows.df[FIGURE].isna().all() and rows.df[data.PAPER].iloc[0] == "Swain & Otu 2011"


def test_load_of_an_empty_db(conn):
    rows = data.load(conn)
    assert rows.df.empty and rows.excluded == {}


# --------------------------------------------------------------------------- #
# Chart specs
# --------------------------------------------------------------------------- #

def test_an_element_keeps_its_style_whatever_else_is_shown():
    assert charts.style_indices(["Nd", "Lu"], ["Nd", "Lu"]) == [3, 13]
    assert charts.style_indices(["a", "b"], ["Nd", "Nd"]) == [0, 1]      # not one per element: by order


def test_sixteen_series_are_sixteen_distinct_styles():
    labels = [str(i) for i in range(charts.MAX_SERIES)]
    color, shape, dash = charts._scales(labels, None, dark=False)
    combos = set(zip(color.range, shape.range))
    assert len(combos) == charts.MAX_SERIES


def test_curve_chart_spec_marks_computed_points_and_grows_with_its_legend():
    df = frame([{EL: el, "pH": p, "Extract%": e, "Phase ratio (O/A)": 1.0, FIGURE: "F"}
                for el in data.LANTHANIDES for p, e in ((1, 20), (2, 50), (3, 80))])
    cs = curves.assemble(df, "pH", "D")
    spec = charts.curve_chart(cs.points, cs.curves, x_title="pH", y_title="D", log_x=False,
                              log_y=True, percent=False).to_dict()
    lines, filled, hollow = spec["layer"]
    assert filled["mark"]["filled"] is True and hollow["mark"]["filled"] is False
    assert hollow["transform"][0]["filter"] == "datum.derived"
    assert spec["layer"][0]["encoding"]["y"]["scale"]["type"] == "log"
    assert spec["height"] > charts.PLOT_HEIGHT + 60                     # 14-entry legend added on
