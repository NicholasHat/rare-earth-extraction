"""validation.report: row-level flags round-trip, and the review tier."""
from validation.report import QAReport, Severity, review_tier


def test_flag_rows_round_trip_and_flagged_rows_union():
    r = QAReport()
    r.add("off_curve", Severity.AMBER, "x", rows=[7, 3])
    r.add("duplicate_rows", Severity.AMBER, "y", rows=(3, 9))
    r.add("monotonicity", Severity.AMBER, "series-level, names no rows")
    assert r.flagged_rows == [3, 7, 9]
    back = QAReport.from_json(r.to_json())
    assert [f.rows for f in back.flags] == [(7, 3), (3, 9), ()]
    assert back.flagged_rows == [3, 7, 9]


def test_from_json_accepts_reports_written_before_rows_existed():
    r = QAReport.from_json('[{"check": "vocabulary", "severity": "amber", "message": "m"}]')
    assert r.flags[0].rows == () and r.flagged_rows == []


def test_review_tier_fast_track_needs_anchor_and_green():
    assert review_tier(Severity.GREEN, has_anchor=True, is_raster=False).label == "Fast track"
    assert review_tier(Severity.AMBER, has_anchor=True, is_raster=False).label == "Standard"
    assert review_tier(Severity.GREEN, has_anchor=False, is_raster=False).label == "Standard"


def test_review_tier_raster_or_red_is_full_review():
    assert review_tier(Severity.GREEN, has_anchor=True, is_raster=True).label == "Full review"
    assert review_tier(Severity.RED, has_anchor=True, is_raster=False).label == "Full review"
