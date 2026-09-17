"""validation.report: row-level flags round-trip."""
from validation.report import QAReport, Severity


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

