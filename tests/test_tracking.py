"""Tests for the paper tracking sheet: export naming, the reviewer-entered
tracking fields on `papers`, and browse.tracking_sheet's spreadsheet shape."""
import pandas as pd
import pytest

from database import browse, merge, naming, papers_repo
from validation import schema


# --------------------------------------------------------------------------- #
# naming.export_filename / naming.doi_link
# --------------------------------------------------------------------------- #

def test_export_filename_matches_tracking_sheet_convention():
    assert naming.export_filename("Swain & Otu", 1, 322) == "Swain_&_Otu_322.csv"


def test_export_filename_strips_hostile_characters():
    assert naming.export_filename('a/b\\c:d*e?f"g<h>i|j', 1, 5) == "abcdefghij_5.csv"


def test_export_filename_collapses_whitespace():
    assert naming.export_filename("  Li   et  al. ", 1, 10) == "Li_et_al._10.csv"


def test_export_filename_falls_back_to_paper_id():
    assert naming.export_filename(None, 7, 12) == "paper_7_12.csv"
    assert naming.export_filename("  ", 7, 12) == "paper_7_12.csv"


def test_doi_link_prefixes_bare_doi_and_keeps_urls():
    assert naming.doi_link("10.1016/j.seppur.2011.09.015") == (
        "https://doi.org/10.1016/j.seppur.2011.09.015"
    )
    assert naming.doi_link("https://doi.org/10.1/x") == "https://doi.org/10.1/x"
    assert naming.doi_link(None) is None


# --------------------------------------------------------------------------- #
# papers_repo.update_tracking + merge integration
# --------------------------------------------------------------------------- #

TRACKING = {
    "short_citation": "Swain & Otu",
    "pub_year": "2011",
    "figures_used": "Fig. 2, Fig. 4",
    "known_issues": "La 0.5 M point below its 0.25 M point",
    "short_description": "Competitive extraction of 14 lanthanides with Cyanex 272.",
}


def _sample_df(n=3):
    rows = [
        {
            schema.ELEMENT_COLUMN: "La",
            "pH": 1.0 + i,
            "Extract%": 10.0 * (i + 1),
            "Extractant": "Cyanex 272",
            "Extractant type": "phosphinic acid based",
        }
        for i in range(n)
    ]
    return schema.coerce_schema(pd.DataFrame(rows))


def _commit(conn, tracking=None, prompt_version="extraction_v9"):
    return merge.commit_extraction(
        conn,
        content_sha256="hash1",
        pdf_path="data/incoming/hash1.pdf",
        df=_sample_df(),
        text_endpoints=[],
        prompt_version=prompt_version,
        prompt_sha256="psha",
        model="claude-sonnet-5",
        qa_passed=True,
        qa_report_json="[]",
        raw_response="{}",
        doi="10.1016/j.seppur.2011.09.015",
        reference_no="1",
        title="A Test Paper",
        tracking=tracking,
    )


def test_update_tracking_rejects_unknown_fields(conn):
    s = _commit(conn)
    with pytest.raises(ValueError):
        papers_repo.update_tracking(conn, s["paper_id"], title="nope")


def test_merge_persists_tracking_fields(conn):
    s = _commit(conn, tracking=TRACKING)
    row = papers_repo.get(conn, s["paper_id"])
    for field, value in TRACKING.items():
        assert row[field] == value


def test_reapproval_with_blank_tracking_keeps_earlier_values(conn):
    _commit(conn, tracking=TRACKING)
    s = _commit(conn, tracking={k: "" for k in TRACKING}, prompt_version="extraction_v10")
    row = papers_repo.get(conn, s["paper_id"])
    assert row["short_citation"] == "Swain & Otu"
    assert row["short_description"] == TRACKING["short_description"]


def test_reapproval_with_new_values_updates_them(conn):
    _commit(conn, tracking=TRACKING)
    s = _commit(
        conn,
        tracking={**TRACKING, "known_issues": "resolved on re-extraction"},
        prompt_version="extraction_v10",
    )
    row = papers_repo.get(conn, s["paper_id"])
    assert row["known_issues"] == "resolved on re-extraction"
    assert row["pub_year"] == "2011"


# --------------------------------------------------------------------------- #
# browse.tracking_sheet
# --------------------------------------------------------------------------- #

def test_tracking_sheet_empty_has_spreadsheet_columns(conn):
    df = browse.tracking_sheet(conn)
    assert df.empty
    assert list(df.columns) == browse.TRACKING_SHEET_COLUMNS


def test_tracking_sheet_matches_spreadsheet_layout(conn):
    _commit(conn, tracking=TRACKING)
    df = browse.tracking_sheet(conn)
    assert list(df.columns) == browse.TRACKING_SHEET_COLUMNS
    row = df.iloc[0]
    assert row["Ref No."] == "1"
    assert row["Status"] == "Extracted"
    assert row["Short citation"] == "Swain & Otu"
    assert row["Year"] == "2011"
    assert row["DOI / link"] == "https://doi.org/10.1016/j.seppur.2011.09.015"
    assert row["Output file"] == "Swain_&_Otu_3.csv"
    assert row["Figures / tables used"] == "Fig. 2, Fig. 4"
    assert row["Rows extracted"] == 3
    assert row["No. of elements"] == 1
    assert row["Extractant"] == "Cyanex 272"
    assert row["Extractant type"] == "phosphinic acid based"
    assert row["Known issues / caveats"] == TRACKING["known_issues"]
    assert row["Short description"] == TRACKING["short_description"]


def test_tracking_sheet_shows_unapproved_paper_as_uploaded(conn):
    papers_repo.insert(
        conn, content_sha256="orphan", pdf_path="data/incoming/orphan.pdf", doi="10.1/orphan"
    )
    df = browse.tracking_sheet(conn)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Status"] == "Uploaded"
    assert pd.isna(row["Rows extracted"])
    assert row["Output file"] is None
