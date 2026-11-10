"""Only openly-licensed papers reach the DB: the manifest lookup, the licence
ids, and the merge gate (database/merge.py) — plus the daily backup the
gate's write path takes."""
import sqlite3

import pandas as pd
import pytest

import config
from database import extractions_repo, merge, papers_repo
from ingestion import licensing, manifest
from ingestion.licensing import License
from validation import schema

HEADER = ("sha256,doi,title,year,venue,license,license_evidence,oa_status,source_url,pdf_path,figure_format,"
          "extractants,elements,facets,n_data_figures,n_data_tables,quality_score,tier,first_batch,notes\n")


def write_manifest(path, rows):
    """rows: dicts of the columns that matter; the rest are left blank."""
    lines = [HEADER]
    for r in rows:
        cols = HEADER.strip().split(",")
        lines.append(",".join(str(r.get(c, "")) for c in cols) + "\n")
    path.write_text("".join(lines))
    return path


def test_manifest_rows_and_licence_lookup(tmp_path):
    m = write_manifest(tmp_path / "manifest.csv", [
        {"sha256": "A" * 64, "license": "cc-by", "license_evidence": "footer: CC BY 4.0",
         "source_url": "https://doi.org/10.1/x", "tier": "A", "first_batch": "true", "quality_score": "0.9",
         "title": "Paper A", "pdf_path": "pdfs/" + "a" * 64 + ".pdf"},
        {"sha256": "b" * 64, "license": "elsevier-oa", "tier": "reject", "first_batch": ""},
    ])
    (a, b) = manifest.read(m)
    assert a.sha256 == "a" * 64 and a.first_batch and a.quality_score == 0.9 and a.tier == "A"
    assert a.pdf == tmp_path / "pdfs" / ("a" * 64 + ".pdf") and not b.first_batch
    assert licensing.from_manifest("a" * 64, m) == License("cc-by", "footer: CC BY 4.0", "https://doi.org/10.1/x",
                                                          "manifest")
    assert licensing.from_manifest("c" * 64, m) is None
    assert not licensing.is_open(licensing.from_manifest("b" * 64, m))
    assert manifest.by_sha(tmp_path / "missing.csv") == {}
    with pytest.raises(manifest.ManifestError):
        (tmp_path / "bad.csv").write_text("sha,licence\n")
        manifest.read(tmp_path / "bad.csv")


def test_an_unverified_manifest_licence_does_not_count(tmp_path):
    m = write_manifest(tmp_path / "manifest.csv", [
        {"sha256": "d" * 64, "license": "cc-by", "tier": "unverified"},
    ])
    assert licensing.from_manifest("d" * 64, m) is None


@pytest.mark.parametrize("raw, lid, is_open", [
    ("CC BY 4.0", "cc-by", True), ("cc-by-nc-nd", "cc-by-nc-nd", True), ("CC0 1.0", "cc0", True),
    ("Public Domain", "public-domain", True), ("cc_by_sa", "cc-by-sa", True),
    ("Elsevier user license", "elsevier-user-license", False), ("", None, False),
])
def test_licence_ids_normalise(raw, lid, is_open):
    assert licensing.normalise(raw) == lid
    assert licensing.is_open(raw) is is_open


def _df():
    return schema.coerce_schema(pd.DataFrame(
        [{schema.ELEMENT_COLUMN: "La", "pH": 1.0 + i, "Extract%": 10.0 * i, "Figure/Table": "Fig. 1"}
         for i in range(3)]))


def _commit(conn, sha="h1", **kw):
    return merge.commit_extraction(
        conn, content_sha256=sha, pdf_path=f"{sha}.pdf", df=_df(), text_endpoints=[],
        prompt_version="extraction_v15", prompt_sha256="p", model="m", qa_passed=True,
        qa_report_json="[]", raw_response="{}", **kw)


def test_merge_refuses_a_paper_without_an_open_licence_and_writes_nothing(conn):
    with pytest.raises(merge.LicenseRequired, match="no recorded licence"):
        _commit(conn)
    with pytest.raises(merge.LicenseRequired, match="licence 'elsevier'"):
        _commit(conn, license=License("elsevier"))
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0


def test_merge_records_the_licence_and_reuses_it_on_a_re_approval(conn):
    _commit(conn, license=License("cc-by-nc", "page footer", "https://x", "manifest"))
    row = papers_repo.find_by_hash(conn, "h1")
    assert papers_repo.get_license(row) == License("cc-by-nc", "page footer", "https://x", "manifest")
    _commit(conn)     # a re-extraction approved later: the paper's recorded licence stands
    assert extractions_repo.count_current_best(conn) == 3
    assert list(extractions_repo.current_best(conn).columns[3:3 + len(schema.COLUMNS)]) == schema.COLUMNS


def test_a_paper_merged_before_licences_keeps_its_rows_and_can_be_licensed_by_hand(conn):
    """Swain & Otu 2011: in the DB with no licence. Nothing about it changes
    until the owner records one; then a re-approval is allowed."""
    conn.execute("INSERT INTO papers (content_sha256, pdf_path, short_citation) VALUES ('swain', 's.pdf', 'Swain & Otu')")
    with pytest.raises(merge.LicenseRequired):
        _commit(conn, sha="swain")
    (paper_id,) = conn.execute("SELECT paper_id FROM papers WHERE content_sha256 = 'swain'").fetchone()
    papers_repo.set_license(conn, paper_id, License.of("CC BY", "checked by hand", origin="manual"))
    _commit(conn, sha="swain")
    assert tuple(conn.execute("SELECT license, license_origin FROM papers").fetchone()) == ("cc-by", "manual")


def test_the_first_merge_of_the_day_backs_the_db_up(tmp_path):
    from database import connection

    db = tmp_path / "master.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.executescript((config.ROOT / "database" / "schema.sql").read_text())
    _commit(conn, license=License("cc-by"))
    backups = list((tmp_path / "backups").glob("master-*.db"))
    assert len(backups) == 1
    assert sqlite3.connect(backups[0]).execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0   # before the merge
    _commit(conn, sha="h2", license=License("cc-by"))
    assert list((tmp_path / "backups").glob("master-*.db")) == backups                         # once a day
    assert connection.backup_once_a_day(sqlite3.connect(":memory:")) is None
