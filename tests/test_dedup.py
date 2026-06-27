"""ingestion.dedup: hash and DOI both catch an already-present paper."""
from database import papers_repo
from ingestion import dedup


def _seed(conn):
    return papers_repo.insert(
        conn,
        content_sha256="abc123",
        pdf_path="data/incoming/abc123.pdf",
        doi="10.1234/test.doi",
        title="A paper",
    )


def test_no_match_returns_none(conn):
    _seed(conn)
    assert dedup.find_existing(conn, "different-hash", "10.9999/other") is None


def test_match_by_content_hash(conn):
    pid = _seed(conn)
    match = dedup.find_existing(conn, "abc123", None)
    assert match is not None
    assert match.paper_id == pid
    assert match.reason == "content_hash"


def test_match_by_doi_when_hash_differs(conn):
    pid = _seed(conn)
    match = dedup.find_existing(conn, "new-file-hash", "10.1234/TEST.DOI")  # case-insensitive
    assert match is not None
    assert match.paper_id == pid
    assert match.reason == "doi"
