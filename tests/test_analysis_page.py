"""pages/4_Analysis.py renders end to end (Streamlit AppTest) on a scratch DB:
empty, a pre-v15 paper with no source figures, and v15 rows with D, phase
ratio and both pH types. No API calls."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
from database import connection, merge
from ingestion.licensing import License
from validation import schema

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "4_Analysis.py")
EL = schema.ELEMENT_COLUMN
FIGURE = schema.FIGURE_COLUMN


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "master.db")
    for name in ("INCOMING_DIR", "STAGING_DIR", "EXPORTS_DIR", "TRANSCRIPTS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    return tmp_path / "master.db"


def _commit(sha, rows, citation):
    conn = connection.get_conn()
    try:
        merge.commit_extraction(
            conn, license=License("cc-by", origin="manifest"), content_sha256=sha,
            pdf_path=f"data/incoming/{sha}.pdf", df=schema.coerce_schema(pd.DataFrame(rows)),
            text_endpoints=[], prompt_version="extraction_v15", prompt_sha256="p", model="m",
            qa_passed=True, qa_report_json="[]", raw_response="{}", doi=f"10.1/{sha}",
            tracking={"short_citation": citation, "pub_year": "2020"},
        )
    finally:
        conn.close()


def _pct(logd, oa=1.0):
    d = 10.0 ** np.asarray(logd, float)
    return 100.0 * d / (d + 1.0 / oa)


def _run():
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _charts(at):
    found = []

    def walk(node):
        if getattr(node, "type", None) == "vega_lite_chart":
            found.append(node)
        children = getattr(node, "children", None)
        for child in (children.values() if isinstance(children, dict) else (children or [])):
            walk(child)
    walk(at._tree)
    return found


def test_empty_db_says_so_in_a_sentence(db):
    at = _run()
    assert "No approved data yet" in at.info[0].value
    assert not _charts(at)


def test_pre_v15_paper_without_figures(db):
    # Swain & Otu shape: a pH sweep at 500 mM plus a concentration sweep at pH 1.75.
    rows = []
    for el, c in (("La", -5.0), ("Ce", -4.6), ("Nd", -4.2)):
        rows += [{EL: el, "pH": p, "Extract%": float(_pct(2.5 * p + c)), "Extractant": "Cyanex 272",
                  "Extractant Conc. (mM)": 500.0} for p in np.arange(1.0, 3.01, 0.25)]
        rows += [{EL: el, "pH": 1.75, "Extract%": float(_pct(2 * np.log10(m) + c - 1)), "Extractant": "Cyanex 272",
                  "Extractant Conc. (mM)": m} for m in (50.0, 100.0, 250.0, 500.0, 1000.0)]
    _commit("swain", rows, "Swain & Otu")
    at = _run()
    captions = " ".join(c.value for c in at.caption)
    assert "Source figure not recorded for Swain & Otu 2020" in captions
    assert "D is not offered" in captions
    assert len(_charts(at)) == 4         # curves, pH½, β, slopes (compare needs two papers)
    assert "two or more papers" in at.info[0].value


def test_v15_rows_with_d_phase_ratio_and_both_ph_types(db):
    rows = []
    for el, c in (("Pr", -4.0), ("Nd", -3.6)):
        for kind, fig in (("equilibrium", "Fig. 2"), ("initial", "Fig. 3")):
            rows += [{EL: el, "pH": p, "Extract%": float(_pct(3 * p + c, oa=0.5)), "Phase ratio (O/A)": 0.5,
                      "Extractant": "Cyanex 272", "Extractant Conc. (mM)": 200.0, "Diluent": "kerosene",
                      "pH type": kind, FIGURE: fig} for p in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8)]
    _commit("v15a", rows, "Lee")
    _commit("v15b", [{**r, "Diluent": "n-heptane"} for r in rows if r["pH type"] == "equilibrium"], "Kim")
    at = _run()
    assert at.radio(key="curves_y").options == ["Extraction (%)", "D"]
    at.radio(key="curves_y").set_value("D").run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "Hollow markers: D computed from %E and the phase ratio" in captions
    md = " ".join(m.value for m in at.markdown)
    assert "Equilibrium pH" in md and "Initial pH" in md       # split, never one axis
    assert at.selectbox(key="cmp_pair").options                 # Pr/Nd with Cyanex 272 in two papers
    assert len(at.get("download_button")) >= 4
