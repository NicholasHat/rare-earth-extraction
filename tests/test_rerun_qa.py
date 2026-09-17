"""runner.rerun_qa: today's checks applied to a staged result, nothing else touched."""
import pandas as pd

from extraction.runner import ExtractionResult, rerun_qa
from validation import schema
from validation.report import QAReport

EL = schema.ELEMENT_COLUMN


def _result(rows):
    return ExtractionResult(
        df=schema.coerce_schema(pd.DataFrame(rows)), text_endpoints=[], qa_report=QAReport(),
        prompt_version="extraction_v9", prompt_sha256="abc", model="m", raw_response="{}",
        coercion_failures=0, curve_analysis="prepass", deterministic_counts=[12],
        input_tokens=1, output_tokens=2, cache_creation_input_tokens=3, cache_read_input_tokens=4,
    )


def test_rerun_qa_replaces_only_the_report():
    rows = [{EL: "Lu", "Extractant": "EHEHPA", "Extractant Conc. (mM)": 1000.0, "pH": 1 + 0.15 * i,
             "Extract%": 100 / (1 + 10 ** (-3 * (1 + 0.15 * i - 1.8)))} for i in range(12)]
    rows[5]["Extract%"] = 97.0   # an off-curve point the stale (empty) report knows nothing about
    stale = _result(rows)
    fresh = rerun_qa(stale, figure_is_curve=True)
    assert fresh.qa_report.flagged_rows == [6]
    assert stale.qa_report.flags == []                     # input untouched
    assert fresh.df is stale.df and fresh.curve_analysis == "prepass" and fresh.cache_read_input_tokens == 4
