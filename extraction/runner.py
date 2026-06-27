"""Orchestrate one extraction run end to end (README §6, Phase A1).

  load pinned prompt -> call Anthropic API -> parse output -> run QA checks

Returns an ExtractionResult for the review UI. It does NOT touch the DB — merge
happens only on human approval (database.merge.commit_extraction).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
from validation import checks
from validation.report import QAReport

from . import anthropic_client, parse_output, prompt_loader


@dataclass
class ExtractionResult:
    df: pd.DataFrame
    text_endpoints: list[dict]
    qa_report: QAReport
    prompt_version: str
    prompt_sha256: str
    model: str
    raw_response: str
    coercion_failures: int


def extract_paper(
    pdf_bytes: bytes,
    *,
    figure_is_curve: bool = True,
    prompt_version: str | None = None,
    model: str | None = None,
) -> ExtractionResult:
    """Run extraction + QA on one PDF's bytes."""
    bundle = prompt_loader.load_prompt(prompt_version)
    model = model or config.EXTRACTION_MODEL

    raw = anthropic_client.extract(bundle.text, pdf_bytes, model=model)
    parsed = parse_output.parse(raw)

    report = checks.run(
        parsed.df,
        parsed.text_endpoints,
        figure_is_curve=figure_is_curve,
        coercion_failures=parsed.coercion_failures,
    )

    return ExtractionResult(
        df=parsed.df,
        text_endpoints=parsed.text_endpoints,
        qa_report=report,
        prompt_version=bundle.version,
        prompt_sha256=bundle.sha256,
        model=model,
        raw_response=parsed.raw_text,
        coercion_failures=parsed.coercion_failures,
    )
