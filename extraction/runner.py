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

from . import anthropic_client, curve_prepass, parse_output, prompt_loader


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
    curve_analysis: str = ""          # the injected deterministic pre-pass block
    deterministic_counts: list[int] = None  # authoritative per-series marker counts
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0  # high value here confirms the cache breakpoint is working


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

    # Deterministic pre-pass: count markers from the PDF's own vector geometry and
    # inject the result as a grounding anchor (plan §6). Pure / no API; a failure
    # here must never block the extraction, so it degrades to "no anchor".
    try:
        prepass = curve_prepass.analyze(pdf_bytes)
        analysis_block = prepass.to_prompt_block()
        deterministic_counts = prepass.authoritative_counts
    except Exception:
        analysis_block, deterministic_counts = "", []

    response = anthropic_client.extract(
        bundle.text, pdf_bytes, model=model, analysis_block=analysis_block or None
    )
    parsed = parse_output.parse(response.text)

    report = checks.run(
        parsed.df,
        parsed.text_endpoints,
        figure_is_curve=figure_is_curve,
        coercion_failures=parsed.coercion_failures,
        deterministic_counts=deterministic_counts,
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
        curve_analysis=analysis_block,
        deterministic_counts=deterministic_counts,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_creation_input_tokens=response.cache_creation_input_tokens,
        cache_read_input_tokens=response.cache_read_input_tokens,
    )
