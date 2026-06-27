"""Pillar C tool-calling agent (README §8).

Thin @beta_tool wrappers around assistant/tools.py's plain functions, wired
through the Anthropic SDK's tool runner (which handles the agentic loop —
calling tools and feeding results back — automatically). Stateless per call:
`run_turn` takes the full message history and returns the updated history
plus the assistant's final text reply, so the caller (the Streamlit chat
page) owns persistence across turns.
"""
from __future__ import annotations

from pathlib import Path

import anthropic
from anthropic import beta_tool

import config
from database import connection

from . import tools as _tools

_SYSTEM_PROMPT = (Path(__file__).resolve().parent / "system_prompt.md").read_text()

# Chat is lightweight Q&A/tool-calling, not a long-output extraction job —
# keep this well under the extraction pipeline's budget.
_MAX_TOKENS = 4096
_MAX_ITERATIONS = 8


@beta_tool
def query_database(sql: str) -> str:
    """Run a single read-only SQL SELECT against the master database.

    Args:
        sql: A single SELECT statement (optionally a WITH/CTE ending in one).
            Query v_current_best for "what data do we have" questions, never
            a raw table, so superseded prompt-version rows never appear.
            Allowed tables: v_current_best, papers, text_endpoints, prompt_runs.
    """
    conn = connection.get_readonly_conn()
    try:
        return _tools.query_database(conn, sql)
    finally:
        conn.close()


@beta_tool
def list_extractants() -> str:
    """List the distinct extractant names already present in the approved dataset."""
    conn = connection.get_readonly_conn()
    try:
        return _tools.list_extractants(conn)
    finally:
        conn.close()


@beta_tool
def calculator(
    operation: str,
    element: str | None = None,
    ppm: float | None = None,
    mM: float | None = None,
    ree_mM: float | None = None,
    molar_ratio_ex_per_ree: float | None = None,
    extractant_mM: float | None = None,
    target_mmol: float | None = None,
    conc_mM: float | None = None,
    volume_mL: float | None = None,
) -> str:
    """Deterministic REE solvent-extraction unit conversions. Never do this math in prose.

    Args:
        operation: one of "ppm_to_mM", "mM_to_ppm", "extractant_conc_from_ratio",
            "molar_ratio_from_conc", "volume_for_target_moles", "mmol_in_volume",
            "mass_mg_in_volume".
        element: REE element symbol (e.g. "La"); required for any operation that
            needs an atomic-mass lookup (ppm_to_mM, mM_to_ppm, mass_mg_in_volume).
        ppm: feed concentration in ppm (mg/L) — for ppm_to_mM.
        mM: a concentration in mM — for mM_to_ppm, mmol_in_volume, mass_mg_in_volume.
        ree_mM: REE concentration in mM — for extractant_conc_from_ratio.
        molar_ratio_ex_per_ree: target EX/REE molar ratio — for extractant_conc_from_ratio.
        extractant_mM: extractant concentration in mM — for molar_ratio_from_conc.
        target_mmol: a target absolute amount in mmol — for volume_for_target_moles.
        conc_mM: a concentration in mM — for volume_for_target_moles, mmol_in_volume,
            mass_mg_in_volume.
        volume_mL: a solution volume in mL — for mmol_in_volume, mass_mg_in_volume.
    """
    return _tools.calculator(
        operation,
        element=element,
        ppm=ppm,
        mM=mM,
        ree_mM=ree_mM,
        molar_ratio_ex_per_ree=molar_ratio_ex_per_ree,
        extractant_mM=extractant_mM,
        target_mmol=target_mmol,
        conc_mM=conc_mM,
        volume_mL=volume_mL,
    )


def run_turn(history: list[dict], user_message: str, *, model: str | None = None) -> tuple[str, list[dict]]:
    """Run one user turn through the agent. Returns (reply_text, updated_history)."""
    client = anthropic.Anthropic()
    messages = [*history, {"role": "user", "content": user_message}]

    runner = client.beta.messages.tool_runner(
        model=model or config.EXTRACTION_MODEL,
        max_tokens=_MAX_TOKENS,
        max_iterations=_MAX_ITERATIONS,
        system=[{"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        tools=[query_database, list_extractants, calculator],
        messages=messages,
    )
    final = runner.until_done()

    # The runner accumulates the full message history (including tool_use/
    # tool_result blocks) on `_params["messages"]`; there's no public accessor
    # for it yet in this beta, so we read the private attribute directly.
    updated_history = runner._params["messages"]

    reply = "\n".join(block.text for block in final.content if block.type == "text").strip()
    return reply, updated_history
