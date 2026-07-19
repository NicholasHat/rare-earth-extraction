"""Thin wrapper over the Anthropic Messages API for figure extraction.

Gives Claude the paper PDF two ways in the same turn: a `document` block (so
it can visually read the figure — legend colours, marker shapes, panel
layout) and a `container_upload` (so the code-execution tool can open the
same file with pdfplumber/numpy for vector/raster detection, axis
calibration, and point digitization, per the prompt's Steps 2-6). Both need
the PDF uploaded once via the Files API first.

Two ways to run an extraction, sharing the same request shape (_message_kwargs):
  - `extract()` — synchronous, streamed (a fully-digitized multi-element,
    multi-figure extraction plus the code-execution transcript is a large
    output; non-streaming would risk the SDK's HTTP timeout).
  - `submit_batch()` / `poll_batch_status()` / `collect_batch_results()` — the
    Message Batches API, 50% cheaper and asynchronous. See the flag on that
    section below before relying on it for a full run.
"""
from __future__ import annotations

from dataclasses import dataclass

import anthropic

import config

_BETAS = ["files-api-2025-04-14", "task-budgets-2026-03-13"]
_CODE_EXECUTION_TOOL = {"type": "code_execution_20260120", "name": "code_execution"}


@dataclass(frozen=True)
class ExtractResponse:
    """One extraction call's text output plus the token usage it billed."""

    text: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


def _usage_from_message(message) -> tuple[int, int, int, int]:
    u = message.usage
    return (
        u.input_tokens,
        u.output_tokens,
        getattr(u, "cache_creation_input_tokens", None) or 0,
        getattr(u, "cache_read_input_tokens", None) or 0,
    )

# A short instruction in the user turn; the real rules live in the system prompt.
_USER_INSTRUCTION = (
    "Extract the data from this paper according to your instructions. The same "
    "PDF is also available in your code execution environment — list the "
    "working directory to find it, install any package you need, and use "
    "pdfplumber/numpy there for axis calibration and point digitization as "
    "Steps 2-6 describe. Return only the single JSON object described in the "
    "OUTPUT CONTRACT."
)


def _build_user_content(file_id: str, analysis_block: str | None) -> list[dict]:
    content: list[dict] = [
        {"type": "document", "source": {"type": "file", "file_id": file_id}},
        {"type": "container_upload", "file_id": file_id},
    ]
    # Inject the deterministic curve pre-pass (plan §6) before the instruction so
    # the model treats the authoritative marker counts as a grounding anchor.
    if analysis_block:
        content.append({"type": "text", "text": analysis_block})
    # Cache breakpoint: the code-execution tool loop re-sends this whole turn
    # (system prompt + this paper's PDF) on every internal iteration. Without
    # this marker only the system prompt is cached (its own breakpoint above)
    # and the PDF gets rebilled at full price on every iteration; with it, the
    # PDF is written to cache once and read back cheaply on every iteration
    # after the first (see prompts/CHANGELOG.md, extraction_v7).
    content.append({
        "type": "text",
        "text": _USER_INSTRUCTION,
        "cache_control": {"type": "ephemeral"},
    })
    return content


def _message_kwargs(prompt_text: str, file_id: str, *, model: str, analysis_block: str | None) -> dict:
    """Build the model-call kwargs shared by the synchronous and Batch API paths."""
    return dict(
        model=model,
        max_tokens=128000,
        thinking={"type": "adaptive"},
        # The extraction prompt is identical across every paper in a batch;
        # cache it so only the first call in a run pays full input price for
        # it (1h TTL since each call's own runtime, or a Batches job's queue
        # time, can exceed the 5min default).
        system=[
            {
                "type": "text",
                "text": prompt_text,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        tools=[_CODE_EXECUTION_TOOL],
        # Loose backstop, not a hard cap (that's max_tokens): the model sees
        # a running countdown across the whole tool loop and self-moderates
        # instead of narrating trial-and-error indefinitely.
        output_config={
            "task_budget": {
                "type": "tokens",
                "total": config.EXTRACTION_TASK_BUDGET_TOKENS,
            }
        },
        messages=[{"role": "user", "content": _build_user_content(file_id, analysis_block)}],
    )


def _response_from_message(message) -> ExtractResponse:
    text = "\n".join(block.text for block in message.content if block.type == "text").strip()
    input_tokens, output_tokens, cache_creation, cache_read = _usage_from_message(message)
    return ExtractResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


def extract(
    prompt_text: str,
    pdf_bytes: bytes,
    *,
    model: str | None = None,
    analysis_block: str | None = None,
) -> ExtractResponse:
    """Run one synchronous extraction. Returns the model's text output plus its
    token usage.

    `analysis_block` is the optional deterministic curve pre-pass text
    (extraction/curve_prepass.py) injected into the user turn as a count anchor.
    """
    model = model or config.EXTRACTION_MODEL
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    uploaded = client.beta.files.upload(file=("paper.pdf", pdf_bytes, "application/pdf"))
    try:
        kwargs = _message_kwargs(prompt_text, uploaded.id, model=model, analysis_block=analysis_block)
        with client.beta.messages.stream(betas=_BETAS, **kwargs) as stream:
            message = stream.get_final_message()
    finally:
        client.beta.files.delete(uploaded.id)

    return _response_from_message(message)


# --------------------------------------------------------------------------- #
# Message Batches API — 50% cheaper token pricing; asynchronous (usually
# minutes, up to 24h).
#
# ASSUMPTION FLAGGED FOR VERIFICATION: the public docs describe code
# execution, Files API document blocks, and task budgets each independently,
# but not this specific combination running inside a *batched* (non-
# streaming, asynchronously processed) request. Test on 1-2 papers before
# relying on this for a full run — see README §"Building next" / the Batch
# API entry in prompts/CHANGELOG.md.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BatchSubmission:
    batch_id: str
    file_ids: dict[str, str]   # custom_id -> uploaded Files API id (cleanup after collection)


def submit_batch(
    items: list[tuple[str, str, bytes, str | None]], *, model: str | None = None
) -> BatchSubmission:
    """Upload each paper's PDF and submit one Batches API job covering all of them.

    `items` is a list of (custom_id, prompt_text, pdf_bytes, analysis_block).
    `custom_id` must be unique within the batch — callers use the paper's
    content sha256.
    """
    model = model or config.EXTRACTION_MODEL
    client = anthropic.Anthropic()

    file_ids: dict[str, str] = {}
    requests = []
    for custom_id, prompt_text, pdf_bytes, analysis_block in items:
        uploaded = client.beta.files.upload(file=("paper.pdf", pdf_bytes, "application/pdf"))
        file_ids[custom_id] = uploaded.id
        kwargs = _message_kwargs(prompt_text, uploaded.id, model=model, analysis_block=analysis_block)
        requests.append({"custom_id": custom_id, "params": kwargs})

    batch = client.beta.messages.batches.create(betas=_BETAS, requests=requests)
    return BatchSubmission(batch_id=batch.id, file_ids=file_ids)


def poll_batch_status(batch_id: str) -> str:
    """Return the batch's processing_status ('in_progress' | 'ended' | ...)."""
    client = anthropic.Anthropic()
    return client.beta.messages.batches.retrieve(batch_id).processing_status


def collect_batch_results(batch_id: str) -> dict[str, ExtractResponse | Exception]:
    """Fetch results once the batch has ended. Keyed by custom_id.

    A result that errored/canceled/expired is surfaced as a RuntimeError value
    rather than raised, so one bad paper doesn't lose the rest of the batch.
    """
    client = anthropic.Anthropic()
    out: dict[str, ExtractResponse | Exception] = {}
    for result in client.beta.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            out[result.custom_id] = _response_from_message(result.result.message)
        else:
            out[result.custom_id] = RuntimeError(
                f"batch item {result.custom_id!r} did not succeed: {result.result.type}"
            )
    return out


def cleanup_batch_files(file_ids: dict[str, str]) -> None:
    """Delete the Files API uploads made for a batch, once results are collected."""
    client = anthropic.Anthropic()
    for file_id in file_ids.values():
        try:
            client.beta.files.delete(file_id)
        except Exception:
            pass
