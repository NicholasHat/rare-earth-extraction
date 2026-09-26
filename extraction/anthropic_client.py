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
    output; non-streaming would risk the SDK's HTTP timeout). Automatically
    resumes through `stop_reason="pause_turn"` (the server-side code-execution
    loop's default 10-internal-iteration cap) by re-sending the assistant's
    own partial response, per the documented continuation pattern — a rich
    multi-element paper can legitimately need more than 10 iterations.
  - `submit_batch()` / `poll_batch_status()` / `collect_batch_results()` — the
    Message Batches API, 50% cheaper and asynchronous. Gets a higher per-turn
    iteration cap than the sync path, so most papers never pause here; a
    `pause_turn` result is transparently finished off with a synchronous
    continuation (`_continue_until_done`) rather than surfaced as an error.

`_continue_until_done` also resumes through `stop_reason="tool_use"`: the only
tool offered is server-side, so a client-side tool call can only be the model
misspelling that tool's name (seen live: `bash_code_execction`). Rather than
lose the whole extraction to a typo, it is answered with an is_error
tool_result naming the mistake and the model carries on.

Only `end_turn` counts as a finished response. Every other `stop_reason` is a
hard failure (surfaced as a specific RuntimeError, not a bare parse error) —
see `_check_stop_reason`.

Context editing (`clear_tool_uses_20250919`) was tried here and reverted: it
clears stale tool-use/tool-result pairs mid-conversation, but that clearing
breaks the prompt-cache prefix for everything downstream of it, forcing a
cache-*write* (1.25x-2x price) instead of the cache-*read* (0.1x price) this
pipeline was already getting on ~96% of its tokens. On a live paper it raised
cache_creation_input_tokens 7.7x and roughly doubled-to-tripled total cost —
worse, not better. Don't re-add it without a plan for the cache invalidation.
"""
from __future__ import annotations

from dataclasses import dataclass

import anthropic

import config

from . import sandbox_toolkit

_BETAS = ["files-api-2025-04-14", "task-budgets-2026-03-13"]
_CODE_EXECUTION_TOOL = {"type": "code_execution_20260120", "name": "code_execution"}


_USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
)


def _sum_usage(messages: list) -> dict[str, int]:
    """Token usage summed over a message chain — every message was a separately
    billed API turn (a mid-stream partial included), so the total is the bill."""
    return {
        f: sum(getattr(getattr(m, "usage", None), f, None) or 0 for m in messages)
        for f in _USAGE_FIELDS
    }


def _describe_usage(usage: dict[str, int]) -> str:
    return (
        f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out / "
        f"{usage['cache_creation_input_tokens']:,} cache-write / "
        f"{usage['cache_read_input_tokens']:,} cache-read tokens"
    )


@dataclass(frozen=True)
class ExtractResponse:
    """One extraction call's text output plus the token usage it billed."""

    text: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int

    def usage(self) -> dict[str, int]:
        return {f: getattr(self, f) for f in _USAGE_FIELDS}


class ExtractionFailed(RuntimeError):
    """An extraction failed after API money was spent. Carries the token usage
    billed up to the failure — every completed turn plus whatever a dying
    stream had already reported — so a failed run leaves a record of what it
    cost instead of vanishing with the exception (2026-09-22: a ~$7.50 sync
    run died on an exhausted credit balance and left nothing to diagnose).
    `cause` is the original error: an SDK error, one of this module's
    stop-reason RuntimeErrors, or a downstream parse failure."""

    def __init__(self, cause: Exception, usage: dict[str, int], turns_completed: int | None = None):
        self.cause = cause
        self.usage = usage
        self.turns_completed = turns_completed
        turns = f" over {turns_completed} completed API turn(s)" if turns_completed is not None else ""
        super().__init__(f"{cause} [billed before failure: {_describe_usage(usage)}{turns}]")


_STOP_REASON_ERRORS = {
    "refusal": "model declined the request (stop_reason=refusal)",
    "max_tokens": (
        "model output was truncated at max_tokens (128000) before finishing — "
        "the paper likely needs more figures/elements digitized than fit in one "
        "turn's output"
    ),
    "model_context_window_exceeded": (
        "the conversation outgrew the model's context window before finishing — "
        "the code-execution loop re-accumulates the PDF and transcript on every "
        "internal iteration, so this is most likely a paper needing many rounds"
    ),
    "pause_turn": (
        "server-side tool loop paused (stop_reason=pause_turn) with no automatic "
        "continuation available here — this paper needs more internal tool "
        "iterations than one turn allows"
    ),
    "tool_use": (
        "model kept calling a client-side tool this pipeline does not provide "
        "(stop_reason=tool_use — usually a misspelled server tool name) even "
        "after being told so"
    ),
}


def _check_stop_reason(stop_reason: str, *, can_continue: bool = False) -> None:
    """Raise a specific, actionable error for a non-finished response instead of
    letting truncated/declined output fall through to an opaque JSON parse
    failure downstream.

    Whitelist, not blacklist: only `end_turn` (and a continuable `pause_turn`)
    means the response is complete. Anything else — including a stop reason the
    API gains after this was written — raises. Falling through is not a safe
    default here: parse_output walks the fenced JSON blocks newest-first and
    returns the first that parses, so a response truncated mid-block silently
    falls back to an earlier, smaller table and reaches the reviewer looking
    like a complete extraction.
    """
    if stop_reason == "end_turn" or (stop_reason == "pause_turn" and can_continue):
        return
    raise RuntimeError(
        _STOP_REASON_ERRORS.get(stop_reason)
        or f"model stopped with an unrecognized stop_reason={stop_reason!r} before "
        "finishing — treating the response as incomplete rather than parsing it"
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


def _build_user_content(
    file_id: str,
    analysis_block: str | None,
    qa_feedback: str | None = None,
    toolkit_file_id: str | None = None,
) -> list[dict]:
    content: list[dict] = [
        {"type": "document", "source": {"type": "file", "file_id": file_id}},
        {"type": "container_upload", "file_id": file_id},
    ]
    # The digitisation toolkit (extraction/sandbox_toolkit.py) rides along as a
    # second sandbox file, described by its own guidance block. Optional only
    # so a request persisted before it existed is rebuilt exactly as sent.
    if toolkit_file_id:
        content.append({"type": "container_upload", "file_id": toolkit_file_id})
        content.append({"type": "text", "text": sandbox_toolkit.guide()})
    # Inject the deterministic curve pre-pass (plan §6) before the instruction so
    # the model treats the authoritative marker counts as a grounding anchor.
    if analysis_block:
        content.append({"type": "text", "text": analysis_block})
    # On-demand re-extraction only (review UI): the previous attempt's QA
    # failures, rendered by runner.qa_feedback_block. Per-run guidance injected
    # like the pre-pass block — never part of the pinned prompt file.
    if qa_feedback:
        content.append({"type": "text", "text": qa_feedback})
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


def _message_kwargs(
    prompt_text: str,
    file_id: str,
    *,
    model: str,
    analysis_block: str | None,
    qa_feedback: str | None = None,
    toolkit_file_id: str | None = None,
) -> dict:
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
        # instead of narrating trial-and-error indefinitely. Effort (thinking
        # depth, tool-call consolidation) only when configured — see config.
        output_config={
            "task_budget": {
                "type": "tokens",
                "total": config.EXTRACTION_TASK_BUDGET_TOKENS,
            },
            **({"effort": config.EXTRACTION_EFFORT} if config.EXTRACTION_EFFORT else {}),
        },
        messages=[{
            "role": "user",
            "content": _build_user_content(file_id, analysis_block, qa_feedback, toolkit_file_id),
        }],
    )


def _response_from_chain(messages: list) -> ExtractResponse:
    """Build the response from a message chain: one message normally, more when
    pause_turn continuations were needed. Only the final message carries the
    finished output, but every message in the chain was a separately billed
    API call, so usage is summed across all of them."""
    _check_stop_reason(messages[-1].stop_reason)
    text = "\n".join(
        block.text for block in messages[-1].content if block.type == "text"
    ).strip()
    return ExtractResponse(text=text, **_sum_usage(messages))


def _failed(cause: Exception, chain: list) -> ExtractionFailed:
    """Wrap a failure with the chain's usage so far; a stream that died part-way
    contributes its partial message (no stop_reason) but not a completed turn."""
    return ExtractionFailed(
        cause, _sum_usage(chain),
        turns_completed=sum(1 for m in chain if getattr(m, "stop_reason", None)),
    )


def _stream_turn(client: anthropic.Anthropic, kwargs: dict, chain: list) -> None:
    """Run one API turn and append its final message to `chain`. If the stream
    dies part-way, the partial message it had accumulated (with the usage
    reported so far) is appended first, so the failure still accounts for
    what was billed."""
    with client.beta.messages.stream(betas=_BETAS, **kwargs) as stream:
        try:
            chain.append(stream.get_final_message())
        except Exception:
            partial = _partial_message(stream)
            if partial is not None:
                chain.append(partial)
            raise


def _partial_message(stream):
    """The stream's accumulated message, if it got far enough to have one with
    real usage on it (the SDK raises before the first message_start event)."""
    try:
        snapshot = stream.current_message_snapshot
    except Exception:
        return None
    return snapshot if isinstance(getattr(snapshot.usage, "output_tokens", None), int) else None


# Server-side tool loops (code execution) pause with stop_reason="pause_turn"
# after a default 10 internal iterations. Bound how many times we resend and
# let it resume — a rich multi-element paper can legitimately need several
# rounds of this; an unbounded loop would not.
_MAX_CONTINUATIONS = 5


def _unknown_tool_calls(message) -> list:
    return [block for block in message.content if block.type == "tool_use"]


def _resumable(message) -> bool:
    """A response that stopped short of end_turn but can be continued in place:
    the server-side tool loop's iteration cap (pause_turn), or a client-side
    tool call — which, with no client-side tools on offer, is the model
    misspelling the server tool's name."""
    return message.stop_reason == "pause_turn" or (
        message.stop_reason == "tool_use" and bool(_unknown_tool_calls(message))
    )


def _transcript(user_content, chain: list) -> list[dict]:
    """The conversation so far, replayed for a continuation call: the original
    user turn, then every response in the chain as an assistant turn (the API
    merges consecutive assistant turns), each unknown tool call answered with
    an is_error tool_result so the model can correct itself."""
    messages = [{"role": "user", "content": user_content}]
    for message in chain:
        messages.append({"role": "assistant", "content": message.content})
        if message.stop_reason == "tool_use":
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "is_error": True,
                        "content": (
                            f"Unknown tool {call.name!r}. There are no client-side tools; "
                            "the only tool available is the server-side code execution "
                            "tool — call it by its exact name (e.g. bash_code_execution) "
                            "and continue."
                        ),
                    }
                    for call in _unknown_tool_calls(message)
                ],
            })
    return messages


def _container_id(chain: list) -> str | None:
    """The code-execution container the chain's latest turn ran in, if any."""
    for message in reversed(chain):
        container = getattr(message, "container", None)
        if getattr(container, "id", None):
            return container.id
    return None


def _continue_until_done(client: anthropic.Anthropic, kwargs: dict, chain: list) -> list:
    """Continue a message chain whose last entry is resumable (see _resumable),
    re-sending the transcript so far — the documented continuation pattern for
    the server-side tool loop's iteration cap, plus an error tool_result for a
    misspelled tool call — until a non-resumable stop reason or
    _MAX_CONTINUATIONS is hit. `chain` must be non-empty; if its last message
    isn't resumable, it's returned unchanged. Appends to and returns `chain`.

    Each continuation runs in the same code-execution container as the turn it
    resumes. A request without `container` gets a fresh, empty sandbox, so the
    replayed transcript would refer to renders, scripts and the unzipped
    toolkit that no longer exist. An idle container is checkpointed and stays
    restorable by id for 30 days, so this also holds for a batch item resumed
    hours after its batch ended."""
    user_content = kwargs["messages"][0]["content"]

    for _ in range(_MAX_CONTINUATIONS):
        if not _resumable(chain[-1]):
            return chain
        turn = {**kwargs, "messages": _transcript(user_content, chain)}
        if container_id := _container_id(chain):
            turn["container"] = container_id
        _stream_turn(client, turn, chain)

    if chain[-1].stop_reason == "pause_turn":
        raise RuntimeError(
            f"extraction did not finish after {_MAX_CONTINUATIONS} pause_turn continuations "
            "(the server-side tool loop kept pausing) — this paper may need more figures/"
            "elements digitized than this pipeline currently handles in one run"
        )
    return chain


def _run_with_continuations(client: anthropic.Anthropic, kwargs: dict, chain: list) -> list:
    """Run one extraction call, automatically resuming through pause_turn.
    Appends every message to `chain` (usually just one) as it arrives, so the
    caller still holds what was billed if a later turn raises."""
    _stream_turn(client, kwargs, chain)
    return _continue_until_done(client, kwargs, chain)


def extract(
    prompt_text: str,
    pdf_bytes: bytes,
    *,
    model: str | None = None,
    analysis_block: str | None = None,
    qa_feedback: str | None = None,
) -> ExtractResponse:
    """Run one synchronous extraction. Returns the model's text output plus its
    token usage. Raises ExtractionFailed, carrying the usage billed so far, for
    anything that goes wrong once the model call has started.

    `analysis_block` is the optional deterministic curve pre-pass text
    (extraction/curve_prepass.py) injected into the user turn as a count anchor.
    `qa_feedback` is the optional previous-attempt QA failure block for an
    on-demand re-extraction (runner.qa_feedback_block).
    """
    model = model or config.EXTRACTION_MODEL
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    uploaded = client.beta.files.upload(file=("paper.pdf", pdf_bytes, "application/pdf"))
    toolkit = _upload_toolkit(client)
    kwargs = _message_kwargs(
        prompt_text, uploaded.id, model=model,
        analysis_block=analysis_block, qa_feedback=qa_feedback, toolkit_file_id=toolkit.id,
    )
    chain: list = []
    try:
        _run_with_continuations(client, kwargs, chain)
        return _response_from_chain(chain)
    except Exception as e:
        raise _failed(e, chain) from e
    finally:
        client.beta.files.delete(uploaded.id)
        client.beta.files.delete(toolkit.id)


def _upload_toolkit(client: anthropic.Anthropic):
    return client.beta.files.upload(
        file=(sandbox_toolkit.FILENAME, sandbox_toolkit.bundle(), "application/zip")
    )


# --------------------------------------------------------------------------- #
# Message Batches API — 50% cheaper token pricing; asynchronous (usually
# minutes, up to 24h). Gets a HIGHER per-turn server-side-tool-loop iteration
# cap than the synchronous path before pausing (Anthropic's docs), so most
# papers never hit pause_turn here at all; the rare one that does is finished
# off synchronously — see collect_batch_results.
#
# Code execution + Files API document blocks + task budgets inside a batched
# request was verified live on 2026-07-29 (~$2.26/paper on Sonnet 5, ~96%
# cache-served, accuracy matching the synchronous baseline). Re-verify on 1-2
# papers after any change to _message_kwargs before trusting a full run.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BatchRequest:
    """One paper's request in a Batches API job — everything needed to build
    (or, for a paused item, rebuild) its Messages API call. `custom_id` must be
    unique within the batch; callers use the paper's content sha256."""
    custom_id: str
    prompt_text: str
    model: str
    analysis_block: str | None = None
    qa_feedback: str | None = None   # a re-extraction's previous-attempt QA block, as in extract()

    def message_kwargs(self, file_id: str, toolkit_file_id: str | None = None) -> dict:
        return _message_kwargs(
            self.prompt_text, file_id, model=self.model,
            analysis_block=self.analysis_block, qa_feedback=self.qa_feedback,
            toolkit_file_id=toolkit_file_id,
        )


@dataclass(frozen=True)
class BatchSubmission:
    batch_id: str
    file_ids: dict[str, str]   # custom_id -> uploaded Files API id (cleanup after collection)
    toolkit_file_id: str       # the one toolkit upload every request in the job shares


def submit_batch(requests: list[BatchRequest], pdfs: dict[str, bytes]) -> BatchSubmission:
    """Upload each paper's PDF (`pdfs` is keyed by custom_id) and submit one
    Batches API job covering every request."""
    client = anthropic.Anthropic()

    toolkit = _upload_toolkit(client)
    file_ids: dict[str, str] = {}
    params = []
    for req in requests:
        uploaded = client.beta.files.upload(
            file=("paper.pdf", pdfs[req.custom_id], "application/pdf")
        )
        file_ids[req.custom_id] = uploaded.id
        params.append({
            "custom_id": req.custom_id,
            "params": req.message_kwargs(uploaded.id, toolkit.id),
        })

    batch = client.beta.messages.batches.create(betas=_BETAS, requests=params)
    return BatchSubmission(batch_id=batch.id, file_ids=file_ids, toolkit_file_id=toolkit.id)


def poll_batch_status(batch_id: str) -> str:
    """Return the batch's processing_status ('in_progress' | 'ended' | ...)."""
    client = anthropic.Anthropic()
    return client.beta.messages.batches.retrieve(batch_id).processing_status


def collect_batch_results(
    batch_id: str,
    requests: list[BatchRequest],
    file_ids: dict[str, str],
    toolkit_file_id: str | None = None,
) -> dict[str, ExtractResponse | Exception]:
    """Fetch results once the batch has ended. Keyed by custom_id.

    `requests` are the same BatchRequests originally submitted, `file_ids`
    their already-uploaded PDFs and `toolkit_file_id` the job's toolkit upload
    (None for a job submitted before the toolkit existed). All three are only
    used to rebuild the request for a paused item (below), which must match
    the one the batch ran; an item that finished cleanly never touches them.

    A result that errored/canceled/expired — or that raises while being read or
    continued — is surfaced as an Exception value rather than raised, so one bad
    paper doesn't lose the rest of the batch.

    A paused item (stop_reason=pause_turn, or a misspelled tool call — see
    _resumable) is NOT treated as a terminal failure: batch requests get a
    HIGHER per-turn iteration cap than synchronous ones, so pausing anyway
    means a genuinely demanding paper.
    Anthropic's docs confirm a paused batch item can be continued via either
    a new batch request or a synchronous one — we use the latter (the same
    `_continue_until_done` the sync `extract()` path uses), so only the rare
    paused item pays synchronous price for its remaining iterations; the rest
    of the batch stays batch-discounted.
    """
    client = anthropic.Anthropic()
    by_id = {req.custom_id: req for req in requests}
    out: dict[str, ExtractResponse | Exception] = {}
    for result in client.beta.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            out[result.custom_id] = RuntimeError(
                f"batch item {result.custom_id!r} did not succeed: {result.result.type}"
            )
            continue
        message = result.result.message
        try:
            chain = [message]
            if _resumable(message):
                req = by_id[result.custom_id]
                kwargs = req.message_kwargs(file_ids[result.custom_id], toolkit_file_id)
                chain = _continue_until_done(client, kwargs, chain)
            out[result.custom_id] = _response_from_chain(chain)
        except Exception as e:
            # Deliberately broad. The continuation above makes live API calls,
            # so this catches anthropic.RateLimitError / APIStatusError /
            # APIConnectionError as well as our own RuntimeErrors — none of
            # which are RuntimeError subclasses. Letting one escape would
            # discard every result already collected in `out`, and the retry
            # re-runs (and re-bills) the synchronous continuations that had
            # already succeeded. The error is not swallowed: it is returned as
            # this paper's value (with the usage billed so far, batch turn
            # included) and surfaced in the review queue.
            out[result.custom_id] = _failed(e, chain)
    return out


def cleanup_batch_files(file_ids: dict[str, str], toolkit_file_id: str | None = None) -> None:
    """Delete the Files API uploads made for a batch, once results are collected."""
    client = anthropic.Anthropic()
    for file_id in [*file_ids.values(), *([toolkit_file_id] if toolkit_file_id else [])]:
        try:
            client.beta.files.delete(file_id)
        except Exception:
            pass
