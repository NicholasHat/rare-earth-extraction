"""Thin wrapper over the Anthropic Messages API for figure extraction.

Gives Claude the paper PDF two ways in the same turn: a `document` block (so
it can visually read the figure — legend colours, marker shapes, panel
layout) and a `container_upload` (so the code-execution tool can open the
same file with the shipped toolkit and numpy for vector/raster detection, axis calibration,
and point digitization, per the prompt's Steps 2-6). Both need the PDF
uploaded once via the Files API first. The model sees figures only through
the document block: images it renders inside the sandbox are pixels for its
code, never shown to it.

Two ways to run an extraction, sharing the same request shape (_message_kwargs):
  - `extract()` — synchronous, streamed (a fully-digitized multi-element,
    multi-figure extraction plus the code-execution transcript is a large
    output; non-streaming would risk the SDK's HTTP timeout). Automatically
    resumes through `stop_reason="pause_turn"` (the server-side code-execution
    loop's per-turn iteration cap) by re-sending the assistant's own partial
    response, per the documented continuation pattern.
  - `submit_batch()` / `poll_batch_status()` / `collect_batch_results()` — the
    Message Batches API, 50% cheaper and asynchronous. A `pause_turn` result is
    transparently finished off with a synchronous continuation
    (`_continue_until_done`) rather than surfaced as an error.

Both paths pause at the same cap: 50 code-execution iterations per turn,
measured on Sonnet 5 (a synchronous turn paused after 50 tool calls on
2026-09-25; batch items paused at 50 on 2026-09-16 and 2026-09-24). Older
notes here said 10 for sync and "higher" for batch — neither holds.

`_continue_until_done` also resumes through `stop_reason="tool_use"`: the only
tool offered is server-side, so a client-side tool call can only be the model
misspelling that tool's name (seen live: `bash_code_execction`). Rather than
lose the whole extraction to a typo, it is answered with an is_error
tool_result naming the mistake and the model carries on.

Every completed turn can be handed to a chain store (extraction/transcripts.py)
as it arrives: collection saves a batch's results to disk before working on
them, and a continuation saves each turn before sending the next, so a
continuation that dies resumes from its last completed turn instead of
re-paying the whole chain.

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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

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
    # The part of the usage above billed by turns after the first — for a
    # batch item, the synchronous continuation, which pays full price.
    continuation_usage: dict = field(default_factory=dict)

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
        self.transcript: str | None = None   # set by the runner: where the paid turns are saved
        turns = f" over {turns_completed} completed API turn(s)" if turns_completed is not None else ""
        super().__init__(f"{cause} [billed before failure: {_describe_usage(usage)}{turns}]")

    @property
    def credit_exhausted(self) -> bool:
        return is_credit_exhausted(self.cause)


class CreditExhausted(RuntimeError):
    """The Anthropic account ran out of credit. Nothing half-done is lost:
    whatever completed is saved, and the same command resumes once topped up."""


def is_credit_exhausted(error: BaseException | None) -> bool:
    """Whether `error` (or anything it wraps) is the API refusing for lack of
    credit — a 402 billing_error, or the 400 'credit balance is too low' the
    API has also returned for it."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, CreditExhausted):
            return True
        if isinstance(error, anthropic.APIStatusError) and (
            error.status_code == 402 or error.type == "billing_error"
            or "credit balance" in str(error).lower()
        ):
            return True
        error = getattr(error, "cause", None) or error.__cause__
    return False


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
# It states the sandbox facts a model otherwise spends iterations discovering:
# the PDF's path, that rendering goes through the toolkit (the installed PDF
# libraries changed between 2026-09-25 and 2026-10-07, see sandbox_toolkit),
# that there is no internet, and that a rendered image is never shown to it
# (opening a PNG with the file viewer returns base64 text that every later
# iteration re-reads).
_USER_INSTRUCTION = (
    "Extract the data from this paper according to your instructions. The same "
    "PDF is also in your code execution environment at $INPUT_DIR/paper.pdf — "
    "render it with the SANDBOX TOOLKIT's `pages.render` and read its vector "
    "objects and text with pdfplumber, for axis calibration and point "
    "digitization as your instructions describe. The sandbox has no internet "
    "access, so nothing can be installed. You cannot see images you create in the sandbox: read "
    "figures (legend, marker shapes, panel layout, in-plot text) from the PDF "
    "document above, and never open an image file with the file viewer. "
    "Return only the single JSON object described in the OUTPUT CONTRACT."
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
    return ExtractResponse(text=text, **_sum_usage(messages), continuation_usage=_sum_usage(messages[1:]))


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
# after 50 iterations in one turn (measured; see the module docstring). Bound
# how many times we resend and let it resume — a demanding paper can need
# more than one turn; an unbounded loop would not.
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


def _continue_until_done(
    client: anthropic.Anthropic, kwargs: dict, chain: list,
    on_turn: Callable[[list], None] | None = None,
) -> list:
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
    hours after its batch ended.

    `on_turn(chain)` runs after each completed turn (never with a dying
    stream's partial message on the end) — the caller's checkpoint. A chain
    reloaded from such a checkpoint resumes here with the turns still
    missing; the continuation budget counts the chain's existing turns, so a
    resumed run gets the same total as one that never died."""
    user_content = kwargs["messages"][0]["content"]

    while len(chain) <= _MAX_CONTINUATIONS:
        if not _resumable(chain[-1]):
            return chain
        turn = {**kwargs, "messages": _transcript(user_content, chain)}
        if container_id := _container_id(chain):
            turn["container"] = container_id
        _stream_turn(client, turn, chain)
        if on_turn is not None:
            on_turn(chain)

    if chain[-1].stop_reason == "pause_turn":
        raise RuntimeError(
            f"extraction did not finish after {_MAX_CONTINUATIONS} pause_turn continuations "
            "(the server-side tool loop kept pausing) — this paper may need more figures/"
            "elements digitized than this pipeline currently handles in one run"
        )
    return chain


def _run_with_continuations(
    client: anthropic.Anthropic, kwargs: dict, chain: list,
    on_turn: Callable[[list], None] | None = None,
) -> list:
    """Run one extraction call, automatically resuming through pause_turn.
    Appends every message to `chain` (usually just one) as it arrives, so the
    caller still holds what was billed if a later turn raises."""
    _stream_turn(client, kwargs, chain)
    if on_turn is not None:
        on_turn(chain)
    return _continue_until_done(client, kwargs, chain, on_turn)


def extract(
    prompt_text: str,
    pdf_bytes: bytes,
    *,
    model: str | None = None,
    analysis_block: str | None = None,
    qa_feedback: str | None = None,
    on_turn: Callable[[list], None] | None = None,
) -> ExtractResponse:
    """Run one synchronous extraction. Returns the model's text output plus its
    token usage. Raises ExtractionFailed, carrying the usage billed so far, for
    anything that goes wrong once the model call has started.

    `analysis_block` is the optional deterministic curve pre-pass text
    (extraction/curve_prepass.py) injected into the user turn as a count anchor.
    `qa_feedback` is the optional previous-attempt QA failure block for an
    on-demand re-extraction (runner.qa_feedback_block). `on_turn(chain)` is
    called after every completed turn (the transcript checkpoint).
    """
    model = model or config.EXTRACTION_MODEL
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    uploads: list[str] = []
    chain: list = []
    try:
        uploads.append(client.beta.files.upload(file=("paper.pdf", pdf_bytes, "application/pdf")).id)
        uploads.append(_upload_toolkit(client).id)
        kwargs = _message_kwargs(
            prompt_text, uploads[0], model=model,
            analysis_block=analysis_block, qa_feedback=qa_feedback, toolkit_file_id=uploads[1],
        )
        _run_with_continuations(client, kwargs, chain, on_turn)
        return _response_from_chain(chain)
    except Exception as e:
        raise _failed(e, chain) from e
    finally:
        _delete_files(client, uploads)


def _delete_files(client: anthropic.Anthropic, file_ids) -> None:
    """Best-effort delete of Files API uploads (an already-deleted or expired
    file is not an error worth surfacing)."""
    for file_id in file_ids:
        if not file_id:
            continue
        try:
            client.beta.files.delete(file_id)
        except Exception:
            pass


def _upload_toolkit(client: anthropic.Anthropic):
    return client.beta.files.upload(
        file=(sandbox_toolkit.FILENAME, sandbox_toolkit.bundle(), "application/zip")
    )


# --------------------------------------------------------------------------- #
# Message Batches API — 50% cheaper token pricing; asynchronous (usually
# minutes, up to 24h). Same 50-iteration per-turn cap as the synchronous path
# (measured); an item that pauses is finished off synchronously — see
# collect_batch_results.
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
    created_at: str | None = None   # the API's creation time (results expire 29 days later)


class BatchNotCreated(RuntimeError):
    """The API definitely did not create the batch (it refused the request).
    Nothing was billed, and the uploads made for it were deleted."""

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(f"the batch was not created: {cause}")


class BatchSubmissionUnknown(RuntimeError):
    """The create request failed in a way that leaves it unknown whether the
    API created the batch (connection dropped, timeout, server error). The
    uploads are kept: if the batch exists it needs them. Reconcile against the
    API's batch list (batch_jobs.reconcile) to find out."""

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(
            f"unknown whether the batch was created ({cause}) — run a status check to reconcile"
        )


def submit_batch(
    requests: list[BatchRequest],
    pdfs: dict[str, bytes],
    *,
    on_upload: Callable[[str | None, str], None] | None = None,
    before_create: Callable[[], None] | None = None,
) -> BatchSubmission:
    """Upload each paper's PDF (`pdfs` is keyed by custom_id) and submit one
    Batches API job covering every request.

    `on_upload(custom_id, file_id)` runs after each upload (custom_id None for
    the toolkit) and `before_create()` just before the paid step, so the
    caller can journal both: a crash at any point leaves a record of what was
    uploaded and whether a batch may exist.

    The create call is sent once (no SDK retry: a retried create whose first
    attempt reached the API would submit, and bill, the batch twice). If the
    API refuses it (4xx — e.g. no credit) the uploads are deleted and
    BatchNotCreated is raised; any other failure raises BatchSubmissionUnknown
    and keeps the uploads. A failed upload deletes the earlier uploads."""
    client = anthropic.Anthropic()
    on_upload = on_upload or (lambda custom_id, file_id: None)

    uploaded: list[str] = []
    file_ids: dict[str, str] = {}
    params = []
    try:
        toolkit = _upload_toolkit(client)
        uploaded.append(toolkit.id)
        on_upload(None, toolkit.id)
        for req in requests:
            pdf = client.beta.files.upload(file=("paper.pdf", pdfs[req.custom_id], "application/pdf"))
            uploaded.append(pdf.id)
            file_ids[req.custom_id] = pdf.id
            on_upload(req.custom_id, pdf.id)
            params.append({"custom_id": req.custom_id, "params": req.message_kwargs(pdf.id, toolkit.id)})
        if before_create is not None:
            before_create()
    except Exception:
        _delete_files(client, uploaded)
        raise

    try:
        batch = client.with_options(max_retries=0).beta.messages.batches.create(betas=_BETAS, requests=params)
    except anthropic.APIStatusError as e:
        if 400 <= e.status_code < 500:
            _delete_files(client, uploaded)
            raise BatchNotCreated(e) from e
        raise BatchSubmissionUnknown(e) from e
    except Exception as e:
        raise BatchSubmissionUnknown(e) from e
    created = getattr(batch, "created_at", None)
    return BatchSubmission(
        batch_id=batch.id, file_ids=file_ids, toolkit_file_id=toolkit.id,
        created_at=created.isoformat() if isinstance(created, datetime) else None,
    )


@dataclass(frozen=True)
class BatchInfo:
    """What the API says about one batch (a free metadata call)."""
    batch_id: str
    processing_status: str           # 'in_progress' | 'canceling' | 'ended'
    created_at: datetime
    ended_at: datetime | None
    requests: int                    # total requests in the batch
    succeeded: int
    errored: int
    canceled: int
    expired: int

    @classmethod
    def from_api(cls, batch) -> "BatchInfo":
        c = batch.request_counts
        return cls(
            batch_id=batch.id, processing_status=batch.processing_status,
            created_at=batch.created_at, ended_at=batch.ended_at,
            requests=c.processing + c.succeeded + c.errored + c.canceled + c.expired,
            succeeded=c.succeeded, errored=c.errored, canceled=c.canceled, expired=c.expired,
        )


def batch_info(batch_id: str) -> BatchInfo:
    client = anthropic.Anthropic()
    return BatchInfo.from_api(client.beta.messages.batches.retrieve(batch_id))


def list_batches(created_after: datetime) -> list[BatchInfo]:
    """Every batch on the account created after `created_after`. Walks every
    page rather than stopping at the first older batch: reconciliation must
    never miss a batch because of how the endpoint happens to order them —
    missing one would get a live batch's uploads cleaned up."""
    client = anthropic.Anthropic()
    return [BatchInfo.from_api(b) for b in client.beta.messages.batches.list(limit=100)
            if b.created_at >= created_after]


def poll_batch_status(batch_id: str) -> str:
    """Return the batch's processing_status ('in_progress' | 'ended' | ...)."""
    client = anthropic.Anthropic()
    return client.beta.messages.batches.retrieve(batch_id).processing_status


class _MemoryStore:
    """A chain store that keeps nothing beyond this call (the default)."""

    def __init__(self):
        self._data: dict = {}

    def load(self, custom_id):
        return self._data.get(custom_id)

    def save(self, custom_id, chain):
        self._data[custom_id] = list(chain)

    def save_error(self, custom_id, error):
        self._data[custom_id] = error


def _item_error(result) -> str:
    """Why a batch item did not succeed, as specifically as the API says."""
    kind = result.result.type
    detail = getattr(getattr(getattr(result.result, "error", None), "error", None), "message", None)
    return f"batch item {result.custom_id!r} did not succeed: {kind}" + (f" ({detail})" if detail else "")


def collect_batch_results(
    batch_id: str,
    requests: list[BatchRequest],
    file_ids: dict[str, str],
    toolkit_file_id: str | None = None,
    *,
    store=None,
) -> dict[str, ExtractResponse | Exception]:
    """Fetch results once the batch has ended. Keyed by custom_id.

    `requests` are the same BatchRequests originally submitted, `file_ids`
    their already-uploaded PDFs and `toolkit_file_id` the job's toolkit upload
    (None for a job submitted before the toolkit existed). All three are only
    used to rebuild the request for a paused item (below), which must match
    the one the batch ran; an item that finished cleanly never touches them.

    `store` (extraction/transcripts.TranscriptStore) is where each item's
    chain of turns lives: every result not yet in it is downloaded and saved
    first — so a batch is fetched from the API once, and its output survives
    the API's 29-day retention — and a continuation saves each turn as it
    completes and resumes from the saved chain. Without a store nothing
    outlives the call.

    A result that errored/canceled/expired — or that raises while being read or
    continued — is surfaced as an Exception value rather than raised, so one bad
    paper doesn't lose the rest of the batch. Once the credit balance runs out
    no further continuation is attempted in this call; those items fail with
    CreditExhausted and resume from their saved turns next time.

    A paused item (stop_reason=pause_turn, or a misspelled tool call — see
    _resumable) is NOT treated as a terminal failure: it has used a full
    50-iteration turn, the mark of a genuinely demanding paper.
    Anthropic's docs confirm a paused batch item can be continued via either
    a new batch request or a synchronous one — we use the latter (the same
    `_continue_until_done` the sync `extract()` path uses), so only the rare
    paused item pays synchronous price for its remaining iterations; the rest
    of the batch stays batch-discounted.
    """
    client = anthropic.Anthropic()
    store = store if store is not None else _MemoryStore()
    by_id = {req.custom_id: req for req in requests}

    missing = {cid for cid in by_id if store.load(cid) is None}
    if missing:
        for result in client.beta.messages.batches.results(batch_id):
            if result.custom_id not in missing:
                continue
            if result.result.type != "succeeded":
                store.save_error(result.custom_id, _item_error(result))
            else:
                store.save(result.custom_id, [result.result.message])

    out: dict[str, ExtractResponse | Exception] = {}
    credit_error: Exception | None = None
    for custom_id, req in by_id.items():
        saved = store.load(custom_id)
        if saved is None:
            out[custom_id] = RuntimeError("no result returned for this paper")
            continue
        if isinstance(saved, str):
            out[custom_id] = RuntimeError(saved)
            continue
        chain = list(saved)
        try:
            if _resumable(chain[-1]):
                if credit_error is not None:
                    raise CreditExhausted(
                        "the Anthropic credit balance ran out earlier in this collection, so this "
                        "paper's continuation was not attempted — top up and collect again; it "
                        f"resumes from its {len(chain)} saved turn(s)"
                    )
                kwargs = req.message_kwargs(file_ids[custom_id], toolkit_file_id)
                chain = _continue_until_done(
                    client, kwargs, chain, on_turn=lambda c, cid=custom_id: store.save(cid, c)
                )
            out[custom_id] = _response_from_chain(chain)
        except Exception as e:
            # Deliberately broad. The continuation above makes live API calls,
            # so this catches anthropic.RateLimitError / APIStatusError /
            # APIConnectionError as well as our own RuntimeErrors — none of
            # which are RuntimeError subclasses. Letting one escape would
            # discard every result already collected in `out`. The error is
            # not swallowed: it is returned as this paper's value (with the
            # usage billed so far, batch turn included) and surfaced in the
            # review queue; the turns that completed are in `store`.
            out[custom_id] = _failed(e, chain)
            if is_credit_exhausted(e):
                credit_error = e
    return out


def cleanup_batch_files(file_ids: dict[str, str], toolkit_file_id: str | None = None) -> None:
    """Delete the Files API uploads made for a batch, once results are collected."""
    client = anthropic.Anthropic()
    _delete_files(client, [*file_ids.values(), toolkit_file_id])
