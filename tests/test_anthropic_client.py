"""Tests for the pause_turn continuation logic — both the synchronous
extract() path and the Batch API's transparent-continuation fallback."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from extraction import anthropic_client, parse_output
from extraction.anthropic_client import BatchRequest, _continue_until_done, collect_batch_results


def _msg(stop_reason: str, text: str = "final text", *, usage=None):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=usage or SimpleNamespace(
            input_tokens=10, output_tokens=20,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        ),
    )


def _fake_client(messages_by_call):
    """A fake anthropic.Anthropic() whose client.beta.messages.stream(...)
    context manager yields the next message in `messages_by_call` on each call."""
    client = MagicMock()
    call_iter = iter(messages_by_call)

    def _stream(**kwargs):
        cm = MagicMock()
        cm.__enter__.return_value.get_final_message.return_value = next(call_iter)
        return cm

    client.beta.messages.stream.side_effect = _stream
    return client


_KWARGS = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}


def test_continue_until_done_returns_immediately_when_not_paused():
    client = _fake_client([])  # no continuation calls expected
    chain = [_msg("end_turn")]
    result = _continue_until_done(client, _KWARGS, chain)
    assert result == chain
    client.beta.messages.stream.assert_not_called()


def test_continue_until_done_continues_until_resolved():
    client = _fake_client([_msg("pause_turn"), _msg("end_turn", "done")])
    chain = [_msg("pause_turn")]
    result = _continue_until_done(client, _KWARGS, chain)
    assert len(result) == 3
    assert result[-1].stop_reason == "end_turn"
    assert client.beta.messages.stream.call_count == 2


def test_continue_until_done_raises_after_max_continuations():
    always_paused = [_msg("pause_turn") for _ in range(anthropic_client._MAX_CONTINUATIONS + 1)]
    client = _fake_client(always_paused)
    chain = [_msg("pause_turn")]
    with pytest.raises(RuntimeError, match="did not finish"):
        _continue_until_done(client, _KWARGS, chain)


def test_check_stop_reason_accepts_only_a_finished_turn():
    assert anthropic_client._check_stop_reason("end_turn") is None
    assert anthropic_client._check_stop_reason("pause_turn", can_continue=True) is None


@pytest.mark.parametrize(
    "stop_reason, expected",
    [
        ("refusal", "declined"),
        ("max_tokens", "truncated"),
        ("model_context_window_exceeded", "context window"),
        ("pause_turn", "paused"),
        ("tool_use", "misspelled"),
    ],
)
def test_check_stop_reason_reports_each_known_failure_specifically(stop_reason, expected):
    with pytest.raises(RuntimeError, match=expected):
        anthropic_client._check_stop_reason(stop_reason)


def test_check_stop_reason_rejects_a_stop_reason_it_has_never_seen():
    """Whitelist, not blacklist: an unknown stop reason means an incomplete
    response, and parse_output would happily accept a truncated one."""
    with pytest.raises(RuntimeError, match="unrecognized stop_reason"):
        anthropic_client._check_stop_reason("some_future_stop_reason")


def test_truncated_response_would_parse_as_complete_if_the_stop_reason_slipped_through():
    """Why the whitelist matters: the parser falls back to the newest block that
    parses, so a run cut off mid-table yields a smaller table, not an error."""
    truncated = (
        '```json\n{"columns": ["a"], "rows": [[1]]}\n```\n'
        '```json\n{"columns": ["a"], "rows": [[1], [2], [3'  # cut off mid-write
    )
    assert parse_output._extract_json_object(truncated) == {"columns": ["a"], "rows": [[1]]}


def test_collect_batch_results_continues_paused_batch_item():
    batch_result = SimpleNamespace(
        custom_id="sha1",
        result=SimpleNamespace(type="succeeded", message=_msg("pause_turn", "partial")),
    )
    resolved = _msg("end_turn", "finished")
    requests = [BatchRequest("sha1", "prompt text", "claude-opus-4-8")]

    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _fake_client([resolved])
        client.beta.messages.batches.results.return_value = [batch_result]
        mock_anthropic.return_value = client
        out = collect_batch_results("batch_1", requests, {"sha1": "file_123"})

    assert client.beta.messages.stream.call_count == 1
    assert isinstance(out["sha1"], anthropic_client.ExtractResponse)
    assert out["sha1"].text == "finished"


def test_collect_batch_results_passes_through_finished_item_untouched():
    batch_result = SimpleNamespace(
        custom_id="sha1",
        result=SimpleNamespace(type="succeeded", message=_msg("end_turn", "already done")),
    )
    requests = [BatchRequest("sha1", "prompt text", "claude-opus-4-8")]

    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _fake_client([])  # no continuation call should happen
        client.beta.messages.batches.results.return_value = [batch_result]
        mock_anthropic.return_value = client
        out = collect_batch_results("batch_1", requests, {"sha1": "file_123"})

    client.beta.messages.stream.assert_not_called()
    assert out["sha1"].text == "already done"


def test_collect_batch_results_surfaces_errored_item():
    batch_result = SimpleNamespace(
        custom_id="sha1", result=SimpleNamespace(type="errored", message=None),
    )
    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _fake_client([])
        client.beta.messages.batches.results.return_value = [batch_result]
        mock_anthropic.return_value = client
        out = collect_batch_results("batch_1", [BatchRequest("sha1", "p", "m")], {"sha1": "f"})

    assert isinstance(out["sha1"], RuntimeError)
    assert "did not succeed" in str(out["sha1"])


class _NotARuntimeError(Exception):
    """Stands in for anthropic.RateLimitError / APIStatusError / APIConnectionError,
    none of which subclass RuntimeError."""


def test_collect_batch_results_isolates_a_failed_continuation():
    """An SDK error while continuing ONE paused item must not discard the
    already-collected results — a re-collect re-bills every continuation."""
    paused = SimpleNamespace(
        custom_id="sha_paused",
        result=SimpleNamespace(type="succeeded", message=_msg("pause_turn", "partial")),
    )
    finished = SimpleNamespace(
        custom_id="sha_ok",
        result=SimpleNamespace(type="succeeded", message=_msg("end_turn", "all good")),
    )
    requests = [
        BatchRequest("sha_paused", "p", "claude-sonnet-5"),
        BatchRequest("sha_ok", "p", "claude-sonnet-5"),
    ]
    file_ids = {"sha_paused": "file_1", "sha_ok": "file_2"}

    with patch("anthropic.Anthropic") as mock_anthropic:
        client = MagicMock()
        client.beta.messages.stream.side_effect = _NotARuntimeError("overloaded")
        # The paused item is collected FIRST, so an escaping exception would
        # take the finished item down with it.
        client.beta.messages.batches.results.return_value = [paused, finished]
        mock_anthropic.return_value = client
        out = collect_batch_results("batch_1", requests, file_ids)

    assert isinstance(out["sha_paused"], _NotARuntimeError)
    assert out["sha_ok"].text == "all good"


# --------------------------------------------------------------------------- #
# Misspelled server-tool call (stop_reason=tool_use) — seen live as
# `bash_code_execction`: answered with an error tool_result, not fatal.
# --------------------------------------------------------------------------- #
def _typo_msg(name="bash_code_execction", tool_id="toolu_1"):
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="text", text="Now let's test pdfplumber."),
            SimpleNamespace(type="tool_use", id=tool_id, name=name, input={"command": "echo test"}),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


def _sent_messages(client, call_index=0):
    return client.beta.messages.stream.call_args_list[call_index].kwargs["messages"]


def test_misspelled_tool_call_is_answered_with_an_error_tool_result_and_resumed():
    client = _fake_client([_msg("end_turn", "done")])
    chain = _continue_until_done(client, _KWARGS, [_typo_msg()])

    assert chain[-1].stop_reason == "end_turn"
    sent = _sent_messages(client)
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    (result,) = sent[2]["content"]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1"
    assert result["is_error"] is True
    assert "bash_code_execction" in result["content"] and "bash_code_execution" in result["content"]


def test_continuation_replays_the_whole_transcript_after_a_typo_then_a_pause():
    # typo -> corrected, then the tool loop pauses -> the next call must carry
    # the typo turn AND its error result AND the paused segment, in order.
    client = _fake_client([_msg("pause_turn", "partial"), _msg("end_turn", "done")])
    chain = _continue_until_done(client, _KWARGS, [_typo_msg()])

    assert len(chain) == 3 and chain[-1].stop_reason == "end_turn"
    roles = [m["role"] for m in _sent_messages(client, 1)]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_tool_use_stop_with_no_tool_call_block_is_not_resumed():
    odd = SimpleNamespace(stop_reason="tool_use", content=[SimpleNamespace(type="text", text="?")],
                          usage=None)
    client = _fake_client([])
    assert _continue_until_done(client, _KWARGS, [odd]) == [odd]
    client.beta.messages.stream.assert_not_called()


def test_repeated_typos_hit_the_continuation_cap_and_fail_specifically():
    client = _fake_client([_typo_msg(tool_id=f"toolu_{i}")
                           for i in range(anthropic_client._MAX_CONTINUATIONS)])
    chain = _continue_until_done(client, _KWARGS, [_typo_msg()])
    assert chain[-1].stop_reason == "tool_use"
    with pytest.raises(RuntimeError, match="misspelled"):
        anthropic_client._response_from_chain(chain)


def test_collect_batch_results_resumes_a_batch_item_that_stopped_on_a_typo():
    batch_result = SimpleNamespace(custom_id="sha1",
                                   result=SimpleNamespace(type="succeeded", message=_typo_msg()))
    requests = [BatchRequest("sha1", "prompt text", "claude-opus-4-8")]
    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _fake_client([_msg("end_turn", "finished")])
        client.beta.messages.batches.results.return_value = [batch_result]
        mock_anthropic.return_value = client
        out = collect_batch_results("batch_1", requests, {"sha1": "file_123"})
    assert isinstance(out["sha1"], anthropic_client.ExtractResponse)
    assert out["sha1"].text == "finished"
    assert client.beta.messages.stream.call_count == 1


def test_batch_request_carries_qa_feedback_into_the_user_turn():
    """A batched re-extraction injects the previous attempt's QA block exactly
    where the synchronous path does: after the pre-pass block, before the
    cache-marked instruction, so the cached prefix shape is unchanged."""
    content = BatchRequest(
        "sha1", "prompt", "claude-sonnet-5", analysis_block="PREPASS", qa_feedback="FEEDBACK",
    ).message_kwargs("file_1")["messages"][0]["content"]
    texts = [b["text"] for b in content if b["type"] == "text"]
    assert texts[:2] == ["PREPASS", "FEEDBACK"]
    assert "cache_control" in content[-1] and content[-1]["text"] not in ("PREPASS", "FEEDBACK")

    without = BatchRequest("sha1", "prompt", "claude-sonnet-5").message_kwargs("file_1")
    assert [b["type"] for b in without["messages"][0]["content"]] == ["document", "container_upload", "text"]
