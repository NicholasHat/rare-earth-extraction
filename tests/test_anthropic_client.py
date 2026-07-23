"""Tests for the pause_turn continuation logic — both the synchronous
extract() path and the Batch API's transparent-continuation fallback."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from extraction import anthropic_client
from extraction.anthropic_client import _continue_until_done, collect_batch_results


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


def test_collect_batch_results_continues_paused_batch_item():
    batch_result = SimpleNamespace(
        custom_id="sha1",
        result=SimpleNamespace(type="succeeded", message=_msg("pause_turn", "partial")),
    )
    resolved = _msg("end_turn", "finished")
    items = [("sha1", "prompt text", "file_123", None, "claude-opus-4-8")]

    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _fake_client([resolved])
        client.beta.messages.batches.results.return_value = [batch_result]
        mock_anthropic.return_value = client
        out = collect_batch_results("batch_1", items)

    assert client.beta.messages.stream.call_count == 1
    assert isinstance(out["sha1"], anthropic_client.ExtractResponse)
    assert out["sha1"].text == "finished"


def test_collect_batch_results_passes_through_finished_item_untouched():
    batch_result = SimpleNamespace(
        custom_id="sha1",
        result=SimpleNamespace(type="succeeded", message=_msg("end_turn", "already done")),
    )
    items = [("sha1", "prompt text", "file_123", None, "claude-opus-4-8")]

    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _fake_client([])  # no continuation call should happen
        client.beta.messages.batches.results.return_value = [batch_result]
        mock_anthropic.return_value = client
        out = collect_batch_results("batch_1", items)

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
        out = collect_batch_results("batch_1", [("sha1", "p", "f", None, "m")])

    assert isinstance(out["sha1"], RuntimeError)
    assert "did not succeed" in str(out["sha1"])
