"""extraction/transcripts.py + the chain store in anthropic_client: every paid
turn is on disk, and a continuation that dies resumes from its last completed
turn instead of re-paying the chain (no API calls — a fake client)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest
from anthropic.types.beta import BetaMessage

import config
from extraction import anthropic_client
from extraction.anthropic_client import BatchRequest, collect_batch_results
from extraction.transcripts import TranscriptStore, final_text


@pytest.fixture(autouse=True)
def _dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", tmp_path / "transcripts")


def _message(stop_reason, text="t", *, container="container_A", out=20):
    return BetaMessage.model_validate({
        "id": f"msg_{stop_reason}_{text}", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig=="},
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "bash_code_execution", "input": {"command": "ls"}},
            {"type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_1",
             "content": {"type": "bash_code_execution_result", "stdout": "a", "stderr": "", "return_code": 0,
                         "content": []}},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": out, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 1000},
        "container": {"id": container, "expires_at": "2026-11-09T00:00:00Z"},
    })


def test_store_round_trips_a_chain_to_the_identical_request_body():
    from anthropic._utils._transform import maybe_transform
    from anthropic.types.beta import BetaMessageParam

    store = TranscriptStore("msgbatch_1")
    chain = [_message("pause_turn", "a"), _message("end_turn", "b")]
    store.save("sha1", chain)
    loaded = store.load("sha1")
    assert loaded == chain
    for orig, back in zip(chain, loaded):   # what a continuation would re-send is byte-identical
        assert maybe_transform({"role": "assistant", "content": orig.content}, BetaMessageParam) == \
            maybe_transform({"role": "assistant", "content": back.content}, BetaMessageParam)
    assert final_text(store, "sha1") == "b"
    assert store.relpath("sha1") == "transcripts/msgbatch_1/sha1.json.gz"
    store.save_error("sha2", "batch item 'sha2' did not succeed: expired")
    assert store.load("sha2") == "batch item 'sha2' did not succeed: expired"
    assert store.load("nope") is None


def _credit_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(402, request=request)
    return anthropic.APIStatusError(
        "Your credit balance is too low", response=response,
        body={"type": "error", "error": {"type": "billing_error", "message": "Your credit balance is too low"}},
    )


def _client(stream_outcomes, batch_results):
    """A fake client whose stream() yields/raises the next outcome, and whose
    batch results are `batch_results`."""
    client = MagicMock()
    outcomes = iter(stream_outcomes)

    def _stream(**kwargs):
        outcome = next(outcomes)
        cm = MagicMock()
        if isinstance(outcome, Exception):
            cm.__enter__.return_value.get_final_message.side_effect = outcome
            cm.__enter__.return_value.current_message_snapshot = MagicMock()
        else:
            cm.__enter__.return_value.get_final_message.return_value = outcome
        return cm

    client.beta.messages.stream.side_effect = _stream
    client.beta.messages.batches.results.return_value = batch_results
    return client


def _result(custom_id, message):
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message))


def test_a_continuation_that_dies_resumes_from_its_last_saved_turn():
    store = TranscriptStore("msgbatch_1")
    requests = [BatchRequest("sha1", "p", "claude-sonnet-5-5")]
    batch = [_result("sha1", _message("pause_turn", "batch"))]

    # 1st collection: one continuation turn completes, the next dies on credit.
    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _client([_message("pause_turn", "cont1"), _credit_error()], batch)
        mock_anthropic.return_value = client
        out = collect_batch_results("msgbatch_1", requests, {"sha1": "file_1"}, "file_kit", store=store)
    failed = out["sha1"]
    assert isinstance(failed, anthropic_client.ExtractionFailed) and failed.credit_exhausted
    assert [m.content[-1].text for m in store.load("sha1")] == ["batch", "cont1"]

    # 2nd collection: nothing re-downloaded, only the missing turn is paid for.
    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _client([_message("end_turn", "done")], [])
        mock_anthropic.return_value = client
        out = collect_batch_results("msgbatch_1", requests, {"sha1": "file_1"}, "file_kit", store=store)
    client.beta.messages.batches.results.assert_not_called()
    assert client.beta.messages.stream.call_count == 1
    sent = client.beta.messages.stream.call_args.kwargs
    assert [m["role"] for m in sent["messages"]] == ["user", "assistant", "assistant"]
    assert sent["container"] == "container_A"
    resp = out["sha1"]
    assert resp.text == "done"
    assert resp.output_tokens == 60 and resp.continuation_usage["output_tokens"] == 40
    assert [m.content[-1].text for m in store.load("sha1")] == ["batch", "cont1", "done"]


def test_after_credit_runs_out_no_further_continuation_is_attempted():
    requests = [BatchRequest("a", "p", "m"), BatchRequest("b", "p", "m"), BatchRequest("c", "p", "m")]
    batch = [_result("a", _message("pause_turn")), _result("b", _message("pause_turn")),
             _result("c", _message("end_turn", "fine"))]
    with patch("anthropic.Anthropic") as mock_anthropic:
        client = _client([_credit_error()], batch)
        mock_anthropic.return_value = client
        out = collect_batch_results("msgbatch_1", requests, {"a": "f", "b": "f", "c": "f"},
                                    store=TranscriptStore("msgbatch_1"))
    assert client.beta.messages.stream.call_count == 1          # b was not tried
    assert isinstance(out["b"].cause, anthropic_client.CreditExhausted) and out["b"].credit_exhausted
    assert out["c"].text == "fine"


def test_a_synchronous_extraction_saves_every_turn_and_names_its_transcript_on_failure():
    from extraction import runner

    with patch("anthropic.Anthropic") as mock_anthropic, \
         patch.object(runner, "_prepass", return_value=("", [])):
        client = _client([_message("pause_turn", "first"), _message("end_turn", "not json")], [])
        client.beta.files.upload.side_effect = [SimpleNamespace(id="file_pdf"), SimpleNamespace(id="file_kit")]
        mock_anthropic.return_value = client
        with pytest.raises(anthropic_client.ExtractionFailed) as exc:
            runner.extract_paper(b"%PDF-1.4 fake", prompt_version="extraction_v15")
    path = config.DATA_DIR / exc.value.transcript
    assert path.exists() and "parseable" in str(exc.value)
    run_id = path.parent.name
    sha = path.name.removesuffix(".json.gz")
    assert [m.content[-1].text for m in TranscriptStore(run_id).load(sha)] == ["first", "not json"]
    assert sorted(c.args[0] for c in client.beta.files.delete.call_args_list) == ["file_kit", "file_pdf"]


def test_a_failed_toolkit_upload_still_deletes_the_pdf_upload():
    with patch("anthropic.Anthropic") as mock_anthropic:
        client = MagicMock()
        client.beta.files.upload.side_effect = [SimpleNamespace(id="file_pdf"), RuntimeError("upload failed")]
        mock_anthropic.return_value = client
        with pytest.raises(anthropic_client.ExtractionFailed):
            anthropic_client.extract("prompt", b"%PDF", model="m")
    assert [c.args[0] for c in client.beta.files.delete.call_args_list] == ["file_pdf"]


def test_a_saved_turn_loads_when_the_api_sent_usage_the_sdk_types_do_not_list():
    """Batch results on 2026-11-09 carried usage.iterations entries without the
    `model` field the SDK's types require: strict validation refused to reload
    all ten saved results. The reloaded turn must still dump identically."""
    saved = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn",
        "container": {"id": "container_A", "expires_at": "2026-12-01T00:00:00Z"},
        "usage": {"input_tokens": 1, "output_tokens": 2,
                  "iterations": [{"type": "message", "input_tokens": 1, "output_tokens": 2,
                                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}]},
    }
    store = TranscriptStore("msgbatch_x")
    store._write("p", {"messages": [saved]})
    (loaded,) = store.load("p")
    assert loaded.content[0].text == "done" and loaded.container.id == "container_A"
    assert loaded.model_dump(mode="json", exclude_unset=True) == saved
    assert final_text(store, "p") == "done"
