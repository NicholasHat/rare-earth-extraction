"""Tests for runner.collect_batch()'s wiring into anthropic_client.collect_batch_results —
specifically that it reloads each item's pinned prompt by version and threads
through the already-uploaded file_id, rather than needing the raw PDF again."""
from unittest.mock import patch

from extraction import runner
from extraction.anthropic_client import BatchRequest
from extraction.runner import BatchItem


def test_collect_batch_builds_request_items_with_reloaded_prompt_and_file_id():
    items = {
        "sha1": BatchItem(
            custom_id="sha1", figure_is_curve=True, analysis_block="block-1",
            deterministic_counts=[19, 19], prompt_version="extraction_v8",
            prompt_sha256="abc", model="claude-opus-4-8",
        ),
        "sha2": BatchItem(
            custom_id="sha2", figure_is_curve=True, analysis_block="",
            deterministic_counts=[], prompt_version="extraction_v8",
            prompt_sha256="abc", model="claude-opus-4-8",
        ),
    }
    file_ids = {"sha1": "file_1", "sha2": "file_2"}

    captured = {}

    def _fake_collect_batch_results(batch_id, requests, file_ids):
        captured["batch_id"] = batch_id
        captured["requests"] = requests
        captured["file_ids"] = file_ids
        return {}

    with patch.object(runner.anthropic_client, "collect_batch_results", side_effect=_fake_collect_batch_results), \
         patch.object(runner.prompt_loader, "load_prompt") as mock_load:
        mock_load.return_value.text = "PROMPT TEXT"
        runner.collect_batch("batch_1", items, file_ids)

    # Prompt reloaded once per distinct version (both items share extraction_v8) — not once per item.
    mock_load.assert_called_once_with("extraction_v8")
    assert captured["batch_id"] == "batch_1"
    assert captured["file_ids"] == file_ids
    by_id = {req.custom_id: req for req in captured["requests"]}
    assert by_id["sha1"] == BatchRequest("sha1", "PROMPT TEXT", "claude-opus-4-8", "block-1")
    assert by_id["sha2"] == BatchRequest("sha2", "PROMPT TEXT", "claude-opus-4-8", None)


def _item(custom_id):
    return BatchItem(
        custom_id=custom_id, figure_is_curve=True, analysis_block="",
        deterministic_counts=[], prompt_version="extraction_v9",
        prompt_sha256="abc", model="claude-sonnet-5",
    )


def test_collect_batch_isolates_a_paper_whose_postprocess_raises():
    """_postprocess does parsing, coercion and QA. Anything it raises for one
    paper must not discard the papers already parsed in the same pass."""
    items = {"sha_bad": _item("sha_bad"), "sha_ok": _item("sha_ok")}
    file_ids = {"sha_bad": "file_1", "sha_ok": "file_2"}
    raw = {"sha_bad": object(), "sha_ok": object()}

    def _fake_postprocess(response, **kw):
        if response is raw["sha_bad"]:
            raise ValueError("unexpected column shape")  # not a ParseError
        return "parsed-ok"

    with patch.object(runner.anthropic_client, "collect_batch_results", return_value=raw), \
         patch.object(runner.prompt_loader, "load_prompt") as mock_load, \
         patch.object(runner, "_postprocess", side_effect=_fake_postprocess):
        mock_load.return_value.text = "PROMPT TEXT"
        out = runner.collect_batch("batch_1", items, file_ids)

    assert isinstance(out["sha_bad"], ValueError)
    assert out["sha_ok"] == "parsed-ok"
