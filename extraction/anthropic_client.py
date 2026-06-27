"""Thin wrapper over the Anthropic Messages API for figure extraction.

Gives Claude the paper PDF two ways in the same turn: a `document` block (so
it can visually read the figure — legend colours, marker shapes, panel
layout) and a `container_upload` (so the code-execution tool can open the
same file with pdfplumber/numpy for vector/raster detection, axis
calibration, and point digitization, per the prompt's Steps 2-6). Both need
the PDF uploaded once via the Files API first.

Streaming is used because a fully-digitized multi-element, multi-figure
extraction plus the code-execution transcript is a large output; non-
streaming would risk the SDK's HTTP timeout.
"""
from __future__ import annotations

import anthropic

import config

_BETAS = ["files-api-2025-04-14"]
_CODE_EXECUTION_TOOL = {"type": "code_execution_20260120", "name": "code_execution"}

# A short instruction in the user turn; the real rules live in the system prompt.
_USER_INSTRUCTION = (
    "Extract the data from this paper according to your instructions. The same "
    "PDF is also available in your code execution environment — list the "
    "working directory to find it, install any package you need, and use "
    "pdfplumber/numpy there for axis calibration and point digitization as "
    "Steps 2-6 describe. Return only the single JSON object described in the "
    "OUTPUT CONTRACT."
)


def extract(prompt_text: str, pdf_bytes: bytes, *, model: str | None = None) -> str:
    """Run one extraction. Returns the concatenated text of the model response."""
    model = model or config.EXTRACTION_MODEL
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    uploaded = client.beta.files.upload(file=("paper.pdf", pdf_bytes, "application/pdf"))
    try:
        with client.beta.messages.stream(
            model=model,
            max_tokens=128000,
            thinking={"type": "adaptive"},
            # The extraction prompt is identical across every paper in a batch;
            # cache it so only the first call in a run pays full input price for
            # it (1h TTL since each call's own runtime can exceed the 5min default).
            system=[
                {
                    "type": "text",
                    "text": prompt_text,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            betas=_BETAS,
            tools=[_CODE_EXECUTION_TOOL],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": uploaded.id},
                        },
                        {"type": "container_upload", "file_id": uploaded.id},
                        {"type": "text", "text": _USER_INSTRUCTION},
                    ],
                }
            ],
        ) as stream:
            message = stream.get_final_message()
    finally:
        client.beta.files.delete(uploaded.id)

    parts = [block.text for block in message.content if block.type == "text"]
    return "\n".join(parts).strip()
