"""The pinned extraction prompt states the rules the 2026-10-07 audit added
and names no PDF library (the sandbox's changed under this pipeline)."""
from pathlib import Path

import config
from extraction import prompt_loader


def test_pinned_prompt_loads_and_has_a_changelog_entry():
    bundle = prompt_loader.load_prompt()
    assert bundle.version == config.EXTRACTION_PROMPT_VERSION
    changelog = (Path(__file__).resolve().parents[1] / "prompts" / "CHANGELOG.md").read_text()
    assert f"## {bundle.version} " in changelog


def test_v14_names_no_pdf_library_and_keeps_the_audit_rules():
    text = prompt_loader.load_prompt("extraction_v14").text
    assert "fitz" not in text and "PyMuPDF" not in text and "get_drawings" not in text
    assert "pages.render" in text
    assert "%E = 100·D / (D + R)" in text and "never converted with\nR = 1" in text   # phase ratio
    assert "La–Lu, Y and Sc" in text                                                 # rare earths only
    assert "never copy that total onto each element" in text
    assert "three significant figures" in text
