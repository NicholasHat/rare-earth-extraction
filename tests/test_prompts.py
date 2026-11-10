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


def test_v15_returns_every_schema_column_in_order_and_says_how_to_fill_the_new_ones():
    import json
    import re

    from validation import schema

    text = prompt_loader.load_prompt("extraction_v15").text
    example = re.search(r"```json\n(\{.*?\})\n```", text[text.index("## What to return"):], re.S).group(1)
    obj = json.loads(example)
    assert obj["columns"] == schema.COLUMNS
    assert all(len(row) == len(schema.COLUMNS) for row in obj["rows"])
    assert set(obj) == {"columns", "rows", "text_endpoints", "figures"}   # row_figures is a column now
    assert "`%E = 100·D / (D + A/O)`" in text and "m_solid / V_solution" in text
    for rule in ("Co-extractant Conc. (mM)", "Total REE feed (mM)", "Contact time (minute)",
                 "`equilibrium`", "Distribution ratio (D)"):
        assert rule in text
    assert "fitz" not in text and "PyMuPDF" not in text


def test_v16_states_the_2026_11_audit_rules_and_keeps_v15s_output_shape():
    import json
    import re

    from validation import schema

    text = prompt_loader.load_prompt("extraction_v16").text
    example = re.search(r"```json\n(\{.*?\})\n```", text[text.index("## What to return"):], re.S).group(1)
    assert json.loads(example)["columns"] == schema.COLUMNS
    assert "`% recovery` → digitise straight into **Extract%**" not in text and "%stripping" not in text
    for rule in ("goes in **`Recovery %`** with\n    `Extract%` blank",          # stripping / scrubbing
                 "A varied axis is never blank", "→ 30.1", "→ 36.5",             # varied axis, vol% → mM
                 "Element groups are not elements", "shared series: Y/Gd/Eu",
                 "**organic-phase** species only", "single batch contact",
                 "Scope is equilibrium solvent extraction", "every numbered figure **and table**",
                 "never a\n  subset of the x values", "FOR PEER REVIEW", "a pre-adjusted feed is `initial`",
                 "`SF Dy/Nd`", "exact schema column\n   name of the x"):
        assert rule in text, rule
    assert "extraction_v15" not in text
