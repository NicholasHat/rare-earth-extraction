"""parse_output: tolerant JSON extraction + schema coercion + coercion-failure count."""
import json

from extraction import parse_output


def test_parses_fenced_json_block():
    raw = """Here is the extraction:

```json
{
  "rows": [
    {"Rare Earth Elements (REY:La, Ce, Nd)": "Yb", "pH": 1.5, "Extract%": 12.0},
    {"Rare Earth Elements (REY:La, Ce, Nd)": "Yb", "pH": 2.0, "Extract%": 30.0}
  ],
  "text_endpoints": [
    {"element": "Yb", "x_value": 3, "x_basis": "pH", "y_value": 95, "y_metric": "Extract%", "source_quote": "95% at pH 3"}
  ]
}
```
"""
    parsed = parse_output.parse(raw)
    assert len(parsed.df) == 2
    assert list(parsed.df.columns) == parse_output.schema.COLUMNS  # full 26-col contract
    assert parsed.df["pH"].tolist() == [1.5, 2.0]
    assert len(parsed.text_endpoints) == 1
    assert parsed.text_endpoints[0]["element"] == "Yb"
    assert parsed.coercion_failures == 0


def test_counts_garbled_numeric_as_coercion_failure():
    raw = '{"rows": [{"pH": "abc", "Extract%": 50}], "text_endpoints": []}'
    parsed = parse_output.parse(raw)
    assert parsed.coercion_failures == 1          # "abc" in a numeric column
    assert parsed.df["pH"].isna().all()           # stored as null, not text


def test_falls_back_to_brace_span_without_fence():
    raw = 'noise {"rows": [{"pH": 1.0}], "text_endpoints": []} trailing'
    parsed = parse_output.parse(raw)
    assert len(parsed.df) == 1


def test_raises_when_no_json():
    import pytest

    with pytest.raises(parse_output.ParseError):
        parse_output.parse("the model said nothing useful")


def _positional_row(overrides: dict) -> list:
    """Build one full 26-value row (in schema.COLUMNS order) with overrides."""
    row = {col: None for col in parse_output.schema.COLUMNS}
    row.update(overrides)
    return [row[col] for col in parse_output.schema.COLUMNS]


def test_parses_compact_positional_rows():
    row1 = _positional_row({"Rare Earth Elements (REY:La, Ce, Nd)": "Yb", "pH": 1.5, "Extract%": 12.0})
    row2 = _positional_row({"Rare Earth Elements (REY:La, Ce, Nd)": "Yb", "pH": 2.0, "Extract%": 30.0})
    raw = "```json\n" + json.dumps({
        "columns": parse_output.schema.COLUMNS,
        "rows": [row1, row2],
        "text_endpoints": [],
    }) + "\n```"
    parsed = parse_output.parse(raw)
    assert len(parsed.df) == 2
    assert list(parsed.df.columns) == parse_output.schema.COLUMNS
    assert parsed.df["pH"].tolist() == [1.5, 2.0]
    assert parsed.df["Extract%"].tolist() == [12.0, 30.0]
    assert parsed.coercion_failures == 0


def test_positional_rows_tolerate_reordered_columns():
    # A reordered (but complete and correctly-spelled) columns list is safe:
    # each row is still keyed by name, not position, before schema coercion.
    reordered = list(reversed(parse_output.schema.COLUMNS))
    row = [
        ("Yb" if col == "Rare Earth Elements (REY:La, Ce, Nd)" else (1.5 if col == "pH" else None))
        for col in reordered
    ]
    raw = json.dumps({"columns": reordered, "rows": [row], "text_endpoints": []})
    parsed = parse_output.parse(raw)
    assert parsed.df["pH"].tolist() == [1.5]
    assert parsed.df["Rare Earth Elements (REY:La, Ce, Nd)"].tolist() == ["Yb"]


def test_positional_rows_without_columns_raises():
    import pytest

    raw = '{"rows": [["Yb", 1.5, 12.0]], "text_endpoints": []}'
    with pytest.raises(parse_output.ParseError):
        parse_output.parse(raw)


def test_positional_rows_with_misspelled_column_raises():
    # A typo'd column name must fail loudly rather than let coerce_schema
    # silently null out the real data that was keyed under the wrong string.
    import pytest

    columns = list(parse_output.schema.COLUMNS)
    columns[columns.index("pH")] = "PH"  # plausible model typo
    row = _positional_row({"pH": 1.5})
    with pytest.raises(parse_output.ParseError, match="does not match the 26-column schema"):
        parse_output.parse(json.dumps({"columns": columns, "rows": [row], "text_endpoints": []}))


def test_positional_rows_with_wrong_row_length_raises():
    # A short/long row must fail loudly rather than let zip() silently
    # misalign every value after the gap.
    import pytest

    raw = json.dumps({
        "columns": parse_output.schema.COLUMNS,
        "rows": [["Yb", 1.5]],  # 2 values against a 26-column header
        "text_endpoints": [],
    })
    with pytest.raises(parse_output.ParseError, match="row length"):
        parse_output.parse(raw)
