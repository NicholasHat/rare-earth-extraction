"""parse_output: tolerant JSON extraction + schema coercion + coercion-failure count."""
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


def test_parses_compact_positional_rows():
    raw = """```json
{
  "columns": ["Rare Earth Elements (REY:La, Ce, Nd)", "pH", "Extract%"],
  "rows": [
    ["Yb", 1.5, 12.0],
    ["Yb", 2.0, 30.0]
  ],
  "text_endpoints": []
}
```"""
    parsed = parse_output.parse(raw)
    assert len(parsed.df) == 2
    assert list(parsed.df.columns) == parse_output.schema.COLUMNS
    assert parsed.df["pH"].tolist() == [1.5, 2.0]
    assert parsed.df["Extract%"].tolist() == [12.0, 30.0]
    assert parsed.coercion_failures == 0


def test_positional_rows_without_columns_raises():
    import pytest

    raw = '{"rows": [["Yb", 1.5, 12.0]], "text_endpoints": []}'
    with pytest.raises(parse_output.ParseError):
        parse_output.parse(raw)
