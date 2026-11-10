"""Which figure each extracted row came from (its `Figure/Table` column —
extraction_v13–v14's parallel `row_figures` list is folded into it on parse
and on load), and why any figure was not digitised (`figures`), summarised
per figure for the review page and the pipeline comparison. Figure labels are
free text ("Fig. 2", "Figure 2(b)"); `figure_number` reduces them to the
number so labels from different sources line up.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from validation.schema import FIGURE_COLUMN

_NUMBER_RE = re.compile(r"(\d+)")


def figure_number(label) -> str | None:
    """"Fig. 2(b)" -> "2"; a table or unlabelled source -> the label itself."""
    if label is None:
        return None
    text = str(label).strip()
    if re.match(r"(?i)^(fig|figure)", text) or text.isdigit():
        m = _NUMBER_RE.search(text)
        return m.group(1) if m else text
    return text or None


def summarise(figures: list[dict], row_figures: list[str]) -> list[dict]:
    """One entry per figure: its label, rows extracted from it, and — when it
    produced none — the model's reason. Figures the rows cite but `figures`
    doesn't list still appear."""
    counts = Counter(figure_number(f) for f in row_figures if f)
    out, seen = [], set()
    for f in figures:
        key = figure_number(f.get("figure"))
        seen.add(key)
        out.append({"figure": f.get("figure"), "rows": counts.get(key, 0),
                    "digitised": bool(f.get("digitised", counts.get(key, 0) > 0)),
                    "reason": f.get("reason") or ""})
    for key, n in sorted(counts.items(), key=lambda kv: str(kv[0])):
        if key not in seen:
            out.append({"figure": f"Fig. {key}" if key and key.isdigit() else key, "rows": n,
                        "digitised": True, "reason": ""})
    return out


def row_figures(df: pd.DataFrame) -> list[str]:
    """The source of each row of `df`, in row order ('' where the row doesn't
    say); empty when no row says."""
    if FIGURE_COLUMN not in df.columns or not df[FIGURE_COLUMN].notna().any():
        return []
    return ["" if pd.isna(v) else str(v) for v in df[FIGURE_COLUMN]]
