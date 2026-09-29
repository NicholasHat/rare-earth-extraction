"""Which figure each extracted row came from, and why any figure was not
digitised (extraction_v13+ `row_figures` / `figures`), summarised per figure
for the review page and the pipeline comparison. Figure labels are free text
("Fig. 2", "Figure 2(b)"); `figure_number` reduces them to the number so
labels from different sources line up.
"""
from __future__ import annotations

import re
from collections import Counter

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
    counts = Counter(figure_number(f) for f in row_figures)
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
