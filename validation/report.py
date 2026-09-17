"""QAReport — the structured result of validation, rendered in the review UI.

A report is a list of flags plus a roll-up verdict. RED flags gate the merge
(approving anyway requires an explicit override + a review_log note, see
plan §6); AMBER flags are advisory.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


@dataclass
class Flag:
    check: str          # short check name, e.g. "row_count_sanity"
    severity: Severity
    message: str        # human-readable explanation for the reviewer
    # 1-based data rows this flag names as individually wrong (the row numbers
    # the reviewer sees in the editor and the exported CSV). Empty when the
    # flag describes a whole series or the paper — those are never safe to
    # act on row by row. Only rows listed here are eligible for the review
    # page's one-click "drop flagged rows".
    rows: tuple[int, ...] = ()


@dataclass
class QAReport:
    flags: list[Flag] = field(default_factory=list)

    def add(self, check: str, severity: Severity, message: str, rows=()) -> None:
        self.flags.append(Flag(check, severity, message, tuple(int(r) for r in rows)))

    @property
    def flagged_rows(self) -> list[int]:
        """Every row some flag names as wrong, sorted, de-duplicated."""
        return sorted({r for f in self.flags for r in f.rows})

    @property
    def reds(self) -> list[Flag]:
        return [f for f in self.flags if f.severity is Severity.RED]

    @property
    def ambers(self) -> list[Flag]:
        return [f for f in self.flags if f.severity is Severity.AMBER]

    @property
    def passed(self) -> bool:
        """True when there are no RED flags (merge is allowed without override)."""
        return len(self.reds) == 0

    @property
    def verdict(self) -> Severity:
        if self.reds:
            return Severity.RED
        if self.ambers:
            return Severity.AMBER
        return Severity.GREEN

    def to_json(self) -> str:
        """Serialize for storage in prompt_runs.qa_report_json."""
        return json.dumps(
            [
                {"check": f.check, "severity": f.severity.value, "message": f.message,
                 **({"rows": list(f.rows)} if f.rows else {})}
                for f in self.flags
            ]
        )

    @classmethod
    def from_json(cls, raw: str | None) -> "QAReport":
        report = cls()
        if not raw:
            return report
        for item in json.loads(raw):
            report.add(item["check"], Severity(item["severity"]), item["message"],
                       rows=item.get("rows", ()))
        return report


# --------------------------------------------------------------------------- #
# Review tier — how much human attention a staged extraction needs.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReviewTier:
    label: str      # "Fast track" | "Standard" | "Full review"
    icon: str
    reason: str


def review_tier(verdict: Severity, has_anchor: bool, is_raster: bool) -> ReviewTier:
    """Review effort proportional to risk, from the three signals the pipeline
    already has: the QA verdict, whether the deterministic pre-pass produced an
    authoritative anchor (machine-verified marker counts and coordinates for
    the figure — only vector figures can), and whether the figures are raster
    (no ground truth at all; every point is the model's visual read).

    - Fast track: anchored vector figure and QA green. The counts and
      coordinates were verified without the model; check the metadata
      columns against the methods section and approve.
    - Full review: raster, or any red flag. Nothing here is machine-verified.
    - Standard: everything else (an anchored figure with warnings, or an
      un-anchored vector figure without reds).
    """
    if verdict is Severity.RED:
        return ReviewTier("Full review", "🔴", "red QA flags gate the merge")
    if is_raster:
        return ReviewTier("Full review", "🔴", "raster figures — no deterministic anchor; every point is a visual read")
    if has_anchor and verdict is Severity.GREEN:
        return ReviewTier("Fast track", "🟢", "vector figure with a verified marker-count anchor and no QA flags — check the metadata columns, then approve")
    if has_anchor:
        return ReviewTier("Standard", "🟡", "anchored vector figure with QA warnings — resolve the flagged rows, check metadata")
    return ReviewTier("Standard", "🟡", "no deterministic anchor for this figure (multi-panel or unverified) — spot-check the curves, check metadata")
