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
