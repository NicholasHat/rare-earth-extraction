"""QAReport — the structured result of validation, rendered in the review UI.

A report is a list of flags plus a roll-up verdict. RED flags gate the merge
(approving anyway requires an explicit override + a review_log note, see
README §6 Phase A4); AMBER flags are advisory.
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


@dataclass
class QAReport:
    flags: list[Flag] = field(default_factory=list)

    def add(self, check: str, severity: Severity, message: str) -> None:
        self.flags.append(Flag(check, severity, message))

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
                {"check": f.check, "severity": f.severity.value, "message": f.message}
                for f in self.flags
            ]
        )

    @classmethod
    def from_json(cls, raw: str | None) -> "QAReport":
        report = cls()
        if not raw:
            return report
        for item in json.loads(raw):
            report.add(item["check"], Severity(item["severity"]), item["message"])
        return report
