"""Licence provenance for a paper, and the open-licence test the merge gate uses.

Only openly-licensed papers may reach the master DB (database/merge.py refuses
anything else). A paper's licence is looked up from the open-access manifest
by its PDF's sha256 when it is ingested (`from_manifest`), or set by hand at
review time (`License(..., origin="manual")` — saved on the staged paper with
extraction.staging.set_license, or on an already-merged paper with
database.papers_repo.set_license). Which ids count as open is config.OPEN_LICENSES.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import config

from . import manifest

_VERSION_SUFFIX = re.compile(r"-\d+(\.\d+)*(-[a-z]+)?$")


@dataclass(frozen=True)
class License:
    id: str                         # normalised lower-case id, e.g. 'cc-by'
    evidence: str | None = None     # where it was read (a licence statement, a page)
    url: str | None = None          # source URL of that evidence
    origin: str = "manual"          # 'manifest' | 'manual'

    @property
    def is_open(self) -> bool:
        return is_open(self.id)

    @classmethod
    def of(cls, raw_id: str, evidence: str | None = None, url: str | None = None,
           origin: str = "manual") -> "License":
        """A License with its id normalised ("CC BY 4.0" -> "cc-by")."""
        lid = normalise(raw_id)
        if lid is None:
            raise ValueError("a licence needs an id")
        return cls(lid, evidence or None, url or None, origin)


def normalise(raw: str | None) -> str | None:
    """'CC BY 4.0' / 'cc_by' / 'CC-BY' -> 'cc-by'; 'Public Domain' -> 'public-domain'."""
    if raw is None:
        return None
    text = re.sub(r"[\s_]+", "-", str(raw).strip().lower())
    text = _VERSION_SUFFIX.sub("", re.sub(r"-+", "-", text)).strip("-")
    return text or None


def is_open(license: "License | str | None") -> bool:
    lid = license.id if isinstance(license, License) else normalise(license)
    return lid is not None and lid in config.OPEN_LICENSES


def from_manifest(sha: str, manifest_path: Path | None = None) -> License | None:
    """The manifest's licence for the PDF with this sha256, if it lists one that
    was confirmed. An `unverified` row's licence is index metadata nobody found
    stated in the PDF or on the publisher's page, so it does not count."""
    entry = manifest.by_sha(manifest_path).get(sha.lower())
    if entry is None or not entry.license or entry.tier == "unverified":
        return None
    return License.of(entry.license, entry.license_evidence, entry.source_url, origin="manifest")


def resolve(sha: str, current: License | None) -> License | None:
    """The licence to record for a paper: the one already on it, else the manifest's."""
    return current if current is not None else from_manifest(sha)
