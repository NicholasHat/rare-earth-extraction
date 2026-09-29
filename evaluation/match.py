"""Row-level comparison of two extractions of the same paper.

Rows are matched one-to-one within the same system (element + extractant,
spelling-normalised): the closest pair by a scaled distance — pH within
~0.15, extractant concentration within ~12 % — is matched first, greedily.
Ported from ree-extraction-local's scorer. Neither table is assumed correct;
`compare` reports what matched, and what each has that the other lacks.
"""
from __future__ import annotations

import math

import pandas as pd

from validation.schema import ELEMENT_COLUMN as EL

CONC = "Extractant Conc. (mM)"
PH_TOL = 0.15
LOG_CONC_TOL = 0.05          # ≈ 12 %
MATCH_MAX_DISTANCE = 1.5     # in units of the tolerances above
TEXT_FIELDS = ["DOI", "Treatment", "Sources", "Material Process", "Extractant", "Extractant type",
               "Acid Solution", "mixing method"]
NUMERIC_FIELDS = ["RRE composition (ppm)", "RRE composition (mM)", "Molar ratio of EX/REE",
                  "Extract Temperature (oC)", "Leaching time (minute)"]
NUMERIC_FIELD_REL_TOL = 0.02


def system(df: pd.DataFrame) -> pd.Series:
    """Element + extractant, spelling-normalised ("Cyanex272" == "Cyanex 272")."""
    extractant = df["Extractant"].fillna("").astype(str).str.lower().str.replace(r"\s+", "", regex=True)
    return df[EL].fillna("").astype(str) + "|" + extractant


def _distance(a: pd.Series, b: pd.Series) -> float:
    d = 0.0
    if pd.notna(a["pH"]) and pd.notna(b["pH"]):
        d += abs(a["pH"] - b["pH"]) / PH_TOL
    elif pd.notna(a["pH"]) != pd.notna(b["pH"]):
        return math.inf
    if pd.notna(a[CONC]) and pd.notna(b[CONC]) and a[CONC] > 0 and b[CONC] > 0:
        d += abs(math.log10(a[CONC] / b[CONC])) / LOG_CONC_TOL
    elif pd.notna(a[CONC]) != pd.notna(b[CONC]):
        return math.inf
    return d


def match(a: pd.DataFrame, b: pd.DataFrame) -> list[tuple[int, int]]:
    """One-to-one (index in a, index in b) pairs."""
    pairs = []
    sa, sb = system(a), system(b)
    for key in set(sa) & set(sb):
        pa, pb = a[sa == key], b[sb == key]
        cands = sorted((_distance(ra, rb), i, j) for i, ra in pa.iterrows() for j, rb in pb.iterrows())
        used_a, used_b = set(), set()
        for d, i, j in cands:
            if d > MATCH_MAX_DISTANCE:
                break
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                pairs.append((i, j))
    return pairs


def _round(x, nd=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """How far tables `a` and `b` agree, symmetrically: matched rows, the
    share of each side matched, value differences on matched rows, constant
    fields, and the systems (element + extractant) only one side has."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    pairs = match(a, b)
    out: dict = {"rows_a": len(a), "rows_b": len(b), "matched": len(pairs),
                 "share_a_matched": _round(len(pairs) / len(a)) if len(a) else None,
                 "share_b_matched": _round(len(pairs) / len(b)) if len(b) else None}
    if pairs:
        ia, ib = zip(*pairs)
        pa, pb = a.loc[list(ia)].reset_index(drop=True), b.loc[list(ib)].reset_index(drop=True)
        err = (pa["Extract%"] - pb["Extract%"]).abs().dropna()
        out["extract_pct_abs_diff"] = {"median": _round(err.median(), 2), "mean": _round(err.mean(), 2),
                                       "share_within_5": _round((err <= 5).mean())} if len(err) else None
        ph = (pa["pH"] - pb["pH"]).abs().dropna()
        out["pH_abs_diff_mean"] = _round(ph.mean()) if len(ph) else None
        fields = {}
        for col in TEXT_FIELDS:
            x = pa[col].fillna("").astype(str).str.strip().str.lower()
            y = pb[col].fillna("").astype(str).str.strip().str.lower()
            fields[col] = _round((x == y).mean())
        for col in NUMERIC_FIELDS:
            x, y = pa[col], pb[col]
            same = ((x - y).abs() <= NUMERIC_FIELD_REL_TOL * y.abs().clip(lower=1e-9)) | (x.isna() & y.isna())
            fields[col] = _round(same.mean())
        out["field_agreement"] = fields
    sa, sb = system(a).value_counts(), system(b).value_counts()
    out["systems_only_in_a"] = {k: int(v) for k, v in sa.items() if k not in sb}
    out["systems_only_in_b"] = {k: int(v) for k, v in sb.items() if k not in sa}
    out["systems_count_differs"] = {k: [int(sa[k]), int(sb[k])] for k in sa.index
                                    if k in sb and abs(int(sa[k]) - int(sb[k])) >= 3}
    return out


def score(pred: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """`agreement` read against a reference: recall = share of the reference
    matched, precision = share of the prediction matched."""
    out = agreement(pred, ref)
    out["recall"], out["precision"] = out.pop("share_b_matched"), out.pop("share_a_matched")
    return out
