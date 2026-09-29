"""Offline evaluation of extraction results — no API calls.

`match` pairs the rows of two 26-column tables of the same paper and measures
how far they agree; `sources` loads a paper's result from this pipeline
(staged, approved or failed) and from the sibling ree-extraction-local
pipeline; `compare` is the CLI that runs both over a set of papers:

    python -m evaluation.compare path/to/papers/ --local-runs ../ree-extraction-local/data/runs

With no hand-checked reference, agreement between two independent pipelines
is the signal: where they agree, both are probably right; where they
disagree, a reviewer looks. A reference CSV per paper (`--ref-dir`) turns the
same report into recall / precision / error against ground truth.
"""
