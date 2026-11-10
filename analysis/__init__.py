"""Analysis view over the approved data (the Analysis page, pages/4_Analysis.py).

Pure and Streamlit-free, so every number the page draws is unit-tested:

  data     — the approved rows, read through `v_current_best` on a read-only
             connection, with a readable paper label per row
  curves   — rows grouped into curves (one paper, figure, element and set of
             fixed conditions — validation.curves' curve key), along a chosen x
  derived  — quantities computed from reported ones: D from %E and the phase
             ratio, the pH of 50 % extraction, separation factors
  slopes   — straight-line fits of log D vs pH and vs log[extractant]
  charts   — the Altair specs the page renders

Nothing here writes to the database or calls a model.
"""
