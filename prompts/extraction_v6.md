# Solvent-Extraction Data-Extraction Prompt — v5 (any paper)

## Task
Extract experimental extraction data from a solvent-extraction paper into an xlsx with this
**exact 26-column schema, fixed order — and nothing appended after it.** The output is these 26
columns only; do not add documentation or confidence columns.

`Reference No. | DOI | Treatment | Sources | Material Process | Si (%) | Al (%) | Zn (%) | Fe (%) | Rare Earth Elements (REY:La, Ce, Nd) | RRE composition (ppm) | RRE composition (mM) | Extractant | Extractant type | Extractant Conc. (mM) | Molar ratio of EX/REE | Extract% | Extract Temperature (oC) | pH | Separation factor (SF%) | Acid Solution | Acid Solution conc. (M) | mixing method | Stripping Temperature (oC) | Leaching time (minute) | Recovery %`

> **What changed from v4 — and why.**
> v4 was tuned by papers whose figures were unreadable, so it made **text/table the primary source
> and the figure a last resort**, and it capped output at a few "high-confidence" rows. Applied to a
> normal isotherm paper that *does* have a clean figure, this is wrong: the **figure holds the actual
> dataset** — a full curve of experimental points for every element (commonly 10–20 points each) —
> while the text usually quotes only the two end-points. v4 therefore produced ~2 rows/element where
> the correct answer is ~18.
>
> v5 fixes four things and keeps everything else:
> 1. **The primary figure is the default data source; digitise the WHOLE curve, every point, every
>    series** (Steps 4–6). Text and tables are used to read conditions, calibrate axes, and *validate*
>    the digitised curve — and they become the data source only when no digitisable figure exists.
> 2. **Molar ratio is per element** (extractant ÷ that element's own moles), never divided by a feed
>    total (Step 8).
> 3. **Output is exactly the 26 columns** — no `Source` / `Read confidence` / `raw value` columns.
>    Confidence is a judgement that guides *how you read*, not a column you ship.
> 4. **Field values are short canonical strings** (Step 8), not long prose.
>
> Everything that made v4 robust is retained: y-axis classification + conversion library (log D / D →
> %E), colour-vs-monochrome branch, vector/raster handling, and axis calibration from real tick labels.

---

## Step 0 — Map the experiments; locate each dataset
List every **distinct experiment** in the paper. An experiment is one varied parameter with the rest
held fixed (e.g. "%E vs pH at 0.5 M extractant"; "%E vs [extractant] at pH 1.75"; "%E vs temperature").
Each distinct experiment → **one sheet**.

For each experiment, decide **where its data lives**, in this order:

1. **Primary figure (default).** A 2-D plot of an extraction quantity (y) vs the varied parameter (x),
   one curve per element (or per extractant). This is normally the dataset — it contains every
   experimental point. Go to Steps 1–7 and digitise it in full.
2. **Table of points / clean model parameters.** If the paper tabulates the actual (x, %E) or (x, D)
   points, those are ground truth — use them and skip digitising. If it gives a curve model with
   *clean* parameters (e.g. `E = 100 − A·exp(−x/B)`), you may regenerate the curve from them **only
   after confirming the parameters reproduce the text end-points** (OCR'd tables are often garbled —
   see Step 6). **Read tables from the rendered PDF, not auto-extracted text.**
3. **Text only.** Fall back to the prose numbers *as the dataset* only when there is no digitisable
   figure and no usable table (figures unreadable / monochrome 3-D / derivative plots only). Then you
   will legitimately have few points per element — that is the data the paper offers, not a shortcut.

Whatever the source, the text end-points and any tabulated values are also your **validation anchors**
(Step 6). It is normal to read conditions from the text, calibrate from the axes, and take the points
from the figure — all in one extraction.

---

## Step 1 — Read the paper: conditions, axes, systems
Before touching the PDF programmatically, capture:

- **All fixed experimental conditions** (you need these for the schema regardless of data source):
  feed composition ("0.1 g/L each" = 100 ppm each; "1.9 mM total"), extractant + its concentration,
  diluent, temperature, contact/mixing time, aqueous acid/medium and its concentration, phase-volume
  ratio, mixing method, feed material/source.
- **x-axis:** pH, −log[H⁺], extractant concentration, temperature, time, phase ratio… **Read the tick
  labels — never assume pH or its range.** (−log[H⁺] at high ionic strength is the acidity proxy used
  when a glass electrode is inaccurate; treat it as the pH-column value.)
- **y-axis — this decides the Step-7 conversion:**
  - `% Extraction` / `% recovery` → digitise straight into **Extract%**.
  - `log D` → digitise, then convert (`D = 10^logD`, then `%E = 100·D/(1+D)`).
  - `D` (distribution ratio) → convert.
  - Anything else (`log D` vs `log[ext]`, slope-vs-radius, parity, 3-D surface, dendrogram) → **not a
    primary data figure; skip it.**
- **Which elements/metals** (4? 14 + Y/Sc? non-REE: Co, Li, U…?).
- **System/panel map:** one figure may hold several extractants (→ one sheet each) and several panels
  (one element per panel, or one extractant per panel). Map it explicitly before digitising.

---

## Step 2 — Detect vector vs raster (per figure page)
```python
import pdfplumber
with pdfplumber.open("paper.pdf") as pdf:
    page = pdf.pages[PAGE_INDEX]
    print(len(page.curves), len(page.lines), len(page.rects), len(page.images))
```
- Many `curves`/`lines`/`rects`, no big `image` → **vector** → Steps 3–6 can use PDF geometry.
- Page is essentially one `image` with ~0 vectors → **raster** → render at 500–600 dpi and read pixels.
- Always also **render the page and look at it** — that is how you learn colour-vs-monochrome, panel
  layout, legend position, and where the curves are separated vs crowded.

---

## Step 3 — Calibrate axes (any axis, vector or raster)
Find the plot frame (a large rect, or the black border in the raster). Detect tick positions (short
tick lines for vector; black-pixel runs at the frame edge for raster). Map **pixel → data** using the
**actual labelled tick values read from the paper**:
```python
import numpy as np
X_TICK_VALUES = [...]   # read from the axis labels (e.g. 1,2,3,4 ; or -0.8...0.0 ; or 0.05...1.0)
X_TICK_PIXELS = [...]
def px_to_x(px): return float(np.interp(px, X_TICK_PIXELS, X_TICK_VALUES))
# same for y
```
**Validate:** convert the frame corners to data units and confirm they bracket the paper's stated range.

---

## Step 4 — Identify series: COLOUR vs MONOCHROME
Look at the rendered legend and build the series map **from this paper's legend every time**.

**A. Colour-coded (typical of modern figures).** Map colour → element from the legend. Traps:
**near-duplicate hues** (several greens/blues/purples) and legends that **list each element twice**
(once for the fitted line, once for the points) — match the **marker** colour, not the line colour.
```python
for c in page.curves:
    if c.get('fill') and IN_LEGEND(c):
        print(tuple(round(v,3) for v in (c.get('non_stroking_color') or ())))
```

**B. Monochrome (all black; series distinguished by marker SHAPE).** Colour is useless; identify by
shape and PDF object type:
- filled square/circle/diamond/star → `page.curves` with `fill=True` (raster: solid blobs; separate
  square↔circle via `extent = area / bbox_area`, ~1.0 square vs ~0.79 circle).
- outline ×/+/* → `page.lines` (cluster 2–4 segments sharing a midpoint).
- open square/diamond/triangle → outline shapes; classify by vertex count / aspect.

---

## Step 5 — Digitise the FULL curve for every series
**Capture every resolvable experimental marker for every element across the whole x-range — not just
the end-points.** A clean isotherm curve typically yields ~10–20 points; that is the target, not two.

- Collect candidate centres per series (filled: bbox centre of each fill curve; outline: clustered
  segment midpoints; raster: morphological opening on each colour/shape mask, then blob centroids —
  disk radius ~5 px kills thin curve lines and keeps markers).
- **Drop legend markers** by excluding the legend region (check where it actually is — right, top,
  inside…).
- Convert every surviving centre with Step 3 and keep the full `(x, y_read)` list per element.
- Sort each element's points by x (low → high) for the output block.

---

## Step 6 — Resolve overlap; validate against text/table
Real figures crowd in the transition zone. Resolve as much as possible **before** thinning:

- Use chemistry to disambiguate crossing curves: with x = pH or −log[H⁺], %E rises with x, and
  **heavier / higher-Z elements extract at lower pH** (their pH½ is lower). A marker assigned to the
  wrong lane will violate this ordering — reassign it.
- A crowded-but-readable colour plot **is** readable; digitise it. Only a genuinely unresolvable region
  (markers fully coincident) gets fewer points — never invent points to fill it.
- **Validate the digitised curve against the paper's own numbers:**
  - low-x and high-x digitised %E should match any text end-points (target ≈ ±1–2 %E, ±0.1 in pH/x).
  - pH½ (or x½) ordering should follow atomic number.
  - if a model with **clean** parameters is given, the regenerated curve and your digitised points
    should agree; if they don't, suspect garbled parameters and trust the digitised points.
- Quick check that OCR'd parameters are usable before relying on them:
  ```python
  import math
  # does Eq. (e.g. E = 100 - A*exp(-x/B)) with the table's A,B reproduce a known text end-point?
  E = 100 - A*math.exp(-x_known/B)
  # if E is wildly off (e.g. -4000%), the table is scrambled -> digitise instead
  ```

---

## Step 7 — Convert the y-quantity to Extract%
**Extract%** holds the final percentage. Digitised %E goes straight in; otherwise convert (equal phase
volumes unless the paper says otherwise):

| Paper reports | Extract% |
|---|---|
| % Extraction / recovery | the read value, as-is |
| `log D` | `D = 10^logD`, then `100·D/(1+D)` |
| `D` | `100·D/(1+D)` |
| unequal volumes (Vorg≠Vaq) | `100·D·(Vorg/Vaq)/(1+D·(Vorg/Vaq))` |

Store the resulting **number** in Extract% (the gold output uses plain values, not formulas). Show the
conversion in your working if helpful, but the cell is the numeric %E. (`%stripping = 100/(1+D)`.)

---

## Step 8 — Experimental constants & derived fields
**Per-element composition.** Feed "0.1 g/L each" → `RRE (ppm) = 100` per element;
`RRE (mM) = ppm / atomic_mass[element]`.

**Molar ratio of EX/REE — PER ELEMENT, every time:**
```
Molar ratio = Extractant_Conc_mM / RRE_mM(this element)
```
Do **not** divide by a feed total, even when the feed is a competitive mixture of many elements. Each
element block gets its own ratio (e.g. 500 mM ÷ 0.72 mM = 694 for La, rising across the series as the
per-element mM falls).

Atomic masses: La 138.91, Ce 140.12, Pr 140.91, Nd 144.24, Sm 150.36, Eu 151.96, Gd 157.25, Tb 158.93,
Dy 162.50, Ho 164.93, Er 167.26, Tm 168.93, Yb 173.04, Lu 174.97, Y 88.91, Sc 44.96, U 238.03,
Th 232.04, Co 58.93, Ni 58.69, Cu 63.55, Zn 65.38, Li 6.94.

**Canonical field values (short strings, not prose):**
| Column | Value convention |
|---|---|
| `Reference No.` | running integer (`1` for a single paper) |
| `DOI` | full `https://doi.org/...` |
| `Treatment` | `Extraction only` / `Leaching + Extraction` / `Stripping` … |
| `Sources` | the feed origin in 1–3 words: `Nitrate Salts`, `Chloride soln`, `Sulfate soln`, `Fly ash`, `Battery waste` … |
| `Material Process` | the prep step: `adjust pH`, `dilution`, `pH adjustment` … |
| `Extractant` | the common name only: `Cyanex 272`, `D2EHPA`, `PC88A` … (no IUPAC name) |
| `Extractant type` | **row 0 of the sheet** = chemical class: `phosphoric acid based`, `phosphonic acid based`, `phosphinic acid based`, `carboxylic acid based`, `amine based`, `ionic liquid`, `solvating (neutral organophosphorus)`. **Every other row** = `"Name (conc.)"`, e.g. `Cyanex 272 (0.5M)`. |

**Experimental-condition columns** (`Extract Temperature`, `Acid Solution`, `Acid Solution conc.`,
`mixing method`, `Leaching time`, `Stripping Temperature`): fill **only when the paper states them**,
once at row 0 or block-level; leave blank otherwise. Note that `Acid Solution conc.` is left **blank
when acidity is the varied axis** (the value lives in the `pH` column). If the varied x-axis IS one of
these (temperature, time), that column carries the per-row varied value instead.

---

## Step 9 — Build the spreadsheet (26 columns, fixed)
**One sheet per experiment/extractant system.** Name it for the system (`Rare earth_C272 Extract`,
`Rare earth_EHEHPA`). Rows are element blocks, ordered light → heavy, each block sorted low-x → high-x.

**Fill pattern:**
| Columns | Filled on |
|---|---|
| `Reference No.`, `DOI`, `Treatment`, `Sources`, `Material Process` | **row 0 of the sheet only** |
| element name (col 10), `Molar ratio of EX/REE` | **first row of each element block only** |
| `RRE (ppm)`, `RRE (mM)`, `Extractant`, `Extractant type`, `Extractant Conc. (mM)` | **every row of the block** |
| `Extract%`, and the **varied-axis column** (`pH`, or `Extractant Conc.`, or `Extract Temperature`, or `Leaching time`) | **every row** |
| condition columns (temp, acid, mixing, time…) | **where reported**, at row 0 / block-level |
| anything not reported | **blank — never "N/A"** |

- One **blank row** between element blocks.
- The fixed (non-varied) parameter that defines the experiment is constant on every row (e.g. pH = 1.75
  for a concentration sweep; [extractant] = 500 mM for a pH sweep).
- **Do not append any column after `Recovery %`.**

---

## Step 10 — Verify
- **Point count reflects the figure:** each digitised curve has roughly as many rows as it has markers
  (~10–20), not 2. (Two rows/element is a red flag that the figure wasn't digitised.)
- Every element/series present in the primary figure (or table/text) is represented.
- All `Extract%` in `[0, 100]`; with x = pH/−log[H⁺], %E rises with x per element.
- Digitised end-points match the text's stated end-points within tolerance; x within the paper's range.
- `pH½` (x½) ordering follows atomic number (extraction increases with Z).
- `Molar ratio = Extractant_mM / this-element_mM` for each block (per element, not a total).
- mM feed values > 0 and physically reasonable.
- Exactly **26 columns**; no "N/A" strings; no appended documentation columns.

---

## Common pitfalls
| ✗ Wrong | ✓ Right |
|---|---|
| Stopping at the text's 2 end-points | Digitise the **whole** curve — every marker, ~10–20 pts/element (Step 5) |
| Treating the figure as a last resort | The primary figure is the default dataset; text/table validate it (Step 0) |
| Molar ratio = ext ÷ feed-total mM | Molar ratio = ext ÷ **this element's** mM, per block (Step 8) |
| Appending `Source` / `Confidence` / `raw value` columns | Ship exactly 26 columns; confidence guides reading, not output |
| Long prose in `Sources` / `Extractant` | Short canonical strings (`Nitrate Salts`, `Cyanex 272`) (Step 8) |
| Assuming y-axis is % Extraction | Could be log D or D — classify and convert (Step 7) |
| Assuming figures are colour | Could be monochrome marker-shape — branch in Step 4 |
| Re-using a previous paper's colour map | Re-derive from this legend every time |
| Matching the legend *line* colour | Match the *marker* colour; legends often list each element twice |
| Trusting OCR'd table parameters | Verify they reproduce a text end-point first; else digitise (Step 6) |
| Assuming pH ticks (0,1,2,3,4) | Read actual tick labels; `np.interp` (Step 3) |
| Splitting a mixed-feed total into per-element ppm | Record per-element ppm if given per element; else total mM, ppm blank |
| Writing "N/A" | Leave blank |

---

# OUTPUT CONTRACT (extraction_v6 additive layer — do not remove)

## Deterministic curve analysis (when present)
The user turn may include a **"DETERMINISTIC CURVE ANALYSIS"** block computed from the PDF's own
vector drawing commands BEFORE you ran. Where it marks a page **authoritative**, the per-series
marker counts it gives are **ground truth** — that figure's drawing commands contain exactly that
many points per series. Your digitised output for those series **must match those counts**; if you
produce fewer rows, you have under-digitised (you missed points, typically in a dense transition
zone) and must recover them before answering. Where it marks a page an **estimate/verify visually**
(multi-panel figures it can't cleanly separate) or a **raster image**, treat its numbers only as a
floor and digitise visually as usual. The block never tells you which series is which element — use
the legend for that, as always.

**On an authoritative page, work to a target, not by trial and error.** The count is the stopping
condition, not a thing to discover:
- Write **one** clustering pass per series (a single reasonable distance tolerance based on that
  series' own marker size) and run it. Do not try several tolerance values, compare them, and pick
  one — that guessing loop is exactly what under/over-counted in the past, and the count makes it
  unnecessary.
- If your result already matches the stated count, **trust it and move on** — do not re-render the
  page as an image to visually re-confirm a count you already have correctly. Re-rendering to "double
  check" a matching count burns vision-input tokens for no informational gain.
- If your result falls short, find and add only the missing points (look specifically in dense /
  crowded regions) — do not discard your pass and restart with a different tolerance from scratch.
- Don't narrate the search in prose ("trying eps=2... that gives 17... trying eps=3...") — run the
  pass, get the count, report the result. The reasoning that matters is *which* legend entry each
  series is, not how you arrived at a point count you were already given.
- This shortcut applies **only** to authoritative pages. Estimate/raster pages have no verified count
  to target, so give them the full visual diligence Steps 2–6 describe, including re-rendering to
  check crowded/ambiguous regions when genuinely needed.

## How you receive the paper
You are given the paper **two ways in this conversation**: as a PDF `document` block (read it
visually — legend colours, marker shapes, panel layout, axis labels) **and** as a file available in
your code execution environment. For Steps 2–6, actually run the `pdfplumber`/`numpy`/morphology code
those steps describe against that file — do vector/raster detection, axis-tick calibration, and curve
digitization programmatically rather than by eye. List the working directory first to find the
uploaded PDF's filename; install any package you need (e.g. `pip install pdfplumber`) before using it.
Use your visual read of the page only for what code can't tell you — which colour/marker maps to which
legend entry, whether a region is genuinely unresolvable versus just crowded — then digitise through
code so every point is a real measured coordinate, not an eyeballed estimate. Digitise **every distinct
experiment/figure in the paper** (Step 0), not just the first one you find.

## What to return
Return your entire answer as a **single JSON object inside one ```json fenced code block, and nothing
else outside the block.** The object has exactly two keys, `rows` and `text_endpoints`:

```json
{
  "rows": [
    {
      "Reference No.": "1",
      "DOI": "https://doi.org/10.1016/j.seppur.2011.09.015",
      "Treatment": "Extraction only",
      "Sources": "Nitrate Salts",
      "Material Process": "dilution",
      "Si (%)": null, "Al (%)": null, "Zn (%)": null, "Fe (%)": null,
      "Rare Earth Elements (REY:La, Ce, Nd)": "La",
      "RRE composition (ppm)": 100, "RRE composition (mM)": 0.72,
      "Extractant": "Cyanex 272",
      "Extractant type": "phosphinic acid based",
      "Extractant Conc. (mM)": 500,
      "Molar ratio of EX/REE": 694,
      "Extract%": 11.6,
      "Extract Temperature (oC)": 25,
      "pH": 2.0,
      "Separation factor (SF%)": null,
      "Acid Solution": "HClO4", "Acid Solution conc. (M)": null,
      "mixing method": "stirring",
      "Stripping Temperature (oC)": null, "Leaching time (minute)": 25,
      "Recovery %": null
    }
  ],
  "text_endpoints": [
    {"element": "Lu", "x_value": 0.90, "x_basis": "pH", "y_value": 39.96, "y_metric": "Extract%",
     "source_quote": "Percent extraction increases ... from 39.96% ... with increasing equilibrium pH from 0.90 ... for Lu"},
    {"element": "Lu", "x_value": 4.00, "x_basis": "pH", "y_value": 99.16, "y_metric": "Extract%",
     "source_quote": "... to 99.16% ... at pH 4.00 for Lu"}
  ]
}
```

## Rules — the JSON REPLACES the Step-9 spreadsheet layout
This flat JSON is the deliverable, not the multi-sheet xlsx of Step 9. The 26-column SCHEMA, the
figure-first digitising rules, the conversions (Step 7), and the per-element molar ratio (Step 8) all
still apply — only the *layout* changes:

1. **One object per digitized data point per element series.** Use the EXACT 26 column names above as
   keys. Numeric fields are JSON numbers (or `null` if the paper does not report them) — never strings
   like `"95%"`. Do not invent values; leave unknown fields `null`.
2. **Fully populate every row — no sparse fill, no blank separator rows.** The Step-9 convention of
   writing sheet-level values only on row 0 and block-level values only on the first row of a block
   does **not** apply here. Repeat the sheet-level values (`Reference No.`, `DOI`, `Treatment`,
   `Sources`, `Material Process`) and the block-level values (`Rare Earth Elements (REY:La, Ce, Nd)`,
   `RRE composition (ppm)`, `RRE composition (mM)`, `Extractant`, `Extractant type`,
   `Extractant Conc. (mM)`, `Molar ratio of EX/REE`) on **every** row of that block, so each row stands
   alone in a database.
3. **`Extractant type` is the chemical class on every row** (e.g. `phosphinic acid based`) — use the
   class consistently, not the `"Name (conc.)"` form; the extractant name + concentration already live
   in `Extractant` and `Extractant Conc. (mM)`.
4. **`Rare Earth Elements (REY:La, Ce, Nd)` carries the element symbol on every row** (`La`, `Ce`, …).
5. **One combined `rows` list across all experiments/sheets.** Do not split into sheets — each row
   already carries its own `pH` / `Extractant Conc. (mM)` / temperature, so the experiments remain
   distinguishable. The fixed parameter that defines an experiment is constant on its rows (e.g.
   `Extractant Conc. (mM)` = 500 for the pH sweep; `pH` = 1.75 for the concentration sweep).
6. **Digitise the WHOLE curve, not the endpoints.** A typical "% Extraction vs. pH" series has ~15–20
   points per element. Stopping at the two text endpoints is the failure this pipeline exists to catch.
7. **`text_endpoints`** — for each element/series, when the paper states a SPECIFIC numeric claim in
   prose (e.g. "39.96% to 99.16% ... for Lu"), emit one entry per stated point. `x_basis` is the
   independent variable (`"pH"` or `"extractant_conc_mM"`); `y_metric` is the dependent metric
   (`"Extract%"`, `"Recovery %"`, `"logD"`). `source_quote` is the exact sentence. These are a QA anchor,
   separate from the 26-column data — do not fold them into `rows`. If the paper states no numeric
   claim for a series, omit it (an empty list is fine).
