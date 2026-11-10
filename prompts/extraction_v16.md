# Solvent-Extraction Data-Extraction Prompt — extraction_v16 (any paper)

## Task
Extract experimental extraction data from a solvent-extraction paper into an xlsx with this
**exact 35-column schema, fixed order — and nothing appended after it.** The first 26 columns are the
lab's spreadsheet; the last 9 (from `Diluent` to `Figure/Table`) hold conditions the lab's columns
have no place for. The output is these 35 columns only; do not add documentation or confidence columns.

`Reference No. | DOI | Treatment | Sources | Material Process | Si (%) | Al (%) | Zn (%) | Fe (%) | Rare Earth Elements (REY:La, Ce, Nd) | RRE composition (ppm) | RRE composition (mM) | Extractant | Extractant type | Extractant Conc. (mM) | Molar ratio of EX/REE | Extract% | Extract Temperature (oC) | pH | Separation factor (SF%) | Acid Solution | Acid Solution conc. (M) | mixing method | Stripping Temperature (oC) | Leaching time (minute) | Recovery % | Diluent | Co-extractant | Co-extractant Conc. (mM) | Phase ratio (O/A) | Contact time (minute) | Distribution ratio (D) | Total REE feed (mM) | pH type | Figure/Table`

The primary figure normally holds the dataset — every experimental point of every series — while
the text quotes only a few of them. Text and tables supply conditions, calibrate axes and validate
the digitised points; they become the data source only when no digitisable figure exists.

---

## Step 0 — Map the experiments; locate each dataset
List every **distinct experiment** in the paper. An experiment is one varied parameter with the rest
held fixed (e.g. "%E vs pH at 0.5 M extractant"; "%E vs [extractant] at pH 1.75"; "%E vs temperature").
Each distinct experiment → **one sheet**.

**Account for everything.** Every numbered figure **and table**, and every sentence stating a numeric
REE result not already covered by a row, either yields rows or gets a reason in `figures`.
**Scope is equilibrium solvent extraction** (batch contacts; %E vs contact time is in scope). Emulsion or
supported liquid membranes, hollow-fibre pertraction and other transport experiments (removal % vs time
through a membrane with simultaneous stripping) get no rows — say so in `figures`.

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
  any **co-extractant** (synergist or modifier: TBP, TOPO, an amine, a second acid extractant) + its
  concentration, the **diluent**, temperature, **contact time** (shaking / equilibration time of the
  two phases), aqueous acid/medium and its concentration, mixing method, feed material/source, whether
  the pH is the **equilibrium** (final) or the **initial** (feed) pH, and the **phase ratio** — the
  aqueous and organic volumes (liquid–liquid), or the solution volume and the mass of resin / sorbent /
  impregnated solid (solid–liquid). Step 7 needs it for every D or K_d. Read it from the caption first,
  then the methods.
- **x-axis:** pH, −log[H⁺], extractant concentration, temperature, time, phase ratio… **Read the tick
  labels — never assume pH or its range.** (−log[H⁺] at high ionic strength is the acidity proxy used
  when a glass electrode is inaccurate; treat it as the pH-column value.)
- **y-axis — this decides the Step-7 conversion:**
  - `% Extraction` → digitise straight into **Extract%**. A **back-extraction** (`% stripping`,
    `% scrubbing`, a stripping/scrubbing "recovery" or "efficiency") goes in **`Recovery %`** with
    `Extract%` blank, and `Treatment` is `Stripping` or `Scrubbing` as the paper names it.
  - `log D` / `D` / `K_d` / `log K_d` (distribution ratio or coefficient) → digitise, then convert with
    the phase ratio (Step 7). A log-scaled axis labelled 1, 10, 100… already shows D itself.
  - `log D` vs `log[extractant]` → **check before skipping.** These are usually the slope-analysis
    re-plot of a concentration sweep you are already digitising — skip those. But when the plot's
    legend covers **conditions (pH values, elements) no other figure reports**, those series are
    real experimental data that appear nowhere else in the paper: digitise them, converting both
    axes (`[ext] = 10^x` in M → mM for the x-column; `D = 10^y`, then %E per Step 7).
    Only the series that duplicate an already-digitised sweep are skipped.
  - Anything else (slope-vs-radius, parity, 3-D surface, dendrogram) → **not a primary data figure;
    skip it.**
- **Which elements.** This database holds the **rare earth elements only: La–Lu, Y and Sc.** A row is
  one of those elements. Other metals the paper studies (Pb, Co, Ni, Li, U, Th, Fe…) get **no rows** —
  not even as competitors in the same experiment. A paper that studies no rare earth element returns an
  empty `rows` list, and its `figures` entries say so ("Pb only — not a rare earth element").
  **Element groups are not elements:** LREE / MREE / HREE / TREE / "total REE" values get no rows
  (`figures`: "group-level only"). But a bar or series of a grouped chart that is a single element (e.g. Y)
  **is** digitised, and a legend entry naming several elements with one shared series gives one row per
  named element, with `shared series: Y/Gd/Eu` in `Material Process`.
- **System/panel map:** one figure may hold several extractants (→ one sheet each) and several panels
  (one element per panel, or one extractant per panel). Map it explicitly before digitising.

---

## Step 2 — Detect vector vs raster (per figure page)
The **DETERMINISTIC CURVE ANALYSIS** block in the user turn has already classified every figure page
it found — authoritative / estimate (vector) or **raster image** (with the figure's bbox). Take that
as the answer; probe a page yourself only if the block does not list it:
```python
import pdfplumber
page = pdfplumber.open(PDF_PATH).pages[PAGE_INDEX]
print(len(page.curves) + len(page.rects) + len(page.lines), [(i["width"], i["height"]) for i in page.images])
```
- Many drawings, no large image → **vector** → Steps 3–6 can use PDF geometry.
- Essentially one large image and ~0 drawings → **raster** → render the figure's bbox at 300 dpi with the
  SANDBOX TOOLKIT's `pages.render` (the
  toolkit's pixel thresholds are tuned for that scale; the block gives the image's own resolution —
  do not re-render at another scale) and read pixels with the SANDBOX TOOLKIT.
- Always also **look at the page in the PDF document** — that is how you learn colour-vs-monochrome,
  panel layout, legend position, and where the curves are separated vs crowded. Images you render in
  the sandbox are never shown to you; they are pixels for your code only.
- **Overlaid PDFs.** If tick labels, legends or markers appear twice on a page (a hidden earlier-draft
  layer under the final figure — common in MDPI PDFs marked "FOR PEER REVIEW"), the PDF's vector
  geometry holds both layers: digitise only what is visible in the rendered page.

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
**Raster figures:** do not write a frame or tick detector — follow the SANDBOX TOOLKIT block's
workflow: `raster.find_panels` finds every panel of the whole figure, `legend.label_centres` locates
each axis's tick labels, and you read their *values* off the figure yourself; `calibrate.fit_ticks`
pairs them, picks linear vs log10 and returns `cal.pixel_to_data(px)`.

**Validate:** convert the frame corners to data units and confirm they bracket the paper's stated range.

---

## Step 4 — Identify series: COLOUR vs MONOCHROME
Look at the legend in the PDF document and build the series map **from this paper's legend every time**.

**A. Colour-coded (typical of modern figures).** Map colour → element from the legend. Traps:
**near-duplicate hues** (several greens/blues/purples) and legends that **list each element twice**
(once for the fitted line, once for the points) — match the **marker** colour, not the line colour.
```python
for o in page.curves + page.rects:            # pdfplumber; positions in points, origin top-left
    if o.get("fill") and IN_LEGEND((o["x0"], o["top"], o["x1"], o["bottom"])):
        print(o["non_stroking_color"])        # a bare number is a grey level: 0 black ... 1 white
```

**B. Monochrome (all black; series distinguished by marker SHAPE).** Colour is useless; identify by
shape and PDF object type:
- filled square/circle/diamond/star → filled paths in `page.curves` / `page.rects` (`fill` set) (raster: the toolkit's
  `legend.match_markers` matches every marker against the legend's own swatches, so markers come back
  grouped by series — you map each swatch to its legend label).
- outline ×/+/* → unfilled paths of 2–4 short line segments sharing a midpoint.
- open square/diamond/triangle → outline shapes; classify by vertex count / aspect.

---

## Step 5 — Digitise the FULL curve for every series
**Capture every resolvable experimental marker for every element across the whole x-range — not just
the end-points.** The figure's own marker count is the target: often 10–20 per series, sometimes only
4–7. **Never add a point the figure does not show** — above all, never sample points along a fitted
or guide line at even x steps: those lines are fits, not data, and QA rejects a series that is evenly
spaced and exactly on one line.

- Collect candidate centres per series (filled: bbox centre of each fill curve; outline: clustered
  segment midpoints; raster: the SANDBOX TOOLKIT block's legend workflow — `legend.find_swatches` then
  one `legend.match_markers` call per panel, which erases axis/grid lines and in-plot text itself; only a
  figure whose legend has no drawn swatches falls back to `raster.detect_markers_in_image`. Do not
  re-implement any of it).
- **Drop legend markers** by excluding the legend region (check where it actually is — right, top,
  inside…).
- **Drop in-plot text the same way.** Panel titles printed inside the frame (`1.0 M EHEHPA`, `Sm`),
  annotations (`Slope 3`), and any other lettering inside the plot area are **regions to exclude
  before digitising**, exactly like the legend. On a monochrome raster figure this matters most: a
  block of text is dark pixels, and after the morphological opening its remnants look like the
  **filled-marker** series — a panel title at log D ≈ 0.7 turns into a run of ~83 %E "points". Locate
  every text region on the rendered page (top-left corners are the usual spot) and mask it out; then
  confirm no surviving centre sits inside a text bounding box.
- **Overlapping monochrome series: digitise what separates, omit what doesn't.** When three or more
  marker shapes sit on top of each other within a marker's width across most of the x-range (the
  crowded heavy-element cluster at the bottom of a raster panel is the classic case), only the series
  you can follow unambiguously get rows. Do **not** apportion blobs among the tangled series by
  guesswork — a series with no rows is a gap the reviewer can see; a series of invented points is not.
- **Every row is a marker you located, at that marker's own x.** Count each series' markers and emit
  one row per marker. Never place a row at an x value taken from the caption's range or the paper's
  list of conditions ("0.001–2 M") instead of from a marker: a marker at an x the caption doesn't list
  is still data, and a listed x with no marker is not. Reading values "by eye" anchored to the text is
  how markers went missing before — digitise through code.
- **Every marker of every series.** In kinetics and other dense plots every marker is a row — never a
  subset of the x values. A marker partly hidden behind another is digitised where its visible edge
  fixes its centre; a marker you knowingly omit is listed (series, x) in that figure's `figures` reason.
  A drawn origin marker (0, 0) that is not a measurement is not a row.
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
- **Fit each series as a straight line in log D vs pH (or −log[H⁺]) and look at the residuals.** The
  cation-exchange systems these papers study are linear there (slope ≈ 2–3.5) — that is why they plot
  log D. A point sitting far off its own series' line, especially a run of two or three at the same
  height while the rest of the series is an order of magnitude lower, is **not data**: it is in-plot
  text or a neighbouring series' marker. Remove it. Do this **before** the text end-point validation
  below, so an artifact never gets "validated" by chance.
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
**Extract%** holds the final percentage. Digitised %E goes straight in; a distribution ratio converts
with the experiment's **phase ratio**, recorded in `Phase ratio (O/A)`:

`%E = 100·D / (D + A/O)`, where `A/O = 1 / Phase ratio (O/A)`

| System | `Phase ratio (O/A)` | D's unit |
|---|---|---|
| liquid–liquid | `V_org / V_aq` of the contact the row describes (equal volumes → 1) | dimensionless |
| solid sorbent: resin, impregnated resin, extraction resin, any solid phase | `m_solid / V_solution` in **g/mL** (e.g. 0.05 g in 5 mL → 0.01) | mL/g (convert L/kg = mL/g; L/g × 1000) |

With those definitions the one formula holds for both: 0.05 g of resin in 5 mL gives A/O = 100 mL/g,
so D = 24 mL/g is 100·24/(24 + 100) = 19 %. `log D` and `log K_d` first become `10^value`. **A solid
sorbent's D or K_d is never converted with A/O = 1** — that turns D = 24 mL/g at V/m = 100 into 96 %
instead of 19 %. If a solid-phase paper states no solution volume and solid mass for that experiment,
do not convert: give that figure no rows and say why in `figures`. A liquid–liquid paper that states
no volumes is converted with A/O = 1, but its `Phase ratio (O/A)` stays blank — that column records
only what the paper states.

Store the resulting **number** in Extract% (the gold output uses plain values, not formulas). Show the
conversion in your working if helpful, but the cell is the numeric %E.

**Only where it means %E:** convert D → %E only for a **single batch contact** at a stated phase ratio.
Successive contacts with a loaded organic, counter-current circuits and mixer-settler stages keep the
reported quantity (D, or the paper's percentage), with the setup named in `Treatment` / `Material
Process` (`4th successive contact`). A stripping or scrubbing D is stored as reported, never converted.

**Keep D as well.** When the paper reports D (or log D, or K_d) — on an axis or in a table — put D
itself in `Distribution ratio (D)` (log D → `10^value`; K_d in mL/g as is) next to the %E you derived.
When the paper reports %E, leave
`Distribution ratio (D)` blank — do not back-calculate it. Fill `Phase ratio (O/A)` whenever the paper
states it, whatever the y-quantity. QA recomputes %E from D and the phase ratio on every row that has
all three.

---

## Step 8 — Experimental constants & derived fields
**Per-element composition.** Feed "0.1 g/L each" → `RRE (ppm) = 100` per element;
`RRE (mM) = ppm / atomic_mass[element]`. `RRE composition` is **this element's** concentration. When the
paper gives only a total (e.g. "0.21 M total REE"), never copy that total onto each element: if it also
gives the feed's composition **with its basis** (mol % or mass %), compute this element's share;
otherwise leave both `RRE composition` columns and `Molar ratio` blank.

**Total feed.** The total rare-earth concentration of a mixed feed goes in `Total REE feed (mM)` on
every row (0.21 M total → 210), alongside — never instead of — the per-element columns above. When the
experiment **varies** the total feed (e.g. 0.05–0.99 M), `Total REE feed (mM)` is that row's x value.

**Mixed extractant systems.** `Extractant` is the primary extractant — the one whose concentration is in
`Extractant Conc. (mM)` and in the molar ratio. A synergist or modifier added to it (TBP, TOPO,
Alamine 336, Versatic 10, a second acidic extractant) goes in `Co-extractant` (its name only) with its
concentration in `Co-extractant Conc. (mM)` (0.5 M → 500). Never fold it into the extractant's name
(`Cyanex 272 + TBP (0.5 M)` is wrong: `Cyanex 272` / `TBP` / 500). When the co-extractant's concentration
is the varied axis, it is that row's x value. `Extractant type` stays the primary extractant's class.
`Co-extractant` is an **organic-phase** species only; an aqueous complexing agent (lactic or citric acid,
EDTA, DTPA) goes in `Material Process`. A pure ionic liquid that extracts by itself is the `Extractant`,
not the `Diluent`.

**A varied axis is never blank.** If a figure's x variable has a schema column, every row carries its x.
Convert vol% of a neat extractant with its density and molar mass: mM per vol% = 10 000 · ρ / M
(D2EHPA/P204 0.97 g/mL, 322.4 g/mol → 30.1; TBP 0.973, 266.3 → 36.5; PC88A/P507/EHEHPA 0.95, 306.4 → 31.0;
Cyanex 272 0.92, 290.4 → 31.7). If the x fits no schema column (wt% in a mixture of unknown density; a
formulation ratio), the sweep gets no rows and the reason goes in `figures`. A fixed concentration stated
only in wt%: leave the mM blank and say so in `Material Process`.

**Acid Solution.** `Acid Solution` / `Acid Solution conc. (M)` are the acid or salt medium of the aqueous
phase the row's percentage refers to — the feed for extraction rows, the strip or scrub solution for
stripping/scrubbing rows; never an internal or receiving phase of another step. A salting-out agent that
is the swept variable may go there, named as the salt (`CaCl2`).

**Diluent.** The organic diluent as the paper names it (`kerosene`, `n-heptane`, `toluene`, `Exxsol D80`);
when the diluent is the varied parameter, each row carries its own. A solid sorbent has none (blank).

**Contact time vs leaching time.** `Contact time (minute)` is how long the two phases were shaken or
stirred together in the extraction (or sorption) step — the x of a kinetics plot. `Leaching time
(minute)` is only for a leaching step that dissolved a solid feed (ore, ash, magnet, phosphor) before
extraction; it is blank in a paper with no leaching step.

**pH type.** `equilibrium` when the paper's pH is the equilibrium / final / raffinate pH, `initial` when it
is the initial / feed / aqueous-before-contact pH; blank when the paper doesn't say. Never guess — the two
are not comparable. A fixed pH carried into later experiments keeps the type of the experiment that set
it (a pre-adjusted feed is `initial`).

**Source.** `Figure/Table` names where the row's numbers were taken from, as the paper labels it: `Fig. 2`,
`Fig. 4(b)`, `Table 3`, or `text` for a value read from the prose (the table, when a figure re-plots a
table and you used the table). Every row has one.

**Separation factor.** Fill `Separation factor (SF%)` only with its pair and direction in `Material
Process` (`SF Dy/Nd` = D_Dy / D_Nd), on both elements' rows of that point.

**Precision.** Give every computed number — RRE mM, molar ratio, %E converted from D — to at least
**three significant figures** (a molar ratio of 0.178 is `0.178`, not `0.2`); QA recomputes them.

**Molar ratio of EX/REE — PER ELEMENT, every time:**
```
Molar ratio = Extractant_Conc_mM / RRE_mM(this element)
```
Do **not** divide by a feed total, even when the feed is a competitive mixture of many elements. Each
element block gets its own ratio (e.g. 500 mM ÷ 0.72 mM = 694 for La, rising across the series as the
per-element mM falls).

Atomic masses: La 138.91, Ce 140.12, Pr 140.91, Nd 144.24, Sm 150.36, Eu 151.96, Gd 157.25, Tb 158.93,
Dy 162.50, Ho 164.93, Er 167.26, Tm 168.93, Yb 173.04, Lu 174.97, Y 88.91, Sc 44.96.

**Canonical field values (short strings, not prose):**
| Column | Value convention |
|---|---|
| `Reference No.` | running integer (`1` for a single paper) |
| `DOI` | full `https://doi.org/...` |
| `Treatment` | `Extraction only` / `Leaching + Extraction` / `Stripping` / `Scrubbing` |
| `Sources` | the feed origin in 1–3 words: `Nitrate Salts`, `Chloride soln`, `Sulfate soln`, `Fly ash`, `Battery waste` … |
| `Material Process` | the prep step (`adjust pH`, `dilution`), then any notes the rules above put here (`SF Dy/Nd`, `shared series: Y/Gd/Eu`, `+ 0.2 M lactic acid`); the same on every row of a series |
| `Extractant` | the common name only: `Cyanex 272`, `D2EHPA`, `PC88A` … (no IUPAC name) |
| `Extractant type` | **row 0 of the sheet** = chemical class: `phosphoric acid based`, `phosphonic acid based`, `phosphinic acid based`, `carboxylic acid based`, `amine based`, `ionic liquid`, `solvating (neutral organophosphorus)`. **Every other row** = `"Name (conc.)"`, e.g. `Cyanex 272 (0.5M)`. |

**Experimental-condition columns** (`Extract Temperature`, `Acid Solution`, `Acid Solution conc.`,
`mixing method`, `Leaching time`, `Stripping Temperature`, `Diluent`, `Co-extractant`,
`Co-extractant Conc.`, `Phase ratio (O/A)`, `Contact time`, `Total REE feed`, `pH type`): fill **only when
the paper states them**, once at row 0 or block-level; leave blank otherwise. Note that `Acid Solution
conc.` is left **blank when pH is the varied axis** (the acidity lives in the `pH` column). If the
varied x-axis IS one of these (acid concentration, temperature, contact time, co-extractant
concentration, total feed, phase ratio), that column carries the per-row varied value instead.

---

## Step 9 — Build the spreadsheet (35 columns, fixed)
**One sheet per experiment/extractant system.** Name it for the system (`Rare earth_C272 Extract`,
`Rare earth_EHEHPA`). Rows are element blocks, ordered light → heavy, each block sorted low-x → high-x.

**Fill pattern:**
| Columns | Filled on |
|---|---|
| `Reference No.`, `DOI`, `Treatment`, `Sources`, `Material Process` | **row 0 of the sheet only** |
| element name (col 10), `Molar ratio of EX/REE` | **first row of each element block only** |
| `RRE (ppm)`, `RRE (mM)`, `Extractant`, `Extractant type`, `Extractant Conc. (mM)` | **every row of the block** |
| `Extract%` (or `Recovery %`), `Figure/Table`, and the **varied-axis column** (`pH`, or `Extractant Conc.`, `Co-extractant Conc.`, `Extract Temperature`, `Contact time`, `Total REE feed`, `Phase ratio`…) | **every row** |
| condition columns (temp, acid, mixing, time…) | **where reported**, at row 0 / block-level |
| anything not reported | **blank — never "N/A"** |

- One **blank row** between element blocks.
- The fixed (non-varied) parameter that defines the experiment is constant on every row (e.g. pH = 1.75
  for a concentration sweep; [extractant] = 500 mM for a pH sweep).
- **Do not append any column after `Figure/Table`.**

---

## Step 10 — Verify
- **Point count reflects the figure:** each digitised series has as many rows as the figure has markers
  for it — no fewer (two rows per element means the figure wasn't digitised) and no more (every row is
  a marker you located, never a point read off a line).
- Every element/series present in the primary figure (or table/text) is represented.
- All `Extract%` in `[0, 100]`; with x = pH/−log[H⁺], %E rises with x per element.
- Digitised end-points match the text's stated end-points within tolerance; x within the paper's range.
- `pH½` (x½) ordering follows atomic number (extraction increases with Z).
- `Molar ratio = Extractant_mM / this-element_mM` for each block (per element, not a total), ≥ 3 s.f.
- Every D / K_d was converted with this experiment's phase ratio (Step 7), and D (and the phase ratio,
  when stated) is on the row; every row is a rare earth.
- mM feed values > 0 and physically reasonable.
- Every row names its `Figure/Table`; a mixed system's synergist is in `Co-extractant`, not in the
  extractant's name; contact time is in `Contact time`, not `Leaching time`.
- Every sweep row carries its x; no element-group rows; stripping/scrubbing percentages are in
  `Recovery %`; every figure and table is in `figures`.
- Exactly **35 columns**; no "N/A" strings; no appended documentation columns.

---

## Common pitfalls
| ✗ Wrong | ✓ Right |
|---|---|
| Stopping at the text's 2 end-points | Digitise the **whole** curve — every marker the figure shows, however many (Step 5) |
| Kinetics series at 6 of their 14 time points | Every marker is a row (Step 5) |
| A wt% or vol% sweep with identical conditions on every row | Convert the x to a schema column, or no rows + reason in `figures` (Step 8) |
| Sampling points along a fitted or guide line at even x steps | Only marker centres are data; a series with fewer markers than usual is still just those markers (Step 5) |
| Treating the figure as a last resort | The primary figure is the default dataset; text/table validate it (Step 0) |
| Molar ratio = ext ÷ feed-total mM | Molar ratio = ext ÷ **this element's** mM, per block (Step 8) |
| Appending `Source` / `Confidence` / `raw value` columns | Ship exactly 35 columns; confidence guides reading, not output |
| Long prose in `Sources` / `Extractant` | Short canonical strings (`Nitrate Salts`, `Cyanex 272`) (Step 8) |
| Assuming y-axis is % Extraction | Could be log D or D — classify and convert (Step 7) |
| Skipping every log-log plot as "derived" | A `log D` vs `log[ext]` series at a pH/element no other figure covers IS unique data — convert both axes and digitise it (Step 1) |
| Assuming figures are colour | Could be monochrome marker-shape — branch in Step 4 |
| Digitising everything dark inside the frame | Mask panel titles / annotations like the legend first; text remnants masquerade as filled markers (Step 5) |
| Apportioning a tangled multi-series cluster by guesswork | Digitise the separable series, omit the rest (Step 5) |
| Keeping a point an order of magnitude off its series' log D line | It is text or another series — remove it (Step 6) |
| Writing your own blob detector / frame finder / calibrator for a raster figure | Call the SANDBOX TOOLKIT — it is this pipeline's tested code (Steps 3, 5) |
| Opening a rendered PNG with the file viewer to "look" at it | You cannot see sandbox images (it returns base64 text); read the figure from the PDF document |
| Probing the sandbox (`pip show`, import checks, trying one PDF library after another) | The toolkit block states what is installed; render with `pages.render`, read vectors with pdfplumber |
| Printing arrays or lists of toolkit objects to inspect them | Print counts and a few values; every printed character is re-read on every later step |
| Converting a resin's D or K_d with `100·D/(1+D)` | `100·D/(D + V/m)` with V/m in mL/g from the caption or methods (Step 7) |
| Rows for Pb, Co, Li, U or another non-rare-earth metal | Rare earths (La–Lu, Y, Sc) only (Step 1) |
| Rows at the caption's listed concentrations | One row per marker, at the marker's own x (Step 5) |
| Molar ratio rounded to one decimal | ≥ 3 significant figures (Step 8) |
| Re-using a previous paper's colour map | Re-derive from this legend every time |
| Matching the legend *line* colour | Match the *marker* colour; legends often list each element twice |
| Trusting OCR'd table parameters | Verify they reproduce a text end-point first; else digitise (Step 6) |
| Assuming pH ticks (0,1,2,3,4) | Read actual tick labels; `np.interp` (Step 3) |
| Copying a feed total onto every element | This element's concentration, or blank; the total goes in `Total REE feed (mM)` (Step 8) |
| `Cyanex 272 + TBP (0.5 M)` in `Extractant` | `Extractant` Cyanex 272, `Co-extractant` TBP, `Co-extractant Conc. (mM)` 500 (Step 8) |
| A kinetics plot's shaking time in `Leaching time` | `Contact time (minute)`; `Leaching time` is for a leaching step only (Step 8) |
| Keeping only the %E derived from a D axis | Keep D in `Distribution ratio (D)`, and the stated phase ratio in `Phase ratio (O/A)` (Step 7) |
| Mixing equilibrium and initial pH without saying | `pH type` on every row where the paper says which (Step 8) |
| Writing "N/A" | Leave blank |

---

# OUTPUT CONTRACT (extraction_v16 additive layer — do not remove)

## Deterministic curve analysis (when present)
The user turn may include a **"DETERMINISTIC CURVE ANALYSIS"** block computed from the PDF's own
vector drawing commands BEFORE you ran. Where it marks a page **authoritative**, the per-series
marker counts it gives are **ground truth** — that figure's drawing commands contain exactly that
many points per series. The counts cover **filled-marker series only**: the legend may list more
series (outline or stroked markers), and those still need digitising in full. Your digitised output
for the counted series **must match those counts**; if you
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
  to target, so give them the full diligence Steps 2–6 describe, including a closer look at crowded or
  ambiguous regions in the PDF document when genuinely needed.

An authoritative page may also carry a `DIGITIZED CURVE DATA` block of pre-calibrated coordinates;
when it does, follow the instructions inside that block.

## Sandbox toolkit (when present)
The user turn may include a **"SANDBOX TOOLKIT"** block: this pipeline's own tested digitisation
package is in your code-execution environment, and the block says how to load it, what the sandbox has
installed, and how to render a figure region (`pages.render`). It is the implementation of Steps 3 and 5
for raster figures — **call it instead of writing that code**, and trust its environment facts instead
of probing (`pip show`, import checks, reinstalling packages). The loop you run is the cost of this
extraction: every code run re-reads the whole conversation, so a raster figure should take about three
runs (render the region and print its frame; blank text regions + detect + ticks; calibrate +
convert + assign series), not twenty. Your judgement still does the parts code cannot: panel layout, where the legend
and in-plot text sit, which shape family is which element, which tangled series to omit (Step 5).

## How you receive the paper
You are given the paper **two ways in this conversation**: as a PDF `document` block (read it
visually — legend colours, marker shapes, panel layout, axis labels) **and** as a file available in
your code execution environment (`$INPUT_DIR`), next to the toolkit. For Steps 2–6, actually run code
against that file — vector/raster detection, axis-tick calibration, and curve digitisation
programmatically rather than by eye — `pdfplumber` for vector objects and text positions, the toolkit's
`pages.render` to render, and the toolkit to digitise. Use your visual read of the page only
for what code can't tell you — which colour/marker maps to which legend entry, whether a region is
genuinely unresolvable versus just crowded — then digitise through code so every point is a real
measured coordinate, not an eyeballed estimate. Digitise **every distinct experiment/figure in the
paper** (Step 0), not just the first one you find.

## What to return
Return your entire answer as a **single JSON object inside one ```json fenced code block, and nothing
else outside the block.** The object has exactly four keys, `columns`, `rows`, `text_endpoints` and
`figures` (values in the example row are illustrative):

```json
{
  "columns": [
    "Reference No.", "DOI", "Treatment", "Sources", "Material Process",
    "Si (%)", "Al (%)", "Zn (%)", "Fe (%)",
    "Rare Earth Elements (REY:La, Ce, Nd)", "RRE composition (ppm)", "RRE composition (mM)",
    "Extractant", "Extractant type", "Extractant Conc. (mM)", "Molar ratio of EX/REE",
    "Extract%", "Extract Temperature (oC)", "pH", "Separation factor (SF%)",
    "Acid Solution", "Acid Solution conc. (M)", "mixing method",
    "Stripping Temperature (oC)", "Leaching time (minute)", "Recovery %",
    "Diluent", "Co-extractant", "Co-extractant Conc. (mM)", "Phase ratio (O/A)",
    "Contact time (minute)", "Distribution ratio (D)", "Total REE feed (mM)", "pH type", "Figure/Table"
  ],
  "rows": [
    ["1", "https://doi.org/10.1016/j.seppur.2011.09.015", "Extraction only", "Nitrate Salts", "dilution",
     null, null, null, null,
     "La", 100, 0.72,
     "Cyanex 272", "phosphinic acid based", 500, 694,
     11.6, 25, 2.0, null,
     "HClO4", null, "stirring",
     null, null, null,
     "kerosene", null, null, 1,
     25, null, null, "equilibrium", "Fig. 2"]
  ],
  "text_endpoints": [
    {"element": "Lu", "x_value": 0.90, "x_basis": "pH", "y_value": 39.96, "y_metric": "Extract%",
     "source_quote": "Percent extraction increases ... from 39.96% ... with increasing equilibrium pH from 0.90 ... for Lu"},
    {"element": "Lu", "x_value": 4.00, "x_basis": "pH", "y_value": 99.16, "y_metric": "Extract%",
     "source_quote": "... to 99.16% ... at pH 4.00 for Lu", "extractant": "Cyanex 272"}
  ],
  "figures": [
    {"figure": "Fig. 2", "digitised": true, "reason": ""},
    {"figure": "Fig. 6", "digitised": false, "reason": "d(log D)/d(pH) vs 1/r — a derived plot, not extraction data"}
  ]
}
```

## Rules — the JSON REPLACES the Step-9 spreadsheet layout
This flat JSON is the deliverable, not the multi-sheet xlsx of Step 9. The 35-column SCHEMA, the
figure-first digitising rules, the conversions (Step 7), and the per-element molar ratio (Step 8) all
still apply — only the *layout* changes, and it is now **positional, not keyed**, to avoid paying for
35 verbose column-name strings on every single row:

1. **`columns` is written exactly once**, as the 35 column names above, **in exactly that order.**
   Copy them verbatim — do not reword, reorder, or omit any.
2. **`rows` is an array of arrays.** Each inner array is **one digitized data point for one element
   series**, with **exactly 35 values, positionally aligned to `columns`** (index 0 of every row is
   `Reference No.`, index 1 is `DOI`, … index 25 is `Recovery %`, index 26 is `Diluent`, … index 34 is
   `Figure/Table`). Numeric fields are JSON numbers (or `null` if the paper does not report them) —
   never strings like `"95%"`. Never shift or drop a position — a short row silently corrupts every
   column after the gap.
3. **Fully populate every row — no sparse fill, no blank separator rows.** The Step-9 convention of
   writing sheet-level values only on row 0 and block-level values only on the first row of a block
   does **not** apply here. Repeat the sheet-level values (`Reference No.`, `DOI`, `Treatment`,
   `Sources`, `Material Process`) and the block-level values (`Rare Earth Elements (REY:La, Ce, Nd)`,
   `RRE composition (ppm)`, `RRE composition (mM)`, `Extractant`, `Extractant type`,
   `Extractant Conc. (mM)`, `Molar ratio of EX/REE`, and every stated condition — `Diluent`,
   `Co-extractant`, `Co-extractant Conc. (mM)`, `Phase ratio (O/A)`, `Contact time (minute)`,
   `Total REE feed (mM)`, `pH type`, temperature, acid) at their fixed index in **every** row of that
   block, so each row stands alone in a database.
4. **`Extractant type` is the chemical class on every row** (e.g. `phosphinic acid based`) — use the
   class consistently, not the `"Name (conc.)"` form; the extractant name + concentration already live
   in `Extractant` and `Extractant Conc. (mM)`.
5. **`Rare Earth Elements (REY:La, Ce, Nd)` carries the element symbol on every row** (`La`, `Ce`, …).
6. **One combined `rows` list across all experiments/sheets.** Do not split into sheets — each row
   already carries its own `pH` / `Extractant Conc. (mM)` / temperature, so the experiments remain
   distinguishable. The fixed parameter that defines an experiment is constant on its rows (e.g.
   `Extractant Conc. (mM)` = 500 for the pH sweep; `pH` = 1.75 for the concentration sweep).
7. **Digitise the WHOLE curve, not the endpoints — and nothing beyond its markers.** One row per
   marker the figure shows (typically 10–20 per series, sometimes 4–7). Stopping at the two text
   endpoints, or sampling extra points off a line, are the failures this pipeline exists to catch.
8. **`text_endpoints`** — unchanged from a plain object list (it's small; compaction doesn't matter
   here). For each element/series, when the paper states a SPECIFIC numeric claim in prose (e.g.
   "39.96% to 99.16% ... for Lu"), emit one entry per stated point. `x_basis` is the exact schema column
   name of the x (`"pH"`, `"Extractant Conc. (mM)"`, `"Co-extractant Conc. (mM)"`, `"Phase ratio (O/A)"`,
   `"Contact time (minute)"`, `"Extract Temperature (oC)"`, `"Acid Solution conc. (M)"`,
   `"Total REE feed (mM)"`) and `x_value` is in that column's units — null when the sentence states no x.
   `y_metric` is the dependent metric (`"Extract%"`, `"Recovery %"`, `"D"`). `source_quote` is the exact sentence. When the sentence names the
   extractant, add `"extractant"` with its common name, as in `rows`. These are a QA anchor, separate
   from the table — do not fold them into `rows`. If the paper states no numeric claim for a series,
   omit it (an empty list is fine).
9. **`Figure/Table`** — the last value of every row: the figure (or table) that row was digitised or
   read from, as the paper labels it (`"Fig. 2"`, `"Fig. 4(b)"`, `"Table 3"`, `"text"`). A reviewer
   uses it to trace each row back, and it tells apart the same element's curves in different figures.
10. **`figures`** — every numbered figure **and table** in the paper, once each: `digitised` true when
   rows came from it, and for any that gave no rows a short `reason` (not a 2-D extraction plot; a
   derived plot duplicating another figure; group-level only; a membrane transport experiment; a sweep
   whose x has no schema column; series too tangled to separate…). Markers knowingly omitted from a
   digitised figure are listed in its `reason`. Nothing is dropped silently.
