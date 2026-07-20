# Validation report: `area()`

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint (the programmatic interface
behind the EVALIDator web tool), queried live via `fetch_evalidator.R` in this directory. rFIA was
run against the real, current FIADB extracts already cached at `~/Dropbox/data/fia/` (pulled via
`getFIA()`), using `clipFIA(mostRecent = TRUE)` to match EVALIDator's "current" evaluation.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest) — the same four states used for `tpa()` (see `tpa.md`).

`tests/testthat/test-area.R` calls the FIADB-API live at test time rather than hard-coding the
reference numbers below — this section of the report is illustrative, not a source of truth the
tests are pinned to. The EVAL_GRP code for each state is read directly off
`clipFIA(..., mostRecent = TRUE)$POP_EVAL_GRP$EVAL_GRP`, never hard-coded. Tests are skipped (not
failed) when the local data cache or network access to `apps.fs.usda.gov` is unavailable.

Unlike `tpa()`, `area()`'s numerator (`fa = CONDPROP_UNADJ * aDI`) and denominator (`fad =
CONDPROP_UNADJ * pDI`) are both COND-level area quantities, not a TREE-level sum ratio'd against
area. This has two methodological consequences:

- **EVAL_TYP matters for `landType` choice of ground truth.** `area()`'s own code always calls
  `handlePops(db, evalType = c('CURR'), ...)`. EVALIDator attribute 1 ("area of sampled and
  nonsampled land and water") is tagged `EVAL_TYP = EXPALL`, a *different* EVALID/stratification
  than `EXPCURR`, so it is not a valid ground truth for `area()`'s own `EXPCURR`-based estimates
  despite superficially describing the same quantity. **Attribute 79** ("area of sampled land and
  water") is the correct `EXPCURR`-tagged equivalent (its own SQL definition excludes
  `COND_STATUS_CD = 5`, i.e. hazardous/denied-access plots) and is used throughout this report and
  `test-area.R` for every `landType` value other than `'forest'`/`'timber'` (which have their own
  dedicated, already-`EXPCURR` attributes 2 and 3).
- **`wnum` (numerator-only tree filter) cannot validate `treeDomain`.** `tpa.md` used `wnum` to
  restrict tpa()'s TREE-level sum directly. Confirmed empirically that this does *not* work for
  `area()`: `fetch_evalidator(wc, snum = 2, wnum = "TREE.SPCD = 129")` returns an estimate byte-
  identical to the same query with `wnum` omitted, because attribute 2's SQL definition never joins
  the `TREE` table — `wnum` has nothing to filter. EVALIDator's `fullreport` endpoint has no
  attribute exposing "area of forest land containing at least one tree matching an arbitrary
  species/diameter filter" (the quantity `area(treeDomain = ...)` computes), so `treeDomain`
  validation for `area()` uses grouping-consistency checks instead (see below), not a direct
  EVALIDator numeric match.

## Results: numeric match

All point estimates and percent standard errors below match the FIADB-API to full double precision
unless noted.

### Core default case (`landType = 'forest'`/`'timber'`), 4 states

| State | `landType = 'forest'` AREA_TOTAL | `landType = 'timber'` AREA_TOTAL | nPlots (forest/timber) |
|---|---|---|---|
| RI | 377491.6 | 357475.5 | 132 / 126 |
| NC | 18509817 | — | — |
| CO | 22718076 | — | — |
| OR | 29754801 | — | — |

All four states: **exact match** against EVALIDator attribute 2 (forest land area) / 3 (timberland
area), including `AREA_TOTAL_SE` and both `nPlots_AREA_NUM`/`nPlots_AREA_DEN` against
`plotCount` (the EVALIDator ground truth's numerator and denominator plot counts are identical here,
since `landType = 'forest'`/`'timber'` with no `treeDomain`/`polys` restriction has `aDI == pDI` for
every condition).

### `landType` variants previously affected by the bugs fixed this pass, 4 states

| Case | EVALIDator attr / filter | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `landType = 'water'` | 79, `COND_STATUS_CD in (3,4)` | exact | exact | exact | exact |
| `landType = 'non-forest'` | 79, `COND_STATUS_CD = 2` | exact | exact | exact | exact |
| `landType = 'all'` | 79 (unfiltered) | exact | exact | exact | exact |

Before the fixes in this pass, `'water'`/`'non-forest'` undercounted by one to two orders of
magnitude (`PLOT_STATUS_CD == 1` dropping non-forest plots before land-type logic ran), and `'all'`
overcounted by including nonsampled conditions. All three now match EVALIDator exactly across all
four regions.

### `areaDomain` filter interaction, 4 states

| Case | Mechanism | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `areaDomain = PHYSCLCD %in% 21:29` (mesic), `landType = 'forest'` | `strFilter` on attr 2 | exact | exact | exact | exact |
| `areaDomain = COUNTYCD == 7`, `landType = 'water'` (RI only) | `strFilter` on attr 79 | exact | — | — | — |
| `areaDomain = COUNTYCD == 7`, `landType = 'non-forest'` (RI only) | `strFilter` on attr 79 | exact | — | — | — |

`AREA_TOTAL`, `AREA_TOTAL_SE`, and `nPlots_AREA_NUM` all match exactly. The `COUNTYCD`-based checks
specifically validate the `udAreaDomain()` fix (bug #3 below): before that fix, any non-forest
`landType` combined with `areaDomain` silently returned zero area regardless of the filter.

### `treeDomain` + `grpBy` interaction (v1.1.1 bug pattern), 4 states

No direct EVALIDator equivalent exists (see Methodology). Instead validated via internal
consistency, which is exactly what the historical v1.1.1 bug broke:

1. **The filter has a genuine effect.** `area(landType = 'forest', treeDomain = DIA > 20)` is
   35%–66% of the unrestricted forest area in all four states (never equal to it, which the v1.1.1
   bug would have produced).
2. **`grpBy` does not silently drop the filter for any group.** Summing `AREA_TOTAL` across
   `grpBy = OWNGRPCD` groups (with the same `treeDomain`) exactly reproduces the ungrouped, filtered
   total in all four states (differences at or below floating-point noise, ~1e-11 in the largest
   case).

**Pass** in all four states — the v1.1.1 bug pattern is confirmed still fixed for `area()`.

### `byLandType` grouping and `PERC_AREA` semantics, 4 states

- `byLandType = TRUE`'s four mutually exclusive categories (Timber / Non-Timber Forest /
  Non-Forest / Water) sum their `AREA_TOTAL` to exactly `landType = 'all'`'s total, in all four
  states (RI: 781730.1, NC: 34444234, CO: 66619618, OR: 62962807) — and `landType = 'all'` itself
  matches EVALIDator attribute 79 exactly (see above).
- Grouping by a mutually exclusive COND-level variable (`FORTYPCD`) on `landType = 'forest'`
  produces `PERC_AREA` values summing to exactly 100% in all four states, confirming the documented
  design (`man/area.Rd`): percentages are relative to the full `landType` land base, not
  renormalized per group.
- Internal identity `AREA_TOTAL / <ungrouped landType total> * 100 == PERC_AREA` holds exactly
  (max abs difference 0) in all four states.

### `polys`/`returnSpatial` (RI, by county)

- `returnSpatial = TRUE` vs `FALSE`: all non-geometry columns match exactly. **Pass.**
- **Finding (not a bug, not `area()`-specific — traced to the bundled `countiesRI` polygon data):**
  summing `AREA_TOTAL` across `polys = countiesRI` does *not* reproduce the state total exactly
  (348029.3 vs. the true 377491.6 for `landType = 'forest'`) — a ~7.8% shortfall. Root-caused to the
  spatial point-in-polygon join (`arealSumPrep1`/`sf::st_join`) failing to match 17 of RI's 257
  forest plots to any county polygon. This is *not* an `area()` bug: the identical shortfall
  (348029.3 vs 377491.6, 123 vs 132 plots) reproduces exactly with `tpa(polys = countiesRI)`, which
  was already fully validated in `tpa.md` without this being flagged.

  An initial hypothesis (FIA's public-coordinate perturbation pushing plots outside their true
  county) was **checked and ruled out**: FIA's fuzzing procedure only displaces a plot within its
  true county (it does not relocate plots across county lines), and — more directly — repeating the
  join with the plot coordinates and `countiesRI` both left in native WGS84 (i.e. with the NAD27
  reprojection removed entirely) still produced the identical 17 unmatched plots. So the datum
  transform isn't the cause either.

  The actual cause is the **`countiesRI` polygon geometry itself**: it is extremely coarse (140
  total vertices across all 5 counties — 18 to 43 per county — for a state with a genuinely
  irregular coastline and many bays/islands), consistent with a heavily generalized/simplified or
  old-vintage source with no `data-raw/` script or provenance recorded in the package to confirm
  which. Measuring each unmatched plot's distance to the nearest county polygon (in UTM 19N, meters)
  shows two clusters:
  - 15 of the 17 plots (all in **Kent**, **Providence**, and **Washington** counties — RI's three
    counties bordering Connecticut) are 2–970 m from the nearest polygon edge, and their longitudes
    sit right at or just west of the corresponding county's bounding-box edge. This is consistent
    with the digitized western county boundaries in `countiesRI` sitting slightly east of the true
    RI/CT state line, so plots genuinely near that border fall just outside the simplified polygon.
  - The remaining 2 plots (**Newport** and **Washington**) are ~3.7 km from the nearest polygon —
    too far for a boundary-precision issue. Both are on the eastern side of the state near
    Narragansett Bay, where Newport and Washington counties include islands and peninsulas
    (Aquidneck/Conanicut Islands, Point Judith); the low vertex count suggests the shapefile likely
    generalizes or omits some of this coastal detail.

  Grouping by the COND-level `COUNTYCD` attribute instead of spatial `polys` does not have this
  issue (confirmed: summing `area(grpBy = COUNTYCD)` reproduces the exact state total), since it
  uses FIADB's own recorded county assignment rather than a geometric join against a third-party
  polygon. No fix is proposed as part of *this* validation pass (`area()`'s own logic is not at
  fault — the same shortfall reproduces identically for `tpa()`), but replacing `countiesRI` with a
  higher-resolution county boundary source (e.g. current-vintage US Census TIGER/Line) would likely
  resolve it, and might be worth a follow-up given it affects every `rFIA` function that supports
  `polys`, not just `area()`.

### `byPlot = TRUE`

Basic sanity check only (produces a non-empty, well-formed per-plot/per-condition
`PROP_FOREST` table). Full reconciliation against the population-level estimate is deferred to
follow-up, consistent with `tpa.md`, which deferred the same check for `tpa()`.

### Empty-domain edge case

`area(treeDomain = SPCD == 999)` (matches no trees) returns a clean 0-row tibble with no warning,
confirming the shared `combineMR()` fix (`tpa.md`, "Fixed" #2) applies correctly to `area()` too.

## Fixed

### 1. `PLOT_STATUS_CD == 1` filter silently dropped non-forest plots before land-type logic ran [FIXED]

`R/areaStarter.R`'s PLOT-level filter (`dplyr::filter(PLOT_STATUS_CD == 1 & sp == 1)`) was copied
from `tpaStarter.R`, where it's valid — trees only exist on forest land, so `tpa()` never needs
non-forest plots. `area()` is documented (`man/area.Rd`) and validated to support `landType` values
of `'forest'`, `'timber'`, `'non-forest'`, `'water'`, `'census water'`, `'non-census water'`, and
`'all'`, plus a `byLandType = TRUE` breakdown — and the forest-only PLOT filter silently broke every
one of those except `'forest'`/`'timber'`, since it ran *before* the (already-correct) COND-level
`landD`/`aD` domain indicators got a chance to restrict to the requested land type.

Confirmed empirically (RI, before fix): `landType = 'water'` reported 927 acres (true value
~104,200, two orders of magnitude too low); `byLandType = TRUE` summed to 450,847 acres against a
true state total of 781,730 acres (42% missing). `landType = 'forest'`/`'timber'` were unaffected,
which is why this went undetected until other `landType` values were tested.

**Fix**: dropped `PLOT_STATUS_CD == 1 &` from the filter, leaving `dplyr::filter(sp == 1)`. This
exactly mirrors `R/areaChangeStarter.R`'s existing, already-correct PLOT filter, which has an
explicit comment for exactly this reason ("we want plots that were forested and non-forested").

**Verification**: after the fix, `landType = 'water'` matches EVALIDator exactly (104200) in all
four states, and `byLandType = TRUE` sums to the exact state total (RI: 781730.1) in all four
states. `landType = 'forest'`/`'timber'` outputs are byte-identical to pre-fix. Full package test
suite re-run with no regressions.

### 2. `nPlots_AREA_DEN` phantom-row inflation [FIXED]

Same class of bug as the `nPlots_AREA` fix already applied to `tpaStarter.R` (see `tpa.md`, "Fixed"
#1). The denominator-construction (`a`) block in `R/areaStarter.R` lacked the `!is.na(CONDID)` guard
that `tpaStarter.R`'s equivalent block already has.

Reproduced on RI (before fix): `landType = 'timber'` reported `nPlots_AREA_DEN = 132` (the broader
forest-land count) instead of the correct 126 — matching EVALIDator's `plotCount` exactly, and
matching what `nPlots_AREA_NUM` already correctly reported (protected by a later, unrelated filter).
Point estimates and SEs were unaffected — purely a plot-count/reporting bug — but this bug becomes
far more consequential once bug #1 (above) ships, since every non-forest-only or nonsampled plot now
generates this exact kind of phantom `CONDID = NA` row via the `left_join` from `db$PLOT` to a
narrower, domain-filtered `db$COND`.

**Fix**: added `dplyr::filter(!is.na(CONDID))` immediately after `distinct()` in the `a` block,
mirroring `tpaStarter.R`'s fix exactly.

**Verification**: after the fix, `landType = 'timber'` returns `nPlots_AREA_DEN = 126` — exact match
to EVALIDator's `denPlotCount` — with `nPlots_AREA_NUM == nPlots_AREA_DEN` in every landType/state
combination with no `treeDomain`/`polys` restriction (confirmed: this equality is expected whenever
`aDI == pDI`, i.e. `sp = 1` and `tD = 1` for every condition). Full package test suite re-run with no
regressions.

### 3. `udAreaDomain()` hard-coded a forest-only filter, silently zeroing `areaDomain` for non-forest `landType` [FIXED]

The shared internal utility used to evaluate a user-supplied `areaDomain` expression
(`R/util.R::udAreaDomain()`) hard-coded `PLOT_STATUS_CD == 1`/`COND_STATUS_CD == 1` when building the
plot/condition context (`pcEval`) the expression is evaluated against. Any non-forest condition
therefore got no row in `pcEval`, meaning its resulting `aD` indicator was `NA` (not `0`) after the
`left_join` back onto `db$COND` — and `db$COND`'s downstream `filter(aD == 1 & landD == 1)` drops
`NA` rows exactly like `0` rows, but silently, for every non-forest condition regardless of whether
the `areaDomain` expression itself would have evaluated to `TRUE` or `FALSE` for it.

Reproduced empirically (RI, before fix): `area(landType = 'water'/'non-forest', areaDomain =
COUNTYCD == 7)` returned 0 rows / 0 acres regardless of the actual `COUNTYCD` restriction, instead of
the correct, non-zero, county-restricted estimate.

**Fix**: removed both hard-coded filters (`plt <- db$PLOT`; `cnd <- db$COND`, evaluating over all
plots/conditions in the population). `rlang::eval_tidy()` evaluates row-wise with no cross-row
aggregation, so adding more rows to `pcEval` cannot change any existing row's own `aD` value — this
is a no-op for `tpa()`'s already-validated forest-only behavior (confirmed: full package test suite,
including `test-tpa.R`, re-run with no regressions).

**Verification**: after the fix, `area(landType = 'water', areaDomain = COUNTYCD == 7)` returns
4303.0 acres, and `area(landType = 'non-forest', areaDomain = COUNTYCD == 7)` returns 114226 acres —
both exact matches to EVALIDator (attribute 79 + an equivalent `COND_STATUS_CD`/`COUNTYCD`
`strFilter`), and both exactly reproducing the corresponding row of `area(landType = ...,
grpBy = COUNTYCD)`. Full package test suite re-run with no regressions.

### 4. `landType = 'all'` incorrectly counted nonsampled conditions as land area [FIXED]

`R/util.R::landTypeDomain()`'s `'all'` branch set `landD <- 1` unconditionally, with no exclusion of
`COND_STATUS_CD == 5` (nonsampled — e.g. hazardous or denied-access plots). This meant
`landType = 'all'` did not actually mean "all *sampled* land", unlike every other `landType` value
and unlike `byLandType = TRUE`'s own four categories (whose `case_when` simply has no branch for
`COND_STATUS_CD == 5`, giving `NA`, which is then correctly dropped).

Reproduced empirically (RI, before fix): `landType = 'all'` returned 808712.2 acres, while summing
`byLandType = TRUE`'s four categories (which already correctly excluded nonsampled conditions) gave
781730.1 acres — matching EVALIDator attribute 79's total exactly. The discrepancy (a ~3.5%
overcount) is exactly the nonsampled-condition acreage.

**Fix**: rather than modifying the shared `landTypeDomain()` utility (also called by
`areaChangeStarter.R`, which hasn't been validated yet this round), the fix is scoped to
`R/areaStarter.R`: immediately after the `landTypeDomain()` call, `db$COND$landD[db$COND$COND_STATUS_CD
== 5] <- 0` when `landType == 'all'`. This has no effect on `byLandType = TRUE` (which resets
`landD <- 1` uniformly for all conditions afterward and relies on its own `NA`-drop for the same
exclusion) or on any other `landType` value (`COND_STATUS_CD == 5` never matches any of their
existing branches regardless).

**Verification**: after the fix, `landType = 'all'` returns 781730.1 acres in RI — exact match to
`byLandType = TRUE`'s sum and to EVALIDator attribute 79 — in all four states. All other `landType`
values and `byLandType = TRUE` itself are byte-identical to pre-fix. Full package test suite re-run
with no regressions.

## Notes

### EVALIDator attribute 1 vs. 79 (`EXPALL` vs `EXPCURR`)

See Methodology above. Attribute 1 happens to equal attribute 79 numerically for RI (both 781730.1
at the time of writing), which could easily be mistaken for confirmation that either is a valid
ground truth for `area()`. It is not, in general: they are tagged different `EVAL_TYP`s and can
resolve to different EVALIDs/stratifications for other states or evaluations. Attribute 79 is the
correct choice for any future `area()`/`areaChange()` ground-truth comparison outside `landType =
'forest'`/`'timber'` (which have their own dedicated, already-`EXPCURR` attributes 2/3).

### `wnum` cannot filter attributes with no `TREE` join

See Methodology above — confirmed by direct API test (`fetch_evalidator(snum = 2, wnum = "TREE.SPCD
= 129")` byte-identical to the same call without `wnum`). Relevant for anyone extending this
validation to `areaChange()`, which shares the same `treeDomain` mechanism.

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population-level estimate (only a structural sanity
  check was done) — same deferral as `tpa.md`.
- The `polys`/spatial-join plot-matching shortfall described above under "`polys`/`returnSpatial`"
  traces to the coarse geometry of the bundled `countiesRI` dataset, not to `area()`'s own logic
  (confirmed present in already-validated `tpa()` too, and reproduces identically with no datum
  reprojection involved). Replacing `countiesRI` with a higher-resolution boundary source is a
  plausible fix, but is a package-data change affecting every `rFIA` function that supports `polys`,
  not something to fix as part of `area()`'s own validation pass.
- `areaChange()` validation, explicitly deferred to a separate pass (see project instructions for
  this initiative).
