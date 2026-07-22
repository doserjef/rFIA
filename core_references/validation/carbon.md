# Validation report: `carbon()`

## Scope

This pass covers `carbon()`, which handles carbon stock estimation exclusively (split out from
`biomass()`, which no longer reports carbon -- see `biomass.md`). `carbon()` estimates the standard
IPCC forest carbon pools: live aboveground (`AG_LIVE`), live belowground (`BG_LIVE`), dead wood
(`DEAD_WOOD` = standing dead trees + down dead wood), litter (`LITTER`), and soil organic
(`SOIL_ORG`), plus a component-level breakdown (`byComponent = TRUE`) of each pool into its
condition-level (modeled understory/litter/soil) and tree-level (measured live/dead overstory)
pieces.

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint (the programmatic interface
behind the EVALIDator web tool), queried live via `fetch_evalidator.R` in this directory. rFIA was
run against the real, current FIADB extracts already cached at `~/Dropbox/data/fia/` (pulled via
`getFIA()`), using `clipFIA(mostRecent = TRUE)` to match EVALIDator's "current" evaluation.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest) -- the same four states used for `tpa()`/`biomass()`/`area()`.

`tests/testthat/test-carbon.R` calls the FIADB-API live at test time rather than hard-coding the
reference numbers below (same approach as the other validated functions) -- this report is
illustrative, not a source of truth the tests are pinned to. The EVAL_GRP code for each state is
read directly off `clipFIA(..., mostRecent = TRUE)$POP_EVAL_GRP$EVAL_GRP`. Tests are skipped (not
failed) when the local data cache or network access to `apps.fs.usda.gov` is unavailable.

### `carbon()` has no `treeDomain`

Unlike `tpa()`/`biomass()`, `carbon()` does not expose a `treeDomain` argument -- its estimate
mixes condition-level modeled components (understory, litter, soil) with tree-level measured
components (live/dead overstory), and a tree-level filter has no defined meaning for the
condition-level pieces. The domain-filter test matrix for `carbon()` is therefore `areaDomain`
only.

### Attribute mapping

EVALIDator's attribute library (`~/Dropbox/data/fia/EVALIDATOR_POP_ESTIMATE.csv`) has a purpose-built
set of "IPCC forest carbon pool" attributes already expressed in metric tonnes on forest land,
which map directly onto `carbon()`'s `POOL` categories with no unit conversion needed:

| `POOL` | EVALIDator attribute | Description |
|---|---|---|
| `AG_LIVE` | 98 | Forest carbon pool 1: live aboveground |
| `BG_LIVE` | 99 | Forest carbon pool 2: live belowground |
| `DEAD_WOOD` | 100 | Forest carbon pool 3: dead wood |
| `LITTER` | 101 | Forest carbon pool 4: litter |
| `SOIL_ORG` | 102 | Forest carbon pool 5: soil organic |
| (sum of all 5) | 103 | Forest carbon total: all 5 pools |

These have no timberland equivalent, so `landType = 'timber'` is instead validated at the
`byComponent = TRUE` level against short-ton (converted with the same `0.90718474` factor
`carbon()` uses internally) component attributes, which exist for both forest land and timberland:

| `COMPONENT` | Forest land attr | Timberland attr |
|---|---|---|
| `AG_UNDER_LIVE` | 48 | 62 |
| `BG_UNDER_LIVE` | 49 | 63 |
| `DOWN_DEAD` | 50 | 64 |
| `LITTER` | 51 | 65 |
| `SOIL_ORG` | 52 | 66 |
| `STAND_DEAD` | 47000 | 61000 |

Confirmed via each attribute's SQL definition (`VBA_SUMFROMWHERE` column) before trusting the
mapping:
- Attribute 98 (`AG_LIVE`) sums `COND.CARBON_UNDERSTORY_AG` plus live trees' (`STATUSCD = 1`)
  `CARBON_AG`, weighted by `TPA_UNADJ` -- exactly `AG_UNDER_LIVE + AG_OVER_LIVE` as computed in
  `carbonStarter.R`.
- Attribute 100 (`DEAD_WOOD`) sums `COND.CARBON_DOWN_DEAD` plus standing dead trees'
  (`STATUSCD = 2 AND STANDING_DEAD_CD = 1`) `CARBON_AG + CARBON_BG` -- exactly `DOWN_DEAD +
  STAND_DEAD`.
- Attribute 50/64 ("Carbon in stumps, coarse roots, and coarse woody debris") is, despite its
  label, defined purely as `SUM(COND.CONDPROP_UNADJ * COND.CARBON_DOWN_DEAD * ...)` -- i.e. exactly
  `carbon()`'s `DOWN_DEAD` component, not a DWM-transect-based total. Confirmed by inspecting the
  raw SQL directly rather than trusting the description text.
- Attribute 47000/61000 requires `TREE.STATUSCD = 2 AND TREE.STANDING_DEAD_CD = 1`, i.e. the same
  "standing dead" tally-tree criterion already fixed for `treeType = 'dead'` elsewhere in the
  package (NEWS.md). This raised a specific concern that `carbon()`'s own `dead` indicator
  (`case_when(STATUSCD == 2 ~ 1, ...)` in `carbonStarter.R`) might have the same omission -- checked
  explicitly (see Results) and confirmed **not** to be a live bug: `STAND_DEAD` matched EVALIDator
  to full double precision in all four states, meaning every `STATUSCD == 2` tree that survives
  `carbon()`'s existing `aDI`/plot/condition filters already satisfies `STANDING_DEAD_CD == 1` in
  practice for the data checked.

## Results: numeric match

All point estimates below match the FIADB-API to full double precision (14+ significant digits)
unless noted.

### Core default case (`landType = 'forest'`, `byPool = TRUE`), 4 states

| State | AG_LIVE | BG_LIVE | DEAD_WOOD | LITTER | SOIL_ORG | Total (5 pools) |
|---|---|---|---|---|---|---|
| RI | 34.42765 | 6.33153 | 8.698515 | 6.389466 | 63.56611 | 119.4133 |
| NC | 31.61490 | 5.89436 | 6.840101 | 3.651650 | 41.01900 | 89.02001 |
| CO | 12.93917 | 2.39610 | 6.405225 | 4.533311 | 44.44756 | 70.72137 |
| OR | 35.43623 | 7.61459 | 14.168579 | 4.353167 | 52.90858 | 114.4811 |

All four states, all five pools, and the grand total: **exact match** against EVALIDator attributes
98-103, including `nPlots_AREA` against `denPlotCount`.

### `landType = 'timber'`, `byComponent = TRUE`, 4 states

| Component | Timberland attr | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `AG_UNDER_LIVE` | 62 | exact | exact | exact | exact |
| `BG_UNDER_LIVE` | 63 | exact | exact | exact | exact |
| `LITTER` | 65 | exact | exact | exact | exact |
| `SOIL_ORG` | 66 | exact | exact | exact | exact |
| `DOWN_DEAD` | 64 | exact | exact | exact | exact |
| `STAND_DEAD` | 61000 | exact | exact | exact | exact |

### `byComponent = TRUE` on forest land (default `landType`), 4 states

| Component | Forest land attr | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `AG_UNDER_LIVE` | 48 | exact | exact | exact | exact |
| `BG_UNDER_LIVE` | 49 | exact | exact | exact | exact |
| `DOWN_DEAD` | 50 | exact | exact | exact | exact |
| `LITTER` | 51 | exact | exact | exact | exact |
| `SOIL_ORG` | 52 | exact | exact | exact | exact |
| `STAND_DEAD` | 47000 | exact | exact | exact | exact |

`STAND_DEAD` matching exactly (despite `carbon()`'s `dead` indicator not explicitly checking
`STANDING_DEAD_CD`) confirms this is not a live bug for `carbon()` in the data checked -- see
Methodology.

### `areaDomain` filter interaction (mesic physiographic classes), 4 states

| State | CARB_ACRE (total, byPool = FALSE) | nPlots_AREA |
|---|---|---|
| RI | exact match to attr 103 + `strFilter` | exact match to `denPlotCount` |
| NC | exact | exact |
| CO | exact | exact |
| OR | exact | exact |

This test doubles as the primary regression check for the `nPlots_AREA` fix below -- see "Fixed".

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `CARB_TOTAL / AREA_TOTAL` reproduces `CARB_ACRE` exactly, across all four states.
  **Pass.**
- `returnSpatial = TRUE` vs `FALSE` (RI, by county): all non-geometry columns match exactly.
  **Pass.**

### Empty-domain edge case

Before the fix below, `carbon(areaDomain = STATECD == 999)` (a domain matching no conditions)
returned 5 rows (one per pool) of `CARB_ACRE = NaN`, with `nPlots_AREA` equal to RI's full,
unfiltered plot count (132) -- rather than a clean empty result. After the fix, it returns a clean
0-row tibble with no warning, matching the behavior already established for `tpa()`/`area()`/
`biomass()`.

## Fixed

### `nPlots_AREA` phantom-row inflation, and a related NaN/garbage-row result for a fully-empty `areaDomain` [FIXED]

Same class of bug as the `nPlots_AREA` fix already applied to `tpaStarter.R` and `areaStarter.R`
(see `area.md`, "Fixed" #2) -- but not previously caught for `carbon()` or `biomass()`, both of
which shared the same missing guard.

**Root cause.** `carbon()`'s (and `biomass()`'s) population-estimation code builds a condition-level
data frame (`a`) via:

```r
a <- data %>%
  dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
  ...
```

where `data <- db$PLOT %>% left_join(db$COND, ...) %>% left_join(db$TREE, ...)`, and `db$COND` has
already been filtered down to only conditions satisfying `landType`/`areaDomain`
(`dplyr::filter(aD == 1 & landD == 1)`). When a plot's condition(s) don't satisfy that filter, the
`left_join` still preserves the plot as a single row with `CONDID = NA` (standard left-join
fill-in behavior) rather than dropping it -- and without a guard against this, that phantom row's
`PLT_CN` is still counted toward `nPlots_AREA` (via `length(unique(PLT_CN))` inside `sumToEU()`),
even though it contributes zero real area.

`tpaStarter.R`'s equivalent block already carries a `dplyr::filter(!is.na(CONDID))` immediately
after the `distinct()` call, with a comment explaining exactly this (added during the `tpa()`
validation pass, and mirrored in `areaStarter.R` during the `area()` pass -- see `area.md`, "Fixed"
#2). That guard was never propagated to `biomassStarter.R` or `carbonStarter.R`.

**Impact.** For `carbon()`, this is worse than a plot-count-only bug: because `carbon()`'s
non-`byPlot`/non-`condList` code path re-anchors its tree-level numerator onto this same
condition-level frame (`tPlt <- aPlt %>% ... %>% left_join(tPlt, ...)`), a *fully* empty
`areaDomain` produced 5 rows (one per pool) of `CARB_ACRE = NaN` instead of the clean 0-row result
every other validated estimator returns for this case. For `biomass()`, the point estimate itself
was unaffected (confirmed: `BIO_ACRE` matched EVALIDator exactly with and without the fix, in every
case checked) -- only the reported `nPlots_AREA` was wrong.

Reproduced empirically (RI, `areaDomain = COUNTYCD == 7`, before fix):

| Function | CARB_ACRE / BIO_ACRE | `nPlots_AREA` (rFIA) | `nPlots_AREA` (EVALIDator `denPlotCount`) |
|---|---|---|---|
| `carbon()` | 125.7212 (already exact) | 132 | 61 |
| `biomass()` | 86.28505 (already exact) | 132 | 61 |
| `tpa()` (unaffected, for comparison) | -- | 61 | 61 |

**Fix.** Added `dplyr::filter(!is.na(CONDID))` immediately after `distinct(PLT_CN, CONDID, .keep_all
= TRUE)` in the population-estimation `a` block of both `R/carbonStarter.R` and
`R/biomassStarter.R`, mirroring `tpaStarter.R`/`areaStarter.R`'s existing fix exactly. (The `byPlot`
branches of both functions already aggregate per-plot via `group_by(PLT_CN, ...) %>%
summarize(..., na.rm = TRUE)`, which handles a phantom row correctly without this guard, so no
change was needed there -- matching `tpaStarter.R`'s `byPlot` branch, which also has no such guard.)

**Verification.** After the fix: `nPlots_AREA` matches EVALIDator's `denPlotCount` exactly for both
functions under `areaDomain`/`landType = 'timber'` restrictions (confirmed RI county filter: 61/61
for both); a fully-empty `areaDomain` now returns a clean 0-row result for `carbon()` with no
warning; all previously-exact point estimates (`CARB_ACRE`, `BIO_ACRE`, and every pool/component/
`landType` variant validated in this report and in `biomass.md`) remain byte-identical to
pre-fix. Full package test suite (`test-biomass.R`, `test-carbon.R`, `test-tpa.R`, `test-area.R`,
plus the rest of `tests/testthat/`) re-run with no regressions.

## Notes

### Why `STAND_DEAD` didn't need the `STANDING_DEAD_CD` fix that `treeType = 'dead'` needed elsewhere

`carbon()`'s tree-level `dead` indicator (`case_when(STATUSCD == 2 ~ 1, ...)`, `carbonStarter.R`)
does not itself check `STANDING_DEAD_CD`, unlike EVALIDator's own standing-dead-carbon attributes
(47000/61000), which explicitly require it. This was flagged as a plausible bug candidate before
testing (matching the exact `treeType = 'dead'` bug pattern already fixed elsewhere in the
package -- see NEWS.md), but empirically, `STAND_DEAD` matched EVALIDator to full double precision
in all four states with no discrepancy. This means every dead tree (`STATUSCD == 2`) that survives
`carbon()`'s existing plot/condition filters in the data checked already has `STANDING_DEAD_CD ==
1`, so the missing check happens to be a no-op here -- but this has not been proven true in
general (e.g. for states/conditions not in this four-state sample), only confirmed absent in the
data actually tested. Worth a quick re-check if `carbon()`'s `STAND_DEAD` component is ever
extended or refactored.

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population-level estimate (only structural/smoke
  coverage exists, pre-existing `test-carbon.R` Test 4) -- same deferral as `tpa()`/`biomass()`.
- `method` options other than `'TI'` (EVALIDator has no equivalent) -- only structural coverage
  exists (pre-existing `test-carbon.R` Test 6, `method = 'LMA'`), not the internal-consistency
  checks called for by the plan.
- `landType = 'all'` has no EVALIDator equivalent at all (no "all land" carbon attribute exists) --
  only structural coverage exists (pre-existing `test-carbon.R` Test 3).
- `condList = TRUE` output was not separately re-validated after the `nPlots_AREA` fix (the fix
  changes what rows appear in `condList` output too, since it shares the same `a` block) -- worth a
  follow-up spot check.
- Given this pass found the same root-cause bug independently in both `carbon()` and `biomass()`
  despite `biomass()` having already been through its own validation pass, it would be worth
  auditing the remaining, not-yet-validated `*Starter.R` files (`areaChangeStarter.R`,
  `volumeStarter.R`, `dwmStarter.R`, etc.) for the same missing `!is.na(CONDID)` guard before their
  own validation passes, rather than relying on each pass to independently rediscover it.
