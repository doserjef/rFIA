# Validation report: `volume()`

## Scope

This pass covers `volume()` -- net/gross/sound cubic-foot bole volume, cubic-foot sawlog volume,
and board-foot sawlog volume of standing trees.

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint (the programmatic interface
behind the EVALIDator web tool), queried live via `fetch_evalidator.R` in this directory. rFIA was
run against the real, current FIADB extracts already cached at `~/Dropbox/data/fia/` (pulled via
`getFIA()`), using `clipFIA(mostRecent = TRUE)` to match EVALIDator's "current" evaluation.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest).

`tests/testthat/test-volume.R` calls the FIADB-API live at test time rather than hard-coding the
reference numbers below (same approach as `tpa()`/`area()`/`biomass()`/`carbon()`) -- this report is
illustrative, not a source of truth the tests are pinned to. The EVAL_GRP code for each state is
read directly off `clipFIA(..., mostRecent = TRUE)$POP_EVAL_GRP$EVAL_GRP`, so the tests always query
whichever evaluation `mostRecent` actually selected. Tests are skipped (not failed) when the local
data cache or network access to `apps.fs.usda.gov` is unavailable.

`treeDomain`/`areaDomain` filter semantics use the same two API mechanisms established during the
`tpa()` validation pass (see `tpa.md`): `wnum` (numerator-only) for `treeDomain`-style filters,
`strFilter` (numerator + denominator) for `areaDomain`-style filters.

### Attribute mapping

`volume()`'s three metrics are read directly off pre-computed FIADB `TREE` columns
(`VOLCFNET`/`VOLCSNET`/`VOLBFNET` for `volType = 'NET'`, the default), so -- as with `biomass()` --
this pass primarily validates rFIA's post-stratified aggregation logic rather than a from-scratch
volume-equation reimplementation.

| rFIA output | FIADB column | EVALIDator attr (forest / timber) |
|---|---|---|
| `BOLE_CF_ACRE` (`treeType = 'live'`) | `VOLCFNET` | 574171 / 574172 |
| `SAW_CF_ACRE` | `VOLCSNET` | 16 / 19 |
| `SAW_MBF_ACRE` (x1000) | `VOLBFNET` | 20 / 21 |
| `BOLE_CF_ACRE` (`treeType = 'gs'`) | `VOLCFNET`, `TREECLCD = 2` | 15 / 18 |
| `BOLE_CF_ACRE` (`treeType = 'dead'`) | `VOLCFNET`, standing dead | 11252 / 11253 |

`SAW_MBF_ACRE` is expressed by `volumeStarter.R` in *thousand* board feet (`/1000`), so it's
multiplied back up by 1000 when compared against EVALIDator's raw board-foot attributes. The board-
foot rule was not assumed -- it was determined empirically by comparing RI's `SAW_MBF_ACRE` against
all three EVALIDator board-foot attribute families (International 1/4-inch rule: 20/21; Scribner:
574200/574201; Doyle: 1020/1021). Only the International 1/4-inch rule matched exactly, confirming
`VOLBFNET` is FIADB's standard/default board-foot volume equation.

No direct EVALIDator "current stock" attribute exists for growing-stock (`treeType = 'gs'`) sawlog
volume as a category distinct from the core `SAW_CF_ACRE`/`SAW_MBF_ACRE` case -- EVALIDator's own
sawlog attributes are already restricted to "sawtimber trees" (a subset of growing-stock), so the
core `treeType = 'live'` sawlog match already exercises that restriction; a separate `'gs'`-specific
sawlog check was not needed.

## Results: numeric match

All point estimates and percent standard errors below match the FIADB-API to full double precision
(14+ significant digits) after the fixes described below.

### Core default case (`treeType = 'live'`, `landType = 'forest'`, `volType = 'NET'`), 4 states

| State | BOLE_CF_ACRE | SAW_CF_ACRE | SAW_MBF_ACRE (x1000) | nPlots_TREE | nPlots_AREA |
|---|---|---|---|---|---|
| RI | 2544.254 | 1634.700 | 8479.459 | 129 | 132 |
| NC | 2727.280 | 1792.252 | 10029.91 | 3379 | 3561 |
| CO | 1035.158 | 697.2632 | 4296.832 | 2491 | 3925 |
| OR | 3670.365 | 3250.741 | 20889.65 | 9841 | 10410 |

All four: **exact match** against EVALIDator attribute 574171 (net bole cubic-foot volume of live
trees, forest land), 16 (net sawlog cubic-foot volume), and 20 (net sawlog board-foot volume,
International 1/4-inch rule), ratio'd against attribute 2 (forest land area).

### `landType`/`treeType` variants, 4 states

| Case | EVALIDator attr (num/denom) | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `landType = 'timber'` | 574172 / 3 | exact | exact | exact | exact |
| `treeType = 'gs'` | 15 / 2 | exact | exact | exact | exact |
| `treeType = 'dead'` | 11252 / 2 | exact | exact | exact | exact |

### Domain filter interactions, 4 states

| Case | Mechanism | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `treeDomain = DIA >= 20` (large trees) | `wnum` | exact | exact | exact | exact |
| `areaDomain = PHYSCLCD %in% 21:29` (mesic) | `strFilter` | exact | exact | exact | exact |

Both matched EVALIDator attribute 574171 (bole cubic-foot volume, forest land) exactly across all
four states, including `nPlots_TREE`/`nPlots_AREA`.

### `bySpecies` grouping (RI)

Cross-checked a random sample of individual species rows from `volume(bySpecies = TRUE)` against
independent single-species EVALIDator queries (`wnum = "TREE.SPCD = <code>"`) -- validates that the
`grpBy = SPCD` join/aggregation path doesn't silently drop the domain filter for some groups (the
historical `area()`/`areaChange()` bug pattern from v1.1.1). **Pass** for the species sampled
(SPCD 901, SPCD 43).

### `returnSpatial` (RI, by county)

`volume(polys = countiesRI, returnSpatial = TRUE)` vs. `returnSpatial = FALSE`: all non-geometry
columns match exactly. **Pass.**

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `BOLE_CF_TOTAL`/`SAW_CF_TOTAL`/`SAW_MBF_TOTAL` divided by `AREA_TOTAL` reproduce
  `BOLE_CF_ACRE`/`SAW_CF_ACRE`/`SAW_MBF_ACRE` exactly, across all four states. **Pass.**

### Empty-domain edge case

`volume(treeDomain = SPCD == 999)` returns a clean 0-row tibble with no warning. **Pass.**

## Fixed

Three bugs were found and fixed this pass, all in `volumeStarter.R`'s population-estimation branch.
`BOLE_CF_ACRE`/`SAW_CF_ACRE`/`SAW_MBF_ACRE`/`BOLE_CF_ACRE_SE` were unaffected by any of them --
they were caught entirely by extending the numeric tests to check `nPlots_TREE`/`nPlots_AREA`
directly (per this validation pass's methodology), which the pre-existing test suite never did.

**1. `nPlots_AREA` phantom-row bug** (matches the class of bug already fixed in `tpa()`/`area()`/
`biomass()`/`carbon()`; see their respective reports and `NEWS.md`). `volumeStarter.R`'s condition
list (`a`, used for the area denominator) was missing the `dplyr::filter(!is.na(CONDID))` guard
present in every other estimator's equivalent code. A plot whose conditions were entirely dropped
by `landType`/`areaDomain` upstream survived the `PLOT` x `COND` left_join as a phantom
`CONDID = NA` row, which correctly contributed 0 area (via `na.rm = TRUE` downstream) but still
inflated `nPlots_AREA`. Confirmed via `landType = 'timber'` and an `areaDomain` physiographic filter
across all four states -- both now match EVALIDator's `denPlotCount` exactly (e.g. OR
`areaDomain`: rFIA 8523 = EVALIDator 8523, would have been inflated before the fix).

**2. `nPlots_TREE` inflation from trees with no defined volume.** `VOLCFNET` (and, as a consequence,
`VOLCSNET`/`VOLBFNET`, which are never non-`NA` when `VOLCFNET` is `NA`) is only populated for trees
at least 5in DBH (timber species) or the woodland-species d.r.c. equivalent -- e.g. dead trees under
5in DBH never get a bole volume computed. The tree list feeding `nPlots_TREE` counted these trees
anyway (their 0 contribution to `BOLE_CF_ACRE` etc. was already handled correctly via
`na.rm = TRUE`, so only the plot count was wrong) -- the same class of bug just fixed for
`biomass()`'s woodland-species components, here triggered by tree diameter instead of species.
Example (Rhode Island, `treeType = 'dead'`, pre-fix): `nPlots_TREE` = 107 vs. EVALIDator's 103.

**3. `nPlots_TREE` inflation from trees with an exactly-zero net volume.** After fixing #2, RI and
CO matched EVALIDator exactly everywhere, but NC and OR still showed small residual `nPlots_TREE`
mismatches (e.g. NC core case: 3380 vs. 3379; NC `treeType = 'dead'`: 2006 vs. 1998). Root cause:
some trees have a *defined* but exactly-zero net volume (a 100% cull/defect deduction can legally
zero out `VOLCFNET`), and EVALIDator's own attribute definitions require the underlying volume
column to be strictly positive, not just non-missing, to count a tree as contributing. Confirmed
directly against North Carolina's raw `TREE` table: e.g. 112 dead trees across 88 plots have
`VOLCFNET == 0` exactly, and excluding them (rather than just excluding `NA`) reproduces
EVALIDator's plot counts exactly.

**Fix (all three)**: added filters scoped to the tree-list (`t`) construction only -- **not** to the
shared `data` object that also feeds the area/condition list `a` (the biomass validation pass tried
scoping to `data` first and caused a real regression -- `BIO_ACRE` itself drifted off of EVALIDator,
since conditions with no qualifying tree were silently dropped from the area denominator too; see
`biomass.md`, "Fixed"). The final filter is `dplyr::filter(!is.na(bcf) & bcf > 0)` (`bcf` is
`volumeStarter.R`'s internal name for the bole-volume column selected by `volType`), applied in both
the `byPlot` and population-estimation branches, immediately after `data %>%` and before any other
mutation. This is mathematically a no-op for every reported point estimate/SE (`sum(..., na.rm =
TRUE)` already treats dropped rows as 0), confirmed by rerunning the full numeric suite (including
`test-tpa.R`, `test-carbon.R`, `test-biomass.R`, `test-area.R`, `test-areaChange.R`) after the change
with no regressions.

## Notes

### Why `tpa()`/`biomass()`'s core attributes don't have this same `> 0` requirement

`tpa()`/`biomass()`'s EVALIDator attributes (e.g. attribute 4, live TPA; attribute 10, live AG
biomass) have no `> 0` requirement in their SQL definitions -- a live tree's tree count and
biomass are essentially never exactly zero, so this class of discrepancy is specific to volume
attributes, where a defect deduction can legitimately zero out net merchantable volume for an
otherwise-real, tallied tree. This was confirmed by inspecting the SQL metadata in
`EVALIDATOR_POP_ESTIMATE.csv` for both families of attributes before implementing the fix, per the
project's bug-handling protocol.

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population estimate (only totals-vs-per-acre was
  checked numerically, same as prior passes).
- `bySizeClass` was only checked structurally (pre-existing `test-volume.R` coverage), not against
  an EVALIDator size-class breakdown.
- `volType = 'GROSS'`/`'SOUND'` were not numerically validated against EVALIDator's gross/sound
  attributes this pass (only `volType = 'NET'`, the default).
- `method` options other than `'TI'` (EVALIDator has no equivalent; internal-consistency-only checks
  per the plan, not yet added).
