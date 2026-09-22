# Validation report: `seedling()`

## Scope

`seedling()` estimates trees per acre (TPA) of live seedlings (< 1 inch DBH) from the `SEEDLING`
table, structurally the closest function to `tpa()` already validated (same dispatcher/Starter
split, same population-estimation machinery -- `sumToPlot()`/`sumToEU()`/`combineMR()` are shared
verbatim), but with several real differences that turned out to matter:

- `SEEDLING` has no per-stem identifier analogous to `TREE.TREE` -- `TPA_UNADJ` is already
  pre-aggregated to the `PLT_CN`/`SUBP`/`CONDID`/`SPCD` grain by FIA. This makes the tree-list
  `distinct()` key a different (and, as it turned out, more fragile) problem than in `tpa()`.
- No `treeType`/basal area equivalent -- seedlings are always live and always microplot-basis, so
  there's only one output metric (`TPA`) to check, not two (`TPA`/`BAA`).
- No `DIA` column at all (seedlings are by definition < 1"), so diameter-threshold `treeDomain`
  filters (like `tpa()`'s `DIA >= 20`) aren't meaningful here; species-code filters were used instead.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest) -- the same four used for `tpa()`, all of which have real
`SEEDLING` data locally.

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint via `fetch_evalidator.R` (see
`tpa.md` for the general methodology, `wnum`/`strFilter` mechanics, and why a leading `AND` breaks
the API). EVALIDator attribute **45** ("Number of live seedlings ... on forest land",
`SUM(SEEDLING.TPA_UNADJ * POP_STRATUM.ADJ_FACTOR_MICR)`) and **46** (the timberland equivalent) map
directly onto `seedling()`'s `TPA` column -- both are a straight sum of `TPA_UNADJ` at microplot
adjustment, exactly matching `seedlingStarter.R`'s hard-coded `TREE_BASIS = 'MICR'`. Ratio'd against
attribute 2 (forest land area) / 3 (timberland area), as in `tpa()`.

`tests/testthat/test-seedling.R` calls the FIADB-API live at test time (reference values are not
hard-coded), for the same reasons given in `tpa.md`.

## Results: numeric match (after fixes below)

All point estimates, percent standard errors, and plot counts below match the FIADB-API to full
double precision unless noted. **All of the mismatches found in the first pass (see "Fixed" below)
were real bugs, not data-vintage or methodology differences** -- once fixed, every case matches
exactly.

### Core default case (`landType = 'forest'` / `'timber'`), 4 states

| State | TPA (forest) | TPA_SE (forest) | nPlots_TREE (forest) | nPlots_AREA (forest) | TPA (timber) | nPlots_AREA (timber) |
|---|---|---|---|---|---|---|
| RI | 662.1822 | 14.61254 | 81 | 132 | 662.2197 | 126 |
| NC | 1230.528 | 2.438309 | 3162 | 3561 | 1240.909 | 3436 |
| CO | 1609.045 | 2.58837 | 3068 | 3925 | 1401.09 | 1829 |
| OR | 889.7643 | 2.7841 | 7317 | 10410 | 877.5122 | 8986 |

**Exact match**, all four states, both land types, against attribute 45/2 and 46/3.

### `areaDomain` (mesic `PHYSCLCD %in% 21:29`), 4 states

| State | TPA | nPlots_TREE | nPlots_AREA |
|---|---|---|---|
| RI | 703.5731 | 79 | 124 |
| NC | 1278.035 | 2658 | 2997 |
| CO | 1702.031 | 1834 | 2121 |
| OR | 981.7755 | 6122 | 8523 |

**Exact match**, all four states, via `strFilter`, mirroring `tpa()`'s `areaDomain` mechanism.

### `treeDomain` (species filter)

| Case | Mechanism | TPA | nPlots_TREE |
|---|---|---|---|
| RI, white pine (`SPCD == 129`) | `wnum` | 331.7753 | 42 |
| NC, loblolly pine (`SPCD == 131`) | `wnum` | 136.3563 | 600 |

**Exact match** in both cases. Diameter-threshold `treeDomain` filters (used for `tpa()`) don't apply
here since `SEEDLING` has no `DIA` column.

### `bySpecies` grouping (RI)

Cross-checked 3 randomly sampled species rows from `seedling(bySpecies = TRUE)` against independent
single-species EVALIDator queries (`wnum = "SEEDLING.SPCD = <code>"`), same rationale/limitation as
`tpa.md`'s equivalent check (EVALIDator's own `rselected` row-grouping is a no-op on `fullreport`).
**Exact match** for all 3 species sampled (SPCD 531, 261, 43).

### `returnSpatial` (RI, by county)

`seedling(polys = countiesRI, returnSpatial = TRUE)` vs. `FALSE`: all non-geometry columns match
exactly. **Pass.**

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `TREE_TOTAL / AREA_TOTAL` reproduces `TPA` exactly, all four states. **Pass.**
- Empty `treeDomain` (`SPCD == 999`) and empty `areaDomain` (`STATECD == 999`): both return a clean
  0-row result with no warning. **Pass.**

## Fixed

Three bugs were found and fixed this pass, all in `R/seedlingStarter.R`. All three are variants of
bug classes already found and fixed in other estimators during this validation initiative, but each
had a `seedling()`-specific manifestation not caught by the existing fixes elsewhere.

### 1. `nPlots_AREA` phantom-row bug (same class as `tpa()`/`area()`/`biomass()`/`carbon()`/`volume()`/`dwm()`/`invasive()`) [FIXED]

Reproduced on RI: `landType = 'timber'` reported `nPlots_AREA = 132` (same as `'forest'`), but
EVALIDator's timberland denominator plot count is **126**. Identical root cause to every prior
instance of this bug (see `tpa.md`, "Fixed" #1): the condition list (`a`) in the population-estimation
branch was missing `dplyr::filter(!is.na(CONDID))`, so a plot whose only condition(s) failed the
`landType`/`areaDomain` filter survived the `PLOT`-to-`COND` left-join as a phantom `CONDID = NA` row,
contributing correctly to the area sum (`na.rm = TRUE`) but incorrectly inflating the plot count.

**Fix**: added `dplyr::filter(!is.na(CONDID))` to the condition list, identical to the fix already
applied to every other affected estimator.

### 2. `nPlots_TREE` counted every forest plot, not just plots with at least one seedling [FIXED]

RI: rFIA reported `nPlots_TREE = 132` (all forest plots); EVALIDator's numerator plot count is **81**
(only plots where at least one live seedling was actually tallied). `tpa()`'s equivalent tree list
avoids this because its `TREE_BASIS` is derived from `DIA` (`case_when(is.na(DIA) ~ NA_character_,
...)`), which is naturally `NA` for a phantom "no tree" join row, and the tree list already filters
`!is.na(TREE_BASIS)`. `seedling()`'s tree list instead hard-codes `TREE_BASIS = 'MICR'` unconditionally
(seedlings only ever use the microplot adjustment factor), so it has no column that's naturally `NA`
for a plot/condition with zero seedlings recorded -- the phantom row (created by the same
`PLOT`-`COND`-`SEEDLING` left-join structure `tpa()` uses) survived undetected.

**Fix**: added `dplyr::filter(!is.na(SPCD))` to the tree list, dropping phantom rows where the
`SEEDLING` join found nothing to match (mirroring the fix `invasive()` needed for its analogous
`!is.na(SYMBOL)` phantom-row problem). `nPlots_TREE` now reflects plots with at least one qualifying
seedling, matching EVALIDator's numerator plot count exactly.

### 3. Missing `CONDID` in the tree list's `distinct()` key silently undercounted seedlings on split-condition subplots [FIXED]

This was the one genuinely new bug class this pass, not previously seen in `tpa()` or any other
estimator, and it's a real point-estimate error (not just a plot-count cosmetic issue): before any
fix, `seedling(landType = 'forest')` gave `TPA = 1228.331` for NC vs. EVALIDator's `1230.528` -- small
(~0.18%) but real, and RI still matched exactly, which is what made it easy to miss initially (small,
simple states rarely have the triggering condition).

**Root cause**: `SEEDLING` has no per-stem ID -- `TPA_UNADJ` is already a count pre-aggregated to the
`PLT_CN`/`SUBP`/`CONDID`/`SPCD` grain by FIA (unlike `TREE`, where `TREE` is a genuine per-stem ID and
`distinct(PLT_CN, SUBP, TREE)` is always a safe dedup key regardless of `CONDID`). `seedlingStarter.R`'s
tree list used `dplyr::distinct(PLT_CN, SUBP, SPCD, .keep_all = TRUE)` -- omitting `CONDID` from the
key. Whenever a subplot straddles two conditions (a real, if uncommon, FIA sampling situation) and the
same species has seedlings recorded under both conditions for the same subplot, this collapsed the two
distinct `SEEDLING` rows into one, silently discarding the other condition's count entirely.

Confirmed directly in NC's raw `SEEDLING` extract: plot `1150115978290487`, subplot 3, red maple
(`SPCD 316`) has one row under `CONDID 1` (`TPA_UNADJ = 149.9306`) and a separate row under `CONDID 2`
(`TPA_UNADJ = 149.9306`) -- two real, distinct observations that `distinct(PLT_CN, SUBP, SPCD)`
collapsed into one, dropping 149.9306 TPA of red maple seedlings for that plot alone. A targeted scan
of NC's full `SEEDLING` table found 127 more `(PLT_CN, SUBP, SPCD)` combinations with exactly this
pattern (2 distinct `CONDID` values each) -- rare relative to NC's ~110k raw seedling records, but
each one a real, silent undercount, and NC/CO/OR (larger, more heterogeneous states) hit this pattern
often enough to move the state-level `TPA` measurably; RI (small, mostly single-condition plots)
never hit it, which is why it alone matched EVALIDator exactly even with this bug present.

**Fix**: added `CONDID` to the `distinct()` key in both the tree list's population-estimation branch
and its `byPlot` branch: `dplyr::distinct(PLT_CN, SUBP, CONDID, SPCD, .keep_all = TRUE)`. In the
population-estimation branch, `sumToPlot()` re-aggregates by `PLT_CN` + `grpBy` afterward (which
doesn't include `CONDID` unless the user explicitly groups by it), so the two condition-specific rows
correctly sum back together at the plot level -- this is a pure bugfix with no schema/output change.

**Verification**: after the fix, NC's `TPA` moved from `1228.331` to `1230.528`, an exact match to
EVALIDator (RI/CO/OR, already exact, were unaffected). Hand-verified independently via
`seedling(byPlot = TRUE, bySpecies = TRUE)` for the specific plot above: raw data shows red maple
seedlings on SUBP 3/CONDID 1 (149.9306), SUBP 3/CONDID 2 (149.9306), and SUBP 4/CONDID 2 (374.8264),
summing to `674.6875`; `seedling()`'s reported plot-level `TPA` for this plot/species now matches
exactly (regression test added, see below). Full package test suite re-run with no regressions.

## Non-TI method validation (SMA/LMA/EMA/ANNUAL)

EVALIDator has no equivalent for these, so correctness here means: the shared weighting machinery
(`maWeights()`/`filterAnnual()`/`combineMR()` in `R/util.R`, used by every `sumToEU()`-based
estimator) does what its own math says it does, and `seedling()`'s output behaves sanely and
consistently with the already-validated TI estimates wherever the documentation actually claims a
relationship. See `tests/testthat/test-util.R` for the underlying unit-level checks on this shared
machinery, and `tpa.md` (the template this section follows) for the full non-TI methodology.
`tpa.md`'s "Fixed" #6 documents a package-wide `combineMR()`/`ANNUAL` bug found and fixed during
`area()`'s non-TI pass; `seedling()` shares the same call site (`combineMR(tEst, method)` /
`combineMR(aEst, method)` in `seedlingStarter.R`) and was already covered by that fix before this
section was written (confirmed below, not just assumed).

**`seedling()` cannot have the `mergeSmallStrata()` P2Veg-style area-inflation defect found in
`vegStruct()`/`invasive()`.** That bug (see `vegStruct.md`'s `AREA_TOTAL` section) requires `db$PLOT`
to be pre-restricted to a P2-ancillary protocol subsample (`P2VEG_SAMPLING_STATUS_CD`,
`INVASIVE_SAMPLING_STATUS_CD`) before `handlePops()` runs, which leaves `INVYR = NA` rows in `pops`
and a pathological 1-2-of-5+-strata-present coverage pattern for `mergeSmallStrata()` to mishandle.
Confirmed by inspection (`grep` of `seedlingStarter.R`): `seedling()` applies no such filter to
`db$PLOT` and draws its population through `evalType = 'VOL'` (EXPVOL, the standard full-population
tree/volume evaluation), the same clean pattern already confirmed safe for `tpa()`/`diversity()` in
`diversity.md`'s "Findings" section -- structurally identical to those two, not merely untested.

### Results

- **EMA(lambda -> 1) vs. SMA (RI)**: `|EMA_TPA - SMA_TPA|` shrinks monotonically as lambda increases
  (328.84 -> 48.90 -> 4.30 -> 0.42 for lambda = 0.5/0.9/0.99/0.999) -- **pass**, confirms the
  vignette's documented limiting relationship at the `seedling()` output level.
- **TI vs. SMA bounded agreement, 4 states**: reuses the 10% relative tolerance established in
  `tpa.md` (panel-count CV is a state/data property, not an estimator property, so it's not
  recomputed per function -- same reuse `standStruct.md`/`carbon.md` already made).

  | State | TI TPA | SMA TPA | Relative diff |
  |---|---|---|---|
  | RI | 575.16 | 549.92 | −4.39% |
  | NC | 1230.53 | 1274.31 | +3.56% |
  | CO | 1609.05 | 1589.36 | −1.22% |
  | OR | 889.76 | 861.00 | −3.23% |

  All four states land well within the 10% bound -- **pass**.
- **Totals-vs-per-acre consistency under SMA/LMA/EMA/ANNUAL, 4 states**: `TREE_TOTAL / AREA_TOTAL ==
  TPA` to `1e-9` tolerance in all 16 state x method combinations -- **pass**.
- **`byPlot = TRUE` + non-TI method (RI, SMA)**: runs cleanly, returns 132 per-plot rows (not a
  population-level estimate) -- **pass**.
- **Domain filter (`treeDomain = SPCD < 300`, `areaDomain` mesic) + `bySpecies` under each of
  SMA/LMA/EMA/ANNUAL, 4 states**: no errors, no warnings, non-negative `TPA` in all 16 combinations
  (4/16/15/30 rows for SMA/LMA/EMA respectively per state, up to 257 for OR's ANNUAL) -- **pass**.
  Re-runs the historically-buggy filter/grpBy interaction from the TI validation (species-filter
  checks above) under every non-TI method.
- **`method = 'EMA'` with default arguments, 4 states**: runs without error in all four -- **pass**.
- **`method = 'ANNUAL'` on a `clipFIA(mostRecent = TRUE)` db returns one row per real panel, not a
  pooled row (NC, not RI -- see "RI's `SEEDLING` data lags its `TREE` data by one panel" below)**:
  confirms `tpa.md`'s "Fixed" #6 `combineMR()` fix covers `seedling()`. NC's clipped `ANNUAL` output
  returns 8 rows (2017-2024); the latest (2024: `TPA = 1227.72`, `nPlots_TREE = 605`) matches, to full
  precision, the same year computed independently from the full unclipped NC history -- **pass**, no
  re-pooling.

### RI's `SEEDLING` data lags its `TREE` data by one panel (data-cache observation, not a bug)

While setting up the `ANNUAL` regression check above, RI (used for every other check in this section
and in the original EVALIDator pass) turned out not to be usable for it. `seedling(db_ri, method =
'ANNUAL')` on the clipped db tops out at panel 2024, one year behind `tpa(db_ri, method = 'ANNUAL')`
on the identical db, which reaches 2025 (RI's true most-recent, self-hosting panel). Confirmed via the
raw, *unclipped* extract, independent of anything in this pass: all 40 plots measured in RI's 2025
panel (`INVYR = 2025`) have **zero** matching rows in `SEEDLING`, not just zero seedlings recorded
for every species -- i.e. the `SEEDLING` table itself has not yet been populated for that panel in the
current local cache, while `TREE` has. This is the same class of issue as `dwm.md`'s "RI is excluded
from every population-estimation check in this section" (FIA's phase-data publication for one table
lagging behind the core evaluation cycle for another), just one panel-year rather than a whole
evaluation. A knock-on effect: since RI's clipped-vs-full `ANNUAL` match is only guaranteed for a
panel that is genuinely self-hosted by the db's most-recent evaluation (`tpa.md`'s "Fixed" #6 only
claims this for the true latest year, not every row -- confirmed directly: even `tpa()`'s own 2024 row
differs slightly between the clipped and full-history runs, only 2025 matches exactly), RI's
seedling-data-lag-shifted "latest" row (2024) is *not* self-hosted and does **not** exactly match a
standalone full-history computation (211.2554 vs. 211.1668 TPA, same `nPlots_TREE = 12` -- confirmed a
`filterAnnual()` hosting-evaluation-choice difference, not a numeric bug). NC, CO, and OR's `SEEDLING`
and `TREE` most-recent panels agree (checked directly: same max `YEAR` under `ANNUAL` for both
functions in all three), so NC was substituted for the `ANNUAL` regression test instead. This is a
live-data-cache observation, not a package bug, and needs no fix -- flagged here (rather than under
"Findings") since it's specific to today's cached extract, matching the precedent set by
`dwm.md`'s RI section and the "Unrelated, pre-existing, noticed-in-passing" CO/OR drift note in the
validation-initiative memory.

## Findings (reported, not fixed -- see bug-handling protocol)

1. **`method = 'ANNUAL'` emits a spurious `max()`-on-empty-group warning when a domain filter matches
   zero rows**, the same shared `filterAnnual()` defect already reported (not fixed) for `dwm()` in
   `dwm.md`'s "Findings" section. Reproduced directly on RI: both
   `seedling(db_ri, treeDomain = SPCD == 999, method = 'ANNUAL')` and
   `seedling(db_ri, areaDomain = STATECD == 999, method = 'ANNUAL')` emit
   `"no non-missing arguments to max; returning -Inf"` from inside a `dplyr::mutate()` call, while
   still returning the correct, clean 0-row result. Root cause and fix scope are identical to the
   `dwm()` write-up (an unguarded `max()` inside `filterAnnual()`, shared by every `sumToEU()`-based
   estimator, not `seedling()`-specific) -- not re-investigated here since it would be a duplicate
   analysis; not fixed here for the same reason `dwm()`'s instance wasn't (narrow trigger, no
   regression test pinned to it, per Jeff's existing sign-off on that approach).

## Deferred to follow-up (not covered this pass, flagged for sign-off before touching)

- **`treeList = TRUE` output duplicates seedling counts across conditions on multi-condition plots.**
  Found while verifying fix #3 above, but this is a distinct, pre-existing bug (present before this
  pass's changes too, just silently producing a different wrong number), and fixing it properly
  requires restructuring how the condition list (`a`) and tree list (`t`) are joined -- out of scope
  for a targeted validation pass, and likely shared by other estimators' `treeList` branches (e.g.
  `tpa()`'s), not `seedling()`-specific. Concretely: `seedlingStarter.R`'s `treeList = TRUE` branch
  joins `a` (one row per `PLT_CN`/`CONDID`) to `t` (one row per `PLT_CN`/`SPCD`, `CONDID` no longer
  present after fix #3's `select()`) via `left_join(t, by = c('PLT_CN', aGrpBy))` -- `CONDID` is not
  part of the join key. For a 2-condition plot, this is a many-to-many join: NC plot
  `1150115978290487` (the same plot used in fix #3) returns **two** rows for red maple, one per
  `CONDID`, both reporting the *full* plot-level total (`TPA = 2848.681`, the all-species pooled
  total for that plot) rather than splitting it by condition -- i.e. the same value is double-counted
  across the two `CONDID` rows. `seedling(byPlot = TRUE)` does **not** have this problem -- its
  condition list is pre-aggregated to one row per plot before the join, so there's no many-to-many
  join to trigger it (confirmed: `byPlot = TRUE`'s value for the same plot/species correctly reflects
  the summed total, see fix #3's verification). Needs explicit sign-off before changing, given it
  touches the `treeList`/`customPSE()` contract and may be shared architecture.

  **Follow-up check: does this actually corrupt `customPSE()` output?** Tested directly by feeding
  `seedling(treeList = TRUE)` into `customPSE()` (numerator `xVars = TPA`, denominator
  `yVars = PROP_FOREST`, mirroring the exact pattern `test-customPSE.R` already uses for `tpa()`) and
  comparing against `seedling()`'s own population-level `TPA`, across all four validation states, both
  `landType`s, and a full `bySpecies = TRUE` breakdown (including red maple/SPCD 316 on the NC plot
  above). **Result: exact match (diff = 0) in every case**, including `nPlots_x`/`nPlots_y` vs.
  `nPlots_TREE`/`nPlots_AREA`. The duplicate-row bug turns out to be silently absorbed rather than
  propagated: `customPSE()` keys tree-basis data by `SUBP`/`TREE` (both hard-set to `NA` for
  seedlings, per the `mutate()` two blocks above this one) and does not use `CONDID` to distinguish
  rows unless the caller explicitly adds it via `xGrpBy`/`yGrpBy`. Since the duplicated `CONDID` rows
  carry byte-identical `TPA` (and no other retained column differs), `customPSE()`'s internal
  `dplyr::distinct()` collapses them back into one row before summing to the plot level -- so the
  standard, documented numerator/denominator workflow is safe today.

  **This is conditional, not a green light to ignore the bug.** If a caller explicitly retains
  `CONDID` (e.g. `xGrpBy = c(SPCD, CONDID)`, which is a reasonable thing to want for condition-level
  detail), the bug is fully exposed: on the same NC plot, red maple's true total is 136.5 TPA/acre,
  but grouping by `(SPCD, CONDID)` returns 4 phantom per-condition rows summing to 2204.3 TPA/acre
  (~16x inflation). The raw `treeList = TRUE` output is also just wrong on its face for any consumer
  that doesn't route through `customPSE()`'s dedup-by-`(SUBP, TREE)` behavior. Still needs the proper
  join fix; downgraded here from "may corrupt `customPSE()` calculations" to "safe for the standard
  `customPSE()` pattern, unsafe if `CONDID` is added to `xGrpBy`/`yGrpBy`."
- `byPlot = TRUE` aggregating to reproduce the population-level estimate exactly (only the specific
  split-condition-plot hand calculation above was checked, not a full aggregation reconciliation --
  same limitation noted in `tpa.md`/`invasive.md`).
- A national audit of how often the split-condition-subplot pattern (fix #3) occurs beyond the four
  states checked here.

## Notes

### Documentation drift [FIXED]

`man/seedling.Rd`'s `\value{}` section documented a `TPA_PERC` output column and an `nPlots_SEEDLING`
column; neither exists in `seedling()`'s actual output (`nPlots_TREE` is what's actually returned,
matching `tpa()`'s naming) -- this looked like documentation copied from `tpa.Rd` and not fully
adapted. Not an estimation bug, so no test/NEWS.md entry, but corrected directly in `man/seedling.Rd`:
removed the nonexistent `TPA_PERC` bullet and renamed `nPlots_SEEDLING` to `nPlots_TREE`, matching the
column `seedling()` actually returns.
