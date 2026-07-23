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
  across the two `CONDID` rows. This would corrupt any downstream `customPSE()` calculation that
  treats each row as an independent observation. `seedling(byPlot = TRUE)` does **not** have this
  problem -- its condition list is pre-aggregated to one row per plot before the join, so there's no
  many-to-many join to trigger it (confirmed: `byPlot = TRUE`'s value for the same plot/species
  correctly reflects the summed total, see fix #3's verification). Needs explicit sign-off before
  changing, given it touches the `treeList`/`customPSE()` contract and may be shared architecture.
- `method` options other than `'TI'` (no EVALIDator equivalent; internal-consistency-only checks per
  the plan, not yet added).
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
