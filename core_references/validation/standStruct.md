# Validation report: `standStruct()`

## Scope

`standStruct()` estimates the percent of forest/timberland area in each stand structural stage
(`pole`/`mature`/`late`/`mosaic`), classified per condition from live-tree basal-area proportions in
three diameter classes (a method similar to Frelich & Lorimer 1991, substituting basal area for
exposed crown area). Structurally, it's close to `tpa()` (same `PLOT`/`COND`/`TREE` tables, same
`landD`/`aD` domain-indicator machinery), but with two real differences:

- No `treeType`/`treeDomain` argument at all -- only `landType` and `areaDomain`. Trees are filtered
  to crown class `{2,3,4}` (dominant/co-dominant/intermediate; suppressed and open-grown stems are
  excluded) internally by `structHelper()`, not user-configurable.
- The unit of estimation is the **condition**, not the tree: every forest condition gets exactly one
  `STAGE` classification (falling back to `'mosaic'` whenever there's insufficient basal-area
  information to classify it, e.g. a condition with zero qualifying trees), so `COVER_PCT` summed
  across all four `STAGE` categories should always equal exactly 100% of the area total -- a strong,
  function-specific internal-consistency check not available for tree-count estimators like `tpa()`.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest) -- the same four used for `tpa()`/`seedling()`.

## Methodology: no EVALIDator ground truth exists for this function

Like `invasive()`, stand structural stage has **no EVALIDator equivalent at all** --
`EVALIDATOR_POP_ESTIMATE.csv` (719 attributes) has zero matches for "structural stage", "stand
structure", "structure class", or similar; EVALIDator's own report categories have nothing analogous.
This is an rFIA-specific derived classification (`structHelper()` in `R/util.R`), not a standard
FIADB/EVALIDator population attribute. Validation here is therefore:

1. **Cross-checks against `tpa()`'s own `nPlots_AREA`/`AREA_TOTAL`** (already validated against
   EVALIDator; see `tpa.md`) for the same `landType`/`areaDomain`/`grpBy` restriction -- valid because
   both functions share the identical area-denominator machinery (`PLOT`/`COND` domain indicators,
   `sumToPlot()`/`sumToEU()`).
2. **A hard internal-consistency invariant specific to this function**: `COVER_PCT` summed across all
   `STAGE` categories must equal exactly 100%, since every forest condition gets exactly one
   classification (unlike `tpa()`/`seedling()`, where "percent of trees" isn't naturally exhaustive).
3. **Manual hand-calculations from raw `TREE`/`COND` data**, replicating `structHelper()`'s basal-area
   -proportion formula independently for a specific plot.

## Results

### `COVER_PCT` sums to exactly 100%, 4 states

| State | sum(COVER_PCT) |
|---|---|
| RI | 100 |
| NC | 100 |
| CO | 100 |
| OR | 100 |

**Exact match** (tolerance `1e-6`) in all four states, after the fixes below. Before fix #2, NC/CO/OR
(but not RI) fell short of 100% by a small amount (NC: 99.98587, CO: 99.98833, OR: 99.95732) -- see
"Fixed" below for root cause.

### `nPlots_AREA`/`AREA_TOTAL` cross-check against `tpa()`, 4 states

| State | `landType='forest'` | `landType='timber'` | `areaDomain` (mesic) |
|---|---|---|---|
| RI | 132 = 132 | 132\* &rarr; 126 = 126 | 132\* &rarr; 124 = 124 |
| NC | 3561 = 3561 | 3561\* &rarr; 3436 = 3436 | 3561\* &rarr; 2997 = 2997 |
| CO | 3925 = 3925 | 3925\* &rarr; 1829 = 1829 | 3925\* &rarr; 2121 = 2121 |
| OR | 10410 = 10410 | 10410\* &rarr; 8986 = 8986 | 10410\* &rarr; 8523 = 8523 |

\* value before fix #1 (identical to the unrestricted `landType = 'forest'` count in every case,
i.e. `landType`/`areaDomain` had no effect on `nPlots_AREA` at all before the fix). `AREA_TOTAL`
(not shown, all four states/cases) also matches `tpa()`'s `AREA_TOTAL` exactly, both before and after
fix #1 -- this was purely a plot-count bug, not a point-estimate bug.

### `grpBy` interaction (`OWNGRPCD`, NC)

Each ownership group's `COVER_PCT` sums to exactly 100%, and each group's `AREA_TOTAL` matches
`tpa(grpBy = OWNGRPCD)`'s grouped `AREA_TOTAL` exactly (4 ownership groups checked: 226/146/247/2982
plots). **Pass** -- confirms the `grpBy` join doesn't silently drop or misattribute area for some
groups (the historical `area()`/`areaChange()` bug pattern from v1.1.1).

### Hand calculation (RI, `pltID = "1_44_3_233"`)

25 live trees on the plot's one forested condition (`CONDID 2`, `CONDPROP_UNADJ = 0.326472`; the
plot's other condition, `CONDID 1`, is non-forest). By hand, restricting to crown class `{2,3,4}` and
`DIA >= 5"` (14 qualifying trees): pole-class (5"-10.23622") basal-area share = 0.6739, mature-class
(10.23622"-18.11024") share = 0.3261, large-class share = 0. Since pole + mature > 0.67 and
pole > mature, `STAGE = 'pole'` per the documented classification rules -- matches
`standStruct(byPlot = TRUE)`'s reported `STAGE = 'POLE'`, `PROP_STAGE = 0.326472` exactly (full
`CONDPROP_UNADJ` of the one forested condition). Also confirms the diameter thresholds in
`man/standStruct.Rd` (12.7-25.9cm pole / 26-45.9cm mature / 46+cm large) convert exactly to
`structHelper()`'s inch thresholds (5 / 10.23622 / 18.11024"), no discrepancy.

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `STAGE_AREA_TOTAL / AREA_TOTAL * 100` reproduces `COVER_PCT` exactly, across all
  four states. **Pass.**
- `returnSpatial` (RI, by county): all non-geometry columns match exactly between
  `returnSpatial = TRUE`/`FALSE`. **Pass.**

## Fixed

Three bugs were found and fixed this pass, all in `R/standStructStarter.R`. All three are variants of
bug classes already found and fixed in other estimators during this validation initiative.

### 1. `nPlots_AREA` phantom-row bug (same class as `tpa()`/`area()`/`biomass()`/`seedling()`/etc) [FIXED]

Identical root cause and fix to every prior instance of this bug (see `tpa.md`, "Fixed" #1;
`seedling.md`, "Fixed" #1): the condition list (`a`) in the population-estimation branch was missing
`dplyr::filter(!is.na(CONDID))`, so a plot whose only condition(s) failed the `landType`/`areaDomain`
filter survived the `PLOT`-to-`COND` left-join as a phantom `CONDID = NA` row, contributing correctly
to the area sum (`na.rm = TRUE`) but incorrectly inflating the plot count. Reproduced on all four
states: `landType = 'timber'`/`areaDomain` restrictions had *zero* effect on `nPlots_AREA` before the
fix (always equal to the unrestricted `'forest'` count).

**Fix**: added `dplyr::filter(!is.na(CONDID))` to the condition list, identical to the fix already
applied to every other affected estimator.

### 2. Missing `CONDID` in the STAGE tree list's `distinct()` key silently dropped area from a zero-tree condition whenever a plot had two or more of them [FIXED]

The same bug class found in `seedling()` this initiative (see `seedling.md`, "Fixed" #3), but with a
different trigger condition: `standStructStarter.R`'s STAGE-classification tree list used
`dplyr::distinct(PLT_CN, SUBP, TREE, .keep_all = TRUE)`. A forest condition with zero qualifying trees
(e.g. a young/sparse/non-stocked stand) survives the `TREE` join as a phantom row with
`SUBP = NA`/`TREE = NA` (structurally identical to `tpa()`'s equivalent phantom row, which `tpa()`
simply discards since it only wants a tree count). `standStruct()`, unlike `tpa()`, does *not* discard
this phantom row -- it's meaningful here, since `structHelper()`'s `NaN`-proportion fallback correctly
classifies a zero-tree condition as `'mosaic'` and its `CONDPROP_UNADJ` area needs to be counted
towards that category. The bug: whenever a single plot had **two or more** such zero-tree conditions,
their phantom rows were indistinguishable by `(PLT_CN, SUBP, TREE)` alone (both have `SUBP = NA`,
`TREE = NA`), so `distinct()` silently collapsed them into one, dropping every zero-tree condition's
area past the first from the `STAGE` classification entirely -- while their area still correctly
counted in `tpa()`'s/the condition list `a`'s `AREA_TOTAL`, which is what made `COVER_PCT` fall short
of 100% (rather than just be wrong within one category).

Confirmed directly in NC's raw `COND`/`TREE` extracts: plot `1150116756290487` has two zero-tree
forest conditions (`CONDID 2` and `CONDID 3`, `CONDPROP_UNADJ = 0.25` each, total `0.5`). Before the
fix, `standStruct(byPlot = TRUE)` reported `PROP_STAGE = 0.25` (`MOSAIC`) for this plot against a
correct `PROP_FOREST = 0.5` -- half of the plot's forest area silently missing from every stage
category. A full scan of NC's raw data found 10 plots total with this exact pattern (2+ zero-tree
forest conditions) -- rare, but enough to move the state-level `COVER_PCT` sum measurably (NC:
99.98587% instead of 100%); RI apparently has none, which is why it alone summed to exactly 100% even
with the bug present.

**Fix**: added `CONDID` to the `distinct()` key in both the tree list's population-estimation branch
and its `byPlot` branch: `dplyr::distinct(PLT_CN, SUBP, CONDID, TREE, .keep_all = TRUE)`.

**Verification**: after the fix, all four states' `COVER_PCT` sums to exactly 100% (previously NC
99.98587, CO 99.98833, OR 99.95732 -- RI was already 100 and unaffected). The specific NC plot above
now reports `PROP_STAGE = 0.5`, matching `PROP_FOREST` exactly (regression test added). Full package
test suite re-run with no regressions.

### 3. Empty `areaDomain` produced a spurious surviving row instead of a clean empty result [FIXED]

```r
standStruct(db_ri, areaDomain = STATECD == 999)
```
Returned a 1-row result (`STAGE = 'MOSAIC'`, `COVER_PCT = NA`, `YEAR = -Inf`) with a
`"no non-missing arguments to max"` warning, instead of a clean 0-row result like every other
validated estimator's empty-domain case (same warning class already fixed in `tpa()`/`invasive()`/
`seedling()`, and the same underlying `combineMR()` guard those fixes rely on -- but that guard only
helps if the population estimate genuinely has 0 rows going in, which wasn't the case here).

**Root cause**: same phantom-row pattern as fix #1/#2, but manifesting through `structHelper()`'s
`NaN`-proportion fallback specifically: the STAGE tree list didn't filter `!is.na(CONDID)` the way
the condition list `a` now does (fix #1), so when *every* condition in the domain is a phantom
`CONDID = NA` row (as happens when `areaDomain` matches nothing at all, not just some conditions),
`structHelper(NA, NA)` still returns `'mosaic'` (its designed fallback for "insufficient information
to classify"), producing a single surviving `STAGE = 'MOSAIC'` row with `nrow(x) > 0` -- which bypasses
`combineMR()`'s existing 0-row guard, letting its `max(YEAR, na.rm = TRUE)` reach an all-`NA` `YEAR`
column and emit the warning (same failure mode as `invasive.md`, "Fixed" #3, just a different tree
list producing the phantom row).

**Fix**: added `dplyr::filter(!is.na(CONDID))` to the STAGE tree list (`t`) in the
population-estimation branch, immediately after its `distinct()` step (mirroring fix #1's placement
in `a`).

**Verification**: after the fix, the empty-domain case above returns a clean 0-row tibble with no
warning. Full package test suite re-run with no regressions. Regression test added:
`tests/testthat/test-standStruct.R` now asserts `expect_no_warning()` around this call.

## Deferred to follow-up (not covered this pass)

- `method` options other than `'TI'` (no EVALIDator equivalent; internal-consistency-only checks per
  the plan, not yet added).
- `byPlot = TRUE` aggregating to reproduce the population-level estimate exactly (only the specific
  hand-calculated/regression-tested plots above were checked, not a full aggregation reconciliation --
  same limitation noted in `tpa.md`/`seedling.md`/`invasive.md`).
- A national audit of how often the multi-zero-tree-condition pattern (fix #2) occurs beyond the four
  states checked here.

## Notes

### Documentation drift [FIXED]

`man/standStruct.Rd`'s `\value{}` section documented only two output columns, `STAGE` and `PERC`;
`standStruct()`'s actual output is `STAGE`, `COVER_PCT` (not `PERC`), `COVER_PCT_SE`, and
`nPlots_AREA`, plus `STAGE_AREA_TOTAL`/`AREA_TOTAL` (and their `_SE` columns) when `totals = TRUE`.
Same class of drift as `seedling.Rd` (see `seedling.md`'s Notes). Not an estimation bug, so no
test/NEWS.md entry, but corrected directly in `man/standStruct.Rd`: replaced the `STAGE`/`PERC` bullet
list with `YEAR`/`STAGE`/`COVER_PCT`/`nPlots_AREA`, matching the columns `standStruct()` actually
returns by default (the `totals = TRUE`-only `_TOTAL` columns are left undocumented in the itemized
list, consistent with `tpa.Rd`'s existing convention of not itemizing those either).
