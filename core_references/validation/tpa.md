# Validation report: `tpa()`

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint (the programmatic interface
behind the EVALIDator web tool), queried live via `fetch_evalidator.R` in this directory. rFIA was
run against the real, current FIADB extracts already cached at `~/Dropbox/data/fia/` (pulled via
`getFIA()`), using `clipFIA(mostRecent = TRUE)` to match EVALIDator's "current" evaluation.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest).

`tests/testthat/test-tpa.R` calls the FIADB-API live at test time rather than hard-coding the
reference numbers below — this section of the report is illustrative, not a source of truth the
tests are pinned to. The EVAL_GRP code for each state (e.g. RI 442024) is likewise not hard-coded
anywhere: it's read directly off `clipFIA(..., mostRecent = TRUE)$POP_EVAL_GRP$EVAL_GRP`, so the
test always queries whichever evaluation `mostRecent` actually selected, even as the local data
cache is refreshed over time. The tests are skipped (not failed) when the local data cache or
network access to `apps.fs.usda.gov` is unavailable.

Two API mechanisms were needed to correctly mirror rFIA's domain-filter semantics:
- **`wnum`** (numerator-only filter) for `treeDomain`-style filters — these should not change the
  area denominator, matching rFIA's design (`tD`/`typeD` only affect the tree sum, not `fa`).
- **`strFilter`** (applies to both numerator and denominator) for `areaDomain`-style filters — these
  should shrink both the tree sum and the area denominator, matching rFIA's `aD`/`aDI` design.
- A leading `AND` in either parameter, as shown in the FIADB-API docs' own example, reproducibly
  caused an "Estimate Error: Parameters resulted in no data being retrieved" even for filters that
  clearly match data (confirmed with `COND.OWNCD = 40` and `TREE.CCLCD in (1,2,3)`). Omitting the
  leading `AND` resolved this. Worth reporting upstream to FIA if this becomes a recurring friction
  point, but not an rFIA issue.

## Results: numeric match

All point estimates and percent standard errors below match the FIADB-API to full double precision
(14+ significant digits) unless noted.

### Core default case (`treeType = 'live', landType = 'forest'`), 4 states

| State | TPA | BAA | TPA_SE | BAA_SE | nPlots_TREE |
|---|---|---|---|---|---|
| RI | 365.206954414683 | 117.303549525083 | 6.31384177386282 | 3.22513090470676 | 129 |
| NC | 712.908125719543 | 118.678811889536 | 1.40717941990529 | 0.830279989984542 | 3455 |
| CO | 481.196172301869 | 91.4134316843006 | 1.9072805017832 | 1.0172171069138 | 3774 |
| OR | 347.605500973174 | 127.288376448323 | 1.34375054278634 | 0.708291919603776 | 9968 |

All four: **exact match** against EVALIDator attribute 4 (live trees, forest land) / 1004 (basal
area, live, forest land), ratio'd against attribute 2 (forest land area).

### `landType`/`treeType` variants, 4 states

| Case | EVALIDator attr (num/denom) | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `landType = 'timber'` | 7 / 3 | exact | exact | exact | exact |
| `treeType = 'gs'` (growing-stock) | 5 / 2 | exact | exact | exact | exact |
| `treeType = 'dead'` | 11264 / 2 | exact | exact | exact | exact |

`treeType = 'all'` has no direct EVALIDator equivalent (EVALIDator doesn't expose a combined
live+dead tree count); validated instead via an internal-consistency check across all four states.
Since the fix in finding #3 below, `all` is no longer expected to equal `live` + `dead` exactly —
`all` still includes every tree regardless of status, while `dead` now excludes non-standing dead
trees, so the check is `all >= live + dead` (a strict lower bound, not an equality) — **pass**.

### Domain filter interactions, 4 states

| Case | Mechanism | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `treeDomain = DIA >= 20` (large trees) | `wnum` | exact | exact | exact | exact |
| `areaDomain = PHYSCLCD %in% 21:29` (mesic) | `strFilter` | exact | exact | exact | exact |
| `treeDomain = SPCD == 129` (white pine, RI only) | `wnum` | exact | — | — | — |

Point estimates, SEs, `nPlots_TREE`, and `nPlots_AREA` (where checked) all match exactly across all
four states for both the diameter-based `treeDomain` and the physiographic-class `areaDomain`. The
white pine species filter is RI-specific (not a nationally meaningful filter for the other three
states) and was left as the original single-state check.

### `bySpecies` grouping (RI)

Cross-checked a random sample of individual species rows from `tpa(bySpecies = TRUE)` against
independent single-species EVALIDator queries (`wnum = "TREE.SPCD = <code>"`) — this validates that
the `grpBy = SPCD` join/aggregation path doesn't silently drop the domain filter for some groups
(the historical `area()`/`areaChange()` bug pattern from v1.1.1). **Pass** for the species sampled.
EVALIDator's own row-grouping mechanism (`rselected`) could not be made to return grouped rows via
the `fullreport` endpoint — see "API grouping (`rselected`/`cselected`) appears to be a no-op"
below — so this is a per-species cross-check rather than a single grouped-report comparison.

### `returnSpatial` (RI, by county)

`tpa(polys = countiesRI, returnSpatial = TRUE)` vs. `returnSpatial = FALSE`: all non-geometry
columns match exactly (`expect_equal` on the two data frames, geometry column dropped). **Pass.**

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `TREE_TOTAL / AREA_TOTAL` reproduces `TPA` exactly, across all four states.
  **Pass.**

## Fixed

### 1. `nPlots_AREA` did not reflect `landType` or `areaDomain` restrictions [FIXED]

Reproduced on RI 2024 (EVAL_GRP 442024) in three independent ways:

- `landType = 'timber'`: rFIA reported `nPlots_AREA = 132` (same as `landType = 'forest'`).
  EVALIDator's timberland-area denominator plot count is **126**
  (`fetch_evalidator(wc=442024, snum=7, sdenom=3)$denPlotCount`).
- `areaDomain = PHYSCLCD %in% 21:29`: rFIA reported `nPlots_AREA = 132`. EVALIDator's filtered-area
  denominator plot count is **124** (`fetch_evalidator(wc=442024, snum=4, sdenom=2,
  strFilter="COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)")$denPlotCount`).

Confirmed as a genuine bug (not intentional design) and fixed. **Point estimates and SEs were
unaffected in all cases** — this was purely a plot-count/reporting bug, but it mattered because
`tpa.Rd` explicitly tells users to use `nPlots_AREA` as the degrees of freedom for a t-based
confidence interval, so an inflated count would understate the true margin of error for any
`landType = 'timber'` or `areaDomain`-restricted estimate.

**Root cause**: in `R/tpaStarter.R`, `db$COND` is filtered to `aD == 1 & landD == 1` (the area/land
domain) *before* being joined to `db$PLOT`, but `db$PLOT` itself is only filtered to
`PLOT_STATUS_CD == 1 & sp == 1` — a broader set. The subsequent `left_join` from `db$PLOT` to the
now-narrower `db$COND` means plots whose only condition(s) failed the domain filter survive as a
row with `CONDID = NA` (the unmatched side of the join). That phantom row correctly contributes
`fa = NA`, which `sum(..., na.rm = TRUE)` correctly reduces to a 0 area contribution downstream (so
the point estimate was always right) — but its `PLT_CN` was still present in the table that
`nPlots_AREA` counts via `length(unique(PLT_CN))`, so it was counted as if it were a real
contributing plot. The tree-list branch a few lines below already guarded against exactly this
failure mode (`dplyr::filter(!is.na(TREE_BASIS))`), which is why `nPlots_TREE` always matched
EVALIDator correctly — the condition-list branch was simply missing the equivalent filter.

**Fix**: added `dplyr::filter(!is.na(CONDID))` to the condition-list (`a`) construction in
`R/tpaStarter.R`, mirroring the existing `!is.na(TREE_BASIS)` filter on the tree list a few lines
below. One line, no change to any point estimate or SE logic.

**Verification**: after the fix, `nPlots_AREA` returns 126 for `landType = 'timber'` and 124 for
the mesic `areaDomain` case — exact matches to EVALIDator's `denPlotCount` in both cases. Full
package test suite (all 17 estimator test files) re-run with no regressions. Regression tests
added: `tests/testthat/test-tpa.R` now asserts `nPlots_AREA` for both cases (previously left
unchecked, flagged as an open finding).

### 2. Empty-domain edge case produced a spurious warning, not a clean result [FIXED]

```r
tpa(db_mr, treeType = 'live', landType = 'forest', treeDomain = SPCD == 999)
```
Returned a 0-row tibble (reasonable) but also emitted:
```
Warning: no non-missing arguments to max; returning -Inf
  (In argument: `YEAR = max(YEAR, na.rm = TRUE)`)
```

**Root cause**: `combineMR()` (`R/util.R`) relabels `YEAR` to the max across states when combining
most-recent-subset population estimates (e.g. 2016 in MI, 2017 in WI → both labeled 2017). It's
called unconditionally whenever `mr = TRUE`, including when a `treeDomain`/`areaDomain` matches no
rows and `tEst`/`aEst` are 0-row tibbles — at which point `max(YEAR, na.rm = TRUE)` has no
non-missing input, R warns, and the (already-empty) result gets a `YEAR` of `-Inf`.

This isn't tpa()-specific: `combineMR()` is shared by every estimator (`area()`, `biomass()`,
`growMort()`, `volume()`, etc. — see `R/util.R`), so the same warning would occur for any of them
given an empty domain and `mostRecent = TRUE`. The fix is in the shared utility, since there's no
tpa()-only copy of this logic to scope it to, but it's a pure no-op for every existing non-empty
call path (the guard only short-circuits the already-degenerate 0-row case), so it carries no risk
to any other estimator's numeric output.

**Fix**: added a `nrow(x) == 0` guard to `combineMR()` that returns `x` unchanged (still 0 rows,
just without attempting the `YEAR` relabel). One line, no change to the non-empty path.

**Verification**: the empty-domain case above now returns a clean 0-row tibble with no warning.
Confirmed no change to the core default-case numeric output (RI TPA still matches EVALIDator to
full precision). Full package test suite re-run with no regressions. Regression test added:
`tests/testthat/test-tpa.R` now asserts `expect_no_warning()` around this call.

### 3. `treeType = 'dead'` did not match EVALIDator's standing-dead definition (NC) [FIXED]

```r
tpa(db_nc, treeType = 'dead', landType = 'forest')
```
rFIA reported `TPA = 152.9567`. EVALIDator (attribute 11264, "Number of standing dead trees ... on
forest land", ratio'd against attribute 2): `TPA = 37.5248`. Roughly 4x too high. RI, CO, and OR all
matched EVALIDator exactly for this same case — the discrepancy was state-specific, not universal,
which is why it wasn't caught until `treeType = 'dead'` was checked against a state other than RI.

**Root cause**: `treeTypeDomain()` (`R/util.R`) defined `treeType = 'dead'` as `STATUSCD == 2` only.
EVALIDator attribute 11264's SQL definition additionally requires `TREE.STANDING_DEAD_CD = 1` (and
`DIA >= 1`) — i.e. the dead tree must meet the "standing dead tally tree" criteria (unbroken bole
length >= 4.5 ft, leaning less than 45 degrees from vertical; see the attribute's metadata in
`EVALIDATOR_POP_ESTIMATE.csv`, and `man/tpa.Rd`'s existing `treeType` documentation, which already
described `'dead'` as "leaning less than 45 degrees" even though the implementation never actually
checked it). Checked all four states' local extracts directly: RI, CO, and OR each have
`STATUSCD == 2` rows with `STANDING_DEAD_CD != 1`, but in every one of those states those particular
rows also have `DIA = NA`, so rFIA's own `!is.na(DIA)` filter already excluded them — masking the
difference by coincidence. NC has ~29,700 `STATUSCD == 2` rows with `STANDING_DEAD_CD != 1` that *do*
have real diameters (range 1"–45.6"), contributing real weight (`sum(TPA_UNADJ)` ~918,000) that rFIA
included and EVALIDator excluded.

**Fix**: `treeTypeDomain()` (`R/util.R`) now requires `STATUSCD == 2 & STANDING_DEAD_CD == 1` for
`treeType = 'dead'`, matching EVALIDator and the pre-existing (but previously unenforced)
documentation. `treeTypeDomain()` is shared by `tpa()`, `diversity()`, `biomass()`, `volume()`, and
`fsi()` (`R/*Starter.R`), so all five call sites were updated to pass `STANDING_DEAD_CD` through;
this affects every estimator's `treeType = 'dead'` output, not just `tpa()`'s.

**Side effect**: `treeType = 'all'` (`STATUSCD` unrestricted) is no longer expected to equal
`treeType = 'live'` + `treeType = 'dead'` exactly, since `'dead'` now excludes non-standing dead
trees that `'all'` still includes. The internal-consistency check for this was changed from an
equality to a `all >= live + dead` lower bound (see "Results" above) — this is expected and
intentional, not a new issue.

**Verification**: after the fix, `treeType = 'dead'` matches EVALIDator exactly in all four states,
including NC (was 152.9567, now 37.5248, matching to full double precision). Full package test
suite (all 17 estimator test files) re-run with no regressions. Regression tests updated:
`tests/testthat/test-tpa.R`'s `treeType = 'dead'` test for NC (previously `skip()`-ped) now runs and
passes; the `treeType = 'all'` consistency check now asserts `>=` instead of `==`.

## Notes

### API grouping (`rselected`/`cselected`) appears to be a no-op on `fullreport`

Attempted to validate `bySpecies`/`grpBy` against EVALIDator's own row-grouping feature
(`rselected = "SPCD"`, or various other candidate values: `"SPECIES"`, `"FORTYPCD"`, `"UNITCD"`,
`"COUNTYCD"`) via the `fullreport` endpoint. In every case the response was identical to
`rselected = "Total"` — a single ungrouped row — and the `numSql`/`denSql` echoed back in the
response metadata confirmed the underlying `GROUP BY` clause never included a grouping column
regardless of the `rselected` value supplied. This suggests row/column grouping is only available
through the stateful multi-step EVALIDator web wizard (confirmed via the FIADB-API's own
`EVALIDator_User_Guide-v2.1.pdf`, linked from `https://apps.fs.usda.gov/fiadb-api/`, which documents
row/column variable selection as a web-form step, not a `fullreport` query parameter), not the
single-shot `fullreport` API used here. Worked around by cross-checking individual species via
`wnum` instead (see "`bySpecies` grouping" above).

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population estimate (only totals-vs-per-acre was
  checked).
- `method` options other than `'TI'` (EVALIDator has no equivalent; these need
  internal-consistency-only checks per the plan).
