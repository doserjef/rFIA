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

### 4. `maWeights()` accepted out-of-range/boundary `lambda` and silently returned degenerate weights [FIXED]

Found during non-TI method validation (see "Non-TI method validation" below), not the original
EVALIDator pass. `man/tpa.Rd` (and every other estimator's man page) documents `lambda` as
`numeric (0,1)`, but `maWeights()` (`R/util.R`) never validated it. At the exact boundaries,
`lambda = 1` produced `NaN` weight for every panel (`l = 1-lambda = 0`, so every weight term is
`0 * 1^(...) = 0` and `sumwgt = 0`, giving `0/0`); `lambda = 0` produced weight `0` for the oldest
panel and `NaN` for every other panel (`(1-l) = 0`, so `0^(negative exponent) = Inf` for every panel
but the oldest, making `sumwgt = Inf`). Out-of-range values were worse: `lambda = -0.5` produced a
negative weight (not a convex combination), and `lambda = 1.5` produced an *inverted* recency
ordering (oldest panel weighted highest, most recent weighted lowest) — the opposite of what EMA is
for.

**Root cause**: no input validation in `maWeights()`'s `EMA` branch (`R/util.R:1113-1156`); `lambda`
was plugged directly into the weighting formula regardless of value.

**Fix**: added a guard at the top of the `EMA` branch that `stop()`s with a clear message if any
element of `lambda` is `NA`, `<= 0`, or `>= 1`. `maWeights()` is shared by every `sumToEU()`-based
estimator plus `fsi()` (via `fsiHelper2`) and `customPSE()`, so this validates `lambda` once for all
of them rather than needing a per-`*Starter.R` copy.

**Verification**: `lambda ∈ {0, 1, -0.5, 1.5, NA}` now error immediately with a clear message instead
of silently returning degenerate weights; in-range `lambda` (including the default `0.5` and the
`0.01`–`0.99` grid used elsewhere in this report) is unaffected. Full package test suite re-run with
no regressions (see "Verification" note at the end of this section). Regression tests added:
`tests/testthat/test-util.R` now asserts `expect_error()` for each of the boundary/out-of-range
values.

### 5. `filterAnnual()` did not correctly select the best hosting evaluation for a panel lacking one of its own [FIXED]

Also found during non-TI method validation. A single panel (`INVYR`) is very commonly a constituent
of *more than one* FIA evaluation's multi-panel window — confirmed directly against RI's real
`POP_EVAL` table, not assumed: RI's 2013 evaluation (`EVALID` 441300, `EXPVOL`) covers panels
2009–2013, and its 2014 evaluation (`EVALID` 441400) covers panels 2009–2014, so panel 2009's plot
data is "hosted" by both. `filterAnnual()`'s job, for `method = 'ANNUAL'`, is to pick — for **each
panel** — the single best hosting evaluation to draw that panel's standalone estimate from: the
evaluation whose own nominal year equals the panel's `INVYR` (a "self-hosting" eval, `INVYR == YEAR`)
if one exists, otherwise whichever hosting eval gives it the most plots. The output is then labeled
with the panel's own `INVYR`, since `ANNUAL` reports one row per actually-sampled panel-year, not one
row per evaluation.

The function's `keep` filter compared each candidate only within its own singleton
`(STATECD, INVYR, YEAR, ...)` group (`YEAR` included), making the comparison a no-op that trivially
kept every candidate; the real effect then came from a final `dplyr::first()`-based dedup (described
in the function's own comment as a fallback, not the primary mechanism), which picked among duplicate
same-`INVYR` rows in whatever order they happened to arrive — not by plot count. This is a faithful
reading of the function's own long-standing `TODO` comment: *"It doesn't result in selecting a bad
annual panel, but it may not get the most optimal one."*

**An incorrect first attempt at this fix, and how it was caught**: the first fix inverted the wrong
axis — it compared candidate *panels* competing for the same reporting `YEAR`, instead of candidate
*hosting evaluations* competing for the same panel. That version passed its own (also-incorrect) unit
tests and the full local test suite, but silently discarded valid data: it required a panel to have a
*self-hosting* evaluation to appear in the output at all, so `tpa(fiaRI, method = 'ANNUAL')` against
the full unclipped RI history dropped years 2009–2012 and 2003–2004 entirely, and other estimators'
`ANNUAL` output shrank the same way. This was caught by direct user review before being finalized —
skepticism about *why* an annual estimator would need a full accumulated cycle before reporting a
year prompted re-checking the assumption against RI's actual `POP_EVAL`/`POP_PLOT_STRATUM_ASSGN`
tables (see above), which confirmed panels like 2009 have real, valid hosted data available and
should appear in the output. The corrected version restores this data.

**Root cause**: `R/util.R`'s `filterAnnual()` grouped its candidate comparison by
`(STATECD, INVYR, YEAR, ...)` — i.e. including `YEAR` — which makes every group a singleton (one row
per unique panel/hosting-eval combination) and the `keep` comparison a no-op against itself, in either
direction (comparing panels-for-a-year or evals-for-a-panel).

**Fix**: the comparison group excludes `YEAR` (`group_by(STATECD, INVYR, <user-requested groups>)`),
so candidates are compared across hosting evaluations for a *fixed panel*; the boolean `keep`
expression is written directly and vectorized rather than via a scalar-condition `ifelse()` (which
would otherwise collapse the whole group's `keep` column to length 1); the "does a self-hosting eval
exist" check uses `%in%` rather than `any(... == ...)`, since a small number of incomplete
estimation-unit rows with `YEAR = NA` can appear in the input (an unrelated, pre-existing upstream
join-completeness artifact, also confirmed directly against real RI data) and `==` against `NA`
propagates through `any()`, silently nulling out `keep` for the whole group — this was a second,
narrower bug caught during re-verification of the corrected version. The `mutate(YEAR = INVYR)`
relabel (originally suspected to be the bug) is retained, since it's actually required: it's how a
panel drawn from a non-self-hosting eval ends up correctly labeled with its own year rather than the
hosting eval's year. The final `first()`-across-everything dedup is narrowed to `slice_head(n = 1)`,
now used only to break genuine `nplts` ties between hosting evals rather than as the primary
selection mechanism.

**Verification**: `tpa(fiaRI, method = 'ANNUAL')` against the full unclipped RI toy dataset returns
all 10 years (2009–2018) again — 2013–2018 numerically unchanged (natural, self-hosted panels), and
2009–2012 now drawing from each panel's *optimal* (highest-plot-count) hosting eval rather than the
original code's arbitrary first-encountered one. Against the full local RI validation extract
(2003–2025, un-clipped), all 23 years are present with no gaps or duplicates, including years 2003
and 2004, which have no self-hosting eval and were the case that surfaced the `NA`-propagation bug
above. Full state validation set (RI/NC/CO/OR, `clipFIA(mostRecent = TRUE)`) still runs cleanly,
returning exactly one row each, since clipping to a single evaluation leaves no candidate ambiguity.
`biomass()`, `growMort()`, `vitalRates()`, and `standStruct()` spot-checked under `method = 'ANNUAL'`
on the unclipped `fiaRI` dataset, each also restored to full coverage. `tests/testthat/test-util.R`
rewritten to test the corrected semantics directly: a panel with no self-hosting eval choosing between
two real hosting evals (matching RI's actual eval structure above), self-hosting-eval preference over
a higher-`nplts` non-self-hosting one, estimation-unit-level aggregation feeding the hosting-eval
comparison, an exact-`nplts` tiebreak, and the `NA`-hosting-candidate regression case.

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

## Non-TI method validation (SMA/LMA/EMA/ANNUAL)

EVALIDator has no equivalent for these, so correctness here means: the shared weighting machinery
(`maWeights()`/`filterAnnual()` in `R/util.R`, used by every `sumToEU()`-based estimator) does what
its own math says it does, and `tpa()`'s output behaves sanely and consistently with the
already-validated TI estimates wherever the documentation actually claims a relationship. See
`tests/testthat/test-util.R` for the underlying unit-level checks on `maWeights()`/`filterAnnual()`
themselves (shared by all 15 `sumToEU()`-based estimators, i.e. every estimator except `fsi()`,
which reimplements this branching independently and needs separate treatment); this section covers
only the `tpa()`-level checks built on top of them.

### `maWeights()`/`filterAnnual()` unit-level findings (informing the checks below)

Verified directly against the real `maWeights()`/`filterAnnual()` code (not just its documentation),
using synthetic data shaped to match a real captured call from `tpa(fiaRI, method = 'ANNUAL')`:

- SMA/LMA/EMA weights sum to 1 for every in-range `lambda`, as expected.
- EMA(`lambda` → 1) monotonically approaches SMA's uniform weight; EMA(`lambda` → 0) concentrates
  weight on the most recent panel — both only as **limits**, matching
  `vignettes/alternativeEstimators.Rmd:28`.
- At the *exact* boundaries, `lambda = 1` gave `NaN` weight for every panel, and `lambda = 0` gave
  weight `0` for the oldest panel and `NaN` for every other panel — the opposite of a clean
  "concentrates on the most recent panel" result. No `lambda` range check existed anywhere in the
  package despite the documented `(0,1)` range. **Fixed** — see "Fixed" #4 below.
- `filterAnnual()`'s `keep` filter was a no-op in the case that matters most (multiple candidate
  panels competing to represent one "edge" reporting year with no natural `INVYR == YEAR` panel of
  its own): every candidate survived the filter, and a subsequent `mutate(YEAR = INVYR)` relabeled
  each surviving candidate under its own native INVYR rather than the edge year it was borrowed to
  represent — so the edge year itself never appeared in the output, and the "pick the panel with the
  most plots" behavior described in code comments didn't actually happen. This directly confirmed the
  self-flagged TODO at `R/util.R:1210-1211` ("its not actually filtering things properly"). **Fixed**
  — see "Fixed" #5 below.

### Results

- **EMA(lambda → 1) vs. SMA (RI)**: `|EMA_TPA - SMA_TPA|` shrinks monotonically as lambda increases
  (3.29 → 0.98 → 0.10 → 0.01 for lambda = 0.5/0.9/0.99/0.999) — **pass**, confirms the vignette's
  documented limiting relationship at the `tpa()` output level, not just in the raw weights.
- **TI vs. SMA bounded agreement, 4 states**: panel plot-count CV (`P2POINTCNT_INVYR` across INVYRs
  within the most-recent evaluation, via `handlePops()`) was computed per state to classify
  "balanced" vs. "imbalanced" panels, per the original plan:

  | State | Panel-count CV | TI TPA | SMA TPA | Relative diff |
  |---|---|---|---|---|
  | RI | 0.61 | 347.46 | 356.67 | 2.65% |
  | NC | 1.08 | 712.91 | 742.50 | 4.15% |
  | CO | 0.66 | 481.20 | 471.98 | −1.92% |
  | OR | 1.48 | 347.61 | 363.69 | 4.63% |

  None of these four states have tightly balanced panels (CV ranges 0.6–1.5, no clean "balanced"
  cluster near 0), yet TI and SMA landed within ~5% of each other in every case. Rather than build a
  two-tier balanced/imbalanced tolerance as originally planned, a single flat 10% relative tolerance
  is used for all four states — simpler, and empirically well-justified by this measurement. **Pass**
  in all four states.
- **Totals-vs-per-acre consistency under SMA/LMA/EMA/ANNUAL, 4 states**: `TREE_TOTAL / AREA_TOTAL ==
  TPA` and `BA_TOTAL / AREA_TOTAL == BAA` to `1e-9` tolerance in all 16 state × method combinations —
  **pass**. The totals/per-acre plumbing is not TI-specific.
- **`byPlot = TRUE` + non-TI method (RI, SMA)**: runs cleanly, returns 132 per-plot rows (not a
  population-level estimate), confirming `mergeSmallStrata()`'s `byPlot`-skip gate doesn't break this
  combination — **pass**.
- **Domain filter (`treeDomain = DIA >= 20`, `areaDomain` mesic) + `bySpecies` under each of
  SMA/LMA/EMA/ANNUAL, 4 states**: no errors, no warnings, non-negative `TPA` in all 16 combinations —
  **pass**. Re-runs the same historically-buggy filter/grpBy interaction from the TI validation (Test
  15 above) under every non-TI method.
- **`method = 'EMA'` with default arguments, 4 states**: runs without error in all four — **pass**.
  Closes the gap that the v1.1.1 "error when setting `method = 'EMA'`" fix (`NEWS.md`) previously had
  zero dedicated regression coverage anywhere in the package.

## Findings (reported, not fixed — see bug-handling protocol)

1. **Inconsistent `method`-argument validation across the package** (not specific to non-TI methods,
   but surfaced while reviewing this code path): most `*Starter.R` files warn-and-silently-fall-back
   to TI on an invalid `method` string (e.g. `tpaStarter.R:49-52`), `fsiStarter.R:42` hard-`stop()`s
   instead, and `customPSE.R` has no check at all.
2. **`vegStruct(method = 'SMA')` errors** (`replacement has length zero`) on the bundled `fiaRI`
   toy dataset — found via a spot-check of the other 15 estimators' non-TI paths while verifying the
   `maWeights()`/`filterAnnual()` fixes above didn't introduce regressions. Confirmed pre-existing
   (reproduces identically on the pre-fix code) and unrelated to either fix in this report (`SMA`
   touches neither the `EMA`-only `lambda` validation nor `filterAnnual()`, which only runs for
   `method = 'ANNUAL'`). Out of scope for this report; deferred to the `vegStruct()` non-TI validation
   pass (Phase 2 of the plan).

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population estimate for the TI/default case (only
  totals-vs-per-acre was checked for TI; non-TI + byPlot was checked structurally above but not for
  numeric aggregation-reproduces-population-estimate).
- Finding #2 above (`vegStruct(method = 'SMA')` error) — root cause not yet investigated.
