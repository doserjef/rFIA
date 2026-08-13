# Validation report: `customPSE()`

## Methodology

`customPSE()` has no EVALIDator equivalent — it estimates arbitrary user-defined variables from a
tree- or condition-list, not one of FIA's standard published attributes. It was instead validated
**internally**: feeding the tree-/condition-list produced by `tpa()`, `area()`, or `volume()` (via
their `treeList`/`condList = TRUE` argument) back into `customPSE()` should exactly reproduce that
same function's own population-level point estimates, standard errors, and plot counts
(`nPlots_TREE`/`nPlots_AREA` in the source function, `nPlots_x`/`nPlots_y` in `customPSE()`), since
`customPSE()` is meant to be a drop-in generalization of the same post-stratified estimator these
functions already use internally (`sumToPlot()`/`sumToEU()` in `R/util.R` are shared by both code
paths).

rFIA was run against the real, current FIADB extracts cached at `~/Dropbox/data/fia/` (pulled via
`getFIA()`), using `clipFIA(mostRecent = TRUE)`. Four states were used, one per FIA region: **RI**
(Northern), **NC** (Southern), **CO** (Interior West), **OR** (Pacific Northwest) — the same four
states used for the `tpa()` validation (`tpa.md`).

## Results: numeric match

All point estimates and percent standard errors below match the source function (`tpa()`/`volume()`)
to full double precision across all four states, in every case tested. `area()`'s condition-only path
was included as a confirmatory negative control (see "Fixed," below, for why condition-only inputs
were never affected).

### `tpa()` tree-area ratio (TPA, BAA per acre of forest land), default case, 4 states

| State | TPA | BAA | TPA_SE | BAA_SE | nPlots_x (nPlots_TREE) | nPlots_y (nPlots_AREA) |
|---|---|---|---|---|---|---|
| RI | 347.464598 | 116.690516 | 5.940478 | 3.317766 | 129 | 132 |
| NC | 712.908126 | 118.678812 | 1.407179 | 0.830280 | 3455 | 3561 |
| CO | 481.196172 | 91.413432 | 1.907281 | 1.017217 | 3774 | 3925 |
| OR | 347.605501 | 127.288376 | 1.343751 | 0.708292 | 9968 | 10410 |

Exact match to `tpa()`'s own population estimate in every column, all four states.

### `tpa()` tree-area ratio, `treeDomain = DIA >= 20` (large trees), 4 states

A restrictive `treeDomain` produces many forested conditions with zero qualifying trees, which is
the case that most clearly exposed the bug below (see "Fixed").

| State | TPA | nPlots_x (nPlots_TREE) | nPlots_y (nPlots_AREA) |
|---|---|---|---|
| RI | 7.049980 | 55 | 132 |
| NC | 5.432391 | 1284 | 3561 |
| CO | 4.242607 | 1220 | 3925 |
| OR | 11.347223 | 7363 | 10410 |

Exact match, all four states.

### `volume()` tree-area ratio (`BOLE_CF_ACRE`), `treeDomain = DIA >= 20`, 4 states

| State | BOLE_CF_ACRE | SE | nPlots_x (nPlots_TREE) | nPlots_y (nPlots_AREA) |
|---|---|---|---|---|
| RI | 652.473384 | 13.201187 | 55 | 132 |
| NC | 608.434202 | 3.138949 | 1284 | 3561 |
| CO | 153.852051 | 5.144286 | 642 | 3925 |
| OR | 2023.179744 | 1.660960 | 7353 | 10410 |

Exact match, all four states. Confirms the fix (scoped to shared utility code in `R/util.R`) applies
correctly to `volume()`'s treeList as well as `tpa()`'s.

### `area()` condition-only self-ratio, `areaDomain` (mesic classes), 4 states

Used as a negative control: `area()`'s `condList` output never contains a tree join, so it should
never have exhibited the bug below. `x = y` = the same domain-restricted `condList`, i.e. a trivial
ratio of 1.

| State | nPlots_x | nPlots_y | area()'s own `nPlots_AREA_NUM`/`nPlots_AREA_DEN` |
|---|---|---|---|
| RI | 125 | 125 | 125 / 125 |
| NC | 2997 | 2997 | 2997 / 2997 |
| CO | 2121 | 2121 | 2121 / 2121 |
| OR | 8523 | 8523 | 8523 / 8523 |

Exact match in all four states, confirming the condition-only path was never affected and remains
correct after the fix.

### `xGrpBy` grouping (RI): `tpa(bySpecies = TRUE)`

Validates that a domain filter survives `customPSE()`'s own `xGrpBy`/join path rather than being
silently dropped for some groups (the historical `area()`/`areaChange()` bug pattern from v1.1.1),
and that the plot-count fix holds per-group, not just in the ungrouped case. All 45 species present
in `tpa(db_ri, bySpecies = TRUE)` matched exactly (point estimate, SE, and `nPlots_x` vs.
`nPlots_TREE`); a sample:

| SPCD | TPA_RATIO | TPA_RATIO_SE | nPlots_x | tpa() TPA | tpa() TPA_SE | tpa() nPlots_TREE |
|---|---|---|---|---|---|---|
| 126 | 6.49382854 | 70.91357 | 8 | 6.49382854 | 70.91357 | 8 |
| 837 | 8.37983797 | 18.40637 | 46 | 8.37983797 | 18.40637 | 46 |
| 43 | 0.16229333 | 76.85996 | 2 | 0.16229333 | 76.85996 | 2 |
| 802 | 13.70570276 | 16.14575 | 62 | 13.70570276 | 16.14575 | 62 |
| 543 | 0.06271955 | 93.36050 | 1 | 0.06271955 | 93.36050 | 1 |

**Pass.**

### Internal consistency (no comparison function needed)

- `TOTAL / denominator TOTAL` reproduces the ratio estimate exactly, across all four states (mirrors
  the `totals = TRUE` check in `tpa.md`). **Pass.**
- An impossible `treeDomain` (`SPCD == 999`) — every condition's `TREE_BASIS` becomes `NA` — returns
  a clean 0-row result from `customPSE()`, with no warning. **Pass.**

### Multi-state `mostRecent` mask (GitHub issue #47): WV/OH/KY

Investigated at the user's request as a follow-up to a still-open GitHub issue reporting that
`customPSE()` returns separate per-state rows (sometimes with different `YEAR`s) for a spatial mask
spanning multiple states, instead of one combined estimate — unlike `area()`, which combines correctly
for the same mask. Reproduced the reporter's own MRE (a mask spanning WV/OH/KY) against the local
FIADB cache: `area(landType = 'forest', condList = TRUE)`'s output fed into `customPSE()` as a
self-ratio (`x = y` = the same condList).

| | YEAR(s) | NUM_TOTAL per row | nPlots_x per row |
|---|---|---|---|
| Before fix | 2023, 2024 (2 rows) | 353372.6 (KY, 2023) + 1304772.2 (OH/WV, 2024) | 61 (KY) + 234 (OH/WV) |
| After fix | 2024 (1 row) | 1658144.8 | 295 |
| `area()` (same mask) | 2024 (1 row) | `AREA_TOTAL` = 1658144.8 | `nPlots_AREA_NUM` = 295 |

The pre-fix rows' totals/plot-counts sum exactly to the post-fix combined row (353372.6 + 1304772.2 =
1658144.8; 61 + 234 = 295) — confirming the underlying per-estimation-unit arithmetic was always
correct, and the bug was purely that `combineMR()` never ran to unify the rows before the final
`group_by(YEAR, ...)`.

After the fix, `customPSE()`'s single combined row matches `area()`'s own combined `AREA_TOTAL` and
`nPlots_AREA_NUM` for the same mask exactly. **Pass** (post-fix).

## Fixed

### 1. `nPlots_x`/`nPlots_y` inflated to the full area-plot count whenever fed a tree-based treeList [FIXED]

Point estimates and standard errors were **always correct** — this was purely a plot-count/reporting
bug, like `tpa.md`'s "Fixed #1" — but it mattered for the same reason: `customPSE.Rd` explicitly
documents `nPlots_x`/`nPlots_y` as "number of **non-zero** plots used to compute
numerator/denominator estimates," so an inflated count understates the true margin of error for any
`t`-based confidence interval built from a tree-based `x`/`y`.

Reproduced on RI (current cache extract, EVAL_GRP read off `clipFIA(mostRecent = TRUE)`) in the
**default, unrestricted case** — not just as an edge case under a restrictive domain:

```r
tl <- tpa(db_ri, treeList = TRUE)
pop <- tpa(db_ri)   # nPlots_TREE = 129, nPlots_AREA = 132
out <- customPSE(db_ri,
                  x = dplyr::select(tl, -AREA_BASIS), xVars = TPA,
                  y = dplyr::select(tl, -TREE_BASIS), yVars = PROP_FOREST)
# out$nPlots_x was 132 (should have been 129)
```

Confirmed across all four states/regions, both with and without a `treeDomain` restriction (see
tables above): `nPlots_x` always equaled `nPlots_y` (the full forested-plot count), never the true
tree-contributing count reported by `tpa()`/`volume()` as `nPlots_TREE`. The discrepancy grows with
how restrictive `treeDomain` is (e.g. RI: 129 vs. 132 by default, but 55 vs. 132 under
`treeDomain = DIA >= 20`), since a more restrictive filter produces more forested conditions with
zero qualifying trees.

**Root cause**: `tpaStarter.R`'s (and `volumeStarter.R`'s) `treeList = TRUE` output is built as
`a %>% left_join(t, ...)` — every forested condition (`a`) left-joined against its trees (`t`) — by
design, since a treeList must represent every forested condition, including ones with zero qualifying
trees, so a user can still compute area-based ratios from it. Conditions with no qualifying tree get
`TREE_BASIS = NA` in the resulting row. Each Starter function's own *population-estimate* path
(non-treeList) never encounters this: `t` is filtered to `!is.na(TREE_BASIS)` *before* being passed
to the shared `sumToPlot()` — this is precisely the fix already applied for the analogous
`nPlots_AREA`/`CONDID` bug documented in `tpa.md`'s "Fixed #1," which by inspection had already been
applied to the `TREE_BASIS`/`CONDID` filtering in both `tpaStarter.R` and `volumeStarter.R`'s
non-treeList branches (comments in both files reference it directly). `customPSE()`, however, calls
the shared `sumToPlot()`/`sumToEU()` directly on whatever `x`/`y` the user supplies — typically a
`treeList = TRUE` output, which (by necessary design) still contains the `NA`-basis rows.
`sumToEU()`'s `nPlots.x <- length(unique(PLT_CN))` then counted every one of those rows, even though
each contributes exactly 0 to the estimate (the mean/variance/covariance sums in `sumToEU()` are
unaffected by explicit-zero vs. absent rows, which is why point estimates and SEs were never wrong —
verified by comparing `customPSE()` output with the phantom rows present vs. manually stripped first:
identical to full precision in both cases, only `nPlots_x` differed).

Verified `area()`'s `condList` path was never subject to this: `AREA_BASIS` is never `NA` in a
condList (there's no tree join involved), confirmed by an `area()`-only self-ratio check matching
`area()`'s own `nPlots_AREA_NUM` exactly, both before and after the fix.

**Fix**: added `dplyr::filter(!is.na(TREE_BASIS))` / `dplyr::filter(!is.na(AREA_BASIS))` to
`sumToPlot()` (`R/util.R`), immediately after `dtplyr::lazy_dt()`, mirroring the filters each
`*Starter.R` file already applies manually before calling `sumToPlot()`. Since every existing
`*Starter.R` call site already pre-filters its input the same way, this is a no-op for every
population-level estimator (`tpa()`, `area()`, `volume()`, `biomass()`, `carbon()`, `dwm()`,
`diversity()`, `seedling()`, `standStruct()`, `vegStruct()`, `growMort()`, `vitalRates()`) — it only
changes behavior for `customPSE()`'s direct calls on user-supplied tree-/condition-lists, which is
the only call path that could pass an unfiltered `NA`-basis row into `sumToPlot()` in the first place.

**Verification**: after the fix, `nPlots_x`/`nPlots_y` match `tpa()`'s/`volume()`'s own
`nPlots_TREE`/`nPlots_AREA` exactly in all cases tested above (default case, `treeDomain`-restricted
case, `xGrpBy = SPCD` per-species case, all four states). Point estimates and SEs, already correct,
are unchanged (re-verified to full double precision post-fix). Full package test suite re-run with no
regressions. Regression tests added: `tests/testthat/test-customPSE.R`.

### 2. A spatial mask spanning states with different `mostRecent` evaluation years returned separate rows instead of one combined estimate (GitHub issue #47) [FIXED]

Reported by a user (`doserjef/rFIA#47`, still open, no comments) using a polygon mask spanning West
Virginia and Ohio: `customPSE()` returned two rows with different `YEAR`s and different, apparently
per-state, estimates, whereas `area()` on the same masked data correctly returned one combined row.
The reporter noted this mainly affected certain state combinations and wasn't always accompanied by a
`YEAR` mismatch, suggesting the underlying cause wasn't `YEAR` itself.

Reproduced the reporter's own MRE (adapted to the local FIADB cache, mask spanning WV/OH/KY, KY's
`mostRecent` evaluation landing in 2023 vs. OH/WV's 2024 — see "Results," above, for the exact
numbers): `customPSE()` returned 2 rows (2023, 2024); `area()` on the same masked `db` returned 1.

**Root cause**: unlike every other exported estimator, `customPSE()` is not split into a thin
dispatcher + `*Starter.R` pair (see CLAUDE.md, "Architecture pattern") — it's a single function that
does its own remote/in-memory handling inline. It called:

```r
db <- readRemoteHelper(db$states, db, remote, req.tables, nCores = 1)   # pares db down to req.tables
mr  <- checkMR(db, remote = ...)                                        # then checks db for 'mostRecent'
```

`readRemoteHelper()`'s non-remote branch (`R/util.R`) does `db <- db[names(db) %in% reqTables]` —
keeping only the named FIA tables (`PLOT`, `POP_EVAL`, etc.) and silently dropping the `mostRecent`
marker `clipFIA(mostRecent = TRUE)` attaches to the top-level list. `checkMR()` — which just tests
`'mostRecent' %in% names(db)` — therefore always saw `FALSE` for a normal in-memory `FIA.Database`,
regardless of whether the user had actually called `clipFIA(mostRecent = TRUE)`. Confirmed directly:
`checkMR()` on the original `db` returned `TRUE`; on the same `db` after the `readRemoteHelper()`
subset (i.e. as `customPSE()` itself actually sees it), `FALSE`. Since `mr` was always `FALSE`,
`combineMR()` — which relabels every row's `YEAR` to the max across states before the final
`group_by(YEAR, ...)`, so that states with different "most recent" years get unified into one row —
never ran. Every other estimator dispatcher (e.g. `R/tpa.R`: `mr <- checkMR(db, remote)`) computes
`mr` on the original, unsubsetted `db`, before any table-paring happens, which is why `area()` and
every other function don't exhibit this. This also explains why the reporter observed it "mainly" for
certain state combinations and not always tied to a `YEAR` difference: any multi-state call needed a
`combineMR()`-driven merge for a reason other than a same-labeled `YEAR` (e.g. distinct `ESTN_UNIT_CN`
values that otherwise legitimately collapse together downstream) would also have silently failed to
merge, not just the `YEAR`-mismatch case that happened to be visible in the reporter's own MRE.

**Fix**: reordered the two lines in `R/customPSE.R` so `mr <- checkMR(db, remote)` runs on the
original `db` before `readRemoteHelper()` pares it down, matching the pattern every other dispatcher
already uses. No change to any estimation math — `handlePops()` and everything downstream still
receives the exact same (post-`readRemoteHelper()`) `db` as before; only the value of `mr` itself
changes (from always-`FALSE` to correctly reflecting whether `clipFIA(mostRecent = TRUE)` was used).

**Verification**: after the fix, the WV/OH/KY reproduction above returns a single row
(`YEAR = 2024`, the max across the two states) whose `NUM_TOTAL`/`nPlots_x` exactly equal the sum of
the two pre-fix rows, and exactly match `area()`'s own combined `AREA_TOTAL`/`nPlots_AREA_NUM` for the
same mask. Regression test added: `tests/testthat/test-customPSE.R`.

## Notes

- `customPSE()`'s ratio-variance formula (inline in `R/customPSE.R`) is structurally identical to the
  shared `ratioVar()` helper (`R/util.R`) that `tpa()`/`volume()`/etc. call — same
  `(1/y^2)(x.var + (x/y)^2 y.var - 2(x/y) cv)` formula — so no formula-level discrepancy was expected
  or found between `customPSE()`'s ratio SE and a source function's own SE.
- Column-name collisions are a real footgun when both `x` and `y` share a variable name (e.g. an
  area-area self-ratio using `PROP_FOREST` on both sides) — `xVars = c(NUM = PROP_FOREST)` (renaming)
  is required in that case, per the existing renaming example in `?customPSE`. Not a bug, just a
  usage note surfaced while writing the area() cross-check above.

## Deferred to follow-up (not covered this pass)

- `biomass()`/`carbon()`/`diversity()` treeList outputs share the same `a %>% left_join(t, ...)`
  construction pattern (confirmed by inspection of the relevant `*Starter.R` files) and are therefore
  presumably subject to the same fix, but were not empirically cross-checked against `customPSE()`
  here (only `tpa()`, `area()`, and `volume()` were, per the scope of this validation pass).
- `method` options other than `'TI'` were not exercised (mirrors the same deferral in `tpa.md`).
- `yTransform`/`xTransform` were not exercised.
