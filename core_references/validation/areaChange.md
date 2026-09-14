# Validation report: `areaChange()`

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint (the programmatic interface
behind the EVALIDator web tool), queried live via `fetch_evalidator.R` in this directory. rFIA was
run against the real, current FIADB extracts already cached at `~/Dropbox/data/fia/` (pulled via
`getFIA()`), using `clipFIA(mostRecent = TRUE)` to match EVALIDator's "current" remeasurement
evaluation. Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO**
(Interior West), **OR** (Pacific Northwest) — the same four states used for `tpa()`/`area()`.

`tests/testthat/test-areaChange.R` calls the FIADB-API live at test time rather than hard-coding the
reference numbers below — this section of the report is illustrative, not a source of truth the
tests are pinned to. The EVAL_GRP code for each state is read directly off
`clipFIA(..., mostRecent = TRUE)$POP_EVAL_GRP$EVAL_GRP`, never hard-coded (this happens to be the
same `EVAL_GRP` code used for `area()`'s `EXPCURR` evaluation — each state has one `POP_EVAL_GRP` per
year, shared across `EVAL_TYP`s). Tests are skipped (not failed) when the local data cache or network
access to `apps.fs.usda.gov` is unavailable.

**`areaChange()`'s EVALIDator ground truth works differently than `area()`/`tpa()`'s.** The
`EXPCHNG`-tagged attributes in `EVALIDATOR_POP_ESTIMATE.csv` (126-139) are *not* signed net-change
deltas. Reading their actual `VBA_SUMFROMWHERE` SQL (via the CSV's `SQL_SUM`/`SQL_WHERE` fields)
shows each is a plain, unsigned `SUM()` over `SUBP_COND_CHNG_MTRX.SUBPTYP_PROP_CHNG` proportions,
filtered by whether the remeasured condition pair was forest/timberland at **both** measurements
(attributes 127/129) or **either** measurement (128/130) — i.e. these are base-population *area*
totals for specific change categories, not deltas. Confirmed the `135`-`139` "average annual"
variants are simply `126`-`130` divided by the average `REMPER` (each pair's ratio is a consistent
~6.6 across attributes, and the SQL has no subtraction), so there is **no direct EVALIDator attribute
for rFIA's signed `AREA_CHNG`/`PERC_CHNG`** — a permanent limitation of this API path, analogous to
the `wnum`-can't-filter-non-`TREE`-joined-attributes limitation already documented in `area.md`.

Given that, validation here uses two complementary strategies:
1. **Direct EVALIDator match** on `areaChange(chngType = 'component')`'s `PREV_AREA` column, which
   *is* a plain area total (not a delta) for each `STATUS1`/`STATUS2` category — attribute 127/129
   ("both") matches the `STATUS1 == STATUS2` (no-change) row exactly, and summing `PREV_AREA` across
   *all three* categories (no-change + diversion + reversion) matches attribute 128/130 ("either")
   exactly.
2. **Internal consistency** for the signed `AREA_CHNG`/`PERC_CHNG` values themselves: `chngType =
   'net'`'s `AREA_CHNG` must equal `chngType = 'component'`'s reversion `AREA_CHNG` minus its
   diversion `AREA_CHNG` (the definition given in `man/areaChange.Rd`'s "Estimation Details").

## Results: numeric match

### `chngType = 'component'`, `landType = 'forest'`/`'timber'`, 4 states

| State | attr 127/129 ("both") | rFIA `PREV_AREA` | attr 128/130 ("either") | rFIA `sum(PREV_AREA)` |
|---|---|---|---|---|
| RI, forest | 370426.0 | 370426.0 | 384741.9 | 384741.9 |
| RI, timber | 348115.5 | 348115.5 | 362431.4 | 362431.4 |
| NC, forest | 18196004.1 | 18196004.1 | 19023084.3 | 19023084.3 |
| NC, timber | 17456814.0 | 17456814.0 | 18367506.3 | 18367506.3 |
| CO, forest | 22058098.2 | 22058098.2 | 23013968.0 | 23013968.0 |
| CO, timber | 9623592.8 | 9623592.8 | 11006043.5 | 11006043.5 |
| OR, forest | 28863113.1 | 28863113.1 | 30518505.9 | 30518505.9 |
| OR, timber | 23251660.7 | 23251660.7 | 24242102.9 | 24242102.9 |

All 16 comparisons: **exact match** (to the displayed precision), including `PREV_AREA_SE` and
`nPlots_AREA` against `sePercent`/`plotCount` for the "both" category.

### Internal consistency: net `AREA_CHNG` = reversion − diversion, 4 states × 2 landTypes

| State | landType | net `AREA_CHNG` | reversion − diversion |
|---|---|---|---|
| RI | forest | 75.4074 | 75.4074 |
| RI | timber | 75.4074 | 75.4074 |
| NC | forest | -38465.8547 | -38465.8547 |
| NC | timber | -53093.9640 | -53093.9640 |
| CO | forest | 16887.1597 | 16887.1597 |
| CO | timber | -29778.4376 | -29778.4376 |
| OR | forest | 9380.4424 | 9380.4424 |
| OR | timber | 14552.6323 | 14552.6323 |

All 8: **exact match**. (RI's forest and timber values happen to coincide — RI's specific
diversion/reversion plot set has no reserved/low-site-class forest among it, so `landType =
'forest'` vs `'timber'` filtering has no effect on *those particular* transitioning plots even though
it does on the much larger "stayed forest"/"stayed timber" population.)

### `returnSpatial`/`polys` consistency (RI, by county)

`returnSpatial = TRUE` vs `FALSE` (both with `polys = countiesRI`, `landType = 'forest'`,
`chngType = 'net'`): all non-geometry columns match exactly. **Pass.** (The `countiesRI`
spatial-join plot-matching shortfall already documented in `area.md` — traced to that dataset's
coarse polygon geometry, not to any estimator logic — applies identically here; not re-litigated in
this report.)

### Empty-domain edge case

`areaChange(treeDomain = SPCD == 999)` (matches no trees) returns a clean 0-row tibble with no
warning, confirming the shared `combineMR()` fix (`tpa.md`, "Fixed" #2) applies correctly to
`areaChange()` too.

## Fixed

### 1. Nonsampled conditions misclassified as genuine forest ↔ non-forest change events [FIXED]

`R/util.R::landTypeDomain()` defines `landType = 'forest'` as `COND_STATUS_CD == 1`; anything else —
including `COND_STATUS_CD == 5` ("nonsampled": hazardous, denied access, etc., which is not a real
land classification) — falls into the "not forest" bucket (`landD = 0`). For a single-point-in-time
estimate (`area()`) this is mostly harmless: a nonsampled condition simply contributes no area either
way. For `areaChange()`, which classifies each remeasured condition pair by its `landD` value at
*both* time points, this is far more consequential: a condition that goes from forest to nonsampled
(or nonsampled to forest) between measurements was being classified as a genuine `Forest →
Non-forest` diversion (or `Non-forest → Forest` reversion) event — fabricating land-use change that
never actually happened, since "nonsampled" only means the condition wasn't reliably observed, not
that it became non-forest.

EVALIDator's own SQL for every `EXPCHNG` area-change attribute explicitly excludes any remeasurement
pair where either side is nonsampled (`COALESCE(COND.COND_NONSAMPLE_REASN_CD, 0) = 0` on both the
current and previous `COND` row). rFIA's `R/areaChangeStarter.R` had no equivalent exclusion.

**Reproduced empirically** (RI, before fix, via direct query of the raw FIA tables): 18
`SUBP_COND_CHNG_MTRX` rows across 10 distinct plots had `COND_STATUS_CD == 5` on one side of a
remeasurement pair — a large contamination of a diversion+reversion population that totaled only
~24 plot-categories. Comparing `areaChange(landType = 'forest', chngType = 'component')`'s three
categories against EVALIDator's attributes 127/128 confirmed the effect: the `Forest → Forest`
("both", uncontaminated) category matched attribute 127 exactly (370426.0 acres, 108 plots), but
summing all three categories gave 397312.5 acres across a union of 132 plot-appearances — larger than
EVALIDator's "either" population (attribute 128: 384741.9 acres, 113 plots).

**Real-world impact confirmed** (RI, `landType = 'forest'`, default `chngType = 'net'`): the sign of
the reported estimate flipped — from `AREA_CHNG = -426` acres/year (apparent net forest loss,
pre-fix) to `AREA_CHNG = +75` acres/year (apparent net forest gain, post-fix). Both are statistically
insignificant individually (`AREA_CHNG_SE` > 100% either way, small-state small-sample noise), but
the point estimate a user would see was materially different, not a rounding change.

**Fix**: `R/areaChangeStarter.R`'s previous-condition `COND` selection (feeding the join that builds
`landD1`/`aD1`/etc.) didn't carry `COND_STATUS_CD` through at all — only the current-condition
selection did. Added `COND_STATUS_CD` to that selection (so `data` carries both `COND_STATUS_CD1`
and `COND_STATUS_CD2`, matching the existing `landD1`/`landD2` naming convention), then added
`dplyr::filter(!(COND_STATUS_CD1 %in% 5 | COND_STATUS_CD2 %in% 5))` immediately after the full
condition-list join, before any `chngType`-specific logic runs. This is unconditional — applied for
every `landType`/`chngType` combination, not just `'forest'` — since it excludes invalid transition
rows from the population entirely, mirroring what EVALIDator does at the SQL level, rather than
changing how any specific `landType` is classified. (`byLandType = TRUE` already incidentally
excluded nonsampled conditions via its own `NA`-drop of `db$COND$landType`, applied before these
joins run — this fix makes that same exclusion happen unconditionally, not just for
`byLandType = TRUE`.)

**Verification**: after the fix, summing `landType = 'forest'`'s three component categories'
`PREV_AREA` gives 384741.916 (RI) — exact match to attribute 128's 384741.9 — with the union
`nPlots_AREA` now exactly 113, matching EVALIDator's `plotCount` (previously 114, using the inflated
union). The `Forest → Forest` category is unaffected (as expected — no nonsampled contamination
there to begin with). Confirmed the same exact-match pattern for `landType = 'timber'` (vs. attributes
129/130) and in a second, much larger/more complex state (NC) — see tables above. Net `AREA_CHNG`
continues to equal reversion − diversion exactly, in all four states and both land types. Full
package test suite re-run with no regressions (`test_full.log`: 330 pass, 0 fail).

## Notes

### No direct EVALIDator attribute for signed net area change

See Methodology above. `AREA_CHNG`/`PERC_CHNG` (rFIA's headline output) has no single matching
EVALIDator attribute via the FIADB-API `fullreport` endpoint — the `EXPCHNG`-tagged attributes are
unsigned base-population area totals, not deltas. Validation instead relies on (a) `PREV_AREA`
matching those base-population totals exactly, and (b) internal consistency between the `net` and
`component` `chngType`s, which is exactly the invariant that the bug fixed in this pass would have
broken had it affected the signed values asymmetrically (it happens to cancel out in `net` mode
today, but would not necessarily for a different pattern of nonsampled contamination — this is why
the direct `PREV_AREA` match in (a) is the more load-bearing check).

### Shared `landTypeDomain()`/`udAreaDomain()` are unaffected by this fix

This fix lives entirely in `R/areaChangeStarter.R` (the new `COND_STATUS_CD1`/`COND_STATUS_CD2`
columns and the exclusion filter), not in the shared `R/util.R` utilities also used by `area()`. The
already-validated `area()` behavior (`area.md`) is untouched by this change.

## Non-TI method validation (SMA/LMA/EMA/ANNUAL)

EVALIDator has no equivalent for these, so correctness here means: the shared weighting machinery
(`maWeights()`/`filterAnnual()`/`combineMR()` in `R/util.R`, used by every `sumToEU()`-based
estimator) does what its own math says it does, and `areaChange()`'s output behaves sanely and
consistently with the already-validated TI estimates wherever the documentation actually claims a
relationship. See `tests/testthat/test-util.R` for the underlying unit-level checks on this shared
machinery, and `tpa.md` (the template this section follows) for the full non-TI methodology, the two
invariants that must not be over-asserted, and `tpa.md`'s "Fixed" #6 for a package-wide
`combineMR()`/`ANNUAL` bug found and fixed during `area()`'s non-TI pass — confirmed below to have
affected `areaChange()` too (it shares the same `combineMR()` call in `R/areaChange.R`), but already
fixed at the shared-utility level before this section was written, so no new bug was found here.

`PREV_AREA` (a plain, nonnegative area total, the same role `AREA_TOTAL` plays for `area()`) is used
for the EMA/SMA convergence and TI-vs-SMA bounded-agreement checks below, not the signed `AREA_CHNG`:
confirmed empirically that `AREA_CHNG` is driven by a small subpopulation of transitioning plots and
can differ substantially in relative terms between methods even when `PREV_AREA` agrees closely (RI:
TI vs. SMA `AREA_CHNG` differ by ~49% relatively — `538.9` vs. `801.6` — vs. ~6.5% for `PREV_AREA`), so
a relative-tolerance bound on `AREA_CHNG` itself is not meaningful. This is consistent with this
report's own pre-existing deferral (below) of a numeric `treeDomain` effect check on `AREA_CHNG` for
the same reason.

### A real bug found during `area()`'s pass, confirmed to affect `areaChange()` too (already fixed)

`method = 'ANNUAL'` on this report's 4-state `clipFIA(mostRecent = TRUE)` harness was, before the fix
described in `tpa.md` "Fixed" #6, affected by the same `combineMR()` bug found while validating
`area()`: every constituent panel's estimate was silently pooled into a single mislabeled row instead
of one row per real sampled panel. `R/areaChange.R` calls `combineMR()` at the same unconditional
`if (mr) { tEst <- combineMR(tEst); aEst <- combineMR(aEst) }` site as every other estimator dispatcher
(not independently verified with a full before/after repro the way `area()`/`tpa()` were, since the fix
was already applied package-wide by the time this section was written — instead verified directly that
the *fixed* behavior is correct, below). No `areaChange()`-specific fix was needed.

### Results

- **EMA(lambda → 1) vs. SMA (RI), `PREV_AREA`**: `|EMA_PREV_AREA - SMA_PREV_AREA|` shrinks
  monotonically as lambda increases (14228 → 1732 → 150 → 15 for lambda = 0.5/0.9/0.99/0.999) —
  **pass**, confirms the same limiting relationship already established in `tpa.md`/`area.md` holds at
  the `areaChange()` output level too.
- **TI vs. SMA bounded agreement, 4 states, `PREV_AREA`**: reusing the flat 10% relative tolerance
  established empirically in `tpa.md` (a property of each state's panel structure, not the estimator):

  | State | TI PREV_AREA | SMA PREV_AREA | Relative diff |
  |---|---|---|---|
  | RI | 376692.8 | 352238.0 | −6.49% |
  | NC | 18720574.6 | 18929685.6 | 1.12% |
  | CO | 22449461.8 | 22062854.3 | −1.72% |
  | OR | 29640989.1 | 28943907.8 | −2.35% |

  All four states land within the 10% bound — **pass** in all four.
- **Internal identity net `AREA_CHNG` = reversion − diversion, under SMA/LMA/EMA/ANNUAL, 4 states ×
  2 landTypes**: re-running this report's central identity (see "Results" above, TI-only) under every
  non-TI method, checked *per YEAR* for `ANNUAL`. A year's diversion or reversion category can be
  legitimately absent from the `component` breakdown (zero qualifying plots that year — confirmed on
  RI, where several individual annual panels have, e.g., a diversion event but no reversion event
  that year, or vice versa) and is treated as `AREA_CHNG = 0` for that category rather than requiring
  both rows to be present. **Pass** in all 32 state × landType × method combinations (holds to floating
  point precision in every case, matching the exact-match precedent from the TI-only version of this
  check).
- **`byPlot = TRUE` + non-TI method (RI, SMA)**: runs cleanly, returns 212 per-plot rows (not a
  population-level estimate) with the documented `PROP_CHNG`/`PREV_PROP_FOREST` columns present,
  confirming `mergeSmallStrata()`'s `byPlot`-skip gate doesn't break this combination for
  `areaChange()` either — **pass**.
- **`treeDomain` + `grpBy` interaction under each of SMA/LMA/EMA/ANNUAL, 4 states**: specifying
  `treeDomain` expands the output with `TREE_DOMAIN1`/`TREE_DOMAIN2` indicator columns (whether the
  domain was satisfied at each measurement), so unlike `area()`'s single-row-per-group case, the
  "genuine restriction" and "`grpBy` preserves the total" checks sum `PREV_AREA` across *all* rows
  (every `TREE_DOMAIN1`/`TREE_DOMAIN2` × `STATUS1`/`STATUS2` combination) per year, not one
  `STATUS1`/`STATUS2` subset. With that correction, the filter still restricts the total (`filtered <
  base` for every year) and `grpBy = OWNGRPCD` does not silently drop it for any group (summing across
  groups reproduces the filtered total exactly, per year) — **pass** in all 16 state × method
  combinations. (An initial version of this check, comparing only one `STATUS1 == STATUS2 == 'Forest'`
  subset without accounting for the `TREE_DOMAIN1`/`TREE_DOMAIN2` expansion, spuriously appeared to
  fail — this was a test-construction mistake, not a package bug, caught and corrected before finalizing
  this report.)
- **`method = 'EMA'` with default arguments, 4 states**: runs without error in all four — **pass**,
  same v1.1.1 regression coverage as `tpa.md`/`area.md`.
- **`method = 'ANNUAL'` with default arguments, 4 states**: runs without error and returns multiple
  distinct-year rows (not pooled) in all four — **pass**. New regression coverage for the
  `combineMR()`/`ANNUAL` bug described above.

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population-level estimate (only a structural sanity
  check was done) — same deferral as `tpa.md`/`area.md`.
- `treeDomain`/`grpBy` interaction numeric validation against EVALIDator directly (as opposed to the
  internal-consistency check now covered above and under non-TI methods) — not re-verified with a
  dedicated EVALIDator-backed test in this pass, since `EXPCHNG` attributes have no `TREE` join to
  filter via `wnum` (same limitation as `area()`'s `treeDomain`, see `area.md`, "Notes"). Existing
  structural tests (1, 3, 5 in `test-areaChange.R`) confirm `treeDomain` runs without erroring; a
  deeper EVALIDator-backed numeric check is left for a future pass.
