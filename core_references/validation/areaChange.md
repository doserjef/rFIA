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

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population-level estimate (only a structural sanity
  check was done) — same deferral as `tpa.md`/`area.md`.
- `treeDomain`/`grpBy` interaction numeric validation (the historical v1.1.1 bug pattern) — not
  re-verified with a dedicated EVALIDator-backed test in this pass, since `EXPCHNG` attributes have no
  `TREE` join to filter via `wnum` (same limitation as `area()`'s `treeDomain`, see `area.md`,
  "Notes"), and constructing an internal-consistency check analogous to `area.md`'s (filter has a
  genuine effect + survives `grpBy`) for the *signed change* case is nontrivial (a `treeDomain`
  restriction doesn't have an obviously predictable directional effect on `AREA_CHNG` the way it does
  on `area()`'s always-nonnegative `AREA_TOTAL`). Existing structural tests (1, 3, 5 in
  `test-areaChange.R`) confirm `treeDomain` runs without erroring; a deeper numeric check is left for
  a future pass.
- `method =` options other than `'TI'` (SMA/LMA/EMA/annual) — only exercised structurally (Test 7,
  `method = 'EMA'`), consistent with the broader initiative's plan (EVALIDator only validates the TI
  estimator directly).
