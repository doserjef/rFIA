# Validation report: `dwm()`

## Scope

This pass covers `dwm()` -- down woody material: fine woody debris (1hr/10hr/100hr), coarse woody
debris (1000hr), slash piles, litter, and duff, in cubic-foot volume (woody fuel types only),
dry-short-ton biomass, and short-ton carbon. `dwm()` is structurally distinct from every estimator
validated so far: it's condition-based (no `TREE` table, no `treeDomain`/`treeType`), and it draws on
`COND_DWM_CALC`, a table not yet touched by any prior validation pass.

## Methodology

Same approach as `tpa()`/`area()`/`biomass()`/`carbon()`/`volume()`: ground truth from the FIADB-API
`fullreport` endpoint via `fetch_evalidator.R`, run against real FIADB extracts at
`~/Dropbox/data/fia/` with `clipFIA(mostRecent = TRUE)`, across RI (Northern), NC (Southern), CO
(Interior West), and OR (Pacific Northwest). `tests/testthat/test-dwm.R` calls the FIADB-API live at
test time rather than hard-coding reference numbers.

DWM is a **phase 3 (P3) measurement**, collected on only a subset of FIA plots (not every forested
condition) -- so plot counts here are much smaller than `tpa()`/`biomass()`/`volume()`'s for the same
state (e.g. RI has only 5-6 DWM-sampled plots total).

EVALIDator has no timberland-specific or `areaDomain`-agnostic DWM attributes beyond a handful of
forest-land totals/fuel-type breakdowns, and no duff-specific attribute at all -- so `landType =
'timber'` was checked only for internal (non-EVALIDator) plot-count consistency, and `areaDomain` was
checked against the forest-land total attribute with a `strFilter` restriction.

## Results: numeric match

All point estimates and percent standard errors below match the FIADB-API to full double precision
after the fixes described below.

### Core default case, totaled across fuel types (`landType = 'forest'`), 4 states

| State | VOL_ACRE | VOL_ACRE_SE | nPlots_DWM | nPlots_AREA |
|---|---|---|---|---|
| RI | 968.1802 | 33.30714 | 5 | 6 |
| NC | 665.3382 | 34.52443 | 208 | 210 |
| CO | 835.7931 | 2.21021 | 3897 | 3919 |
| OR | 1793.515 | 2.421219 | 10273 | 10397 |

All four: **exact match** against EVALIDator attribute 123 (total volume of DWM: FWD + CWD + piles,
forest land), ratio'd against attribute 2 (forest land area).

### `byFuelType = TRUE` fuel-type-specific variants, 4 states

| Case | EVALIDator attr | RI | NC | CO | OR |
|---|---|---|---|---|---|
| CWD (`1000HR`): volume/biomass/carbon | 114/115/116 | exact | exact | exact | exact |
| FWD small (`1HR`): volume | 104 | exact | exact | exact | exact |

Both checked with `nPlots_DWM` matching exactly, per fuel type -- only possible because the
zero-value qualifying filter (see "Fixed" below) is applied per fuel-type row, not just to the
combined total.

### `areaDomain` filter interaction (mesic physiographic classes), 4 states

Matched EVALIDator attribute 123 with a `strFilter` restriction, including `nPlots_DWM` and
`nPlots_AREA`, exactly across all four states -- the primary regression check for both fixes below
(neither plot count shrinks with `landType`/`areaDomain` restrictions without them).

### `landType = 'timber'` (internal consistency only -- no EVALIDator attribute exists)

`nPlots_AREA` and `nPlots_DWM` never exceed the corresponding forest-land count, across all four
states. **Pass.**

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `VOL_TOTAL`/`BIO_TOTAL`/`CARB_TOTAL` divided by `AREA_TOTAL` reproduce
  `VOL_ACRE`/`BIO_ACRE`/`CARB_ACRE` exactly, across all four states. **Pass.**

### `returnSpatial` (RI, by county)

`dwm(polys = countiesRI, returnSpatial = TRUE)` vs. `returnSpatial = FALSE`: all non-geometry
columns match exactly. **Pass.**

### Empty-domain edge case

`dwm(areaDomain = STATECD == 999)` returns a clean 0-row tibble with no warning. **Pass.**

## Fixed

Three bugs were found and fixed this pass, all in `dwmStarter.R`. `VOL_ACRE`/`BIO_ACRE`/`CARB_ACRE`
and their SEs were unaffected by any of them -- all three were caught by checking `nPlots_DWM`/
`nPlots_AREA` directly, which the pre-existing test suite never did.

**1. `nPlots_AREA` phantom-row bug.** Same class of bug already fixed in `tpa()`/`area()`/`carbon()`/
`biomass()`/`volume()`: `dwmStarter.R`'s condition list (`a`) was missing the `!is.na(CONDID)` guard
present in every other estimator's equivalent code. Fixed identically -- confirmed via `areaDomain`
across all four states.

**2. `COND_DWM_CALC` multi-EVALID duplication -- much more severe than #1.** `COND_DWM_CALC` is
denormalized: a single `(PLT_CN, CONDID)` can legitimately appear as *multiple rows*, one per
`EVALID`, because consecutive annual panels can each report the same not-yet-remeasured plot as
their current DWM data (confirmed directly: one Colorado plot/condition had 4 rows, for EVALIDs
82107/82007/82307/81907, with `pops` -- the current evaluation being estimated -- specifying only
82307 as relevant). `dwmStarter.R` filtered `COND_DWM_CALC` by `PLT_CN %in% pops$PLT_CN` alone,
*after* dropping the `EVALID` column, so every EVALID's copy of every condition survived. This
inflated `nPlots_DWM` by ~4-5x (Colorado core case: 17775 reported vs. 3897 actual) without changing
the point estimate/SE at all -- the duplicate rows' `STRATUM_CN` values didn't match the current
evaluation's population table, so their contribution was `NA` and dropped via `na.rm = TRUE`
downstream, but the phantom rows still inflated the reported plot count and created spurious
zero-area estimation-unit groups.

  **Fix**: `dplyr::semi_join(dplyr::select(pops, PLT_CN, EVALID), by = c('PLT_CN', 'EVALID'))`
  applied to `db$COND_DWM_CALC` *before* dropping the `EVALID` column, restricting to only the row(s)
  relevant to the evaluation actually being estimated.

**3. `nPlots_DWM` inflation from all-zero-volume plots -- the same class of fix just made in
`volume()` (`bcf > 0`), applied twice here for two different reported quantities:**

  - **Combined total (`byFuelType = FALSE`)**: a domain-qualifying, DWM-sampled plot can have
    exactly zero down woody material of every kind (no FWD, no CWD, no piles) -- confirmed directly:
    Colorado has exactly 22 such plots, matching a residual `nPlots_DWM` gap of 3919 vs. 3897 (after
    fix #2, before this one). EVALIDator's "Total volume of DWM" attribute requires this sum to be
    strictly positive.
  - **Per-fuel-type (`byFuelType = TRUE`)**: a plot's *specific* fuel type can independently be zero
    even when its total isn't (e.g. no CWD present, but FWD and litter both are) -- EVALIDator's
    per-fuel-type attributes each require their own column to be positive, not the combined total.

  **Fix**: a per-fuel-type-row filter (`VOL > 0` for the five woody fuel types; `BIO > 0` for
  `DUFF`/`LITTER`, which have no volume column at all -- confirmed `VOL`/`BIO` are co-zero/co-positive
  for the woody types in all but 8 of 141,592 `CWD` rows checked nationally, but `VOL` is used
  directly there rather than by approximation), applied right after the wide-to-long fuel-type pivot;
  plus a second, separate filter on the re-collapsed total (`VOL > 0` again, after summing back
  across fuel types) for the `byFuelType = FALSE` path specifically, since a plot can survive the
  per-row filter via `DUFF`/`LITTER` alone (nonzero biomass, zero woody volume) but EVALIDator's
  combined-total attribute explicitly excludes duff/litter from its definition. Verifying both paths
  independently (rather than assuming one filter would serve both) is what caught this: an early,
  single-filter version of the fix matched EVALIDator's per-fuel-type attributes exactly but silently
  broke the combined-total case by ~0.5% in Colorado/Oregon before this second filter was added.

## Notes

### Why this pass found three bugs where `biomass()`/`carbon()` found fewer

`dwm()` shares `landTypeDomain()`/`udAreaDomain()`/`sumToPlot()`/`sumToEU()` with every other
estimator (hence bug #1, the same recurring gap), but its use of `COND_DWM_CALC` -- a table no other
estimator reads -- is unique to `dwm()`, and its own filtering logic (not inherited from any shared
utility) had never been exercised by a numeric test before. This meant `dwm()`'s condition-specific
code carried its own, previously-undetected bugs (#2 and #3) in exactly the way the validation plan's
methodology is designed to surface: `VOL_ACRE`/`BIO_ACRE`/`CARB_ACRE` matched EVALIDator throughout,
so a structural/point-estimate-only test suite (the pre-existing `test-dwm.R`) would never have
caught any of these.

## Non-TI method validation (SMA/LMA/EMA/ANNUAL)

EVALIDator has no equivalent for these, so correctness here means: the shared weighting machinery
(`maWeights()`/`filterAnnual()`/`combineMR()` in `R/util.R`, used by every `sumToEU()`-based
estimator) does what its own math says it does, and `dwm()`'s output behaves sanely and consistently
with the already-validated TI estimates wherever the documentation actually claims a relationship. See
`tests/testthat/test-util.R` for the underlying unit-level checks on this shared machinery, and
`tpa.md` (the template this section follows) for the full non-TI methodology. `tpa.md`'s "Fixed" #6
documents a package-wide `combineMR()`/`ANNUAL` bug found and fixed during `area()`'s non-TI pass;
`dwm()` shares the same call site and was already covered by that fix before this section was written.

**RI is excluded from every population-estimation check in this section**, a departure from every
other function's 4-state pattern. Confirmed directly, independent of anything in this validation
pass: RI's most-recent `EXPDWM` evaluation (`EVALID` 442507, nominal year 2025) currently has **zero**
matching rows in the locally cached `COND_DWM_CALC` extract -- the most recent `EVALID` that table
actually has data for is 442407 (2024). This is FIA's real-world phase-3 (DWM) data publication
lagging behind the core `EXPCURR` evaluation cycle, not a package bug: `dwm(db_ri, method = 'TI')`
itself already returns a clean, correct 0-row result under the current data cache, and the
*pre-existing, unmodified* EVALIDator-comparison tests above (Tests 9-12) already fail for RI against
today's live cache for the identical reason -- this predates and is unrelated to this pass.
`byPlot = TRUE` output is unaffected (it doesn't depend on the current population-estimation eval the
same way) and still uses RI, matching every other function's pattern.

**A flat relative-tolerance TI-vs-SMA bound (used for every other function) is not meaningful for
`dwm()`** and was not applied. Confirmed empirically: NC's TI vs. SMA `VOL_ACRE` differ by ~127%
(665.3 vs. 1508.2), which looks alarming in isolation but is fully explained by both estimates' own
enormous sampling error (`VOL_ACRE_SE` = 34.5% for TI, 67.7% for SMA) -- the two point estimates are
well within a couple of standard errors of each other. This is down woody material's inherently
clumpy spatial distribution (a handful of logs or slash piles can dominate a small phase-3 sample)
interacting with `dwm()`'s already-small sample sizes, not a bug -- the same class of deferral
`areaChange.md` made for its similarly noisy `AREA_CHNG` metric. Only a finite/non-negative sanity
check was used instead.

### Results

- **EMA(lambda → 1) vs. SMA (NC)**: `|EMA_VOL_ACRE - SMA_VOL_ACRE|` shrinks monotonically as lambda
  increases, and by more than 99% from lambda = 0.5 to lambda = 0.999 (1004.6 → 345.8 → 36.5 → 3.7) —
  **pass**, confirming the same limiting relationship already established in `tpa.md` holds at the
  `dwm()` output level too, once accounting for `VOL_ACRE`'s larger absolute scale (a relative rather
  than absolute final-distance bound was used for this reason).
- **TI and SMA both finite/non-negative, 3 states (NC/CO/OR)**: **pass** in all three (see above for
  why a numeric bound was not applied).
- **Totals-vs-per-acre consistency under SMA/LMA/EMA/ANNUAL, 3 states, all three metrics**:
  `VOL_TOTAL`/`BIO_TOTAL`/`CARB_TOTAL` divided by `AREA_TOTAL` reproduce
  `VOL_ACRE`/`BIO_ACRE`/`CARB_ACRE` to `1e-9` tolerance in all 12 state × method combinations —
  **pass**.
- **`byPlot = TRUE` + non-TI method (RI, SMA)**: runs cleanly, returns 42 per-plot rows (not a
  population-level estimate), confirming `mergeSmallStrata()`'s `byPlot`-skip gate doesn't break this
  combination — **pass**. RI's byPlot data is real and usable despite the population-estimation gap
  noted above.
- **`areaDomain` (mesic physiographic classes) + `grpBy = OWNGRPCD` under each of SMA/LMA/EMA/ANNUAL,
  3 states**: the filter restricts (or exactly reproduces, in a legitimate edge case per `carbon.md`'s
  precedent) the unfiltered total for every year, and `grpBy` does not silently drop it for any group
  — **pass** in all 12 state × method combinations.
- **`method = 'EMA'` with default arguments, 3 states**: runs without error in all three — **pass**,
  same v1.1.1 regression coverage as `tpa.md`.
- **`method = 'ANNUAL'` with default arguments, 3 states**: runs without error and returns multiple
  distinct-year rows (not pooled) in all three — **pass**. Confirms `dwm()` is unaffected by the
  `combineMR()` bug described in `tpa.md`.

## Findings (reported, not fixed — see bug-handling protocol)

1. **`method = 'ANNUAL'` emits an unguarded `max()`-on-empty-group warning when a state has zero
   population-estimation data for the requested attribute.** Reproduced on RI specifically, as a
   direct consequence of the data-availability gap described above (RI's current `EXPDWM` evaluation
   has zero `COND_DWM_CALC` rows): `dwm(db_ri, method = 'ANNUAL')` emits `"no non-missing arguments to
   max; returning -Inf"` from inside `filterAnnual()`'s `dplyr::mutate(keep = ...)` step, then
   correctly returns a clean 0-row result (the same correct output `method = 'TI'` already produces
   for RI with no warning at all). Not fixed this pass: the *output* is correct, only a diagnostic
   warning is spurious, and the only known trigger is a state having literally zero rows for the
   specific attribute/evaluation combination requested -- an unusual condition currently reachable via
   RI's DWM data-publication gap, but not exercised as a deliberate user-facing empty-domain case (that
   class of bug, matching-nothing `treeDomain`/`areaDomain`, was already fixed elsewhere -- see `tpa.md`
   "Fixed" #2 -- and is confirmed unrelated: this warning originates inside `filterAnnual()`, a
   different code path). No regression test is pinned to this, since it depends on today's transient
   local data cache state rather than a stable, reproducible package condition.

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population estimate (only totals-vs-per-acre was
  checked numerically, same as prior passes).
- `landType = 'timber'` was only checked for internal plot-count consistency; no EVALIDator
  timberland DWM attribute exists to check point estimates against.
- `DUFF`-specific point estimates were not checked against an EVALIDator attribute (none exists in
  the attribute library at the per-fuel-type level for duff alone); duff's `nPlots_DWM`
  qualifying-filter behavior was verified via the `BIO > 0` mechanism only.
