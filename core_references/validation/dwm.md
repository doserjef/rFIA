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

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population estimate (only totals-vs-per-acre was
  checked numerically, same as prior passes).
- `method` options other than `'TI'` (EVALIDator has no equivalent; internal-consistency-only checks
  per the plan, not yet added).
- `landType = 'timber'` was only checked for internal plot-count consistency; no EVALIDator
  timberland DWM attribute exists to check point estimates against.
- `DUFF`-specific point estimates were not checked against an EVALIDator attribute (none exists in
  the attribute library at the per-fuel-type level for duff alone); duff's `nPlots_DWM`
  qualifying-filter behavior was verified via the `BIO > 0` mechanism only.
