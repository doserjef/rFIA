# Validation report: `growMort()`

## Scope

This pass covers `growMort()` -- annual recruitment, natural mortality, harvest removal, and
survivor-growth rates, plus net change, for a chosen state variable (`TPA`, `BAA`, volume, biomass,
or carbon). `growMort()` is a remeasurement-based (growth-accounting) estimator, sharing its
`TREE_GRM_COMPONENT`/`TREE_GRM_MIDPT`/`TREE_GRM_BEGIN`/`SUBP_COND_CHNG_MTRX` machinery with
`vitalRates()` (validated in a prior pass, see `vitalRates.md`) -- per the project plan, growth-based
functions were scheduled last, and `growMort()` specifically has the worst prior bug history of any
rFIA estimator (`NEWS.md`/the project plan cite a v1.1.1 bug where `growMort()` reported zero survivor
growth).

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint (the programmatic interface behind
the EVALIDator web tool), queried live via `fetch_evalidator.R` in this directory. rFIA was run against
the real, current FIADB extracts already cached at `~/Dropbox/data/fia/` (pulled via `getFIA()`), using
`clipFIA(mostRecent = TRUE)` to match EVALIDator's "current" evaluation.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest).

`tests/testthat/test-growMort.R` calls the FIADB-API live at test time rather than hard-coding the
reference numbers below (same approach as every prior validation pass) -- this report is illustrative,
not a source of truth the tests are pinned to. The EVAL_GRP code for each state is read directly off
`clipFIA(..., mostRecent = TRUE)$POP_EVAL_GRP$EVAL_GRP`, so the tests always query whichever evaluation
`mostRecent` actually selected. Tests are skipped (not failed) when the local data cache or network
access to `apps.fs.usda.gov` is unavailable.

`treeDomain`/`areaDomain` filter semantics use the same two API mechanisms established during the
`tpa()` validation pass (see `tpa.md`): `wnum` (numerator-only) for `treeDomain`-style filters,
`strFilter` (numerator + denominator) for `areaDomain`-style filters.

### Attribute mapping

`growMort()`'s `RECR_*`/`MORT_*`/`REMV_*` outputs (in trees, `stateVar = 'TPA'`, the default) are
matched against EVALIDator's "number of trees" mortality/removal attributes:

| rFIA output | `treeType`/`landType` | EVALIDator attr | denom |
|---|---|---|---|
| `MORT_TPA` | all / forest | 901 | 2 |
| `REMV_TPA` (harvest only) | all / forest | 913 | 2 |
| `MORT_TPA` | all / timber | 904 | 3 |
| `REMV_TPA` | all / timber | 916 | 3 |
| `MORT_TPA` | gs / forest | 902 | 2 |
| `REMV_TPA` | gs / forest | 914 | 2 |

`REMV_TPA` corresponds specifically to EVALIDator's **harvest** removals attributes (913/914/916), not
the broader "removals" (907/908/910/911, harvest + other/diversion combined) -- `growMort()`'s
`TPAREMV_UNADJ`-based removal accounting only tallies the `CUT%` growth-accounting component, matching
"harvest removals" exactly (confirmed by exact-match results below).

**There is no EVALIDator attribute for recruitment ("ingrowth") counts at all** -- an exhaustive
keyword search of `EVALIDATOR_POP_ESTIMATE.csv` for "recruit"/"ingrowth" in any field returned zero
rows. `RECR_TPA`/`GROW_TPA`/`CHNG_TPA` therefore have no numeric ground truth under the default
`stateVar = 'TPA'`; only the internal identity `CHNG = GROW + RECR - MORT - REMV` (which holds by
construction) is checked for those three.

For continuous state variables, EVALIDator publishes matching mortality/harvest-removal/net-growth
attributes. `stateVar = 'BIO_AG'` (aboveground biomass, dry short tons/acre, all trees >= 5in DBH) was
used as the primary continuous-variable cross-check across all four states:

| rFIA output | EVALIDator attr (forest) | denom |
|---|---|---|
| `CHNG_BIO_AG_ACRE` | 2635 (net growth) | 2 |
| `MORT_BIO_AG_ACRE` | 2637 | 2 |
| `REMV_BIO_AG_ACRE` | 2649 (harvest removals) | 2 |

`stateVar = 'NETVOL'`/`'SAWVOL_BF'` (growing-stock/sawtimber volume, `treeType = 'gs'`) were checked on
RI against attributes 202 (net growth, merch. bole cubic volume, growing-stock) and 203 (net growth,
sawlog board-foot volume, International 1/4-inch rule) -- these specifically stress-tested the
NA-alignment fix described below (see "Fixed" #4), since board-foot/cubic-foot volume is undefined for
some trees (below merchantability thresholds), unlike TPA/BAA/biomass.

`growMort()`'s **`CHNG_*`** column (population net change, `(CURR_TOTAL - PREV_TOTAL) / REMPER`) is the
column that corresponds to EVALIDator's "net growth" attributes -- EVALIDator's own net-growth SQL
(confirmed directly from attribute 202's `VBA_SUMFROMWHERE` in `EVALIDATOR_POP_ESTIMATE.csv`) credits a
harvested/diverted tree's growth up to the point of removal (valuing it at its *midpoint* measurement on
the "ending" side) while subtracting every departed tree's (`SURVIVOR`, `CUT1`, `DIVERSION1`,
`MORTALITY1`) *begin* measurement on the "starting" side -- i.e. "net growth" already nets out
mortality/removals and includes ingrowth, matching `CHNG`'s definition (see `vitalRates.md` for the
same identification, made for `vitalRates()`'s analogous `BIO_GROW_AC`). `growMort()`'s **`GROW_*`**
column (survivor-only growth, per `growMort.Rd`) has no EVALIDator equivalent (same situation as
`vitalRates()`'s per-stem growth columns) and is checked only via the `CHNG = GROW + RECR - MORT - REMV`
identity.

## Results

### Core default case (`stateVar = 'TPA'`, `treeType = 'all'`, `landType = 'forest'`), 4 states

| State | MORT_TPA | EVALIDator | REMV_TPA | EVALIDator | nPlots_AREA | EVALIDator |
|---|---|---|---|---|---|---|
| RI | 3.997492 | 3.997492 | 0.712488 | 0.712488 | 108 | 108 |
| NC | 2.888280 | 2.888280 | 3.588947 | 3.588947 | 3489 | 3489 |
| CO | 3.278287 | 3.278287 | 0.139925 | 0.139925 | 3704 | 3704 |
| OR | 1.712469 | 1.712469 | 1.464287 | 1.464287 | 9083 | 9083 |

**Exact match** (to full double precision) in all four states, after the fix documented below under
"Fixed" #1 (pre-fix, OR was `MORT_TPA = 1.710672`, `nPlots_AREA = 9031` -- a small but real -0.105%/
-52-plot miss, the same pattern and magnitude already found and fixed for `vitalRates()`).

### `landType`/`treeType` variants, 4 states

`treeType = 'gs'` (attrs 902/914): **exact match**, all four states.

`landType = 'timber'` (attrs 904/916): **exact match** in RI, NC, CO. **OR shows a small residual
mismatch** (`MORT_TPA` rFIA `1.672704` vs. EVALIDator `1.671130`, +0.09%; `nPlots_AREA` 7788 vs. 7749)
-- this is the identical "Known Issue A" from `vitalRates.md` (macroplot-heavy states' timberland
denominator), inherited via the shared `SUBP_COND_CHNG_MTRX`-based area-change logic. Not
re-investigated independently here; see `vitalRates.md`'s "Known issues and intentional divergences
from EVALIDator" section for the full root-cause writeup -- **root-caused in a 2026-08-11 follow-up
session and confirmed to be a documented, intentional divergence from EVALIDator, not a bug**:
EVALIDator's own generated SQL for the timberland growth-accounting denominator (obtained live via the
FIADB-API `fullreport` endpoint's `metadata.denSql` field) hardcodes `SCCM.SUBPTYP=1` with no
macroplot (`SUBPTYP=3`) branch, silently excluding macroplot-basis timberland area from its own
denominator, while `growMortStarter.R`'s `aChng` (fix #1 above) correctly implements the FIA
Population Estimation Guide's documented dual-branch rule (`SUBPTYP` matched to `PROP_BASIS`)
uniformly for both land bases -- the more statistically defensible behavior. `growMort()`'s test suite
checks this case with a loose (<1%) tolerance rather than exact equality, and no code change is planned
to force an exact match here.

### `areaDomain` (mesic physiographic classes), 4 states

| State | MORT_TPA | EVALIDator | nPlots_AREA | EVALIDator |
|---|---|---|---|---|
| RI | 4.065875 | 4.065875 | 102 | 102 |
| NC | 2.878684 | 2.878684 | 2934 | 2934 |
| CO | 4.727470 | 4.727470 | 2024 | 2024 |
| OR | 1.846811 | 1.846811 | 7408 | 7408 |

**Exact match** in all four states, after the fix documented below under "Fixed" #3 (pre-fix: RI exact
by coincidence, NC -0.25%, CO -4.8%, OR -6.2% -- scaling with each state's rate of physiographic-class
turnover between remeasurements, the same signature as the identical bug already found and fixed for
`vitalRates()`).

### `treeDomain` (species filter, RI)

`treeDomain = SPCD == 129` (eastern white pine): `MORT_TPA` exact match both before and after this
pass's fixes (`0.522632` both sides) -- unaffected by any of the four bugs found this pass.

A `DIA >= 20` `treeDomain` (large trees) was also checked and does **not** match EVALIDator exactly
(rFIA `0.107990` vs. EVALIDator `0.072275`) -- expected and not investigated further, per the same
timing-semantics caveat already documented in `vitalRates.md`: `growMort()`'s comprehensive tree-domain
indicator evaluates `treeDomain` against the tree's *previous*-measurement attributes (`tD.prev`), while
EVALIDator's growth-accounting SQL applies the filter to the *current* measurement -- immaterial for a
time-invariant attribute like `SPCD` (confirmed above), not necessarily for a time-varying one like
`DIA`. The test suite uses the species-based filter for its numeric cross-check, matching `tpa.md`'s and
`vitalRates.md`'s precedent.

### `bySpecies` grouping (RI)

Two sampled species (`SPCD` 837, 12) matched EVALIDator's per-species `MORT_TPA` exactly via independent
single-species queries, confirming the domain filter survives rFIA's internal `grpBy`/join path (the
historical `area()`/`areaChange()` bug pattern from v1.1.1) -- **pass**.

### `stateVar = 'BIO_AG'`, 4 states

| State | CHNG_BIO_AG_ACRE | EVALIDator | MORT_BIO_AG_ACRE | EVALIDator | REMV_BIO_AG_ACRE | EVALIDator |
|---|---|---|---|---|---|---|
| RI | 0.182267 | 0.182267 | 1.550537 | 1.550537 | 0.284628 | 0.284628 |
| NC | 1.962002 | 1.962002 | 0.678590 | 0.678590 | 0.965588 | 0.965588 |
| CO | -0.276256 | -0.276256 | 0.704083 | 0.704083 | 0.025058 | 0.025058 |
| OR | 1.114638 | 1.114638 | 0.625426 | 0.625426 | 0.766729 | 0.766729 |

**Exact match** (to full double precision) in all four states, after the fixes documented below under
"Fixed" #2 and #4. Pre-fix, `MORT_BIO_AG_ACRE`/`REMV_BIO_AG_ACRE` were exactly 2000x too large (a units
bug -- see "Fixed" #2) and `CHNG_BIO_AG_ACRE` was wrong in both magnitude and sign (RI: rFIA `-405.18`,
i.e. `-0.203` after manually correcting for the units bug alone, vs. the correct `0.182267` -- see
"Fixed" #4).

### `stateVar = 'NETVOL'`/`'SAWVOL_BF'`, `treeType = 'gs'` (RI)

`CHNG_NETVOL_ACRE`: rFIA `13.80089`, EVALIDator (attr 202) `13.80089` -- **exact match**.
`CHNG_SAWVOL_BF_ACRE`: rFIA `98.01274`, EVALIDator (attr 203) `98.01274` -- **exact match**, with no
`*1000` rescaling needed (unlike `vitalRates()`'s `SAWVOL_GROW_AC`, `growMort()` already reports
`SAWVOL_BF` in raw board feet, not MBF).

These two were the primary regression targets for "Fixed" #4's NA-alignment sub-fix: `VOLCFNET`/
`VOLBFNET` are undefined for some trees (below merchantability thresholds), unlike `DRYBIO_AG`, so they
exposed a bug invisible in the `BIO_AG` checks above (see "Fixed" #4 for detail).

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: every `*_TOTAL` column divided by `AREA_TOTAL` reproduces its corresponding ratio
  column exactly, across all four states -- **pass**.
- `CHNG = GROW + RECR - MORT - REMV` identity: holds to floating-point precision for every state and
  every `stateVar` tested (`TPA`, `BAA`, `NETVOL`, `SAWVOL_BF`, `BIO_AG`) -- **pass** after "Fixed" #4's
  NA-alignment sub-fix (pre-fix, this identity failed for `NETVOL`/`SAWVOL_BF` specifically; see below).
- `nPlots_AREA` responds to `landType`/`areaDomain`, `nPlots_TREE` responds to `treeDomain`: **pass** in
  all cases checked -- `growMort()` did **not** have the `vitalRates()`-style bugs where these columns
  were completely unresponsive to domain restriction (`vitalRates.md`, "Fixed" #1/#2); those were
  already correct here.
- Empty `treeDomain`/`areaDomain` (matching zero trees/area): clean 0-row result, no warning, in both
  cases -- **pass** (the generic `combineMR()` fix from `tpa.md`, "Fixed" #2, already covers this).
- `returnSpatial` (RI, by county): all non-geometry columns match exactly against `returnSpatial =
  FALSE` -- **pass**.

## Fixed

Four bugs were found and fixed this pass, all in `R/growMortStarter.R`. Two (#1, #3) are the exact same
defect class already found and fixed in `vitalRatesStarter.R` during that function's own validation pass
-- that report's "shared-risk note" under fix #3 explicitly flagged #1 as a likely latent issue here,
which this pass confirms. The other two (#2, #4) are `growMort()`-specific.

### 1. Hardcoded `SUBPTYP == 1` in the growth-accounting area-change calculation

Identical to `vitalRatesStarter.R`'s already-fixed bug (see `vitalRates.md`, "Fixed" #3). The FIA
Population Estimation User Guide's worked example for growth-accounting area (using the
`SUBP_COND_CHNG_MTRX` table) requires matching `SUBPTYP` to the *current* condition's `PROP_BASIS`:
`SUBPTYP = 3` when `PROP_BASIS = 'MACR'`, `SUBPTYP = 1` when `PROP_BASIS = 'SUBP'`.
`growMortStarter.R`'s `aData$aChng` calculation (~line 433) hardcoded `SUBPTYP == 1` unconditionally,
silently discarding all area-change information for any condition measured on the macroplot -- invisible
in RI/NC/CO (zero `MACR`-basis conditions) and a small, consistent miss in OR (predominantly
macroplot-based).

**Fix**: changed the `aChng` condition from `SUBPTYP == 1` to `(SUBPTYP == 1 & PROP_BASIS == 'SUBP') |
(SUBPTYP == 3 & PROP_BASIS == 'MACR')`, matching the guide's worked example exactly (same fix as
`vitalRatesStarter.R`).

**Verification**: `landType = 'forest'` now matches EVALIDator exactly (point estimate and
`nPlots_AREA`) in all four states, including OR (was `MORT_TPA = 1.710672`/`nPlots_AREA = 9031`, now
`1.712469`/`9083`, both exact). As with `vitalRates()`'s fix, this exposes a smaller residual mismatch
in `landType = 'timber'` for OR specifically (`vitalRates.md`'s "Known Issue A", not re-investigated
here -- see "Results" above). Full package test suite re-run with no regressions.

### 2. Missing lbs -> short-tons unit conversion for weight-based state variables

`stateVar = 'BIO_AG'`, `'BIO_BG'`, `'BIO'`, `'CARB_AG'`, `'CARB_BG'`, `'CARB'` all report `DRYBIO_AG`/
`DRYBIO_BG`-derived quantities directly, with no unit conversion -- unlike `biomass()`/`carbon()`
(`R/biomassStarter.R`, `R/carbonStarter.R`), which both divide by 2000 to convert FIADB's native pounds
to the short tons/acre that `biomass()`/`carbon()`/EVALIDator report. Confirmed on RI:
`MORT_BIO_AG_ACRE` was exactly 2000x EVALIDator's value (`3101.074` vs. `1.550537`); same 2000x factor
for `REMV_BIO_AG_ACRE`.

**Fix**: added `/ 2000` to all three `state`/`state_recr` assignments (`TREE_GRM_MIDPT`,
`TREE_GRM_BEGIN`, `TREE`) for each of the six affected `stateVar` branches in `growMortStarter.R`'s
state-variable block, matching `biomassStarter.R`'s existing comment ("2000 is to convert from
pounds/acre to short tons/acre").

**Verification**: `MORT_BIO_AG_ACRE`/`REMV_BIO_AG_ACRE` now match EVALIDator exactly in all four states
(see "Results" above). `CARB_*`/`BIO_BG`/`BIO` were not independently checked against EVALIDator
(no `bySpecies`-independent carbon-fraction cross-check was set up this pass) but share the identical
code path, so the fix applies uniformly. Full package test suite re-run with no regressions.

### 3. `areaDomain` filter evaluated against the previous-period condition instead of the current one

Identical to `vitalRatesStarter.R`'s already-fixed bug (see `vitalRates.md`, "Fixed" #4).
`growMortStarter.R`'s comprehensive tree-domain indicator (`tDI = landD.prev * aD.prev * tD.prev *
typeD.prev * sp.prev * tChng`, ~line 413) used `aD.prev` -- the user's `areaDomain` expression evaluated
against the tree's *previous*-measurement condition -- while the area list's own indicator (`aDI =
landD * aD * sp * aChng`) already correctly used `aD` (current condition). EVALIDator's `strFilter`
mechanism restricts both the numerator and denominator using the *current* condition consistently.
Confirmed pre-fix: `areaDomain = PHYSCLCD %in% 21:29` was off by -0.25% (NC), -4.8% (CO), -6.2% (OR)
(RI matched by coincidence), scaling with each state's rate of physiographic-class turnover between
remeasurements -- the identical signature documented for `vitalRates()`'s pre-fix state.

**Fix**: changed `tDI`'s `aD.prev` to `aD` (one-token change), matching the area-side `aDI`'s existing
(correct) use of the current condition.

**Verification**: `areaDomain = PHYSCLCD %in% 21:29` now matches EVALIDator exactly (point estimate and
`nPlots_AREA`) in all four states (see "Results" above). The core default case, `landType`/`treeType`
variants, and the `treeDomain = SPCD == 129` case were re-checked and remain exact -- this fix only
affects the numerator side of `areaDomain`-restricted calls. Full package test suite re-run with no
regressions.

### 4. `GROW_*`/`CHNG_*` (growth/net-change) computation wrong for every continuous state variable

The most serious finding this pass. `CHNG_TPA` (and every other `stateVar`'s `CHNG_*`) is defined as
`(CURR_TOTAL - PREV_TOTAL) / REMPER` -- the literal net change in the state variable's population total
between measurements. Under `stateVar = 'TPA'` (`state` constant `= 1`), this happened to look
reasonable; under any continuous `stateVar`, it was numerically wrong. Confirmed via `NETVOL` (`treeType
= 'gs'`, chosen specifically because it involves no unit-conversion factor, isolating this bug from
Fixed #2): `growMort(stateVar = 'NETVOL', treeType = 'gs')$CHNG_NETVOL_ACRE` was `-1.899745`, while the
same underlying quantity computed by `vitalRates()` (already validated exactly against EVALIDator
attribute 202) was `13.80089` -- while `MORT_NETVOL_ACRE`/`REMV_NETVOL_ACRE` (computed independently)
both already matched EVALIDator exactly, isolating the bug specifically to the growth/change residual
term, not the separately-reported mortality/removal statistics.

**Root cause (part A -- wrong measurement basis for departed trees)**: `growMortStarter.R`'s
`TPA_UNADJ.prev` (the previous-period/T1 population-total building block) was defined only for
`COMPONENT == 'SURVIVOR'` rows (valued at the tree's *begin* measurement, via `state.prev`). To account
for mortality/harvest trees' T1 contribution, the code instead added `(mPlot + hPlot) * REMPER` onto
`pPlot` -- but `mPlot`/`hPlot` are valued at the tree's *midpoint* measurement (correct for, and shared
with, the separately-reported `MORT_*`/`REMV_*` columns, confirmed to match EVALIDator exactly), not its
*begin* measurement. EVALIDator's own growth-accounting SQL (confirmed directly from attribute 202's
`VBA_SUMFROMWHERE` in `EVALIDATOR_POP_ESTIMATE.csv`) subtracts the *begin* value for every departed tree
(`SURVIVOR`, `CUT1`, `DIVERSION1`, `MORTALITY1`) on the "starting" side, and separately credits
`CUT`/`DIVERSION` trees' *midpoint* value on the "ending" side (crediting their growth up to the point of
removal -- mortality trees get no such credit). For `stateVar = 'TPA'` (`state` constant `= 1`),
midpoint and begin values are identically `1`, so this distinction was invisible; for any state variable
that actually changes over the remeasurement period (diameter, volume, biomass, ...), it was not.

**Fix (part A)**: extended `TPA_UNADJ.prev`'s component list to `c('SURVIVOR', 'CUT1', 'DIVERSION1',
'MORTALITY1')`, valued at `state.prev` (the begin measurement) for all of them, matching EVALIDator's
SQL exactly. Added the missing "ending"-side midpoint credit for harvested trees directly to `tPlot`
(`tPlot = (TPA_UNADJ * tDI) + (hPlot * REMPER)`) -- `hPlot` (`TPAREMV_UNADJ`-weighted, midpoint-valued)
is unit-equivalent to the `TPAGROW_UNADJ`-weighted midpoint credit EVALIDator's SQL uses for `CUT`
components, since `TPAREMV_UNADJ` is `TPAGROW_UNADJ`'s pre-annualized counterpart for those same rows
(the same relationship `growMortStarter.R` already relies on for `TPARECR_UNADJ`/ingrowth). Removed the
now-redundant `(mPlot + hPlot) * REMPER` term from `pPlot`, which is now just `TPA_UNADJ.prev * tDI`.
Mortality trees correctly get no "ending"-side credit under this fix (matching EVALIDator, which credits
only `CUT`/`DIVERSION`, not `MORTALITY`, on the ending side). The identical double-mortality-subtraction
bug in the `byPlot = TRUE` branch's separate, simpler tree summary (`PREV_TPA = PREV_TPA + (MORT_TPA +
REMV_TPA) * REMPER`) was fixed the same way (`CURR_TPA = CURR_TPA + (REMV_TPA * REMPER)`, no correction
needed on `PREV_TPA` since it already aggregates via `sum(TPA_UNADJ.prev * tDI, na.rm = TRUE)`, which
picks up the extended `TPA_UNADJ.prev` automatically).

**Root cause (part B -- NA misalignment across columns)**: applying part A's fix alone corrected
`stateVar = 'BIO_AG'` exactly, but left `stateVar = 'NETVOL'`/`'SAWVOL_BF'` still off (by ~5% and more,
respectively) and broke the `CHNG = GROW + RECR - MORT - REMV` identity for those two specifically.
`VOLCFNET`/`VOLBFNET` (unlike `DRYBIO_AG`) are undefined (`NA`) for some trees -- e.g. below
merchantability thresholds -- and this `NA` can appear in one of `rPlot`/`mPlot`/`hPlot`/`tPlot`/`pPlot`
for a given tree row without appearing in the others (e.g. `TPAREMV_UNADJ * state` becomes `0 * NA =
NA` in R for a *non*-harvested tree whose midpoint volume happens to be undefined, even though that
tree contributes nothing to harvest removals). Each of `RECR_*`/`MORT_*`/`REMV_*`/`CURR_*`/`PREV_*` was
being aggregated independently via `sum(x, na.rm = TRUE)`, so a different subset of trees was silently
dropped from each column's total -- breaking the algebraic identity between them (each row's own
`cPlot`/`gPlot` arithmetic was internally correct, per direct row-level inspection with a max deviation
of `3.5e-15`, but the *aggregate* sums no longer lined up column-to-column).

**Fix (part B)**: every building-block term (`rPlot`, `mPlot`, `hPlot`, and the `TPA_UNADJ`/
`TPA_UNADJ.prev`-derived pieces of `tPlot`/`pPlot`) is now wrapped in `dplyr::coalesce(..., 0)` at the
row level, before combination, so a `NA` for reasons unrelated to a given row's own contribution can
never propagate into -- or silently vanish from -- a different column than it would have under plain
`na.rm = TRUE` aggregation. This is a no-op for every point estimate that was already correct (`sum(x,
na.rm = TRUE)` is arithmetically identical to `sum(coalesce(x, 0))`), and only changes results in cases
where the previous per-column-independent `na.rm = TRUE` dropping produced a different, misaligned set
of contributing rows across `RECR`/`MORT`/`REMV`/`GROW`/`CHNG`.

**Verification**: `CHNG_NETVOL_ACRE` (RI, `treeType = 'gs'`) now matches EVALIDator's attribute 202
exactly (`13.80089` both sides, was `-1.899745` pre-fix, `13.07034` after part A alone);
`CHNG_SAWVOL_BF_ACRE` matches attribute 203 exactly (`98.01274` both sides). `CHNG_BIO_AG_ACRE` remains
exact in all four states (unaffected by part B, since `DRYBIO_AG` has no relevant `NA`s). The `CHNG =
GROW + RECR - MORT - REMV` identity now holds (to floating-point precision) for every state and every
`stateVar` tested. Full package test suite re-run with no regressions.

### 5. `bySizeClass` silently dropped nearly all removed/harvested trees, and a large share of mortality
   trees, from `growMort()` estimates [FIXED -- issue #40]

Reported upstream as issue #40 ("growMort bySizeClass missing removed stems"). Reproduced on OR's current
GRM evaluation (EVAL_GRP 412022, `stateVar = 'BAA'`): summing `bySizeClass = TRUE` output back across size
classes gave `REMV_BAA = 0.0141` vs. `1.26` without `bySizeClass` (-98.9%) and `MORT_BAA = 0.848` vs.
`1.12` (-24.6%); `RECR_BAA` was unaffected (new ingrowth trees almost always have a valid current-cycle
diameter).

**Root cause**: `growMortStarter.R` computed `sizeClass` from `db$TREE$DIA` -- the tree's current-cycle
(T2) diameter -- and dropped any row where that was `NA`, *before* `db$TREE` was even joined to the GRM
tables. A removed (harvested) or dead tree usually can't be measured at T2: confirmed directly against
OR's `TREE_GRM_COMPONENT` that 17,428 of 17,653 (98.7%) removal-component tree records, and 11,401 of
33,716 (33.8%) mortality-component tree records, have `DIA = NA` in their T2 `TREE` row. Meanwhile
`TREE_GRM_MIDPT.DIA` -- the midpoint diameter `growMortStarter.R` already uses elsewhere in the same
function to compute each row's `state` value for exactly these components -- was populated for 17,652 of
those same 17,653 (99.99%) removal records, and 11,397 of 11,401 (99.96%) of the DIA-NA mortality records;
`TREE_GRM_BEGIN.DIA` covered the remainder (0 rows in OR had all three sources NA).

**Fix**: `sizeClass` is now assigned after `data` (the full joined tree list) has been built and joined to
`TREE_GRM_MIDPT`/`TREE_GRM_BEGIN`, using `makeClasses(dplyr::coalesce(DIA, DIA.mid, DIA.beg), ...)` --
the same diameter sources, in the same priority order, that the row's own `state` value is already drawn
from a few lines earlier. A first attempt at this fix dropped size-class-unclassifiable rows directly from
the shared `data` object, which also backs the forested-area denominator (`a`) via one row per
`(PLT_CN, CONDID)` for conditions with zero tally trees -- this silently shrank the area denominator for
any such condition, inflating every ratio by a small but nonzero uniform amount (~0.84% on RECR/MORT/REMV
alike in OR, confirmed by testing an unrelated grouping variable like `bySpecies` or `grpBy = STATUSCD`,
which showed no such delta). The corrected fix filters the tree list (`t`, in both the `byPlot` and
population/`treeList` branches) instead of `data` itself, leaving the area calculation untouched.

**Verification**: on OR's current GRM evaluation, `bySizeClass = TRUE` summed back across classes now
matches the unrestricted `RECR_BAA`/`MORT_BAA`/`REMV_BAA` to floating-point precision (~1e-14 relative
difference), in both `byPlot = TRUE` and population-level modes. `treeList = TRUE` row counts are
identical with and without `bySizeClass` (317,569 rows either way), confirming no trees are dropped.
Full `test-growMort.R` suite re-run with no regressions. The identical bug (and fix) applies to
`vitalRates()` -- see `vitalRates.md`.

## Notes

### `nPlots_TREE` is a single generic column, not a per-event plot count -- FIXED (follow-up pass)

`growMort.Rd`'s Value section documents `nPlots_RECR`, `nPlots_MORT`, `nPlots_REMV` as separate output
columns (implying a distinct plot count per event type, presumably intended for computing the correct
degrees of freedom for a t-based CI on each individual rate), but the actual implementation only ever
returns a single `nPlots_TREE` (count of plots contributing to *any* of recruitment/mortality/removal/
survivor-growth combined) alongside `nPlots_AREA`. Confirmed: RI's unrestricted `nPlots_TREE = 107`
matches neither EVALIDator's mortality-attribute plot count (86) nor its removals-attribute plot count
(13) -- it's a broader, different quantity by design, not a bug (`nPlots_TREE` does correctly shrink
under `treeDomain`/`bySpecies`, confirming it responds to filtering, just not per-event). This was
originally a pre-existing documentation/implementation gap (the three per-event columns appear to never
have been implemented), flagged rather than fixed during the main pass since deciding whether to
implement the three missing columns or correct the documentation was a design decision outside that
pass's numeric-correctness scope.

**Follow-up fix**: implemented `nPlots_RECR`/`nPlots_MORT`/`nPlots_REMV` in `growMortStarter.R`,
matching the documentation rather than correcting it. A fourth column, `nPlots_GROW` (non-zero plots
for survivor growth), was added alongside them at the user's request even though `growMort.Rd` didn't
originally list it, for consistency with the other three event types. Each is computed by re-using
`sumToEU()`'s existing strata/estimation-unit plot-count machinery (the same `nPlots.x` logic already
powering `nPlots_TREE`/`nPlots_AREA` throughout the package), called on a version of the plot-level
tree table (`tPlt`) restricted to the one relevant column and filtered to plots with a non-zero
RECR/MORT/REMV/GROW contribution, respectively -- rather than reimplementing the strata-weighted
aggregation from scratch. `gPlot` (survivor growth) is a differenced quantity (`cPlot - rPlot + mPlot +
hPlot`), so a plot with no real survivor growth can land a hair off exact zero from floating-point
noise; `nPlots_GROW`'s filter uses the same `abs(.x) < 1e-5` tolerance already applied to `GROW_TPA`
elsewhere in `growMort()` (see "Results", core default case) rather than an exact `!= 0` test, to avoid
counting that noise as a real contribution. `nPlots_TREE` itself is unchanged (still the broader
any-event count).

**Verification**: RI (unrestricted, `landType = 'forest'`), freshly pulled `mostRecent` data:
`nPlots_MORT = 87` matches EVALIDator attribute 901's `numPlotCount` (87) exactly; `nPlots_REMV = 13`
matches attribute 913's `numPlotCount` (13) exactly (both counts confirmed via `fetch_evalidator.R`,
same mechanism used throughout this report). `nPlots_RECR`/`nPlots_GROW` have no EVALIDator equivalent
(ingrowth has no matching attribute at all, per the "Attribute mapping" section above, and `GROW_*` was
already established to have none either) and were checked only for internal sanity
(`nPlots_RECR`/`nPlots_MORT`/`nPlots_REMV`/`nPlots_GROW` all `<= nPlots_TREE` in every case tried;
RI's default case gives `nPlots_GROW = 7`, plausible under `stateVar = 'TPA'` since a tree only
registers nonzero survivor growth there when it crosses the microplot/subplot size threshold between
measurements -- a comparatively rare event -- not on ordinary diameter growth). Also checked:
`bySpecies` (each species' new columns shrink sensibly and independently per species), `totals = TRUE`,
`landType = 'timber'`, and an empty `treeDomain` (clean 0-row result, no warning) -- all pass. `byPlot =
TRUE` is a separate, untouched code path and does not return `nPlots_*` columns (unchanged pre-existing
behavior). Full `test-growMort.R` suite (124 assertions, network-backed) re-run with no
regressions.

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population-level estimate (only totals-vs-per-acre/
  the `CHNG` identity internal consistency was checked, same as every prior pass).
- `method` options other than `'TI'` (EVALIDator has no equivalent; internal-consistency-only checks
  per the plan, not yet added).
- `bySizeClass` was only checked structurally (pre-existing `test-growMort.R` coverage), not against
  an EVALIDator size-class breakdown.
- `CARB_AG`/`CARB_BG`/`CARB`/`BIO_BG`/`BIO`/`SAWVOL`/`SNDVOL` state variables were fixed (Fixed #2)
  but not independently cross-checked against EVALIDator this pass (only `BIO_AG`, `NETVOL`, and
  `SAWVOL_BF` were) -- they share the exact code path already validated for those three, but a direct
  check would be worth adding in a follow-up pass.
- ~~`landType = 'timber'`'s macroplot-heavy-state residual (OR, and per `vitalRates.md`, likely CA/WA)
  -- inherited unresolved from the `vitalRates()` pass; not re-investigated here.~~ Root-caused in a
  2026-08-11 follow-up session (see `vitalRates.md`) and kept as an intentional divergence from
  EVALIDator, not fixed -- see "Results" above.
- ~~The `nPlots_RECR`/`nPlots_MORT`/`nPlots_REMV` documentation/implementation gap noted above.~~ Fixed
  in a follow-up pass -- see "Notes" above.
