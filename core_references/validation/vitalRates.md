# Validation report: `vitalRates()`

## Scope

This pass covers `vitalRates()` -- average annual diameter, basal area, net volume, sawlog volume,
and aboveground biomass growth rates of individual stems, plus basal area/volume/biomass growth per
acre. `vitalRates()` is a remeasurement-based (growth-accounting) estimator, sharing its
`TREE_GRM_COMPONENT`/`TREE_GRM_MIDPT`/`TREE_GRM_BEGIN` machinery with `growMort()` (validated
separately) -- per the project plan, growth-based functions were scheduled last because they carry
the most complex remeasurement logic and the worst bug history of any rFIA estimator family.

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint (the programmatic interface
behind the EVALIDator web tool), queried live via `fetch_evalidator.R` in this directory. rFIA was
run against the real, current FIADB extracts already cached at `~/Dropbox/data/fia/` (pulled via
`getFIA()`), using `clipFIA(mostRecent = TRUE)` to match EVALIDator's "current" evaluation.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest).

`tests/testthat/test-vitalRates.R` calls the FIADB-API live at test time rather than hard-coding the
reference numbers below (same approach as every prior validation pass) -- this report is
illustrative, not a source of truth the tests are pinned to. The EVAL_GRP code for each state is
read directly off `clipFIA(..., mostRecent = TRUE)$POP_EVAL_GRP$EVAL_GRP`, so the tests always query
whichever evaluation `mostRecent` actually selected -- the same code covers every eval type
(including `EXPGROW`, the remeasurement evaluation `vitalRates()` uses), since `POP_EVAL_GRP` groups
all eval types for a state/year together. Tests are skipped (not failed) when the local data cache or
network access to `apps.fs.usda.gov` is unavailable.

**Network note**: `apps.fs.usda.gov` was unreachable (TLS connects, then resets immediately after the
request is sent) for an extended period spanning this pass's first working session; it recovered in a
later session, at which point the full EVALIDator numeric cross-check (Tests 12-17 in
`test-vitalRates.R`) was run live and is reported below. Because those tests fetch reference values
live rather than hard-coding them, they self-validate on every future run rather than drifting.

`treeDomain`/`areaDomain` filter semantics use the same two API mechanisms established during the
`tpa()` validation pass (see `tpa.md`): `wnum` (numerator-only) for `treeDomain`-style filters,
`strFilter` (numerator + denominator) for `areaDomain`-style filters.

### Attribute mapping

`vitalRates()`'s growth attributes are computed via FIA's own `TREE_GRM_COMPONENT` growth-accounting
columns (`SUBP_TPAGROW_UNADJ_{AL,GS}_{FOREST,TIMBER}`, `SUBP_COMPONENT_*`, etc. -- see
`typeDomain_grow()` in `R/util.R`), and `R/vitalRatesStarter.R`'s `vrAttHelper()` reproduces
EVALIDator's own growth-accounting SQL almost line-for-line (confirmed by inspecting attribute 202's
`VBA_SUMFROMWHERE` in `EVALIDATOR_POP_ESTIMATE.csv`: the `ONEORTWO`/`COMPONENT`
SURVIVOR-INGROWTH-REVERSION-vs-CUT-DIVERSION-vs-MORTALITY1 branching matches `vrAttHelper()`
exactly). This means, as with `biomass()`/`volume()`, the numeric match here is primarily testing
rFIA's post-stratified aggregation of a pre-defined FIADB growth quantity, not a from-scratch
reimplementation.

`treeType = 'all'` and `treeType = 'live'` use the **same** underlying `TREE_GRM_COMPONENT` columns
(`AL_FOREST`/`AL_TIMBER`) -- the only difference is that `treeType = 'live'` additionally multiplies
the tree domain indicator by `status` (`COMPONENT == 'SURVIVOR'`), restricting to trees alive at both
measurements and excluding recruitment/mortality/cut from the estimate entirely. EVALIDator's own
"average net growth" attributes are always net-of-mortality, net-of-cut, and inclusive of ingrowth
(i.e. they match rFIA's `treeType = 'all'` design, not `'live'`) -- there is no "survivor-only growth"
attribute published in `EVALIDATOR_POP_ESTIMATE.csv` (confirmed by keyword search for "survivor").
So, matching how `tpa.md` handled `treeType = 'all'` (no EVALIDator equivalent, internal-consistency
check only), **`treeType = 'live'` has no EVALIDator equivalent here** and is only checked
structurally/for internal consistency.

Similarly, EVALIDator publishes net volume/sawlog growth attributes only for **growing-stock**
trees (`treeType = 'gs'`), not for the unrestricted `treeType = 'all'` set -- confirmed by an
exhaustive keyword search of `EVALIDATOR_POP_ESTIMATE.csv` for growth attributes matching "trees (at
least 5 inches ...)" without a "growing-stock"/"sawtimber" restriction: none exist for volume, only
for aboveground/belowground biomass (attributes 2635/2636). So:

| rFIA output (per acre) | `treeType` | EVALIDator attr (forest / timber) | denom (forest / timber) |
|---|---|---|---|
| `BIO_GROW_AC` | `'all'` (default) | 2635 / 2636 | 2 / 3 |
| `BIO_GROW_AC` | `'gs'` | 312 / 315 | 2 / 3 |
| `NETVOL_GROW_AC` | `'gs'` (no `'all'` equivalent) | 202 / 208 | 2 / 3 |
| `SAWVOL_GROW_AC` (x1000, International 1/4-inch rule) | `'gs'` (no `'all'` equivalent) | 203 / 209 | 2 / 3 |

(Attribute 315 -- "aboveground biomass net growth, growing-stock, timberland" -- was originally
mistyped as 318 in this pass's first draft of the test file/report; 318 is *belowground* biomass on
*forest land*, a different attribute entirely. Caught when the timberland `BIO_GROW_AC` comparison
came back 3-5x off in every state, which was large enough to look like a real rFIA bug before the
attribute numbers were re-checked against `EVALIDATOR_POP_ESTIMATE.csv` directly -- see "Fixed" #5.)

`DIA_GROW`/`BA_GROW`/`NETVOL_GROW`/`SAWVOL_GROW`/`BIO_GROW` (the **per-stem**, not per-acre,
columns -- ratioed against total previous-measurement tree count, not area) have **no EVALIDator
equivalent at all**: an exhaustive keyword search of `EVALIDATOR_POP_ESTIMATE.csv` found no
"basal area growth" or "diameter growth" attribute of any kind, and EVALIDator does not publish a
"growth per average tree" statistic for volume/biomass either -- its standard outputs are always
population totals or per-acre ratios. These columns are validated only via internal consistency
(`*_TOTAL / TREE_TOTAL` reproduces the per-stem ratio exactly) in this pass.

## Results

### Internal consistency (no EVALIDator needed), 4 states

`totals = TRUE`: every `*_TOTAL` column divided by the matching denominator (`AREA_TOTAL` for the
per-acre columns, `TREE_TOTAL` for the per-stem columns) reproduces its corresponding ratio column
exactly, across all four states. **Pass** (`tests/testthat/test-vitalRates.R`, Test 8).

### Core default case (`treeType = 'all'`, `landType = 'forest'`, the function defaults), 4 states

| State | BIO_GROW_AC | EVALIDator | nPlots_AREA | EVALIDator |
|---|---|---|---|---|
| RI | 0.182267 | 0.182267 | 108 | 108 |
| NC | 1.962002 | 1.962002 | 3489 | 3489 |
| CO | -0.276256 | -0.276256 | 3704 | 3704 |
| OR | 1.114638 | 1.114638 | 9083 | 9083 |

**Exact match** (to full double precision) against EVALIDator attribute 2635 (forest land, biomass
net growth, all trees >= 5in DBH), ratio'd against attribute 2 (forest land area), in all four
states -- OR only after the fix documented below under "Fixed" #3 (pre-fix, OR was `1.113468` /
`9031`, a small but real -0.105%/-52-plot miss).

`%SE` comparisons matched in magnitude everywhere, but EVALIDator reports a **negative** `%SE` when
the point estimate itself is negative (e.g. CO: rFIA `7.81326`, EVALIDator `-7.81326`) -- this is
purely an EVALIDator display/sign convention for negative ratios, not a discrepancy; confirmed by
checking every CO case (which is always negative, since `treeType = 'all'` growth is net of
mortality/cut and can legitimately go negative -- see `man/vitalRates.Rd`) and finding the magnitude
matches exactly in each one. The test suite compares `abs()` of both sides to avoid false failures
from this convention.

### `landType`/`treeType` variants, 4 states

`landType = 'timber'` (attr 2636/3) matches EVALIDator exactly in RI/NC/CO (unaffected by fix #3
below) and, after that fix, in CA/WA/OR's *forest* case -- but **`landType = 'timber'` in the three
macroplot-heavy Western states (OR, CA, WA) still shows a small residual `nPlots_AREA` over-count**
(OR 7788 vs. 7749, CA 2580 vs. 2562, WA 4899 vs. 4870) not yet root-caused; see "Known issues" below.

`treeType = 'gs'` (attrs 202/208, 203/209, 312/315): **exact match** for `NETVOL_GROW_AC`,
`SAWVOL_GROW_AC`, and `BIO_GROW_AC` in RI/NC/CO, both `landType`s, after fix #5 below; OR shows only
the same Known Issue A residual as every other `landType = 'timber'` metric (~0.09%), not a new
mismatch.

### Domain filter interactions, 4 states

| Case | RI | NC | CO | OR |
|---|---|---|---|---|
| `areaDomain = PHYSCLCD %in% 21:29` (mesic), vs. EVALIDator `strFilter` | exact | exact | exact | exact |

**Exact match** (to full double precision) in all four states after fix #4 below (pre-fix: RI -0.58%,
NC -0.70%, CO -7.36%, OR -5.16%).

`treeDomain = SPCD == 129` (white pine, RI, `wnum`): **exact match** (`BIO_GROW_AC = 0.131097` both
sides), confirming the note below about timing semantics doesn't matter for a time-invariant filter
like `SPCD`.

**A note on `treeDomain` timing semantics**: unlike `tpa()`/`volume()`, `vitalRates()`'s
comprehensive tree-domain indicator (`tDI`) is built from `tD.prev` -- the user's `treeDomain`
expression evaluated against the tree's *previous*-measurement attributes (falling back to the
current measurement only for ingrowth/recruit trees with no previous record; see
`vitalRatesStarter.R` lines ~276-289) -- whereas EVALIDator's growth-accounting SQL (confirmed via
attribute 202's `VBA_SUMFROMWHERE`) applies its `wnum`/`strFilter` WHERE-clause fragment against the
*current* (`TREE` alias) measurement. For a time-invariant attribute like `SPCD`, this distinction is
immaterial (confirmed above). For a time-varying attribute like `DIA`, it is not necessarily
immaterial (a tree's diameter at the previous measurement can differ from its current diameter), so a
`DIA`-based `treeDomain` filter is not assumed to match EVALIDator's `wnum`/`strFilter` the way it
does for `tpa()`/`volume()` -- not yet empirically checked; a species-based filter was used instead
for the numeric cross-check tests.

### `bySpecies` grouping (RI)

Post-fix, `nPlots_TREE` correctly varies by species (e.g. SPCD 129 (white pine): 53; several other
sampled species: single digits) -- before the fix, every species row reported the same unrestricted
plot count regardless of how common that species actually was. **Pass** (structural/internal;
EVALIDator per-species cross-check pending API availability, Test 17).

### `returnSpatial` (RI, by county)

`vitalRates(polys = countiesRI, returnSpatial = TRUE)` vs. `returnSpatial = FALSE`: all non-geometry
columns match exactly. **Pass** (Test 9).

## Fixed

Two bugs were found and fixed this pass, both in `R/vitalRatesStarter.R`'s population-estimation
branch, both of the same general class already documented in `tpa.md`/`volume.md`/`biomass.md`
(a plot that contributes nothing to a ratio's numerator/denominator still gets counted towards the
plot-count/degrees-of-freedom column) but arising from a different mechanism than those prior fixes.
**Point estimates and standard errors were unaffected by either fix** -- confirmed by rerunning the
same calls before and after and comparing to full precision; both fixes only affect the
`nPlots_TREE`/`nPlots_AREA` reporting columns.

### 1. `nPlots_AREA` did not respond to `landType` or `areaDomain` at all

Reproduced on RI: `landType = 'forest'` and `landType = 'timber'` reported *identical*
per-estimation-unit plot counts (85 and 25), and even an `areaDomain` engineered to match zero area
everywhere (`PHYSCLCD == 11`, a physiographic class not present in the data) still reported the same
non-zero counts -- i.e. `nPlots_AREA` was completely invariant to the area/land domain, not merely
inflated by an edge case.

**Root cause**: unlike `tpa()`'s original bug (a join-order issue producing phantom `CONDID = NA`
rows), `vitalRatesStarter.R` never pre-filters `db$COND` by `aD`/`landD` before joining, so the
condition list used for area estimation (`a <- aData %>% dplyr::mutate(fa = SUBPTYP_PROP_CHNG * aDI)
%>% dplyr::select(...)`, ~line 420) never produces `NA` rows for non-qualifying conditions -- it
produces real rows with `fa` correctly computed as exactly `0`. Since `nPlots_AREA` is computed as
`length(unique(PLT_CN))` over this list (via `sumToPlot()`/`sumToEU()`) with no filter excluding
`fa == 0` rows, **every** plot in the remeasurement panel was counted, regardless of whether its area
actually qualified under the current `landType`/`areaDomain`.

**Fix**: filter to `!is.na(fa) & fa > 0` immediately before the area list feeds `sumToPlot()` in the
population-estimation branch only (`aPlt <- sumToPlot(dplyr::filter(a, !is.na(fa) & fa > 0), pops,
aGrpBy)`) -- not on the shared `a` object itself, which also feeds the `treeList = TRUE` output a few
lines above and should retain every condition (including zero-area ones) for that consumer. This is
a no-op for every point estimate/variance sum (`sum(fa, na.rm = TRUE)` already treats a dropped `fa =
0` row identically to a kept one), matching the precedent and caution documented in `volume.md`'s
"Fixed" #3 (scope narrowly; don't touch a shared object feeding a different consumer).

**Verification**: post-fix, RI `nPlots_AREA` is 108 (forest) vs. 102 (timber) vs. 102 (mesic
`areaDomain`) vs. correctly empty (0 rows, no warning) for the impossible-filter case -- all
distinct from each other and from the pre-fix constant 110. All four states show the same pattern
(e.g. OR: 9031 forest vs. 7749 timber). Point estimates (`BIO_GROW_AC` etc.) and internal-consistency
checks (`*_TOTAL / AREA_TOTAL` reproduces the per-acre ratio) are byte-identical before and after the
fix in every case tested. Full package test suite (all estimator test files, including `growMort()`,
which shares no code with this fix) re-run with no regressions.

### 2. `nPlots_TREE` did not respond to `treeDomain` at all

More severe than #1: reproduced on RI, `treeDomain = SPCD == 129` (white pine, a small fraction of
trees) reported the same `nPlots_TREE = 108` as the unrestricted case, and `treeDomain = SPCD == 999`
(matching **zero** trees -- `BIO_GROW_AC` correctly came out as `0`) *also* reported `nPlots_TREE =
108`. For comparison, `tpa()` on the same data and filter correctly shrinks (129 -> 68). This bug was
severe enough that every row of a `bySpecies = TRUE` call reported the identical, unrestricted
`nPlots_TREE`, regardless of how common that species actually was -- defeating the entire purpose of
the column for per-species confidence intervals (`vitalRates.Rd` explicitly tells users to use
`nPlots_AREA`/(implicitly, by the same logic) `nPlots_TREE` as the degrees of freedom for a t-based
CI).

**Root cause**: the tree list's plot-count filter (`dplyr::filter(!is.na(TREE_BASIS))`, ~line 446)
excludes only trees lacking a valid `SUBPTYP_GRM` growth-accounting record for the current
`landType`/`treeType` (i.e. `TREE_BASIS` reflects only the *canned* `landType`/`treeType` domain, via
FIA's own precomputed `TREE_GRM_COMPONENT` columns) -- it has no dependency on `tDI`, the
user-supplied `treeDomain`/`areaDomain` indicator. `tDI` is applied only as a multiplier zeroing
`dPlot`/`bPlot`/`gPlot`/`sPlot`/`bioPlot` for non-qualifying trees; the row (and its `PLT_CN`) stays
in the list and gets counted by `nPlots_TREE = length(unique(PLT_CN))` regardless of whether `tDI`
was ever `1`.

**Fix**: added `tDI` to the tree list's selected columns, then filtered to `!is.na(tDI) & tDI > 0`
immediately before it feeds `sumToPlot()` in the population-estimation branch
(`tPlt <- sumToPlot(dplyr::select(dplyr::filter(t, !is.na(tDI) & tDI > 0), -tDI), pops, grpBy)`),
dropping the helper column again right after so it doesn't get mistaken for a value column by
`sumToPlot()`/`sumToEU()`'s column-scoping logic. Scoped the same way as fix #1: the shared `t`
object (which also feeds the `treeList = TRUE` output) is left untouched. No-op for every point
estimate: `tDI == 0` rows already contribute exactly `0` to every sum.

**Verification**: post-fix, RI `nPlots_TREE` is 107 (unrestricted) vs. 53 (`SPCD == 129`) vs.
correctly empty (`SPCD == 999`, 0 rows, no warning). `bySpecies = TRUE` now shows `nPlots_TREE`
varying per species (e.g. 1, 3, 6, 7, 53, 1 for a sample of species) instead of a single repeated
value. All four states show `nPlots_TREE` shrinking appropriately under `landType = 'timber'` (e.g.
OR: 8860 -> 7638). Point estimates unaffected in every case tested. Full package test suite re-run
with no regressions.

### Incidental fix: empty-domain warning

Before either fix, an `areaDomain` matching zero area (e.g. `PHYSCLCD == 11`) returned a 1-row result
with `NA` values and the `combineMR()` empty-result warning already documented and fixed generically
in `tpa.md` "Fixed" #2 (shared utility, `nrow(x) == 0` guard) -- because `tPlt`/`aPlt` were non-empty
even though every row's contribution was zero. Since fixes #1 and #2 above now correctly empty both
lists in this situation, this case now falls through to the already-existing empty-result path and
returns a clean 0-row tibble with no warning, with no additional code change needed.

### 3. `SUBP_COND_CHNG_MTRX` (SCCM) join hardcoded `SUBPTYP == 1`, silently dropping macroplot-basis
   area change (`landType = 'forest'`, Pacific Northwest / macroplot-heavy states)

Found via the EVALIDator numeric cross-check (once the API became reachable): `BIO_GROW_AC` and
`nPlots_AREA` matched EVALIDator exactly in RI, NC, and CO, but OR was off by a small, consistent
amount (`BIO_GROW_AC` -0.105%; `nPlots_AREA` 9031 vs. EVALIDator's 9083, i.e. -52 plots) under the
default `landType = 'forest'`.

**Root cause**: the FIA Population Estimation User Guide's own worked example for growth-accounting
area (Ch. 7.8, "Using the SCCM table to estimate area change between two measurements") requires
matching `SUBP_COND_CHNG_MTRX.SUBPTYP` to the *current* condition's `PROP_BASIS`:
`SUBPTYP = 3` when `PROP_BASIS = 'MACR'`, `SUBPTYP = 1` when `PROP_BASIS = 'SUBP'`. `aData`'s `aChng`
calculation in `vitalRatesStarter.R` hardcoded `SUBPTYP == 1` unconditionally, silently discarding all
area-change information for any condition whose proportion was measured on the macroplot. Checked
raw `COND.PROP_BASIS` directly across states: **RI, NC, and CO have zero `MACR`-basis conditions**
(100% `SUBP`), while **OR, CA, and WA have zero (or near-zero) `SUBP`-basis conditions** (100%, or
nearly so, `MACR`) -- explaining exactly why this was invisible in three states and a small, uniform
miss in the others (Pacific/Western states more commonly use the macroplot design, e.g. for
large-diameter breakpoint sampling).

**Fix**: changed the `aChng` condition in `vitalRatesStarter.R` (~line 346) from `SUBPTYP == 1` to
`(SUBPTYP == 1 & PROP_BASIS == 'SUBP') | (SUBPTYP == 3 & PROP_BASIS == 'MACR')`, matching the guide's
worked example exactly. Also added the guide's `COALESCE(COND_NONSAMPLE_REASN_CD, 0) = 0` check for
both current and previous condition (empirically confirmed to be a no-op for every case tested here,
but included for methodological completeness with the guide).

**Verification**: after the fix, `landType = 'forest'` matches EVALIDator exactly (point estimate and
`nPlots_AREA`) in all of RI, NC, CO, OR, CA, and WA (the latter two checked specifically because they
share OR's macroplot-heavy `COND` composition -- see "Known issues" below for how that check was
requested and what it found). RI/NC/CO were already exact before this fix and are unaffected by it
(zero `MACR`-basis rows to begin with). Full package test suite re-run with no regressions.

**Shared-risk note**: `growMortStarter.R` (~line 436) has the identical hardcoded `SUBPTYP == 1`
pattern in its own SCCM-based `aData`/`aChng` construction, sharing the same root cause. `growMort()`
is scheduled for its own validation pass later in this project's plan and was not touched here, but
this is flagged now so that pass doesn't need to rediscover it from scratch.

### 4. `areaDomain` filter restricted the tree side using the *previous* condition instead of the current one

`areaDomain = PHYSCLCD %in% 21:29` (mesic physiographic classes) was off by -0.58% (RI), -0.70% (NC),
-7.36% (CO), and -5.16% (OR) -- present in every state, including RI/NC/CO which have no macroplot
conditions at all, so unrelated to fix #3.

**Root cause**: `vitalRatesStarter.R`'s comprehensive tree-domain indicator
(`tDI = landD.prev * aD.prev * tD.prev * typeD.prev * sp.prev * tChng`, ~line 284) used `aD.prev` --
the user's `areaDomain` expression evaluated against the tree's *previous*-measurement condition --
while the area list's own indicator (`aDI = landD * landD.prev * aD * sp * aChng`, in `aData`) already
correctly used `aD` (current condition) alone. EVALIDator's `strFilter` mechanism restricts both the
numerator and denominator using the *current* condition consistently (confirmed by the `tpa()`/
`volume()` methodology, and by this fix's exact-match result below). Since a plot's `PHYSCLCD` rarely
(but sometimes genuinely does) change between measurements, this produced a small-but-real,
state-dependent mismatch that scaled with each state's rate of physiographic-class turnover between
remeasurements (much higher in CO/OR than RI/NC).

**Fix**: changed `tDI`'s `aD.prev` to `aD` (one-token change, ~line 284), matching the area-side
`aDI`'s existing (correct) use of the current condition.

**Verification**: `areaDomain = PHYSCLCD %in% 21:29` now matches EVALIDator to full double precision,
both point estimate and `nPlots_AREA`, in all four states (RI, NC, CO, OR). The core default case,
`landType`/`treeType` variants, and the `treeDomain = SPCD == 129` case were re-checked and remain
exact -- this fix only affects the numerator side of `areaDomain`-restricted calls. Full package test
suite re-run with no regressions.

### 5. `SAWVOL_GROW`/`SAWVOL_GROW_AC` used the wrong growth-accounting component

`treeType = 'gs'` `SAWVOL_GROW_AC` was off in NC (-0.44%), CO (+3.0%), and OR (-0.34%); RI happened to
match by coincidence (small state, few trees near the sawtimber-size threshold during the measurement
window).

**Root cause**: `vitalRatesStarter.R` computes all five growth metrics (`DIA_GROW`, `BA_GROW`,
`NETVOL_GROW`, `SAWVOL_GROW`, `BIO_GROW`) from the same growth-accounting component
(`TREE_GRM_COMPONENT`'s `..._GS_FOREST`/`..._GS_TIMBER` columns for `treeType = 'gs'`, via
`typeDomain_grow()`). But EVALIDator's sawlog-volume growth attributes are defined specifically for
**sawtimber trees**, a size-based subset of growing-stock, with their own dedicated component
(`..._SL_FOREST`/`..._SL_TIMBER`). `growMort()` already handles this correctly -- it always uses the
`SL_FOREST`/`SL_TIMBER` component for its `SAWVOL`/`SAWVOL_BF` state variables, independent of
`treeType`; `vitalRatesStarter.R` had no equivalent, since it had never previously needed to swap
components mid-calculation. Confirmed empirically: temporarily renaming from `SL_FOREST` instead of
`GS_FOREST` (diagnostic only) made `SAWVOL_GROW_AC` match EVALIDator's attribute 203 exactly in all
four states, while `NETVOL_GROW_AC`/`BIO_GROW_AC` broke as expected (confirming those two need to stay
on the growing-stock component).

**Fix**: `typeDomain_grow()` (`R/util.R`, `'vr'` type) now unconditionally also renames the
sawtimber-specific columns (`TPAGROW_UNADJ_SAW`, `SUBPTYP_GRM_SAW`, `COMPONENT_SAW`), alongside
whichever `treeType` component was selected for the other four metrics. `vitalRatesStarter.R` carries
these through the pipeline and uses `COMPONENT_SAW` (not `COMPONENT`) for the `VOLBFNET2`/`VOLBFNET1`
`vrAttHelper()` calls. Because sawtimber-sized trees can be tallied on a different subplot/macroplot
basis than the broader growing-stock set, `SAWVOL_GROW`/`SAWVOL_GROW_AC` are computed via a parallel
`t_saw`/`sumToPlot()` pass (its own `TREE_BASIS` derived from `SUBPTYP_GRM_SAW`), then `left_join`ed
back into the general `tPlt` before `sumToEU()` -- `sumToPlot()`'s output collapses `TREE_BASIS` away
after applying the adjustment factor, so the two objects are compatible to join on `(ESTN_UNIT_CN,
STRATUM_CN, PLT_CN, grpBy)` with no changes needed to `sumToEU()` or the existing tree-total-CV
("ttEst") pass. The `byPlot = TRUE` branch (no adjustment factors) needed only a one-line change
(`TPAGROW_UNADJ` -> `TPAGROW_UNADJ_SAW` in its `svol` sum); the `treeList = TRUE` branch gets
`SAWVOL_GROW` via a `left_join` from the same `t_saw` object.

**A test-script bug found along the way**: the timberland `BIO_GROW_AC` (attr 318) comparison came
back 3-5x off in every state while investigating this fix -- large enough to look like a second real
bug. It turned out to be a mistake in this pass's own test/diagnostic scripts: attribute 318 is
*belowground* biomass net growth on *forest land*, not *aboveground* biomass on *timberland* (that's
attribute 315) -- an easy mixup since both numbers sit in the same attribute-ID neighborhood
(`EVALIDATOR_POP_ESTIMATE.csv`'s 311-322 block alternates aboveground/belowground and
forest/timberland across growing-stock and sawtimber variants). Corrected in both
`test-vitalRates.R` and the attribute mapping table above; no `vitalRates()` code was affected by this
particular finding.

**Verification**: `SAWVOL_GROW_AC` now matches EVALIDator's attributes 203 (forest, exact in all four
states) and 209 (timber, exact in RI/NC/CO; OR shows only the pre-existing Known Issue A residual).
`NETVOL_GROW_AC`/`BIO_GROW_AC` (gs) remain exact (regression-checked before and after). Internal
consistency (`SAWVOL_TOTAL / AREA_TOTAL == SAWVOL_GROW_AC`, `SAWVOL_TOTAL / TREE_TOTAL ==
SAWVOL_GROW`) holds exactly. `byPlot = TRUE` and `treeList = TRUE` both run without error and produce
sane values (`treeList`'s `SAWVOL_GROW` is legitimately `NA` for non-sawtimber-sized trees, matching
how `VOLBFNET` itself is undefined below the sawtimber threshold in raw FIADB data -- not a bug). Full
package test suite re-run with no regressions.

## Known issues and intentional divergences from EVALIDator

### A. `landType = 'timber'` over-counts `nPlots_AREA` by a small margin in macroplot-heavy states -- root-caused, kept intentionally (not a bug)

After fix #3 above made `landType = 'forest'` match EVALIDator exactly in OR/CA/WA, `landType =
'timber'` in those same three states -- which matched EVALIDator exactly *before* fix #3 -- now
over-counts: OR 7788 vs. EVALIDator's 7749 (+39), CA 2580 vs. 2562 (+18), WA 4899 vs. 4870 (+29),
roughly 0.5-0.7% in each case. (RI/NC/CO timber is unaffected and still exact, since they have no
`MACR`-basis conditions to begin with.)

This was investigated in some depth without a confirmed resolution:

- A **fully independent, from-scratch replication** of the qualifying-plot count, written directly
  against the raw `PLOT`/`COND`/`SUBP_COND_CHNG_MTRX` CSVs (no `rFIA` code involved) using the same
  documented rule (current + previous condition both forest, both timber-eligible by
  `SITECLCD`/`RESERVCD`, `SUBPTYP`/`PROP_BASIS` matched per the guide) reproduced rFIA's post-fix
  count (7788) exactly, not EVALIDator's (7749) -- meaning this isn't a join-implementation bug
  specific to `vitalRatesStarter.R`'s code; two independent implementations of the documented rule
  agree with each other and disagree with EVALIDator.
- Ruled out: the guide's `COND_NONSAMPLE_REASN_CD` exclusion (already included in fix #3; no effect
  here). Ruled out: requiring `PROP_BASIS` to match between current and previous period (it already
  always does, for these rows). Ruled out: requiring `SITECLCD` to be stable between periods
  (overcorrects drastically -- drops the count to 6056, far below EVALIDator's 7749 -- so ordinary
  site-class reclassification between remeasurements is not the discriminating factor).
- Not yet tried (at the time): obtaining EVALIDator's actual SQL for the `EXPGROW`-type
  timberland-denominator attribute specifically (the guide's worked example is for the simpler
  `EXPCHNG` area-change eval type, which may not be identical for growth attributes), or documentation
  more specific to macroplot-basis timberland classification during growth accounting.

Point estimates for `landType = 'timber'` in these three states are correspondingly off by a similar
small margin (e.g. OR: rFIA `1.367651` vs. EVALIDator `1.366364`, +0.09%). Known Issues B (`areaDomain`)
and C (`SAWVOL_GROW_AC`, `gs`) from earlier in this pass were root-caused and fixed (see "Fixed" #4 and
#5 above); Known Issue A was root-caused in a follow-up session (below) but deliberately left unfixed.

#### Resolution (follow-up session, 2026-08-11): root-caused via EVALIDator's actual generated SQL; kept as an intentional divergence

The "not yet tried" step above was completed: rather than relying on the static attribute metadata in
`EVALIDATOR_POP_ESTIMATE.csv` (which has no row for a timberland growth-accounting *denominator* at
all -- ratio denominators are generated dynamically by EVALIDator, not stored as their own attribute),
the FIADB-API `fullreport` endpoint's response includes a `metadata.denSql`/`metadata.numSql` field
containing the *actual* SQL EVALIDator ran for that specific query. Pulling this for OR EVALID 412203
(`fetch_evalidator.R`-style query, `snum=2636` (timberland biomass growth) `sdenom=3` vs. `snum=2635`
(forest biomass growth) `sdenom=2`) shows EVALIDator itself treats the two land bases inconsistently:

- **Forest-land denominator** (`sdenom=2`) restricts the `SUBP_COND_CHNG_MTRX` (SCCM) join with
  `((SCCM.SUBPTYP = 3 AND COND.PROP_BASIS = 'MACR') OR (SCCM.SUBPTYP = 1 AND COND.PROP_BASIS =
  'SUBP'))` -- the dual-branch match this pass's fix #3 implements, taken directly from the guide's
  Ch. 7.8 Example 7-12 (`SUBP_COND_CHNG_MTRX` worked example, itself an `EXPCHNG`/forest-land example).
- **Timberland denominator** (`sdenom=3`) has **no such branch**: it hardcodes `WHERE SCCM.SUBPTYP=1`
  unconditionally (plus the `RESERVCD`/`SITECLCD` timberland-productivity filters), while its `SELECT`
  clause still multiplies by `CASE COND.PROP_BASIS WHEN 'MACR' THEN POP_STRATUM.ADJ_FACTOR_MACR ELSE
  POP_STRATUM.ADJ_FACTOR_SUBP END` -- i.e. for a `MACR`-basis condition, it applies the *macroplot*
  adjustment factor to a *subplot*-level (`SUBPTYP=1`) SCCM proportion, a combination that doesn't
  correspond to any physically meaningful quantity. It never even reads the `SUBPTYP=3` (macroplot)
  SCCM rows for timberland.

This is the direct, confirmed cause of the over-count: rFIA's fix #3 (correctly) reads macroplot-basis
SCCM rows for both land bases per the guide's documented rule, while EVALIDator's own timberland
template only ever reads the subplot-basis rows, silently dropping macroplot-basis timberland area
change. Exhaustively searching the FIA Population Estimation User Guide (`core_references/
fia_pop_estimation_user_guide.pdf`, ch. 7) turns up no worked example of a timberland or `EXPGROW`
growth-accounting denominator at all -- the guide's only SCCM dual-branch worked example (Ch. 7.8,
Example 7-12) is for forest land under `EXPCHNG`. There is no documented, principled reason given
anywhere for excluding macroplot-basis timberland conditions from growth-accounting area; combined with
the internally mismatched `SUBPTYP`/`ADJ_FACTOR` pairing found above, this looks like an omission in
EVALIDator's own timberland-ratio query template (the dual-branch fix applied to the forest-land
template apparently was never carried over to the timberland one), not a deliberate methodological
choice.

**Decision (user sign-off, 2026-08-11): do not change `vitalRatesStarter.R` to replicate this.** rFIA's
current `aChng` logic (`(SUBPTYP == 1 & PROP_BASIS == 'SUBP') | (SUBPTYP == 3 & PROP_BASIS == 'MACR')`,
applied uniformly regardless of `landType`) already implements the guide's own documented rule
correctly and consistently for both land bases -- the more statistically defensible behavior, since it
doesn't silently discard sampled macroplot area the way EVALIDator's timberland template does. Matching
EVALIDator's timberland output exactly here would mean deliberately reintroducing the same
class of bug fix #3 fixed for forest land, scoped narrowly to `landType = 'timber'`. This ~0.5-0.7%
residual in OR/CA/WA `landType = 'timber'` is therefore an intentional, understood divergence from
EVALIDator, not an open bug -- no further action planned. The identical shared-code pattern in
`growMortStarter.R` (see `growMort.md`) is covered by the same decision.

## Notes

### `BA_GROW_AC` vs. `BAA_GROW` column-naming inconsistency (not a numeric bug, not fixed this pass)

The population-level (non-`byPlot`) output names the per-acre basal-area-growth column
`BA_GROW_AC`, but `byPlot = TRUE` output names the identical quantity `BAA_GROW` (see
`vitalRatesStarter.R` line ~393), and `man/vitalRates.Rd`'s Value section documents it as `BAA_GROW`
-- matching the `byPlot` naming, not the population-level naming actually produced by the default
call. This is a pure naming/documentation inconsistency (confirmed the underlying values themselves
are correct and internally consistent in both branches) -- flagged here rather than fixed, since it's
a naming/API-surface decision (which name is "correct," and whether changing either is a breaking
change for existing users) rather than a numeric correctness issue, and outside this pass's
bug-handling protocol (which covers numeric mismatches).

## Deferred to follow-up (not covered this pass)

- ~~Known Issue A above (`landType = 'timber'` `nPlots_AREA` over-count in macroplot-heavy states) is
  under active investigation but not yet resolved as of this writing.~~ Root-caused in a follow-up
  session and kept as an intentional divergence from EVALIDator, not fixed -- see "Known issues and
  intentional divergences from EVALIDator" above.
- Whether a `DIA`-based `treeDomain` filter matches EVALIDator despite the `tD.prev`-vs-current-`TREE`
  timing difference noted above -- a species-based filter was used instead for the numeric
  cross-check tests, matching `tpa.md`'s fallback for filters not meaningful nationally.
- `byPlot = TRUE` aggregation reproducing the population-level estimate (only totals-vs-per-acre/
  per-stem internal consistency was checked, same as every prior pass).
- `method` options other than `'TI'` (EVALIDator has no equivalent; internal-consistency-only checks
  per the plan, not yet added).
- `bySizeClass` was only checked structurally (pre-existing `test-vitalRates.R` coverage), not against
  an EVALIDator size-class breakdown.
- The `BA_GROW_AC`/`BAA_GROW` naming inconsistency noted above -- needs a decision on which name is
  canonical before any fix.
- The identical `SUBPTYP == 1` hardcoding in `growMortStarter.R` (see Fixed #3's "shared-risk note")
  -- left untouched, since `growMort()` has its own scheduled validation pass.
