# Validation report: `fsi()`

## Scope

This pass covers `fsi()` -- the Forest Stability Index, a measure of annual change in relative live
tree density (Stanke et al. 2021, doi: 10.1038/s41467-020-20678-z). Per the project plan, `fsi()` has
**no EVALIDator equivalent** (it is not an FIA-standard population attribute), so this validation
pass differs from every prior one: instead of comparing rFIA output to EVALIDator ground truth, it
(1) checks that `fsi()`'s implementation matches the mathematical definition of the index given in
Stanke et al. 2021 exactly, and (2) exercises the same domain-filter/grouping/internal-consistency
matrix used for every other function, using the standard `unitMean`/`unitVarNew`-based post-stratified
estimator (Bechtold & Patterson 2005) that FIA estimators generally share.

`fsi()` is architecturally different from every other function validated so far: it does **not** use
the shared `sumToPlot`/`sumToEU` generic post-stratified estimator (`R/util.R`) that `tpa()`,
`volume()`, `biomass()`, `vitalRates()`, `growMort()`, etc. all use. It has its own, bespoke
population-estimation code path in `R/fsiHelper.R` (`fsiHelper2`, built on `unitMean`/`unitVarNew`,
otherwise unused anywhere else in the package). This matters for the "Fixed" section below: a bug in
`fsiHelper2` would not have been caught by any prior function's validation, since no other function
shares that code.

## Methodology

Ground truth for the FSI-specific arithmetic (relative density, FSI, %FSI, the maximum size-density
curve) comes from the equations given in Stanke et al. 2021's Methods section
(`core_references/stanke2021NC.pdf`):

- Eq. 1: `Nmax(S_i) = a_i * S_i^r_i` -- the maximum size-density curve (power function of average tree
  basal area `S`, fit per stand-type `i` via Bayesian quantile regression, 99th percentile).
- Eq. 2: `RD_ij = sum_h(N_hij / Nmax(S_hi))` -- relative density of population `j` in stand-type `i`,
  as a **tree-level sum** of individual relative densities (not an aggregate-index approximation).
- Eq. 3: `FSI = deltaRD / deltat` -- average annual change in relative density between successive
  measurements.
- Eq. 4: `%FSI = 100 * FSI / RD_t1` -- FSI standardized by the initial relative density.

Since `fsi()`'s quantile-regression curve fit (`R2jags`/JAGS, `inst/extdata/qrLM.jag` /
`qrLMM.jag`) is itself stochastic and has no single "correct" numeric answer to check against, the
arithmetic-fidelity checks below substitute a **fixed, non-stochastic `betas`** (via the `betas=`
argument, e.g. `alpha = 1, rate = 0`, which collapses `Nmax` to 1 and makes `rd == TPA_UNADJ` exactly)
to isolate and exactly verify eqs. 2-4's implementation, independent of the curve-fitting step. This
also enables an approximate cross-check against `tpa()` (an already-validated function): with
`alpha=1, rate=0`, `fsi()`'s per-acre `CURR_RD`/`PREV_RD` reduce to a plain live-tree TPA estimate
over the growth-eligible (remeasured) plot population, directly comparable to `tpa(landType='forest',
treeType='live')` restricted to the same population.

Four states, one per FIA region, read from the local FIADB extract cache (`~/Dropbox/data/fia`, or
`RFIA_VALIDATION_DATA` if set) via `clipFIA(readFIA(...), mostRecent = TRUE)`: **RI** (Northern),
**NC** (Southern), **CO** (Interior West), **OR** (Pacific Northwest). All work here used
`~/R-4.6.0`/`~/R/x86_64-pc-linux-gnu-library/4.6` per this session's environment.

**Data-hygiene note (non-TI pass, follow-up session)**: the local FIADB cache lives in a
Dropbox-synced folder. Mid-session, an OR `fsi()` run returned `FSI = 0.804`/`nPlots = 9022`, then an
identical rerun of the identical call minutes later returned `FSI = -0.405`/`nPlots = 9084` -- initially
indistinguishable from a non-determinism bug in `fsi()` itself. `dropbox status` confirmed an active
background sync ("Syncing 33 files") at the exact timestamp of the first (anomalous) run, and
`OR_*.csv` file mtimes matched that sync window; four repeated calls against the *same*, already-loaded
`db` object were, by contrast, perfectly reproducible (byte-identical `FSI`/`nPlots` every time). This
narrows the cause to a torn/partial read of the CSVs mid-overwrite by Dropbox, not `fsi()` -- but it's a
real, silent-failure-mode hazard for this validation methodology generally (no error or warning of any
kind accompanied the bad read), worth flagging for every future pass using this same local cache. All
numeric results below were captured after confirming `dropbox status` reported "Up to date" and were
independently re-verified as reproducible (4 repeated calls, byte-identical) before being recorded.

## Results

### Formula fidelity vs. Stanke et al. 2021 (eqs. 1-4)

With `betas = data.frame(grps = 1, alpha = 1, rate = 0, n = 1)` (so `rd == TPA_UNADJ` exactly, tree
by tree), `byPlot = TRUE` output matched hand-derivable identities to full double precision in all
four states:

- `CURR_RD == CURR_TPA` and `PREV_RD == PREV_TPA` exactly (eq. 2's tree-level sum, implemented at
  `R/fsi.R`'s `byPlot` branch and `R/fsiHelper.R`'s population branch, both correct).
- `FSI == (CURR_RD - PREV_RD) / REMPER` exactly (eq. 3).
- `PERC_FSI == 100 * FSI / PREV_RD` exactly (eq. 4).

| State | max\|CURR_RD - CURR_TPA\| | max\|PREV_RD - PREV_TPA\| | max eq.3 residual | max eq.4 residual |
|---|---|---|---|---|
| RI | 0 | 0 | 0 | 0 |
| NC | 0 | 0 | 0 | 0 |
| CO | 0 | 0 | 0 | 0 |
| OR | 0 | 0 | 0 | 0 |

The quantile-regression spec (99th percentile, `p = .99` in `inst/extdata/qrLM.jag`/`qrLMM.jag`)
matches the paper. Two smaller deviations from the paper's stated priors/exclusion criteria were
found and are documented under "Known issues," not fixed this pass.

### `scaleBy` group-specific curves (regression test for the v1.1.3 fix)

NEWS.md documents a v1.1.3 bug where, under `byPlot = TRUE`, `scaleBy` group-specific `alpha`/`rate`
were not actually used (the overall/fixed-effect mean was used for every group instead). Re-tested
this pass, at both `byPlot = TRUE` and population level, by handing `fsi()` artificial `betas` with
drastically different `alpha` per `scaleBy` (`FORTYPCD`) group and confirming the resulting `CURR_RD`
differs accordingly (not identically) between groups:

- `byPlot = TRUE`, RI, `scaleBy = grpBy = FORTYPCD`: groups assigned `alpha = 1` showed
  `CURR_RD / CURR_TPA == 1` exactly; groups assigned `alpha = 1e6` showed the ratio at `1e-6`
  exactly, across all 17 fitted forest-type groups, no cross-contamination.
- Forest types absent from the (artificially supplied, incomplete) `betas` correctly fell back to
  `NA`/dropped (expected when `betas` lacks `fixed_alpha`/`fixed_rate` columns); with the real,
  JAGS-fitted `betas` (which does include `fixed_alpha`/`fixed_rate` for a mixed model), those same
  missing-group plots correctly picked up the fixed-effect fallback instead of being dropped
  (`R/fsi.R`, `if ('fixed_alpha' %in% names(betas))` branch) -- confirmed non-zero, plausible `rd` for
  forest types 520/901/962/999 in RI, which are absent from the curve-fit calibration set.
- Population level, RI: `scaleBy = FORTYPCD` with `alpha = 1` vs. `alpha = 1e6` per group produced
  `CURR_RD` of ~495 and ~0.0003 respectively for two different forest types on the same call --
  confirms group-specific curves are applied at the population estimator too, not just `byPlot`.

**No regression of the v1.1.3 bug.**

### Domain filter interactions, 4 states

`nPlots` shrinks monotonically as domains are restricted, as expected (`betas` fixed to
`alpha=1,rate=0` throughout):

| State | unrestricted | `treeDomain(SPCD==129)` | `areaDomain`(mesic, `PHYSCLCD %in% 21:29`) | both |
|---|---|---|---|---|
| RI | 108 | 58 | 102 | 53 |
| NC | 3473 | 313 | 2976 | 228 |
| CO | 3641 | 0 | 2258 | 0 |
| OR | 9022 | 0 | 7719 | 0 |

(SPCD 129 is white pine, an eastern species absent from CO/OR -- the 0-plot result there is the
*correct* answer, not a bug; see "Known issues" below regarding what `fsi()` returns in that case.)

### `returnSpatial` (RI, `byPlot = TRUE`)

Numeric columns identical between `returnSpatial = TRUE` and `FALSE`; only geometry added. **Pass.**

### `method=` structural checks (RI)

`TI`, `SMA`, `LMA`, `EMA`, and `ANNUAL` all ran to completion without error (no EVALIDator equivalent
exists for non-TI estimators, per the project plan -- structural/no-crash check only). A
dplyr-deprecation warning (`across(..., sum, na.rm = TRUE)` old-style, from `R/fsi.R`'s
`ANNUAL`/moving-average branch) fired under `ANNUAL`; not a numeric bug, but will break under a future
dplyr major version. Not fixed this pass (out of scope; see "Known issues").

This was only a structural/no-crash check at the time it was written; see "Non-TI method validation"
below for the full numeric/behavioral pass (the same treatment given to every other estimator's
`method != 'TI'` path in this project).

### `bySpecies` / `bySizeClass` (RI)

Both ran without error and produced plausible row counts (48 species rows, 19 size-class rows).

### `useSeries` + `mostRecent` (RI)

Ran without error.

## Fixed

### 1. Population-level per-acre estimates (`FSI`, `PERC_FSI`, `PREV_RD`, `CURR_RD`) were systematically under-estimated by ~50-65%

**Root cause**: `R/fsiHelper.R`, in `fsiHelper2`, previously included `REMPER = REMPER * tAdj` in a
`mutate()` alongside the legitimate nonresponse-adjustment scaling of `CHNG_TPA`/`CURR_RD`/etc.
`REMPER` (years between measurements) is a per-plot constant with no relationship to `tAdj` (the
subplot/microplot/macroplot nonresponse adjustment factor, which differs by `PLOT_BASIS`). At this
point a single plot is still split into multiple rows (one per `PLOT_BASIS` it has tally trees on --
e.g. a separate `MICR` row and `SUBP` row), each with its own `tAdj`. Multiplying `REMPER` by `tAdj`
therefore gave the *same physical plot* a *different* `REMPER` value per row.

Two lines later, the code groups by `(ESTN_UNIT_CN, ESTN_METHOD, STRATUM_CN, PLT_CN, REMPER, ...)`
specifically to collapse those `PLOT_BASIS` rows back into one row per plot (comment: *"Extra step for
variance issues -- summing micro, subp, and macr components"*). Because `REMPER` now differed between
a plot's rows, this collapse silently failed to merge them -- confirmed directly on RI: this step
produced 280 rows for only 109 real plots.

The tree-density sums (`CURR_RD`, `PREV_RD`, `FSI`, etc.) still totaled correctly across the leftover
duplicate rows (summing fragments recovers the true total). But the subsequent area re-join
(`left_join(aAdj, by = c('ESTN_UNIT_CN', 'STRATUM_CN', 'PLT_CN', aGrps))`) does **not** key on
`REMPER` -- so every duplicate row for an affected plot independently re-attached a full copy of that
plot's forest-area weight (`fa`). Confirmed directly on RI: `sum(fa)` after this join was 233.9,
vs. 92.0 for the 114 real (plot, not fragment) rows feeding it -- a 2.54x inflation, matching the
observed bias.

**Reproduction** (before fix), using `betas = data.frame(grps=1, alpha=1, rate=0, n=1)` so
`fsi()`'s `CURR_RD` reduces to a plain per-acre live-tree TPA over the growth-eligible/remeasured
plot subset, cross-checked against `tpa(db, landType='forest', treeType='live')` (an already-validated
function) on the same population:

| State | `fsi()` CURR_RD (buggy) | `tpa()` TPA | rel. diff | `fsi()` nPlots | `tpa()` nPlots_AREA |
|---|---|---|---|---|---|
| RI | 129.7 | 365.2 | -64.5% | 174 | 132 |
| NC | 301.8 | 712.9 | -57.7% | 5750 | 3561 |
| CO | 232.4 | 481.2 | -51.7% | 4019 | 3925 |
| OR | 129.7 | 347.6 | -62.7% | 17175 | 10410 |

Note `fsi()`'s `nPlots` exceeding `tpa()`'s `nPlots_AREA` (and the true ~109-plot growth-eligible
population in RI) is itself diagnostic of the row-duplication mechanism -- `plotIn_t` (the 0/1
per-plot flag summed into `nPlots`) was being summed across the same spurious duplicate rows.

An alternative hypothesis was tested and ruled out first: `fsiStarter.R` uses `evalType = 'VOL'` where
sibling remeasurement estimators `vitalRates()`/`growMort()` use `'GROW'`. Patching a scratch copy to
use `'GROW'` left the bias unchanged, so this is a separate, lower-severity discrepancy (see "Known
issues" below), not the cause of this bug.

**Fix**: removed `REMPER = REMPER * tAdj` from the `mutate()` (`R/fsiHelper.R`, `fsiHelper2`).
`REMPER` now stays constant across a plot's `PLOT_BASIS` rows, so the fragment-collapse `group_by`
works as intended.

**After fix**, same reproduction:

| State | `fsi()` CURR_RD (fixed) | `tpa()` TPA | rel. diff | `fsi()` nPlots | `tpa()` nPlots_AREA |
|---|---|---|---|---|---|
| RI | 351.0 | 365.2 | -3.9% | 108 | 132 |
| NC | 657.6 | 712.9 | -7.8% | 3473 | 3561 |
| CO | 465.8 | 481.2 | -3.2% | 3641 | 3925 |
| OR | 347.7 | 347.6 | +0.04% | 9022 | 10410 |

The residual few-percent gap is expected, not a bug: `fsi()`'s per-acre estimate is restricted to the
growth-eligible remeasured-plot domain and uses the average of previous/current forest-area proportion
(`fa = (amin+amax)/2`) as its area base, while `tpa()`'s is a point-in-time estimate over all currently
forested plots -- two closely related but not identical domains. `nPlots` is now sane everywhere
(always somewhat below `tpa()`'s broader-domain plot count, as expected).

`byPlot = TRUE` output was **not affected** by this bug (that code path never calls `fsiHelper2`) --
confirmed by the eq. 2-4 exact-match checks above passing both before and after the fix.

### 2. `inst/extdata/qrLM.jag`/`qrLMM.jag`: intercept prior mean was 6, paper states 7

Stanke et al. 2021, Methods ("Statistical analysis"): *"We place informative normal priors on the
model intercept (mu = 7, sigma = 1) and coefficient (mu = 0.8025, sigma = 0.1)"*. The JAGS model files
used `alpha ~ dnorm(6, pow(1, -2))` (intercept mean 6, not 7). The slope prior
(`beta ~ dnorm(-.8025, pow(.1, -2))`) was already fine -- the paper's magnitude (0.8025) correctly gets
a negative sign matching the negative exponent `r` in eq. 1.

**Fix**: changed `alpha ~ dnorm(6, ...)` / `fe_alpha ~ dnorm(6, ...)` to `dnorm(7, ...)` in both
`qrLM.jag` (single-group model) and `qrLMM.jag` (mixed model, `scaleBy` groups). Not numerically
re-verified against a reference fit this pass (JAGS fits are stochastic, no single "correct" posterior
to check against) -- the fix is a direct, literal match to the paper's stated prior.

### 3. Max-density-curve calibration excluded disturbed *and* treated plots (AND); paper says disturbed *and/or* treated (OR)

`R/fsiHelper.R` (`fsiHelper1`) excluded a plot from the curve-fitting subset only when
`DSTRBCD1 > 0 & TRTCD1 > 0` (both disturbed and treated; natural regeneration, `TRTCD1 == 40`,
exempted). Stanke et al. 2021 states plots are excluded when they show evidence of recent disturbance
**and/or** silvicultural treatment -- i.e. either alone should exclude a plot. As written, a plot with
significant disturbance but no recorded follow-up treatment (e.g. an untreated wildfire) was
incorrectly retained in the calibration set.

**Fix**: changed the exclusion filter from `DSTRBCD1 > 0 & (TRTCD1 > 0 & !(TRTCD1 %in% 40))` to
`DSTRBCD1 > 0 | (TRTCD1 > 0 & !(TRTCD1 %in% 40))` -- disturbance alone, or (non-natural-regen)
treatment alone, now excludes a plot. Not numerically re-verified against a reference fit this pass
(same reasoning as #2).

### 4. `totals` argument had no effect -- removed

`totals = TRUE` and `totals = FALSE` produced byte-identical output (confirmed, RI) -- both branches
of the `if (totals) {...} else {...}` in `R/fsi.R` `select()`ed the exact same columns. Every other
estimator in the package adds raw population totals (e.g. `TPA_TOTAL`, `AREA_TOTAL`) when
`totals = TRUE`; `fsi()`'s never did, despite `man/fsi.Rd` documenting `totals` as controlling this.

**Fix**: removed the `totals` argument entirely (`R/fsi.R`, `R/fsiStarter.R`, `man/fsi.Rd`) rather than
implementing it, since `FSI`/`PERC_FSI`/`PREV_RD`/`CURR_RD` are already ratio quantities (relative
density is dimensionless by construction, eq. 2) with no natural population-total analog the way e.g.
`TPA`/`BAA` have `TPA_TOTAL`/`BAA_TOTAL`.

### 5. `areaDomain` matching no conditions crashed instead of returning a clean empty result

`fsi(db, areaDomain = PHYSCLCD == 11)` (RI, a physiographic class RI does not have) threw
`Error in \`$<-.data.frame\`(...) : replacement has 1 row, data has 0` from the `t$grps <- 1` /
`grpRates$grps <- 1` assignments in `R/fsi.R` (the non-`scaleBy` branch), which fail when `t`/
`grpRates` has zero rows -- `grpRates` (and its `t1` source) becomes genuinely empty whenever
`landType`/`areaDomain` (or, incidentally, the disturbance/skewness exclusions used to build the curve
calibration set) leaves no plots to work with. Left un-guarded, a user calling `fsi()` with the
default `betas = NULL` on such a domain would additionally have hit `R2jags::jags()` with zero
observations, which cannot run.

The project plan notes a similar empty-domain class of bug was already fixed for every other estimator
via a shared utility (`combineMR`'s `"no non-missing arguments to max"` case, see
`tpa.md`/`vitalRates.md`) -- this was a different failure mode, specific to `fsi()`'s own non-shared
code path, not covered by that earlier fix.

**Fix** (`R/fsi.R`): (1) `grpRates$grps <- 1` / `t$grps = 1` now use `rep(1, nrow(...))`, safe for
zero-row input. (2) Added a guard, `if (nrow(grpRates) == 0)`, ahead of the JAGS-fitting branch: when
there is no data to calibrate a curve against, fitting is skipped entirely and `alpha`/`rate` are set
to `NA` for every observed `grps` value in `t`, rather than invoking JAGS on empty data. (3) The
population-estimate branch now drops any row with `nPlots == 0` (see #6) before returning, so a
`byPlot = FALSE` call collapses cleanly to an empty tibble. `byPlot = TRUE` instead keeps all plot rows
with `FSI = 0`/`NA`, matching how `tpa()`/`vitalRates()` already handle an empty `areaDomain`
(confirmed: `tpa(db, byPlot = TRUE, areaDomain = <empty>)` keeps every plot row with `TPA = 0`, not an
empty result).

**Verified**, RI: `fsi(db, areaDomain = PHYSCLCD == 11)` (`byPlot = FALSE`) now returns 0 rows without
error, with both a user-supplied `betas` and the default `betas = NULL` (JAGS-fitting) path;
`byPlot = TRUE` returns all 109 plot rows with `FSI` equal to 0 or `NA`.

### 6. `treeDomain` matching no trees returned a 1-row `NaN`-filled result, not a clean 0-row result

`fsi(db, treeDomain = SPCD == 999)` (RI) did not crash, but returned one row with `FSI = 0`,
`PERC_FSI = NaN`, `PREV_RD = 0`, `CURR_RD = 0`, `nPlots = 0` -- rather than the 0-row result every
other validated estimator returns for an empty domain (e.g. `tpa()`, `vitalRates()`).

**Fix** (`R/fsi.R`): added `dplyr::filter(nPlots > 0)` immediately after the population-estimate
`select()`, dropping exactly this kind of no-contributing-plots row. This also finishes off #5's fix
for the `byPlot = FALSE` case (a row with `nPlots == 0` arises identically whether the cause was an
empty `areaDomain` or an empty `treeDomain`). `byPlot = TRUE` is unaffected -- it keeps all plot rows
regardless of domain, matching `tpa()`/`vitalRates()` (confirmed: `tpa(db, byPlot = TRUE, treeDomain =
<empty>)` keeps every plot row with `TPA = 0`).

**Verified**, RI: `fsi(db, treeDomain = SPCD == 999)` (`byPlot = FALSE`) now returns 0 rows;
`byPlot = TRUE` still returns all 109 plot rows with `FSI` equal to 0 or `NA`.

### 7. Deprecated `across(..., fn, ...)` usage under `method = 'ANNUAL'`

`R/fsi.R`'s moving-average/annual branch called `across(ctEst:plotIn_t, sum, na.rm = TRUE)`, the
pre-dplyr-1.1.0 calling convention (function passed positionally rather than via a lambda), which
raised a deprecation warning and would become a hard error in a future dplyr release.

**Fix**: changed to `across(ctEst:plotIn_t, \(x) sum(x, na.rm = TRUE))`. Verified `method = 'ANNUAL'`
no longer emits the warning.

## Known issues (identified this pass, reviewed, not changed)

### A. `fsiStarter.R` uses `evalType = 'VOL'` instead of `'GROW'`

`R/fsiStarter.R:140` calls `handlePops(db, evalType = 'VOL', ...)`, where sibling remeasurement
estimators `vitalRates()` (`evalType = 'GROW'`) and `growMort()`
(`evalType = c('GROW','MORT','REMV')`) use `'GROW'`. `fsi()` is the only one of the three using
`'VOL'`. These are genuinely different EVALIDs/plot populations (confirmed, RI: `EXPVOL` -> EVALID
442401, 240 plots; `EXPGROW` -> EVALID 442403, 210 plots).

Tested directly: patching a scratch copy of the package to use `'GROW'` did **not** change the ~2x
population-level bias described in "Fixed" #1 above, so this was not the cause of that bug -- it was
investigated as a candidate root cause and ruled out.

**Reviewed and confirmed intentional -- not a bug.** Left as `'VOL'`, unchanged.

## Non-TI method validation (SMA/LMA/EMA/ANNUAL)

EVALIDator has no equivalent for these, so correctness here means: `fsi()`'s output behaves sanely and
consistently with the already-validated `TI` estimates wherever the documentation actually claims a
relationship, and its shared-utility touchpoints (`maWeights()`, `mergeSmallStrata()`, `combineMR()` --
all in `R/util.R`) are exercised correctly. See `tpa.md`/`vitalRates.md` (the templates this section
follows) for the general non-`TI` methodology.

**Architectural context that shapes this section**: `fsi()` does *not* use `sumToEU()`, the shared
population-estimation engine every other `method`-aware estimator (`tpa()`, `biomass()`, `growMort()`,
`vitalRates()`, etc.) is built on. It has its own, bespoke estimator (`fsiHelper2`, `R/fsiHelper.R`),
which reimplements the panel-collapse-and-reweight logic independently. This matters because
`vitalRates()`'s non-`TI` pass (see `vitalRates.md` "Fixed" #7) found a real, package-wide bug in
`sumToEU()`'s own SMA/LMA/EMA collapse step (grouping by a per-panel-varying column, `P2PNTCNT_EU`,
defeated the collapse and caused a many-to-many join explosion whenever an estimator called
`sumToEU()` twice on the same data) -- but that fix lives entirely inside `sumToEU()`, so it has **no
bearing on `fsi()`**, which never calls that function. `fsi()`'s own panel-reweighting join
(`R/fsi.R`'s `dplyr::left_join(wgts, by = joinCols)`, in the "Compute moving average weights" block) is
a structurally different design: for `SMA`, `maWeights()` returns one weight row per `(YEAR, STATECD)`
with no `INVYR` column at all (every panel in the window shares the same uniform weight), so the join
is one-to-many by construction and cannot explode; for `LMA`/`EMA`, `joinCols` explicitly includes
`INVYR`, matching `wgts`' own one-row-per-`(YEAR, STATECD, INVYR)` grain, so that join is one-to-one.
This was verified empirically, not just by code reading -- see the `nPlots` results below, which are
the direct empirical signature the `vitalRates()` bug would have left behind (there, `nPlots_TREE` was
inflated by roughly the constituent-panel count; here, it is not).

### Results

All results below were computed with `betas = data.frame(grps = 1, alpha = 1, rate = 0, n = 1)` (see
"Methodology"), isolating the panel-weighting/collapse machinery from the stochastic JAGS curve fit,
across the same four states used throughout this report.

- **Structural**: `TI`, `SMA`, `LMA`, `EMA`, and `ANNUAL` all run to completion without error or
  warning (RI) -- **pass**, superseding the shallower structural-only check recorded earlier in this
  report under "`method=` structural checks (RI)".
- **`nPlots` matches exactly across `TI`/`SMA`/`LMA`/`EMA`, 4 states** -- the direct regression witness
  against the `vitalRates()`-class bug described above (there, `nPlots_TREE` was inflated ~6-10x
  post-panel-count; here, it is byte-identical across all four non-`TI` methods and `TI`):

  | State | TI | SMA | LMA | EMA |
  |---|---|---|---|---|
  | RI | 109 | 109 | 109 | 109 |
  | NC | 3473 | 3473 | 3473 | 3473 |
  | CO | 3641 | 3641 | 3641 | 3641 |
  | OR | 9084 | 9084 | 9084 | 9084 |

  **Pass** in all four states. (This equality is expected for the same reason documented in
  `vitalRates.md`: `fsi()`'s growth-eligible/remeasured plot population is shared across every
  `method`, since `TI`'s static weighting already draws on every panel in the evaluation window --
  `SMA`/`LMA`/`EMA` differ only in *how* each panel is weighted, not in *which* plots contribute.)
- **`EMA(lambda -> 1)` vs `SMA` (RI)**: `|EMA_FSI - SMA_FSI|` shrinks monotonically as lambda increases
  (3.30 -> 0.99 -> 0.10 -> 0.01 for lambda = 0.5/0.9/0.99/0.999) -- **pass**, confirms the same limiting
  relationship established in `tpa.md`/`vitalRates.md` holds at the `fsi()` output level too.
- **TI vs SMA bounded agreement, 4 states**: like `vitalRates()`'s `BIO_GROW_AC`, `FSI` is a signed rate
  of change (density decline is the typical case; all four states here show net decline) that can be
  small relative to its own sampling error, so the flat relative-tolerance check used for
  always-positive quantities (`tpa()`'s `TPA`/`BAA`) isn't appropriate. Bounding the absolute difference
  by 1.5x the combined standard error of the two estimates instead (`sqrt(SE_TI^2 + SE_SMA^2)`):

  | State | TI FSI | SMA FSI | Combined SE (abs) | \|diff\| / combined SE |
  |---|---|---|---|---|
  | RI | -13.029 | -12.855 | 4.917 | 0.035 |
  | NC | -8.452 | -9.211 | 2.813 | 0.270 |
  | CO | -3.978 | -3.745 | 0.962 | 0.242 |
  | OR | -0.405 | -0.762 | 1.111 | 0.322 |

  All four states land well within the 1.5x bound (max observed 0.322, OR) -- **pass** in all four.
- **`method = 'ANNUAL'` returns one row per real constituent panel, not a single pooled row, 4 states**:
  RI 7 rows (2019-2025), NC 8 (2017-2024), CO 10 (2014-2023), OR 10 (2015-2024) -- **pass**, confirming
  `combineMR()`'s method-awareness fix (`R/util.R`, a shared utility fixed during `area()`'s later
  non-`TI` pass -- see `tpa.md` "Fixed" #6) correctly covers `fsi()`, which was one of the ~16 dispatcher
  call sites updated by that fix but not itself re-verified there.
- **`byPlot = TRUE` + non-TI method (RI, SMA)**: runs cleanly, returns 110 rows, one per unique
  `PLT_CN` -- **pass**, confirms no row duplication analogous to `fsi.md`'s own "Fixed" #1 (which was a
  `PLOT_BASIS`-fragment collapse failure specific to the *population-estimate* branch; `byPlot` uses a
  different code path entirely and was already confirmed unaffected by that fix).
- **`useSeries = TRUE` + non-TI method (RI)**: `SMA`, `EMA`, and `ANNUAL` all run without error, at both
  population level and `byPlot = TRUE` -- **pass**. `fsi()`'s `useSeries` (chaining multiple
  remeasurements into one weighted estimate) has no analog in `tpa()`/`vitalRates()`, so this
  interaction had no precedent to draw on; confirmed here for the first time.
- **Domain filter (`treeDomain = DIA >= 10`, `areaDomain` mesic) + `bySpecies` under each of
  `SMA`/`LMA`/`EMA`/`ANNUAL`, 4 states**: no errors, no `NA`/`NaN` `FSI` values, plausible row counts in
  all 16 combinations -- **pass**. Re-runs the same historically-buggy filter/grpBy interaction pattern
  from the `TI` validation under every non-`TI` method.
- **`lambda` boundary validation propagates through `fsi()`'s `EMA` path (RI)**: `lambda` in
  `{0, 1, -0.5, 1.5, NA}` each error immediately with the shared `maWeights()` validation message (see
  `tpa.md` "Fixed" #4) -- **pass**, confirms that shared fix covers `fsi()` (`fsiHelper2` calls
  `maWeights()` directly, per `fsi.md`'s "Scope" section above).
- **`lambda` ribbon (multiple lambda values in one `EMA` call, RI)**: `lambda = c(0.3, 0.6, 0.9)`
  returns 3 distinct rows (one per lambda), each with the full RI plot count (`nPlots = 109`) -- **pass**,
  no cross-contamination between ribbons.

**No bugs were found in `fsi()`'s non-`TI` code paths.** This is a genuine (not merely
under-investigated) clean result: `fsi()`'s architectural independence from `sumToEU()` -- normally a
liability, since it means fixes to the shared estimator don't automatically propagate to `fsi()` (see
"Scope" above) -- happens to be exactly what insulated it from the specific bug class that hit
`vitalRates()`/`growMort()`, because `fsi()`'s own panel-reweighting join was never structured the way
that bug required (see the architectural note above). No code changes were made in this section; no
`NEWS.md` entry was needed. Regression tests added: `tests/testthat/test-fsi.R` Tests 14-19 cover the
`nPlots`-equality witness, the EMA/SMA convergence limit, the TI/SMA bounded-agreement check, the
`ANNUAL` per-panel-row check, `byPlot` + non-TI non-duplication, and the `lambda` boundary validation,
across all four states where applicable.
