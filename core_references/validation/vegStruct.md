# Validation report: `vegStruct()`

## Scope

`vegStruct()` estimates percent areal cover of vegetation, grouped by canopy `LAYER` (4 height
bands, plus an "aerial: all layers" summary) and `GROWTH_HABIT` (tally tree, non-tally tree, shrub,
forb, graminoid, plus several region-specific codes -- see "Fixed" #3). It draws on
`P2VEG_SUBP_STRUCTURE` and `SUBP_COND` (condition/subplot-based, no `TREE` table), and -- like
`invasive()` -- is restricted to plots flagged `P2VEG_SAMPLING_STATUS_CD %in% 1:2`, a P2 ancillary
protocol layered on top of core inventory sampling, not collected identically everywhere. `LAYER` and
`GROWTH_HABIT` are always active grouping variables (there's no argument to turn them off, since
layer/growth-habit *is* the row granularity, the same design `invasive()` uses for species).

Structurally, `vegStructStarter.R` was already noticeably more careful than the other functions
validated earlier in this initiative: it already has a proper `aGrpBy`/`grpBy` split (the bug just
fixed in `diversity()`), and its population-branch tree list `t` already includes `CONDID` in its
`distinct()` key (the bug just fixed in `seedling()`/`standStruct()`). Both of those bug classes were
checked for and confirmed absent here. The bugs found this pass are the two other classes seen
elsewhere in this initiative (`nPlots_AREA` phantom-row, `invasive()`'s `byPlot` `mean()` formula) plus
one new one specific to this function's domain-code mapping.

Four states were used: **RI** (Northern), **NC** (Southern), **CO** (Interior West), **OR** (Pacific
Northwest) -- the same four used throughout this initiative.

## Methodology: no EVALIDator ground truth exists for this function

Like `invasive()`/`standStruct()`/`diversity()`, vegetation structure cover has **no EVALIDator
equivalent at all** -- `EVALIDATOR_POP_ESTIMATE.csv` has zero matches for "P2VEG", "vegetation
structure", "growth habit", or "canopy layer". Validation here is therefore: (1) cross-checks against
`tpa()`'s `nPlots_AREA` (already validated against EVALIDator; see `tpa.md`) for the same
`landType`/`areaDomain`/`grpBy` restriction, valid in **CO**/**OR** where
`P2VEG_SAMPLING_STATUS_CD` doesn't restrict the plot universe at all (confirmed empirically, same
situation `invasive()` found for 3 of its 4 states); **RI**/**NC** are checked only for monotonicity,
since their P2Veg samples are genuinely smaller subsets of their forest plots. (2) A hand calculation
replicating `vegStructStarter.R`'s own per-plot cover formula independently from raw
`P2VEG_SUBP_STRUCTURE`/`SUBP_COND`/`COND` data. (3) Internal consistency (totals/per-acre,
`returnSpatial`, empty-domain).

## Results

### `nPlots_AREA` cross-check against `tpa()`

| State | `landType='forest'` | `landType='timber'` | `areaDomain` (mesic) |
|---|---|---|---|
| CO | 3925 = 3925 | 3925\* &rarr; 1829 = 1829 | 3925\* &rarr; 2121 = 2121 |
| OR | 10410 = 10410 | 10410\* &rarr; 8986 = 8986 | 10410\* &rarr; 8523 = 8523 |

\* value before fix #1 (`landType`/`areaDomain` had no effect on `nPlots_AREA` at all before the fix,
same as every prior instance of this bug). **Exact match** in both states after the fix.

| State | forest | timber | areaDomain (mesic) |
|---|---|---|---|
| RI | 6 | 6 | 6 |
| NC | 210 | 202 | 169 |

RI's tiny 6-plot P2Veg sample doesn't shrink further under `timber`/mesic restrictions (plausible
given its size); NC's genuinely restricted 210-plot sample correctly shrinks under both
restrictions (202, 169) -- **monotonicity holds** in both states, the applicable check given
EVALIDator provides no independent plot-count ground truth for the P2Veg-restricted universe itself.

### `grpBy` interaction (`OWNGRPCD`, CO)

`AREA_TOTAL` per ownership group matches `tpa(grpBy = OWNGRPCD)`'s grouped `AREA_TOTAL` exactly (CO
chosen specifically because `P2VEG_SAMPLING_STATUS_CD` doesn't restrict its universe, making an exact
match meaningful rather than confounded by real P2Veg sampling gaps). **Pass** -- confirms `grpBy`
doesn't silently drop or misattribute area for some groups (the historical
`area()`/`areaChange()` bug pattern from v1.1.1), and confirms `vegStructStarter.R`'s pre-existing
`aGrpBy` split (see "Scope" above) is correctly implemented.

### `byPlot = TRUE` cover, hand-calculated from raw data (NC)

Plot `471569784489998` has three conditions: `CONDID 1` (non-forest, excluded from the `forest`
domain), `CONDID 2` (forest, `CONDPROP_UNADJ 0.23061`), `CONDID 3` (forest, `CONDPROP_UNADJ 0.25`).
Forbs, 0-2ft layer, is recorded on subplot 3 (part of `CONDID 2`, `COVER_PCT 5`, `SUBPCOND_PROP
0.922438`) and subplot 4 (`CONDID 3`, `COVER_PCT 10`, `SUBPCOND_PROP 1.0`) -- not recorded at all on
subplots 1/2 (entirely non-forest `CONDID 1`). By hand, dividing by a fixed 4 subplots:
`(0.05 * 0.922438 + 0.10 * 1.0) / 4 = 0.03653047`. `vegStruct(byPlot = TRUE)` reports exactly this
value after the fix below (previously `0.07306095` -- a 2x inflation, since only 2 of the 4 subplots
had any Forbs/0-2ft record at all).

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `COVER_AREA_TOTAL / AREA_TOTAL * 100` reproduces `COVER_PCT` exactly, across all
  four states. **Pass.**
- `returnSpatial` (RI, by county): all non-geometry columns match exactly. **Pass.**
- Empty `areaDomain` (`STATECD == 999`, RI): clean 0-row result, no warning. **Pass** (see "Fixed" #1).

## Fixed

Three bugs were found and fixed this pass, all in `R/vegStructStarter.R`.

### 1. `nPlots_AREA` phantom-row bug, plus the empty-domain consequence (same class as `tpa()`/`seedling()`/`standStruct()`/`diversity()`/etc) [FIXED]

Identical root cause to every prior instance (see `tpa.md`/`seedling.md`/`standStruct.md`/
`diversity.md`): the condition list (`a`) in the population-estimation branch was missing
`dplyr::filter(!is.na(CONDID))`, inflating `nPlots_AREA` without affecting point estimates. Reproduced
on CO/OR: `landType = 'timber'`/`areaDomain` restrictions had *zero* effect on `nPlots_AREA` before
the fix (always equal to the unrestricted `'forest'` count).

A second, related phantom-row problem: the tree list (`t`, population branch) was *also* missing
`!is.na(CONDID)`, and unlike the condition list's `fa = CONDPROP_UNADJ * aDI` (which is `NA` for a
phantom row and correctly drops out via `na.rm = TRUE`), the tree list's `cover = sum(COVER_PCT/100 *
..., na.rm = TRUE) / 4` formula turns an all-`NA` phantom row into `sum(NA, na.rm = TRUE) / 4 = 0` --
a real-looking zero, not a missing value. When `areaDomain` matched no conditions at all, this
produced a single surviving `COVER_PCT = NA`-ish row (from the ratio `0/0`) with `YEAR = -Inf` and a
`"no non-missing arguments to max"` warning, instead of the clean empty result every other estimator
gives (same failure mode as `standStruct.md`/`diversity.md`'s equivalent fix).

**Fix**: added `dplyr::filter(!is.na(CONDID))` to both the condition list `a` and the tree list `t`
in the population-estimation branch, mirroring the fix already applied to every other affected
estimator.

**Verification**: after the fix, `nPlots_AREA` matches `tpa()` exactly for CO/OR across
`landType`/`areaDomain`; the empty-`areaDomain` case returns a clean 0-row tibble with no warning.
Full package test suite re-run with no regressions.

### 2. `byPlot = TRUE` cover formula inflated whenever a LAYER/GROWTH_HABIT combination wasn't recorded on all 4 subplots [FIXED]

The identical bug class already found and fixed in `invasive()`'s `byPlot` branch (see
`invasive.md`, "Fixed" #2). The `byPlot` branch computed each plot's `PROP_COVER` as
`mean(cover, na.rm = TRUE)` across whatever subplots had a recorded value for a given
`LAYER`/`GROWTH_HABIT` combination -- since that combination's rows come directly from
`P2VEG_SUBP_STRUCTURE` (only subplots where it was actually recorded appear at all, there's no
explicit zero row for the rest), `mean()` divides by however many subplots *did* have a record,
rather than the true fixed denominator of 4 subplots. Since vegetation cover is typically patchy
(rarely recorded on all 4 subplots for a given layer/growth-habit), this is the common case, not an
edge case -- confirmed via hand calculation (see "Results" above) that this inflated `PROP_COVER` by
2x for one NC plot/layer/growth-habit combination (would be up to 4x for a combination recorded on
only 1 of 4 subplots).

**Fix**: changed `PROP_COVER = mean(cover, na.rm = TRUE)` to `PROP_COVER = sum(cover, na.rm = TRUE) /
4`, matching the population-estimation branch's own formula (which already divides by a fixed 4).

**Verification**: after the fix, the NC plot above reports `PROP_COVER = 0.03653047`, matching the
hand calculation exactly (previously `0.07306095`). Full package test suite re-run with no
regressions.

### 3. Incomplete `GROWTH_HABIT_CD` domain mapping silently dropped region-specific vegetation records [FIXED]

`vegStructStarter.R` maps `P2VEG_SUBP_STRUCTURE.GROWTH_HABIT_CD` to a readable `GROWTH_HABIT` label
via `dplyr::case_when()`, covering only the 5 core national codes (`TT`/`NT`/`SH`/`FB`/`GR`). Since
`GROWTH_HABIT` is part of `vegStruct()`'s internal `grpBy`, and the final output step does
`tidyr::drop_na(grpBy)`, any record with an unmapped code gets `GROWTH_HABIT = NA` and its entire row
-- including real `COVER_PCT` data -- is silently dropped. Confirmed real, non-trivial data loss in
two of the four validation states: **CO** has 1172 raw `P2VEG_SUBP_STRUCTURE` rows coded `DS`
(0.33% of its 360,222 total rows); **OR** has 231 rows coded `SS` (0.02% of its 1,105,581 total rows).
Neither code is a data-entry anomaly -- both are legitimate, documented FIADB domain values with a
narrow, region-specific scope (confirmed directly against the FIADB User Guide, Database Description
v9.2, ch. 4.3.10): `DS` = "Dead pinyon species shrubs" (dead pinyon-juniper-associated shrub cover,
populated only by certain Interior West work units, `SURVEY.RSCD = 22` -- exactly CO's region) and
`SS` = "Newly sprouted shrub cover" (post-fire shrub resprouting, populated only for Pacific
Northwest Research Station Fire Effects and Recovery Study plots, `SURVEY.RSCD = 26/27` -- exactly
OR's region). The same FIADB User Guide section documents three further PNWRS-only codes not present
in any of the four validation states' local extracts but equally unmapped in the prior code: `AL`
("All vegetation"), `MO` ("Moss/bryophytes"), `SL` ("Bare soil"), and `ST` ("Seedlings").

**Fix**: extended the `GROWTH_HABIT` `case_when()` to cover the complete documented domain: `DS` ->
"Dead pinyon species shrubs", `AL` -> "All vegetation", `MO` -> "Moss/bryophytes", `SL` -> "Bare
soil", `SS` -> "Newly sprouted shrub cover", `ST` -> "Seedlings", in addition to the 5 pre-existing
core codes. Also fixed a cosmetic typo while in the same `case_when()` block: `LAYER == 5` was
labeled `'Areal: all layers'`; the FIADB User Guide's own wording is "Aerial: Canopy cover for all
layers" (`'Aerial'`, not `'Areal'`) -- corrected to `'Aerial: all layers'`. Not a data-loss bug (this
label always survived the `drop_na(grpBy)` step; it's a display-string spelling fix only), so not
included in the `NEWS.md`/regression-test scope of this fix, but noted here for completeness.

**Verification**: after the fix, `vegStruct(db_co)` includes a "Dead pinyon species shrubs" row with
real, positive cover, and `vegStruct(db_or)` includes a "Newly sprouted shrub cover" row, both
previously entirely absent from the output. Confirmed via `grep` that no other estimator function
reads `P2VEG_SUBP_STRUCTURE.GROWTH_HABIT_CD` (this table/column is specific to `vegStruct()`). Full
package test suite re-run with no regressions.

## Notes

### Why this function had fewer bugs than most prior passes

`vegStructStarter.R` already had a correct `aGrpBy`/`grpBy` split and a correct `CONDID`-inclusive
`distinct()` key in its tree list -- both bug classes found and fixed in earlier functions this
initiative (`diversity()`, `seedling()`/`standStruct()` respectively). This suggests those two bug
classes were understood and addressed by the time `vegStruct()` was written, even though the fix
hadn't yet been back-ported to the earlier functions. The bugs that *were* found here (`nPlots_AREA`
phantom-row, the `byPlot` `mean()` formula, and the domain-mapping gap) are the ones not yet
addressed anywhere in the codebase at the time of this pass -- the first two shared with
`invasive()`, the third unique to this function's specific reference domain.

### Documentation update [DONE]

`man/vegStruct.Rd`'s `\details{}` "Growth habit" section previously documented only the 5 core
national codes (matching the code before fix #3). Updated to also document the region-specific codes
now mapped by the fix (`DS`/`AL`/`MO`/`SL`/`SS`/`ST`), split into two clearly-labeled subsections --
one for the 5 codes recorded by every FIA work unit nationwide, one for the codes populated by only a
single work unit (or small subset), each with the description and scope pulled directly from the
FIADB User Guide (Database Description v9.2, ch. 4.3.10) -- so a user encountering an unfamiliar
`GROWTH_HABIT` value in their output (e.g. "Dead pinyon species shrubs" on an Interior West extract)
can immediately tell it's a real, documented, region-specific category rather than mistaking it for
an error.

### 4. `mergeSmallStrata()` crashed with `"replacement has length zero"` when a too-small stratum's only cross-year neighbor has `INVYR = NA` [FIXED]

Found while starting this function's non-TI (`method`) validation pass — `vegStruct(fiaRI, method =
'SMA')` (and `'LMA'`/`'EMA'`/`'ANNUAL'`) crashed, a pre-existing bug already noted but not
root-caused in `tpa.md`'s "Findings" #2. Not specific to `vegStruct()`: the shared utility that pools
too-small strata together (`mergeSmallStrata()`, `R/util.R`, used by every `sumToEU()`-based
estimator under non-TI methods) picked a stratum's cross-year merge partner via
`which.min(abs(neighbors$INVYR - (dat$INVYR + .01)))` without accounting for a neighbor whose own
`INVYR` is `NA` -- `which.min()` on an all-`NA` input silently returns a length-0 result rather than
erroring, which then propagated into a `pops[pops$stratID == i, 'P2POINTCNT_INVYR'] <- <empty> +
dat$P2POINTCNT_INVYR` assignment and crashed there instead.

**Root cause of the `INVYR = NA` rows themselves**: `handlePops()` (`R/util.R`) attaches `INVYR` to
`pops` via `left_join(select(db$PLOT, PLT_CN, INVYR), by = 'PLT_CN')`. `vegStruct()` (like
`invasive()`) restricts `db$PLOT` to its own P2-ancillary-sampled subset
(`P2VEG_SAMPLING_STATUS_CD %in% 1:2`) *before* this join runs, so any plot present in the general
stratification (`POP_PLOT_STRATUM_ASSGN`) but excluded from that P2Veg-restricted `db$PLOT` gets
`INVYR = NA` from the join, rather than being absent. These rows are real artifacts of the
P2-ancillary restriction, not upstream data corruption -- confirmed directly (`fiaRI`: `pops` has
plots stratified into the population with no corresponding P2Veg-restricted `db$PLOT` row).

**Fix**: three edits to `mergeSmallStrata()`, all excluding `INVYR = NA` rows from participating in
the merge logic -- neither needing a merge partner (the loop driver) nor being selected as one (both
neighbor-candidate searches) -- since such a row represents a plot excluded from this function's
actual estimation, not a real per-year sample needing small-strata pooling.

**Verification**: `vegStruct(fiaRI, method = 'SMA'/'LMA'/'EMA'/'ANNUAL')` no longer crashes -- it now
correctly falls into `mergeSmallStrata()`'s pre-existing "bad stratification" warning path (the same
graceful degradation already used for genuinely too-small real strata) and returns a valid result.
Confirmed byte-identical `tpa()`/`area()` output (all four states, every method) before and after the
fix, since neither pre-restricts `db$PLOT` and so never produces `INVYR = NA` rows in the first place
-- the fix is a true no-op for every already-validated function. Full package test suite re-run with
no regressions. `NEWS.md` entry added.

## Non-TI method validation (SMA/LMA/EMA/ANNUAL) -- PAUSED, not completed this pass

Started this pass following the established template (`tpa.md`), but stopped partway through after
finding something that needs sign-off before continuing, per the project's bug-handling protocol: see
"Open investigation" below. **No non-TI tests were added to `test-vegStruct.R` this pass** -- only the
crash fix above (which is real, verified, and unrelated to the open question) has landed.

## `AREA_TOTAL` under a genuine P2Veg restriction: `TI` confirmed correct against EVALIDator;
## `SMA`'s divergence root-caused to a `mergeSmallStrata()` defect (not a legitimate estimate)

While building the non-TI test matrix, `AREA_TOTAL` under `method = 'SMA'` diverged sharply from
`method = 'TI'` for states where P2Veg sampling *genuinely* restricts the plot universe (confirmed on
NC): `TI` gives 1,135,290 acres, `SMA` gives 17,568,389 -- a 1447% jump. This was initially suspected
to be the same class of legitimate small-sample panel variance already established as correct (not a
bug) for `dwm()`'s TI-vs-SMA divergence, but that explanation was directly ruled out: `dwm()`'s
legitimately-noisy estimates carry correspondingly large standard errors (30-70%+), matching the
uncertainty a thin sample should have. Here, `SMA`'s reported `AREA_TOTAL_SE` is a *tight* 4.4% --
mathematically inconsistent with a real design-based estimate that swung 15x on a genuinely thin
sample.

### Step 1: is `TI`'s `AREA_TOTAL` itself correct under a genuine P2Veg restriction? -- **yes, confirmed exact against EVALIDator**

The report's original "Methodology" section (above) only ever numerically cross-checked `AREA_TOTAL`
against EVALIDator for **CO/OR**, where P2Veg sampling happens not to restrict the plot universe at
all (the easy case); **RI/NC** (the states where it *does* restrict the universe) were checked only
for plot-count *monotonicity* against `tpa()`, never `AREA_TOTAL` itself. This gap is now closed.
Using the same `strFilter` mechanism already validated for `area()`'s `areaDomain` checks (`area.md`,
test 14 -- EVALIDator attribute 2, restricted by a raw-SQL `WHERE` fragment applied identically to
numerator and denominator), a filter of `PLOT.P2VEG_SAMPLING_STATUS_CD in (1,2)` reproduces exactly
`vegStructStarter.R`'s own upstream `db$PLOT` restriction (`R/vegStructStarter.R` line 67):

| State | `vegStruct(method = 'TI')` `AREA_TOTAL` | EVALIDator attr 2 + `strFilter` | `AREA_TOTAL_SE` | `nPlots_AREA` |
|---|---|---|---|---|
| RI | 17,310.84 | 17,310.84 | 41.999% = 41.999% | 6 = 6 |
| NC | 1,135,289.7 | 1,135,290 | 6.900% = 6.900% | 210 = 210 |

**Exact match**, both states, including SE% and plot count. This confirms `TI`'s much-smaller
P2Veg-restricted total (vs. NC's ~18.5M-acre unrestricted forest area, already validated in
`area.md`) is not an under-count -- EVALIDator itself, an independent authoritative source, computes
the same small figure for the same restricted domain. The concern that a correctly-weighted
subsample "should" extrapolate to roughly the full forest area does not hold here, because the
sampling design does not treat the P2Veg-restricted set as a probability subsample of all forest
plots with a correspondingly adjusted weight -- it is evaluated as its own domain, exactly as
`areaDomain`-restricted `area()` calls already are. `TI` reproduces that correctly. **The bug is
confined to the non-TI (moving-average) methods.**

### Step 2: root cause of `SMA`'s inflation -- confirmed defect in `mergeSmallStrata()`'s stratum-weight renormalization

**Mechanism traced and confirmed** (not merely suspected): `SMA`'s 17,568,389 equals exactly
`weight (0.125 = 1/8 panels) × Σ(every stratum's own extrapolated area contribution, across every
panel-year that stratum happened to have at least one P2Veg-sampled plot)`. Directly instrumenting
`mergeSmallStrata()` (`R/util.R`, ~line 838, "Adjust stratum weights when not all strata are sampled
in an INVYR") on NC confirms why: this step renormalizes `stratWgt_INVYR` to **sum to exactly 1
within every (`ESTN_UNIT_CN`, `INVYR`) group**, regardless of how few strata are actually present.
Under the P2Veg-restricted `db$PLOT`, most (estimation unit, year) cells have only **1-2 of the
5+ real strata present** (confirmed directly, all 28 EU/year combinations checked) -- yet
`sumStratWgt` is 1.0 in every single one. Under the unrestricted `db$PLOT` (`tpa()`/`area()`'s
normal case), most cells have **4-6 of the strata present** (the genuine, expected case this logic
was written for -- a stratum boundary reassignment between panels, where the land area itself was
still sampled by *someone* that year), and renormalizing those to sum to 1 is the correct
compensation.

The defect: this renormalization does not distinguish between the two cases. When db$PLOT has been
pre-restricted to a P2-ancillary protocol subsample *before* `handlePops()`/`mergeSmallStrata()` ever
run (`vegStruct()`'s -- and `invasive()`'s -- own upstream filter), a stratum with zero rows in a
given `INVYR` does not mean its land went unsampled that year (as in the boundary-reassignment case)
-- it means the P2Veg protocol simply was not run on any plot in that stratum that year. The land, and
its non-veg core-inventory sample, still existed. Renormalizing the 1-2 present strata's weight up to
1.0 silently attributes the *entire* estimation unit's area to whichever handful of plots happened to
carry a P2Veg flag that particular year, independently in each of ~8 panel-years, then sums them.

**Two independent numeric confirmations that this is an artifact, not a legitimate larger estimate**
(direct answer to the concern that EVALIDator's restricted-domain number might itself be the
suspicious one, and `SMA`'s larger number the more plausible "true" P2Veg-sampled population total):

1. **`nPlots_AREA` is identical between `TI` and `SMA`: 210, both.** If `SMA`'s larger total reflected
   a genuinely bigger pooled P2Veg sample (more distinct plots contributing across more panel-years),
   plot count would grow accordingly -- NC has 335 distinct physical plots ever P2Veg-sampled across
   its full available local-extract history, well above 210. It doesn't grow at all. Same 210 plots,
   15.5x larger area.
2. **`SMA`'s `AREA_TOTAL` (17,568,389) is 94.9% of NC's own *unrestricted* forest area (18,509,817,
   `area()`, already validated exact vs. EVALIDator)** -- not some intermediate value consistent with
   "a larger but still P2Veg-restricted population." That is exactly the number the renormalization
   mechanism predicts: each panel-year's contribution converges toward the *full* estimation-unit
   area (weight forced to sum to 1 off a tiny present subsample), and summing ~8 such full-unit-sized
   contributions at a uniform 1/8 weight converges back toward ~1x the unrestricted total.

**Confirmed: this is a real defect in `mergeSmallStrata()`, not a legitimate alternate estimate of a
genuinely larger population.** It is not specific to `vegStruct()` -- the function is a shared
utility used by every `sumToEU()`-based estimator under `SMA`/`LMA`/`EMA` (`ANNUAL` uses a different
path, not yet checked), and the failure mode specifically requires an upstream domain-restricted
`db$PLOT` of the kind only `vegStruct()`/`invasive()` construct (`P2VEG_SAMPLING_STATUS_CD`/
`INVASIVE_SAMPLING_STATUS_CD`) -- so `tpa()`/`area()`/etc. are not affected (confirmed: their
unrestricted `db$PLOT` has 4-6 of ~5-6 strata present in nearly every EU/year cell, the case this
logic already handles correctly).

**Not yet fixed.** The right remedy needs a domain-judgment call this report defers rather than
guesses at: whether a stratum with zero P2Veg-sampled plots in a given `INVYR` should (a) be excluded
from that year's weight renormalization entirely (effectively giving that year's estimate zero
contribution from the missing stratum's land, rather than redistributing it), (b) cause that whole
`(EU, INVYR)` cell to be dropped from the moving-average window if too few strata are represented, or
(c) something else grounded in `westfall2022USDA.pdf`'s treatment of moving-average estimators over
an incomplete panel -- since `mergeSmallStrata()` is shared code, any fix must also be checked against
its legitimate use case (real cross-year boundary reassignment for `tpa()`/`area()`/etc.) to avoid
regressing it.

### Confirmed present in `invasive()`; confirmed absent in `dwm()`/`diversity()` -- why

**`invasive()` (2026-09-15): confirmed, same defect.** It shares the identical P2-ancillary
`db$PLOT` pre-filter pattern (`INVASIVE_SAMPLING_STATUS_CD %in% 1:2`, `invasiveStarter.R` line 60),
run through the same `handlePops()`/`mergeSmallStrata()`/`sumToEU()` path. Per `invasive.md`'s own
methodology, only RI is a genuinely restricted state for `invasive()` (NC/CO's restriction is a
no-op there). Confirmed on RI: `TI` `AREA_TOTAL` = 17,310.84 (exact match to EVALIDator) vs `SMA` =
62,965.45 (3.6x inflation -- smaller magnitude than `vegStruct()`'s NC case since RI's design has
fewer strata to inflate from, but the identical mechanism: every `(EU, INVYR)` cell has only 1-2 of
7 real strata present, `sumStratWgt` still renormalized to exactly 1 in each; `nPlots_AREA`
unchanged between `TI`/`SMA`, 6 = 6). Full write-up in `invasive.md`'s "Findings" section.

**`dwm()` and `diversity()` (2026-09-15): confirmed clean, and structurally cannot hit this bug --
not merely untested.** Both were suspected candidates (`dwm()` explicitly, since its own
TI-vs-SMA divergence was the original reason this defect's SE signature was checked at all;
`diversity()` because it shares `vegStruct()`'s "no EVALIDator ground truth, P2/P3-adjacent"
profile). Neither pre-filters `db$PLOT` by any sampling-status column -- confirmed via `grep`, the
only `db$PLOT` filter either applies is the routine `prev == 0` MR-clip step every estimator does.
Instead of a manual post-hoc plot-list restriction, each draws its (smaller, for `dwm()`) population
through a **separate, FIA-maintained evaluation type** -- `dwm()` uses `evalType = 'DWM'` (EXPDWM,
NC's current EVALID 372407, a self-contained 348-plot design with its own proper
`POP_STRATUM`/`POP_ESTN_UNIT`/`POP_PLOT_STRATUM_ASSGN` bookkeeping), `diversity()` uses
`evalType = 'VOL'` (EXPVOL, NC's current EVALID 372401 -- the standard, full-population tree/volume
evaluation, 5,674 plots, essentially the same universe as CURR). In both cases every plot the design
expects to find really is present in `db$PLOT`, so `handlePops()`'s `PLT_CN` -> `INVYR` join never
produces an `INVYR = NA` row (confirmed directly: zero for both, vs. genuine ones for
`vegStruct()`/`invasive()`), and instrumenting `mergeSmallStrata()` the same way shows the ordinary,
expected per-year coverage pattern (`dwm()`: 3-5 of 5 real strata present most years, only thin in
2017-2019, the early edge of the current panel cycle; `diversity()`: 4-6 of ~5-6 present in nearly
every cell) -- the genuine cross-year boundary-reassignment case this renormalization logic was
built for, never the "1-2 of 5+, every single year" pattern `vegStruct()`/`invasive()` produce.
Consequently `AREA_TOTAL` for both stays anchored near the true unrestricted forest total under
every method: `dwm()` NC, `TI` = 18,193,439 / `SMA` = 19,121,872 (~5% apart, both near the true
~18.5M); `diversity()` NC, `TI` = 18,509,817 (exact match to `area()`'s already-validated figure) /
`SMA` = 18,650,929 (~0.8% apart, `AREA_TOTAL_SE` widening from 0.64% to 1.26% -- a small, plausible
change, not the suspicious tightening seen in the buggy case). `dwm()`'s own already-documented
TI-vs-SMA divergence (large, honest SEs, 30-70%+) is therefore a genuinely different phenomenon: real
small-P3-sample noise in the fuels-load *numerator* (only 348 of NC's forest plots have a
`COND_DWM_CALC` row at all, handled correctly downstream via `semi_join`/`na.rm` when computing the
mean/variance), never touching `STRATUM_WGT` -- not `mergeSmallStrata()` silently inflating the
*area* itself while reporting a falsely tight SE.

**Practical implication for the eventual fix**: whatever remedy is chosen for `mergeSmallStrata()`
must specifically target the case where an upstream ad hoc plot-list restriction has been layered on
top of a stratification designed for a larger population (`vegStruct()`/`invasive()` only) --
`dwm()`/`diversity()`'s pattern of drawing a genuinely smaller population through its own proper FIA
evaluation type is legitimate and must not be touched by the fix.

## Deferred to follow-up (not covered this pass)

- **`mergeSmallStrata()`'s stratum-weight-renormalization fix** (see above) -- root cause is confirmed,
  but the actual code change needs a domain-judgment call on the correct remedy before it's safe to
  make, given the function is shared with every `SMA`/`LMA`/`EMA` estimator. `vegStruct()`'s non-TI
  method validation (this phase's actual goal) is still blocked on this fix landing.
- `byPlot = TRUE` aggregation reproducing the population estimate exactly (only order-of-magnitude
  agreement via the specific hand-calculated plot above was checked, same limitation as every other
  estimator's `byPlot` output in this initiative).
- A full national audit of `GROWTH_HABIT_CD` coverage beyond the four states checked here (only `DS`
  and `SS` were confirmed present/fixed; `AL`/`MO`/`SL`/`ST` are now mapped defensively per the FIADB
  User Guide's documented domain, but weren't observed in any of the four states' local extracts).
