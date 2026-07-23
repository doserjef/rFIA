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

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population estimate exactly (only order-of-magnitude
  agreement via the specific hand-calculated plot above was checked, same limitation as every other
  estimator's `byPlot` output in this initiative).
- `method` options other than `'TI'` (no EVALIDator equivalent; internal-consistency-only checks per
  the plan, not yet added).
- A full national audit of `GROWTH_HABIT_CD` coverage beyond the four states checked here (only `DS`
  and `SS` were confirmed present/fixed; `AL`/`MO`/`SL`/`ST` are now mapped defensively per the FIADB
  User Guide's documented domain, but weren't observed in any of the four states' local extracts).
