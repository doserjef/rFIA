# Validation report: `diversity()`

## Scope

`diversity()` estimates Shannon's Diversity Index (H), Shannon's Equitability (Eh), and species
richness (S) at alpha (stand), beta (landscape), and gamma (regional) levels, using `TPA_UNADJ`
(default `stateVar`) grouped by `SPCD` (default `grpVar`). Structurally close to `tpa()`/`seedling()`
(same `PLOT`/`COND`/`TREE` tables, same domain-indicator machinery), but with a distinguishing
architectural feature relevant to this pass: `grpBy` can legitimately reference `TREE`-table columns
(species group, size class via `bySizeClass`, or any user-supplied `TREE` column) as well as
`PLOT`/`COND` columns -- unlike `tpa()`'s `bySpecies`/`standStruct()`, which don't mix condition-level
and tree-level groupings into the same area-denominator computation without special handling.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest).

## Methodology: no EVALIDator ground truth exists for this function

Like `invasive()`/`standStruct()`, diversity indices have **no EVALIDator equivalent at all** --
`EVALIDATOR_POP_ESTIMATE.csv` has no Shannon/richness/diversity attributes. Validation here is
therefore: (1) hand calculations replicating `divIndex()`'s own formula independently from raw
`TREE`/`COND` data, (2) cross-checks against `tpa()`'s `nPlots_AREA`/`AREA_TOTAL` (already validated
against EVALIDator; see `tpa.md`) for the same `landType`/`areaDomain`/`grpBy` restriction, and (3) a
mathematical invariant specific to this function's own (documented) formula: per-condition `Eh` is
bounded within `[0, 1/e]` (~0.368), so any area-weighted alpha-level `Eh_a` must be too.

## Results

### `nPlots_AREA`/`AREA_TOTAL` cross-check against `tpa()`, 4 states

| State | `landType='forest'` | `landType='timber'` | `areaDomain` (mesic) |
|---|---|---|---|
| RI | 132 = 132 | 132\* &rarr; 126 = 126 | 132\* &rarr; 124 = 124 |
| NC | 3561 = 3561 | 3561\* &rarr; 3436 = 3436 | 3561\* &rarr; 2997 = 2997 |
| CO | 3925 = 3925 | 3925\* &rarr; 1829 = 1829 | 3925\* &rarr; 2121 = 2121 |
| OR | 10410 = 10410 | 10410\* &rarr; 8986 = 8986 | 10410\* &rarr; 8523 = 8523 |

\* value before fix #1 (`landType`/`areaDomain` had no effect on `nPlots_AREA` at all before the fix,
same as every prior instance of this bug). `AREA_TOTAL` (all four states/cases) matches `tpa()`'s
`AREA_TOTAL` exactly both before and after fix #1.

`nPlots_TREE` is consistently equal to `nPlots_AREA` for `diversity()` in every case checked (e.g. RI
forest: 132/132), rather than the strictly-lower value `tpa()` reports (RI: 129) -- see "Notes" below;
this is a design/semantic observation, not something fixed this pass.

### `grpBy` interaction: COND-level vs. TREE-level grouping variables (the core finding this pass)

| `grpBy` | Variable source | Before fix #2 | After fix #2 |
|---|---|---|---|
| `OWNGRPCD` (RI) | `COND` (constant per condition) | Matched `tpa(grpBy = OWNGRPCD)` exactly | Still matches exactly (regression-checked) |
| `SPGRPCD` (RI) | `TREE` (varies within a condition) | `AREA_TOTAL` fragmented per bin; `sum(AREA_TOTAL)` across 21 bins &asymp; 375,951 (close to, not exceeding, the true 377,491.6 state total) | Every bin reports `AREA_TOTAL = 377,491.6` (the full state total), matching `tpa(bySpecies = TRUE)`'s established convention exactly |
| `bySizeClass = TRUE` (RI) | `TREE`-derived `sizeClass` | `Eh_a` exceeded 1 for several bins (e.g. 1.80, 1.74, 1.61) -- mathematically impossible, since per-condition `Eh` is bounded by `1/e` &asymp; 0.368 | All bins' `Eh_a` &le; 0.213 (well within the `[0, 0.368]` bound) |

**Pass** (after fix #2) in all cases. See "Fixed" #2 for the full root-cause explanation.

### Hand calculations (RI)

- **Alpha-level, single plot** (`pltID = "1_44_1_91"`, 21 live trees, 3 species -- SPCD 316 &times; 14,
  SPCD 129 &times; 1, SPCD 833 &times; 2, each `TPA_UNADJ = 6.018046`): by hand,
  `p = (14/21, 1/21, 2/21)`, `H = -sum(p*log(p)) = 0.4851045`, `S = 3`, `Eh = H/S = 0.1617015` --
  matches `diversity(byPlot = TRUE)`'s reported `H`/`S`/`Eh` exactly.
- **Gamma-level, whole state** (default case, live trees on forest land, RI): pooling every
  qualifying tree statewide (using the exact plot set from `pops$PLT_CN`, i.e. the plots the current
  TI evaluation actually uses) by `SPCD`: `H_g = 2.508803`, `S_g = 44` -- matches exactly.
- **Gamma-level with `treeDomain = DIA > 12`** (RI): same hand-calculation approach with the diameter
  filter applied before pooling: `H_g = 2.284983`, `S_g = 26` -- matches exactly. Confirms
  `treeDomain` survives the gamma-diversity computation path (the historical
  `area()`/`areaChange()` bug pattern from v1.1.1).

### Internal consistency (no EVALIDator needed)

- `returnSpatial` (RI, by county): all non-geometry columns match exactly. **Pass.**
- Empty `areaDomain` (`STATECD == 999`): clean 0-row result, no warning. **Pass** (see "Fixed" #3).
- Empty `treeDomain` (`SPCD == 999`, RI): a genuinely different case from an empty `areaDomain` --
  every forest condition still exists and still contributes real area, there just aren't any
  qualifying trees. `H_a = S_a = Eh_a = 0` (a real, meaningful "no species present" value) and
  `H_g`/`S_g`/`H_b`/`Eh_b`/`S_b` are all `NA` (the pooled "full" tree list used for gamma diversity is
  empty, so the gamma/beta join finds no match) -- both are correct, expected behavior, not a bug.
  **Pass**, pinned as a regression test.

## Fixed

Three bugs were found and fixed this pass, spanning `R/diversityStarter.R` and `R/diversity.R`.

### 1. `nPlots_AREA` phantom-row bug (same class as `tpa()`/`seedling()`/`standStruct()`/etc) [FIXED]

Identical root cause and fix to every prior instance (see `tpa.md`/`seedling.md`/`standStruct.md`):
the condition list (`a`) in the population-estimation branch was missing
`dplyr::filter(!is.na(CONDID))`, letting a phantom `CONDID = NA` row (from a plot whose only
condition(s) failed the `landType`/`areaDomain` filter) inflate the plot count without affecting the
point estimate. **Fix**: added the filter, identical to every other affected estimator.

### 2. Missing `aGrpBy`/`grpBy` split corrupted the area denominator whenever `grpBy` included a `TREE`-table variable [FIXED]

The most significant bug found in this validation pass. Unlike `tpa()`/`seedling()`/`standStruct()`,
which all explicitly separate `aGrpBy` (grouping columns available on `PLOT`/`COND`, i.e. constant
per condition) from the full `grpBy` (which may include `TREE`-table columns, e.g. `tpa()`'s
`bySpecies` adding `SPCD`) before building the condition-area list, `diversityStarter.R` used the
same, full `grpSyms` for **both** the tree list `t` **and** the condition list `a`. This is
semantically wrong for `a`: `dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE)` picks exactly one row
per condition before any `grpBy`-based grouping happens, so whichever tree's `sizeClass`/`SPGRPCD`
value happened to be attached to that one retained row is the *only* bin that condition's area gets
attributed to -- even though the same condition may have qualifying trees spanning many such bins.

Confirmed directly: with `grpBy = SPGRPCD` (RI), summing `AREA_TOTAL` across all 21 species-group
bins came out to ~375,951 acres -- close to (not exceeding) RI's true total forest area of 377,491.6.
Since most forest stands contain trees from *multiple* species groups simultaneously, correctly
attributing each condition's area to every group it belongs to should make the cross-bin sum
substantially *exceed* the true total (each condition's area legitimately counted more than once),
not merely reproduce it -- the near-equality was the signature of each condition's area being
assigned to only one bin. The same root cause manifested even more visibly with
`bySizeClass = TRUE`: `Eh_a` (Shannon's Equitability, alpha level) exceeded 1 for several size-class
bins (e.g. 1.80, 1.74, 1.61) -- mathematically impossible under `divIndex()`'s own `Eh = H/S` formula,
since per-condition `Eh` can never exceed `1/e` (~0.368) and an area-weighted average of bounded
values can't exceed that bound either. The corrupted (understated) `AREA_TOTAL` denominator for
several bins was inflating the `Eh_a` ratio past its own ceiling.

**Fix**: threaded a proper `aGrpBy` through `diversityStarter.R`, mirroring `tpa()`'s exact pattern:
- Computed `aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND), ...)]` right after the
  existing `grpP`/`grpC`/`grpT` table-membership split.
- `byPlot` branch: condition list `a` now groups by `aGrpSyms` (not `grpSyms`); its join to `t` uses
  `aGrpBy` (not `grpBy`); `aGrpBy` is returned in `out`.
- Population branch: condition list `a` now selects `!!!aGrpSyms` (not `!!!grpSyms`); the `condList`
  branch's `a`-`t` join uses `aGrpBy`; `aPlt <- sumToPlot(a, pops, aGrpBy)` (was `grpBy`);
  `aGrpBy <- c('YEAR', aGrpBy)` alongside the existing `grpBy <- c('YEAR', grpBy)`;
  `sumToEU(db, tPlt, aPlt, pops, grpBy, aGrpBy, method, lambda)` (was `grpBy, grpBy`); `aGrpBy`
  returned in `out` for both the `condList` and population-estimate cases.
- `full` (the raw tree list used for gamma/beta diversity) is **untouched** -- it correctly uses the
  full `grpBy` (including e.g. `sizeClass`), which is right: gamma diversity for a given size class
  should pool species *within* that size class, unlike the area denominator, which shouldn't be
  split by it.
- `R/diversity.R` (dispatcher): mirrored the identical split -- `aEst` is now grouped/selected/joined
  by `aGrpSyms`/`aGrpBy` (not `grpSyms`/`grpBy`), matching `tpa.R`'s exact pattern. `tEst`'s
  grouping/gamma-beta join is untouched (still uses the full `grpSyms`/`grpBy`).

**Verification**: after the fix, `grpBy = SPGRPCD` reports the *same* `AREA_TOTAL`
(377,491.6, the full state total) for every group, matching `tpa(bySpecies = TRUE)`'s own
already-validated convention (its `aGrpBy` also excludes `SPCD`) exactly. `bySizeClass = TRUE`'s
`Eh_a` now stays within `[0, 0.213]` for every bin, well inside the `[0, 1/e]` bound. `grpBy =
OWNGRPCD` (a `COND`-level variable, unaffected by this bug in the first place) still matches
`tpa(grpBy = OWNGRPCD)` exactly -- confirming the fix is scoped precisely to the `TREE`-level-grpBy
case without disturbing the already-correct `COND`-level case. The core default-case numbers (no
`grpBy`) -- `H_a`, `S_a`, `Eh_a`, `H_g`, `S_g` for all four states -- are bit-for-bit unchanged from
before this fix, confirming it doesn't touch anything outside the specific `TREE`-level-`grpBy`
scenario. Full package test suite re-run with no regressions.

### 3. Empty `areaDomain` produced a spurious surviving row instead of a clean empty result [FIXED]

Same class of bug as `standStruct.md`, "Fixed" #3 (and the underlying `combineMR()` guard that relies
on the population estimate genuinely having 0 rows going in): the diversity tree list (`t`, population
branch) was missing the `!is.na(CONDID)` filter its condition list `a` already has (fix #1), so when
*every* condition in the domain is a phantom `CONDID = NA` row, `divIndex()`'s empty-species fallback
(`H = S = 0`, not `NA`) still produced a real-looking row rather than a genuinely empty result,
triggering the same `"no non-missing arguments to max"` warning downstream in `combineMR()`.

**Fix**: added `dplyr::filter(!is.na(CONDID))` to the diversity tree list (`t`) in the
population-estimation branch, immediately after its `distinct()` step.

**Verification**: the empty-`areaDomain` case now returns a clean 0-row tibble with no warning. Full
package test suite re-run with no regressions.

## Secondary fix (found while investigating fix #3, same underlying phantom-row bug class)

While fixing #3, the `distinct(PLT_CN, SUBP, TREE)` key in the same tree list (both `byPlot` and
population branches) was found to have the identical `CONDID`-omission problem already fixed in
`standStruct()` (see `standStruct.md`, "Fixed" #2): a zero-tree forest condition survives the `TREE`
join as a phantom `SUBP = NA`/`TREE = NA` row, and a plot with two or more such conditions had all but
one collapsed together. Confirmed via `condList = TRUE`: NC plot `1150116756290487` (two zero-tree
forest conditions, `CONDID` 2 and 3, `CONDPROP_UNADJ = 0.25` each) reported `CONDID 3` as
`H = S = Eh = NA` (dropped from the join entirely) instead of the correct `H = S = 0` (a real,
meaningful zero, matching `CONDID 2`'s value). **Unlike `standStruct()`**, this does *not* change any
population-level point estimate here -- `divIndex()`'s empty-species fallback is `0`, not a
meaningful non-zero classification like `standStruct()`'s `'mosaic'`, so whether the phantom row
survives or is silently collapsed away, its contribution to any area-weighted sum is identically
zero either way. It does, however, matter for `condList = TRUE`'s correctness (a `customPSE()`-facing
output), where `NA` and `0` are not interchangeable. **Fix**: added `CONDID` to the `distinct()` key
in both branches, mirroring `standStruct()`'s fix exactly.

## Notes

### `nPlots_TREE` always equals `nPlots_AREA` for `diversity()` -- a design difference from `tpa()`, not fixed

`tpa()`'s `nPlots_TREE` (and `seedling()`'s, after this initiative's fix) specifically excludes plots
with literally zero qualifying trees, matching EVALIDator's own numerator-plot-count convention. For
`diversity()`, `nPlots_TREE` and `nPlots_AREA` were found to always be numerically identical after fix
#1 (e.g. RI forest: 132/132, vs. `tpa()`'s 129/132) -- because `diversityStarter.R`'s tree list `t`
carries `AREA_BASIS` (not `TREE_BASIS`), and is summed via `sumToPlot()`'s condition-level branch
(area-adjustment factors), not its tree-level branch (which is what naturally excludes `tpa()`'s
zero-tree phantom rows via `!is.na(TREE_BASIS)`). A zero-tree forest condition contributes a real,
meaningful `H = S = 0` value to `diversity()` (unlike `tpa()`, where a "phantom" tree-less condition
contributes nothing to a trees-per-acre ratio's numerator in a way EVALIDator itself excludes from its
plot count) -- so it's not obvious `diversity()`'s `nPlots_TREE` *should* match `tpa()`'s narrower
definition rather than its own broader one. Left unchanged pending explicit design direction; flagged
here rather than silently changed, since EVALIDator provides no independent ground truth to settle it
either way for this function.

### Pre-existing `many-to-many` join warning in gamma/beta computation (not introduced this pass)

`R/diversityStarter.R`'s `full` construction (used for `H_g`/`S_g`/`Eh_g` and the beta-diversity
subtraction) carries a `# TODO: this can lead to a many-to-many join. Likely want to change this`
comment already present in the code before this validation pass. Confirmed via `git stash` that the
`many-to-many relationship` warnings seen in `tests/testthat/test-diversity.R` (Tests 1 and 5) are
byte-for-byte identical before and after every fix in this report -- pre-existing, not introduced or
worsened by this pass. Left as-is; flagged here since it's the same class of warning noise already
present elsewhere in the package (e.g. `test-area.R`), not a new finding.

### Shannon's Equitability formula divides by `S`, not `ln(S)` (documented behavior, not a bug)

`divIndex()`'s `Eh = -sum(p*log(p))/S` differs from the more commonly cited Pielou's evenness index
`J' = H'/ln(S)`. This is `rFIA`'s own documented formula (unchanged by this pass, not something this
validation initiative is scoped to second-guess), but it's worth noting explicitly because it's the
reason `Eh` is bounded within `[0, 1/e]` rather than `[0, 1]` -- the bound this pass's `Eh_a`
regression test (and the bug-#2 discovery) both rely on.

### Documentation drift [FIXED]

`man/diversity.Rd`'s `\value{}` section (unlike `seedling.Rd`/`standStruct.Rd`) correctly listed
`H_a`/`H_b`/`H_g`/`Eh_a`/`Eh_b`/`Eh_g`/`S_a`/`S_b`/`S_g`, but omitted `nPlots_TREE`/`nPlots_AREA`
entirely, and its `\note{}` section referenced a `nStands` column for confidence-interval degrees of
freedom that doesn't exist in the actual output (`nPlots_AREA` is what's actually returned). Not an
estimation bug, so no test/NEWS.md entry, but corrected directly in `man/diversity.Rd`: added
`nPlots_TREE`/`nPlots_AREA` to the value list and replaced the `nStands` reference with
`nPlots_AREA`.

## Deferred to follow-up (not covered this pass)

- `method` options other than `'TI'` (no EVALIDator equivalent; internal-consistency-only checks per
  the plan, not yet added).
- `byPlot = TRUE` aggregating to reproduce the population-level estimate exactly (only the specific
  hand-calculated/regression-tested plot above was checked).
- A broader audit of other `grpBy`/`stateVar`/`grpVar` combinations (e.g. a custom `stateVar` like
  basal area, or `grpVar` set to something other than the default `SPCD`).
