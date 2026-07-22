# Validation report: `biomass()`

## Scope

This pass covers `biomass()` only, following its split from carbon estimation (carbon estimation is
now handled exclusively by `carbon()`, which was not touched here and is validated separately).
`biomass()`'s output no longer includes `CARB_ACRE`/`CARB_TOTAL` or any other carbon-related column.

## Methodology

Ground truth was obtained from the FIADB-API `fullreport` endpoint (the programmatic interface
behind the EVALIDator web tool), queried live via `fetch_evalidator.R` in this directory. rFIA was
run against the real, current FIADB extracts already cached at `~/Dropbox/data/fia/` (pulled via
`getFIA()`), using `clipFIA(mostRecent = TRUE)` to match EVALIDator's "current" evaluation.

Four states were used, one per FIA region: **RI** (Northern), **NC** (Southern), **CO** (Interior
West), **OR** (Pacific Northwest).

`tests/testthat/test-biomass.R` calls the FIADB-API live at test time rather than hard-coding the
reference numbers below (same approach as `tpa()`/`area()`) -- this report is illustrative, not a
source of truth the tests are pinned to. The EVAL_GRP code for each state is read directly off
`clipFIA(..., mostRecent = TRUE)$POP_EVAL_GRP$EVAL_GRP`, so the tests always query whichever
evaluation `mostRecent` actually selected. Tests are skipped (not failed) when the local data cache
or network access to `apps.fs.usda.gov` is unavailable.

A key simplification for `biomass()`, unlike `tpa()`: rFIA's default `component = 'AG'` (and every
other component) is read directly off a FIADB `TREE.DRYBIO_*` column (the pre-computed NSVB
per-tree biomass fields, e.g. `DRYBIO_AG`, `DRYBIO_STEM`, `DRYBIO_FOLIAGE`), not derived via any
rFIA-side biomass model. This means the numeric match here is primarily testing rFIA's
post-stratified ratio-of-means aggregation (join/filter/adjustment-factor/estimator logic), the same
logic already validated for `tpa()`/`area()`, rather than a from-scratch NSVB reimplementation --
which is consistent with every attribute below matching EVALIDator to full double precision with no
fix needed this pass.

`treeDomain`/`areaDomain` filter semantics use the same two API mechanisms established during the
`tpa()` validation pass (see `tpa.md`): `wnum` (numerator-only) for `treeDomain`-style filters,
`strFilter` (numerator + denominator) for `areaDomain`-style filters.

### Component-to-attribute mapping

EVALIDator's attribute library (`~/Dropbox/data/fia/EVALIDATOR_POP_ESTIMATE.csv`) has direct "at
least 1 inch dbh" stock attributes for `AG` (10/13), `ROOT` (belowground: 59/73), `FOLIAGE`
(11020/11052), and `BRANCH` (11032/11064) on forest land/timberland respectively. It does **not**
have 1-inch-threshold stock attributes for `STEM`, `STEM_BARK`, `STUMP_BARK`, `BOLE`, or
`BOLE_BARK` -- only "timber species at least 5 inches d.b.h." variants (11016, 11017, 11018, 11019,
11000). For those components, `treeDomain = DIA >= 5` was added on the rFIA side to make an
apples-to-apples comparison; this doubles as a `treeDomain` interaction check. `treeType = 'gs'`
(growing-stock) has no EVALIDator "current stock" biomass attribute at all (only growth/mortality/
removal deltas, e.g. attribute 2312), so it wasn't numerically validated against EVALIDator this
pass -- it was, however, already exercised structurally by the pre-existing test suite
(`test-biomass.R` Test 2).

## Results: numeric match

All point estimates and percent standard errors below match the FIADB-API to full double precision
(14+ significant digits) unless noted.

### Core default case (`component = 'AG'`, `treeType = 'live'`, `landType = 'forest'`), 4 states

| State | BIO_ACRE | BIO_ACRE_SE | nPlots_TREE | nPlots_AREA |
|---|---|---|---|---|
| RI | 76.96422 | 4.031046 | 129 | 132 |
| NC | 71.07347 | 1.068819 | 3455 | 3561 |
| CO | 27.09236 | 1.331426 | 3774 | 3925 |
| OR | 74.65293 | 0.9377079 | 9968 | 10410 |

All four: **exact match** against EVALIDator attribute 10 (aboveground biomass of live trees, forest
land), ratio'd against attribute 2 (forest land area).

### `landType`/`treeType`/`component` variants, 4 states

| Case | EVALIDator attr (num/denom) | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `landType = 'timber'` | 13 / 3 | exact | exact | exact | exact |
| `treeType = 'dead'` | 11266 / 2 | exact | exact | exact | exact |
| `component = 'ROOT'` | 59 / 2 | exact | exact | exact | exact |
| `component = 'FOLIAGE'` | 11020 / 2 | exact | exact | exact | exact |
| `component = 'BRANCH'` | 11032 / 2 | exact | exact | exact | exact |

### Merchantable-scale components (`treeDomain = DIA >= 5` to match EVALIDator's 5in threshold), 4 states

| Case | EVALIDator attr (num/denom) | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `component = 'STEM'` | 11016 / 2 | exact | exact | exact | exact |
| `component = 'STEM_BARK'` | 11017 / 2 | exact | exact | exact | exact |
| `component = 'STUMP_BARK'` | 11018 / 2 | exact | exact | exact | exact |
| `component = c('BOLE','BOLE_BARK')` (summed) | 11000 / 2 | exact | exact | exact | exact |

### `component = 'TOTAL'` internal consistency (v1.1.3 regression check), 4 states

`biomass(component = 'TOTAL')` reproduces the sum of `biomass(byComponent = TRUE)`'s `ROOT`, `STEM`,
`STEM_BARK`, `BRANCH`, and `FOLIAGE` rows exactly, in all four states -- confirms the v1.1.3
double-counting bug fix (see NEWS.md) has held. **Pass.**

### Domain filter interactions, 4 states

| Case | Mechanism | RI | NC | CO | OR |
|---|---|---|---|---|---|
| `treeDomain = DIA >= 20` (large trees) | `wnum` | exact | exact | exact | exact |
| `areaDomain = PHYSCLCD %in% 21:29` (mesic) | `strFilter` | exact | exact | exact | exact |

Both matched EVALIDator attribute 10 (AG biomass, forest land) exactly across all four states.

### `bySpecies` grouping (RI)

Cross-checked a random sample of individual species rows from `biomass(bySpecies = TRUE)` against
independent single-species EVALIDator queries (`wnum = "TREE.SPCD = <code>"`) -- validates that the
`grpBy = SPCD` join/aggregation path doesn't silently drop the domain filter for some groups (the
historical `area()`/`areaChange()` bug pattern from v1.1.1). **Pass** for the species sampled
(SPCD 833, SPCD 43).

### `returnSpatial` (RI, by county)

`biomass(polys = countiesRI, returnSpatial = TRUE)` vs. `returnSpatial = FALSE`: all non-geometry
columns match exactly (`expect_equal` on the two data frames, geometry column dropped). **Pass.**

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `BIO_TOTAL / AREA_TOTAL` reproduces `BIO_ACRE` exactly, across all four states.
  **Pass.**

### Empty-domain edge case

`biomass(treeDomain = SPCD == 999)` returns a clean 0-row tibble with no warning (this class of bug,
previously affecting all estimators including `biomass()`, was already fixed as part of the `tpa()`
validation pass -- see `tpa.md`, "Fixed" #2). **Pass**, no regression.

## Fixed

**`nPlots_TREE` over-counted plots for component-restricted queries in woodland-heavy states.**
When `tests/testthat/test-biomass.R` Tests 12-15 were extended to also check `nPlots_TREE`/
`nPlots_AREA` (not just point estimates and SEs) for every `component` variant, RI/NC/OR matched
EVALIDator exactly everywhere, but **Colorado** failed `nPlots_TREE` for `component = 'BRANCH'`,
`'STEM'`, `'STEM_BARK'`, `'STUMP_BARK'`, and `c('BOLE', 'BOLE_BARK')` (all restricted to `DIA >= 5`
merch. components except `BRANCH`). `BIO_ACRE`/`BIO_ACRE_SE` matched exactly in every one of these
failing rows -- only the plot count was wrong. A follow-up check against **Arizona** and **Utah**
(not part of the original 4-region matrix, but heavy in pinyon-juniper woodland) showed the same
issue far more severely:

| State | Component (attr) | `nPlots_TREE` (before fix) | EVALIDator `numPlotCount` |
|---|---|---|---|
| CO | `BRANCH` (11032) | 3774 | 2530 |
| CO | `STEM`, DIA>=5 (11016) | 3676 | 2491 |
| AZ | `BRANCH` (11032) | 3137 | 842 |
| AZ | `STEM`, DIA>=5 (11016) | 3113 | 819 |
| UT | `BRANCH` (11032) | 2932 | 1021 |
| UT | `STEM`, DIA>=5 (11016) | 2850 | 993 |

**Root cause**: NSVB does not model `STEM`, `STEM_BARK`, `STUMP_BARK`, `BOLE`, `BOLE_BARK`, or
`BRANCH` for woodland-form species (e.g. pinyon, juniper, Gambel oak, curlleaf mountain-mahogany --
SPCD 65, 106, 814, 66, 69, 475 in Colorado), so `TREE.DRYBIO_<component>` is `NA` for those species,
not 0. In `biomassStarter.R`, the long-format `data` frame (one row per tree x component, filtered
to the requested `component`(s)) fed these `NA`-valued rows straight into the per-tree table `t`/
`tPlt`. `sum(bPlot, na.rm = TRUE)` correctly zeroed out their contribution to `BIO_ACRE` (hence the
point estimate/SE always matched), but `nPlots_TREE` (computed downstream in `sumToEU()` as
`length(unique(PLT_CN))` over that same tree table) still counted a plot whose only tallied trees
were woodland species that don't model the requested component, which EVALIDator's own plot count
excludes. `AG`, `ROOT`, and `FOLIAGE` are modeled for every species (including woodland form), so
the core default case and those components were never affected.

**Fix**: added `dplyr::filter(!is.na(DRYBIO))` in `biomassStarter.R`, scoped to the tree-side
pipeline that builds `t`/`tPlt` (in both the `byPlot` and population-estimation branches) -- *not*
to the shared upstream `data` object, since the condition/area table `a` is also built from `data`
and would otherwise have qualifying conditions silently dropped whenever their only tallied trees
lacked the requested component (this was tried first and caused `BIO_ACRE` itself to drift off of
EVALIDator by up to ~50%, e.g. Arizona `STEM` going from an exact match to 20.49 vs. a true value of
4.95 -- a genuine regression, caught by rerunning the full numeric suite before treating the fix as
done). After the scoped fix, `nPlots_TREE` matches EVALIDator exactly in CO/AZ/UT for every
component tested, and `BIO_ACRE`/`BIO_ACRE_SE` are unchanged (confirmed identical to pre-fix values
in RI/NC/CO/OR/AZ/UT). `carbon()` and `volume()` were checked and do not share this bug: `carbon()`
always sums all of `CARBON_AG`/`CARBON_BG`/etc. before ever building its area/tree tables (no
`component`-driven pre-filter of the shared join), and `volume()` has no `component` argument at
all.

## Notes

### Why the original pass found nothing (and this one did)

The original validation pass (documented above) exercised `component` variants across only 4
states (RI, NC, CO, OR) and checked `BIO_ACRE`/`BIO_ACRE_SE` but not `nPlots_TREE`/`nPlots_AREA` for
most of them -- so the bug above was present but invisible: CO's discrepancy would have shown up
only in a `nPlots_TREE` assertion, which hadn't been added yet, and RI/NC/OR have essentially no
woodland-form species so no discrepancy exists there regardless. Extending the tests to check
sampling error and plot-count columns for every `component` variant (not just the core default
case) is what surfaced it.

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population estimate (only totals-vs-per-acre was
  checked numerically; `byPlot` output was sanity-checked for plausible magnitude/distribution but
  not reconciled to the population estimate via the stratified estimator, same as the `tpa()` pass).
- `treeType = 'gs'` has no direct EVALIDator "current stock" biomass equivalent (see "Component-to-
  attribute mapping" above) -- only structural coverage exists (pre-existing `test-biomass.R` Test 2).
- `method` options other than `'TI'` (EVALIDator has no equivalent; these need
  internal-consistency-only checks per the plan, not yet added).
- `bySizeClass` was only checked structurally (pre-existing `test-biomass.R` Test 6), not against an
  EVALIDator size-class breakdown.
