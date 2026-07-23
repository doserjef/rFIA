# Validation report: `invasive()`

## Scope

This pass covers `invasive()` -- percent areal cover of invasive plant species, reported per species
(species-level grouping is always active; there is no `bySpecies` argument because species *is* the
row granularity). `invasive()` is condition/subplot-based (no `TREE` table), draws on
`INVASIVE_SUBPLOT_SPP` and `SUBP_COND` (not used by any prior validation pass), and is further
restricted to only the subset of plots flagged `INVASIVE_SAMPLING_STATUS_CD %in% 1:2` -- a P2
ancillary protocol, not collected identically to core inventory attributes.

## Methodology: no EVALIDator ground truth exists for this function

Unlike every function validated so far, **EVALIDator has no invasive-species attributes at all.**
`EVALIDATOR_POP_ESTIMATE.csv` (719 total attributes) has zero matches for "invasive", "P2VEG",
"understory", "noxious", "nonnative"/"non-native", or "ground cover", and its `EVAL_TYP` list
(`EXPVOL`, `EXPALL`, `EXPCURR`, `EXPDWM`, `EXPCHNG`, `EXPGROW`, `EXPREMV`, `EXPMORT`) has nothing
invasive-species-related. This isn't a gap in the API mapping -- EVALIDator's web tool itself has no
"Invasive Plants" report category. All validation here is therefore either:

1. **Cross-checks against `tpa()`'s own `nPlots_AREA`** (already validated against EVALIDator; see
   `tpa.md`) for the same `landType`/`areaDomain` restriction. This works because `nPlots_AREA` is
   fundamentally a property of the area denominator (which forest-land conditions/plots qualify),
   not of the specific numerator being estimated -- and empirically, `INVASIVE_SAMPLING_STATUS_CD`
   turns out not to shrink the plot universe at all in 3 of the 4 states checked.
2. **Manual hand-calculations from raw `INVASIVE_SUBPLOT_SPP`/`SUBP_COND`/`COND` data**, replicating
   `invasiveStarter.R`'s own per-plot cover formula independently, for specific (plot, species)
   combinations.
3. **Internal consistency** (totals/per-acre, `returnSpatial`, empty-domain).

Four states were used: RI (Northern), NC (Southern), CO (Interior West). **OR was substituted with
ID** for the Pacific Northwest region -- OR's local `INVASIVE_SUBPLOT_SPP` extract is header-only (no
data rows at all), as is WA's; ID is the nearest state with real invasive-species sampling data.

## Results

### `nPlots_AREA` cross-check against `tpa()`, 3 states (+ RI internal check)

| State | `landType='forest'` | `landType='timber'` | `areaDomain` (mesic) |
|---|---|---|---|
| NC | 3561 = 3561 | 3436 = 3436 | 2997 = 2997 |
| CO | 3925 = 3925 | 1829 = 1829 | 2121 = 2121 |
| ID | 3757 = 3757 | 2855 = 2855 | 2781 = 2781 |

**Exact match** in all three states and all three cases -- `INVASIVE_SAMPLING_STATUS_CD` doesn't
restrict the plot universe further than `landType`/`areaDomain` alone in NC/CO/ID. RI's
invasive-sampled plots (6) are a much smaller subset of its 132 forest plots, so RI is checked only
for monotonicity (`landType='timber'`/`areaDomain` restrictions never *increase* the plot count
relative to `landType='forest'`), which holds.

### `byPlot = TRUE` cover, hand-calculated from raw data (RI)

`PROP_INV_COVER` is deliberately left unadjusted by `CONDPROP_UNADJ`; that's reported separately as
`PROP_FOREST` (the same split `biomassStarter()`'s `byPlot` branch uses for `BIO_ACRE`/`PROP_FOREST`
-- see "Fixed" #2 below). *Rosa multiflora* (`ROMU`) on one RI plot: raw data shows it recorded on
subplot 2 only, 20% cover, subplot fully within a condition that's 25% of the plot's area. By hand:
`0.20 (cover) * 1.0 (SUBPCOND_PROP) * 1 (aDI) / 4 (subplots) = 0.05`, with `PROP_FOREST = 0.25` for
that plot reported alongside it. `invasive(byPlot = TRUE)` reports exactly `0.05`/`0.25` for this plot
after the fix below (previously `0.2` for cover alone, with no separate `PROP_FOREST` weighting
exposed -- a 4x difference once `CONDPROP_UNADJ` is properly factored out rather than folded in, and
16x relative to the pre-fix `mean(na.rm = TRUE)` bug). Checked similarly for *Ligustrum* spp.
(`LIGUS2`) on a North Carolina plot recorded on all 4 subplots across a single condition (90/90/90/5%
cover, one subplot only 72.3% within the condition, `CONDPROP_UNADJ = 0.930791`): hand calculation
gives `PROP_INV_COVER = 0.6840`, `PROP_FOREST = 0.930791`, matching exactly.

### Internal consistency (no EVALIDator needed)

- `totals = TRUE`: `INV_AREA_TOTAL / AREA_TOTAL * 100` reproduces `COVER_PCT` exactly, across all
  four states. **Pass.**
- `returnSpatial` (RI, by county): all non-geometry columns match exactly between
  `returnSpatial = TRUE`/`FALSE`. **Pass.**

### Empty-domain edge case

`invasive(areaDomain = STATECD == 999)` returns a clean 0-row tibble with no warning (after the fix
below). **Pass.**

## Fixed

Three bugs were found and fixed this pass, all in `invasiveStarter.R`, plus one data-completeness fix
to the package's bundled internal reference table.

**1. `REF_PLANT_DICTIONARY` out of date -- silently dropped genus-level species entirely.**
`invasiveStarter.R` joins each detected species' `VEG_SPCD` against the internal
`REF_PLANT_DICTIONARY` table to attach `SCIENTIFIC_NAME`/`COMMON_NAME`, both of which are part of
`invasive()`'s internal `grpBy`. `invasive()`'s final step does `tidyr::drop_na(grpBy)`, so any
species whose code has no dictionary match gets a fully dropped row -- not just a missing name, but
its entire `COVER_PCT` data. The bundled dictionary (37,526 rows) turned out to include only
`SYMBOL_TYPE == "Species"` entries from the full USDA PLANTS-derived reference table; genus-level
codes (`SYMBOL_TYPE == "Genus"`, used whenever a field crew can only identify a plant's genus, not its
exact species -- e.g. `LIGUS2` for *Ligustrum* spp., a major invasive shrub genus in the Southeast)
were entirely absent, along with `"Old"` (deprecated, superseded) and `"Unknown"` category codes.

  | State | Distinct species codes | Missing from old dictionary | Raw records affected |
  |---|---|---|---|
  | RI | 6 | 1 (`LONIC`) | 3 / 14 |
  | NC | 33 | 8 (`ROSA5`, `LIGUS2`, `VINCA`, `HEDER`, `ELAEA`, `LONIC`, `LESPE`, `WISTE`) | **3757 / 10455 (36%)** |
  | CO | 43 | 0 | 0 |
  | ID | 14 | 0 | 0 |

  For North Carolina, over a third of all invasive-species records were silently invisible to every
  `invasive()` user. `tpa()`/`biomass()`/`volume()`'s analogous `REF_SPECIES_DEC_2024` join was
  checked and has zero missing tree species codes across all four states -- this is specific to the
  invasive-plant dictionary, not a systemic pattern (`TREE.SPCD` is a small, FIA-curated code list;
  `VEG_SPCD` draws on the much larger USDA PLANTS system, which is more prone to bundling gaps).

  **Fix**: regenerated `intData$REF_PLANT_DICTIONARY` in `R/sysdata.rda` from the current, complete
  FIA-provided `REF_PLANT_DICTIONARY.csv` (81,489 rows spanning `Genus`/`Species`/`Old`/`Unknown`
  categories, confirmed to have zero duplicate `SYMBOL` values so every code maps unambiguously),
  keeping the same `SYMBOL`/`SCIENTIFIC_NAME`/`COMMON_NAME` structure `invasiveStarter.R`'s join
  already expects. All 8 previously-missing NC codes (and RI's `LONIC`) now resolve correctly and
  appear in `invasive()`'s output with real, positive `COVER_PCT` values (e.g. `LIGUS2`/privet: 1.49%
  cover, detected on 1132 of 3561 NC forest plots -- a substantial, previously wholly-invisible
  signal). Confirmed via `grep` that no other estimator function reads `REF_PLANT_DICTIONARY`, and
  reran `tpa()`/`biomass()`/`volume()`/`carbon()`'s full test suites to confirm the `sysdata.rda`
  update caused no other regressions.

**2. `byPlot = TRUE` cover formula inflated whenever a species wasn't detected on all 4 subplots.**
The population-estimate branch computes each plot's cover contribution as
`sum(COVER_PCT/100 * SUBPCOND_PROP * aDI * CONDPROP_UNADJ, na.rm = TRUE) / 4` -- a fixed denominator
of 4 subplots, treating a subplot with no record for that species as a real 0%-cover observation.
`byPlot = TRUE`'s branch instead computed `mean(cover, na.rm = TRUE)` over only the subplots where the
species *did* have a record -- silently treating "not recorded here" as a missing value to exclude
from the average rather than a zero to include. Since real invasive species are usually patchy (rarely
present on all 4 subplots of a plot), this was the common case, not an edge case: confirmed via hand
calculation (see "Results" above) that this inflated `PROP_INV_COVER` by 16x for one RI plot/species.

  **Fix**: rewrote the `byPlot` branch's cover formula to divide by a fixed 4 (not
  `mean(..., na.rm = TRUE)`) at `PLT_CN`/species grain. `CONDPROP_UNADJ` is deliberately *not* folded
  into this per-plot cover value -- an earlier version of this fix did multiply it in directly
  (matching the population-estimate formula exactly), but per explicit follow-up direction, it's
  instead reported as its own `PROP_FOREST` column via the `a`/`left_join` structure the `byPlot`
  branch already builds for the area denominator, mirroring the same split
  `biomassStarter()`'s `byPlot` branch uses for `BIO_ACRE`/`PROP_FOREST` (`biomassStarter.R`, `byPlot`
  branch). This keeps `PROP_INV_COVER` an unadjusted, directly-interpretable subplot-cover fraction
  rather than one already discounted by how much of the plot is forest -- a user who wants the
  forest-area-weighted quantity can compute `PROP_INV_COVER * PROP_FOREST` themselves. Verified
  against both hand-calculated examples (see "Results" above) to full precision. This only affects
  `byPlot = TRUE` output; the population-level `COVER_PCT` was never affected by this bug and still
  includes `CONDPROP_UNADJ` in its own (separate, stratification-facing) formula.

**3. `nPlots_AREA` phantom-row bug**, plus **a related, invasive()-specific consequence for the
empty-domain contract.** `invasiveStarter.R`'s condition list (`a`) was missing the `!is.na(CONDID)`
guard present in every other estimator's equivalent code (same recurring class of bug already fixed
in `tpa()`/`area()`/`carbon()`/`biomass()`/`volume()`/`dwm()`) -- fixed identically, confirmed via the
`nPlots_AREA` cross-check against `tpa()` above.

  Separately, the population-estimation branch's tree/cover list (`t`) was missing a
  `dplyr::filter(!is.na(SYMBOL))` step that the `byPlot` branch already had. A plot/condition with no
  detected invasive species survives the join as a `SYMBOL = NA` "phantom species" row; since `SYMBOL`
  is part of `grpBy`, this is normally just silently dropped later by `invasive()`'s own
  `tidyr::drop_na(grpBy)` -- harmless. But when *every* species group is empty (e.g. an `areaDomain`
  matching no data), this phantom row is the *only* row left in the intermediate estimate, with an
  all-`NA` `YEAR`, and it has `nrow() > 0` -- so it doesn't trip the existing 0-row guard in the
  shared `combineMR()` utility (the same guard that gives every other estimator a clean, warning-free
  empty-domain result). This reintroduced the exact `"no non-missing arguments to max"` warning
  class already fixed everywhere else, specifically for `invasive()`. **Fix**: added the missing
  `!is.na(SYMBOL)` filter, mirroring the `byPlot` branch.

## Notes

### Why this pass found more issues than most prior passes

`invasive()` is the first function validated with genuinely no EVALIDator ground truth at all, which
forced a closer, from-first-principles look at its own formulas (via hand calculation) rather than
leaning on an external oracle -- and its species-name join (unique among validated functions so far)
introduced a failure mode (silent row-dropping via an incomplete reference table) that a purely
attribute-matching approach would never have surfaced, since EVALIDator has no per-species invasive
attribute to diff against in the first place.

## Deferred to follow-up (not covered this pass)

- `byPlot = TRUE` aggregation reproducing the population estimate exactly (only order-of-magnitude
  agreement was checked, given the extra stratification/adjustment-factor weighting in the population
  branch that `byPlot` intentionally omits, same limitation as every other estimator's `byPlot` output).
- `method` options other than `'TI'` (no EVALIDator equivalent; internal-consistency-only checks per
  the plan, not yet added).
- A full national audit of `REF_PLANT_DICTIONARY` coverage (only the four states already downloaded
  for this validation pass were checked).
