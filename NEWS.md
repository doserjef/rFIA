# rFIA (development version)

+ Fixed a bug in `vitalRates()` where `SAWVOL_GROW`/`SAWVOL_GROW_AC` (sawlog board-foot volume
  growth) was computed from the same growing-stock growth-accounting component used for the other
  four growth metrics (`DIA_GROW`, `BA_GROW`, `NETVOL_GROW`, `BIO_GROW`), rather than the
  sawtimber-specific component EVALIDator's sawlog-volume growth attributes are actually defined
  against -- the same distinction `growMort()` already makes for its own `SAWVOL`/`SAWVOL_BF` state
  variables, independent of `treeType`. This under- or over-stated `SAWVOL_GROW_AC` by roughly
  0.3-3% depending on state. Point estimates and sampling errors for the other four growth metrics
  were not affected.
+ Fixed a bug in `vitalRates()` where an `areaDomain` restriction (e.g. `PHYSCLCD %in% 21:29`) was
  applied to tree-level growth using the *previous* measurement's condition instead of the current
  one, while the area (denominator) side already correctly used the current condition -- creating a
  small, state-dependent mismatch (worse in states with more physiographic-class turnover between
  remeasurements) whenever a plot's `areaDomain`-relevant classification changed between visits.
  Point estimates for `areaDomain`-restricted calls were affected; unrestricted calls were not.
+ Fixed a bug in `vitalRates()` where `landType = 'forest'` silently dropped all area-change
  information for any condition whose proportion was measured on the macroplot (`COND.PROP_BASIS ==
  'MACR'`) rather than the standard subplot -- the internal area-change join hardcoded
  `SUBP_COND_CHNG_MTRX.SUBPTYP == 1`, when the FIA Population Estimation User Guide's own
  growth-accounting methodology requires `SUBPTYP == 3` for macroplot-basis conditions. This was
  invisible in states where forest conditions are exclusively subplot-basis (confirmed for RI, NC,
  CO), but caused a small, consistent undercount of `BIO_GROW_AC` (~-0.1%) and `nPlots_AREA` (~-0.6%)
  in Pacific/Western states that commonly use macroplot sampling (confirmed for OR, CA, WA). A
  residual, smaller discrepancy specific to `landType = 'timber'` in those same three states remains
  under investigation (see `core_references/validation/vitalRates.md`).
+ Fixed a bug in `vitalRates()` where `nPlots_TREE` did not reflect restrictions imposed by
  `treeDomain` at all -- even a `treeDomain` matching zero trees left `nPlots_TREE` unchanged from the
  unrestricted value. Every row of a `bySpecies = TRUE` call reported the same, unrestricted
  `nPlots_TREE` regardless of how common that species actually was, defeating its use as the
  degrees of freedom for a t-based confidence interval. Root cause: the tree list's plot-count filter
  depended only on whether a tree had a valid growth-accounting record for the current
  `landType`/`treeType` (via FIA's precomputed `TREE_GRM_COMPONENT` columns), not on whether the
  user's `treeDomain`/`areaDomain` indicator (`tDI`) actually matched -- `tDI` only zeroed the growth
  values for non-matching trees, without excluding them from the plot count. Point estimates and
  sampling errors were not affected.
+ Fixed a bug in `vitalRates()` where `nPlots_AREA` did not reflect restrictions imposed by
  `landType` or `areaDomain` at all (the same class of bug already fixed in `tpa()`, `area()`,
  `carbon()`, `biomass()`, `volume()`, `dwm()`, `invasive()`, `seedling()`, `standStruct()`,
  `diversity()`, and `vegStruct()`; see above), including the same spurious/empty-result-with-warning
  edge case when a restriction matched no data. Point estimates and sampling errors were not affected.
+ Fixed a bug in `vegStruct()` where the internal `GROWTH_HABIT_CD` domain mapping only covered the 5
  core national vegetation growth-habit codes, silently dropping records coded with a region-specific
  code (`DS` = dead pinyon-species shrubs, populated by certain Interior West units; `SS` = newly
  sprouted post-fire shrub cover, populated only for Pacific Northwest Research Station Fire Effects
  plots; also added the documented but previously unseen `AL`/`MO`/`SL`/`ST` codes) -- since
  `GROWTH_HABIT` is part of `vegStruct()`'s internal grouping and its final output step drops any row
  with a missing group value, every record with one of these codes was silently invisible to users,
  not just mislabeled. Confirmed real data loss in 2 of 4 states checked (Colorado: 1172 raw `DS`
  records; Oregon: 231 raw `SS` records).
+ Fixed a bug in `vegStruct()` where `byPlot = TRUE`'s per-plot cover estimate (`PROP_COVER`) used
  `mean(cover, na.rm = TRUE)` across a layer/growth-habit combination's subplot-level cover values,
  which treats a subplot where that combination wasn't recorded as a missing observation to exclude
  from the average rather than a real 0%-cover observation to include (the same bug already fixed in
  `invasive()`'s `byPlot` branch). This inflated `PROP_COVER` by up to 4x for a combination recorded on
  fewer than all 4 subplots -- the normal case for patchy vegetation. Fixed by dividing by a fixed 4
  subplots instead, matching the population-estimate branch's own formula. `byPlot = TRUE` was the only
  affected output; the main population-level `COVER_PCT` was not affected.
+ Fixed a bug in `vegStruct()` where `nPlots_AREA` did not reflect restrictions imposed by `landType`
  or `areaDomain` (the same class of bug already fixed in `tpa()`, `area()`, `carbon()`, `biomass()`,
  `volume()`, `dwm()`, `invasive()`, `seedling()`, `standStruct()`, and `diversity()`; see below), and
  an `areaDomain`/`landType` restriction matching no data could produce a spurious result instead of a
  clean empty one. Point estimates and sampling errors were not affected by either fix.
+ Fixed a bug in `diversity()` where grouping by a `TREE`-table variable (`bySizeClass = TRUE`, or a
  user-supplied `grpBy` referencing a `TREE` column, e.g. species group) corrupted the area
  denominator: each forest condition's area was collapsed into whichever one grouping bin happened to
  be encountered first, instead of correctly contributing to every bin its trees actually belong to.
  This could push alpha-level Shannon's Equitability (`Eh_a`) above 1, which is mathematically
  impossible under its own formula. Fixed by separating the area-only grouping columns from the full
  grouping columns internally (matching `tpa()`'s existing `aGrpBy`/`grpBy` split), so a `TREE`-level
  grouping variable no longer fragments the area total. `grpBy` restricted to `PLOT`/`COND` columns
  (e.g. `OWNGRPCD`) was not affected.
+ Fixed a bug in `diversity()` where an `areaDomain`/`landType` restriction matching no data produced
  a spurious result (`H = S = 0` with a `"no non-missing arguments to max"` warning) instead of a
  clean empty result. `nPlots_AREA` also did not reflect restrictions imposed by `landType` or
  `areaDomain` (the same class of bug already fixed in `tpa()`, `area()`, `carbon()`, `biomass()`,
  `volume()`, `dwm()`, `invasive()`, `seedling()`, and `standStruct()`; see below). Point estimates and
  sampling errors were not affected by either fix.
+ Fixed a bug in `standStruct()` where a forest condition with zero qualifying live trees (e.g. a
  young/sparse/non-stocked stand) survives the internal tree-list join as a phantom row indistinguishable
  from any other such condition on the same plot by `(PLT_CN, SUBP, TREE)` alone (`SUBP`/`TREE` are both
  `NA`). Whenever a single plot had two or more zero-tree forest conditions, `distinct(PLT_CN, SUBP,
  TREE)` collapsed them into one, silently dropping every zero-tree condition's area past the first from
  its stand structural stage ("mosaic") classification -- even though that area still correctly counted
  in the area total, so `COVER_PCT` summed across all four structural stages fell short of 100% (e.g.
  North Carolina: 99.986% instead of 100%). Fixed by adding `CONDID` to the deduplication key.
+ Fixed a bug in `standStruct()` where an `areaDomain`/`landType` restriction matching no data produced
  a spurious single-row `'mosaic'` result (with a `"no non-missing arguments to max"` warning) instead
  of a clean empty result, because the internal structural-stage classification step could still assign
  a fallback `'mosaic'` label to a plot with no real qualifying conditions. `nPlots_AREA` also did not
  reflect restrictions imposed by `landType` or `areaDomain` (the same class of bug already fixed in
  `tpa()`, `area()`, `carbon()`, `biomass()`, `volume()`, `dwm()`, `invasive()`, and `seedling()`; see
  below). Point estimates and sampling errors were not affected by either fix.
+ Fixed a bug in `seedling()` where the tree list's `distinct(PLT_CN, SUBP, SPCD)` deduplication key
  omitted `CONDID`. Unlike `TREE`, `SEEDLING` has no per-stem ID -- `TPA_UNADJ` is already a count
  pre-aggregated to the `PLT_CN`/`SUBP`/`CONDID`/`SPCD` grain by FIA -- so whenever a subplot straddled
  two conditions and the same species had seedlings recorded under both, this silently discarded one
  condition's count entirely. Confirmed on real data (North Carolina): this undercounted statewide
  seedling TPA by ~0.2%, moving it from 1228.331 to the correct 1230.528 (matching EVALIDator exactly)
  after the fix. Small/simple states (e.g. Rhode Island) rarely hit the triggering condition and were
  unaffected.
+ Fixed a bug in `seedling()` where `nPlots_TREE` counted every forest plot rather than just plots
  with at least one live seedling actually recorded, because its `TREE_BASIS` column (always `'MICR'`
  for seedlings) couldn't detect a phantom "no seedling" row the way `tpa()`'s `DIA`-derived
  `TREE_BASIS` does. Point estimates and sampling errors were not affected, only the reported plot count.
+ Fixed a bug in `seedling()` where `nPlots_AREA` did not reflect restrictions imposed by `landType` or
  `areaDomain` (the same class of bug already fixed in `tpa()`, `area()`, `carbon()`, `biomass()`,
  `volume()`, `dwm()`, and `invasive()`; see below -- `seedling()` was the one remaining estimator
  missing this fix). Point estimates and sampling errors were not affected.

+ Updated the internal `REF_PLANT_DICTIONARY` reference table (used by `invasive()` to attach a scientific/common name to each invasive species code) from an out-of-date snapshot to the current version provided by FIA. The previous version only included species-level PLANTS codes; genus-level codes (used whenever field crews identify an invasive plant's genus but not its exact species, e.g. `LIGUS2` for *Ligustrum* spp., a major invasive shrub genus in the southeastern US) were entirely absent. Because the name columns are part of `invasive()`'s internal grouping and its final output step drops any row with a missing group value, every genus-level species was silently dropped from the output entirely -- not just its name, but its real `COVER_PCT` data. This affected up to 36% of raw invasive-species records in the states checked (North Carolina). `tpa()`/`biomass()`/`volume()`'s analogous tree-species reference table was checked and does not have this problem.
+ Fixed a bug in `invasive()` where `byPlot = TRUE`'s per-plot cover estimate (`PROP_INV_COVER`) used `mean(cover, na.rm = TRUE)` across a species' subplot-level cover values, which treats a subplot where the species wasn't recorded as a missing observation to exclude from the average rather than a real 0%-cover observation to include. This inflated `PROP_INV_COVER` by up to 16x for a species detected on fewer than all 4 subplots -- the normal case for patchy invasive species. Fixed by dividing by a fixed 4 subplots instead. As part of this fix, `byPlot = TRUE` now also returns a `PROP_FOREST` column (the proportion of the plot meeting the land type/area domain, i.e. `CONDPROP_UNADJ`-weighted forest proportion), matching the same split already used by `biomass()`'s `byPlot = TRUE` output (`BIO_ACRE`/`PROP_FOREST`) -- `PROP_INV_COVER` itself is not weighted by plot forest proportion. `byPlot = TRUE` was the only affected output; the main population-level `COVER_PCT` was not affected.
+ Fixed a bug in `invasive()` where `nPlots_AREA` did not reflect restrictions imposed by `landType` or `areaDomain`, instead always reporting the plot count for the broader unrestricted land base (the same class of bug already fixed in `tpa()`, `area()`, `carbon()`, `biomass()`, `volume()`, and `dwm()`; see above -- `invasive()` was the one remaining estimator missing this fix). Point estimates and sampling errors were not affected.
+ Fixed a bug in `invasive()` where an `areaDomain`/`landType` restriction matching no data could produce a spurious `"no non-missing arguments to max"` warning instead of a clean empty result, unlike every other estimation function (which this warning class was already fixed for). Root cause: the population-estimation branch was missing a filter to drop conditions with no detected invasive species, present only in the (separate) `byPlot` branch's code; when every species group was empty, the resulting phantom "no species" row(s), rather than a genuinely empty result, bypassed the existing 0-row guard in the shared `combineMR()` utility.

+ `biomass()` no longer estimates carbon (`CARB_ACRE`/`CARB_TOTAL`/associated SE and variance columns have been removed from its output). Tree biomass estimation is otherwise unchanged. Use `carbon()` for carbon stock estimation, which covers the full suite of forest ecosystem carbon pools (live and dead trees, understory vegetation, down dead wood, litter, and soil organic matter), not just standing tree carbon.
+ Fixed a bug in `areaChange()` where a condition that was nonsampled (`COND_STATUS_CD == 5`, e.g. hazardous or denied-access) at either measurement was misclassified as a genuine forest/non-forest (or timberland/non-timberland) land-use change event, since the shared `landTypeDomain()` helper has no distinct category for "nonsampled" -- it simply isn't forest, indistinguishable from a real non-forest reclassification. This fabricated diversion/reversion events that never actually occurred, and could bias `AREA_CHNG`/`PERC_CHNG` in either direction depending on how the affected plots happened to fall; confirmed on real data that this flipped the sign of the reported net change in forest area for Rhode Island. `area()` was not affected (a nonsampled condition simply contributes no area there, rather than being paired against a different point in time).
+ Fixed a bug in `area()` where a hard-coded `PLOT_STATUS_CD == 1` filter (a leftover from code shared with `tpa()`, where it is valid since trees only occur on forest land) silently dropped every plot with no accessible forest before land-type domain indicators were applied. This caused large undercounts (up to two orders of magnitude) for every `landType` value other than the defaults of `'forest'`/`'timber'` (e.g. `'water'`, `'non-forest'`, `'all'`), and caused the documented `byLandType = TRUE` output to sum to well under the true total land area. `landType = 'forest'`/`'timber'` estimates were not affected.
+ Fixed a bug in `area()` where `nPlots_AREA_DEN` did not reflect restrictions imposed by `landType = 'timber'` or `areaDomain`, instead always reporting the plot count for the broader `landType = 'forest'` land base (the same class of bug as the `tpa()` `nPlots_AREA` fix described below). Point estimates and sampling errors were not affected.
+ Fixed a bug in the shared internal utility used to evaluate a user-supplied `areaDomain` where the plots/conditions used to evaluate the domain expression were hard-coded to forest land (`PLOT_STATUS_CD == 1`/`COND_STATUS_CD == 1`), regardless of `landType`. This caused `area()` (and `areaChange()`) to silently return zero area for any combination of a non-forest `landType` (e.g. `'water'`, `'non-forest'`, `'all'`) with an `areaDomain` filter, instead of applying the filter and returning the correctly restricted estimate. This is a rare use case. 
+ Updated `area()` where `landType = 'all'` to now explicitly remove nonsampled conditions from the land area calculation (e.g. hazardous or denied-access plots). In prior versions, nonsampled conditions were included in the count of `landType = 'all'`, but this resulted in the sum of the different components when `byLandType = TRUE` to not sum to the total when `landType = 'all'`. 
+ Fixed a bug in `tpa()` where `nPlots_AREA` did not reflect restrictions imposed by `landType = 'timber'` or `areaDomain`, instead always reporting the plot count for the broader `landType = 'forest'` land base. Point estimates and sampling errors were not affected, but `nPlots_AREA` is documented as the recommended degrees of freedom for constructing t-based confidence intervals, so an inflated count understated the true margin of error for any `landType = 'timber'` or `areaDomain`-restricted estimate.
+ Fixed a bug where a `treeDomain`/`areaDomain` matching no data, combined with the default `mostRecent = TRUE` behavior, produced a spurious `"no non-missing arguments to max"` warning instead of a clean empty result. This affected all estimation functions (not just `tpa()`), since the underlying cause was in a shared internal utility.
+ Fixed a bug where `treeType = 'dead'` did not require dead trees to meet the "standing dead" tally-tree criteria (`STANDING_DEAD_CD == 1`), instead counting all trees with `STATUSCD == 2` regardless of whether they were still standing. This inflated `treeType = 'dead'` estimates in states with a meaningful number of down or broken dead trees recorded in the tree table (e.g. North Carolina, where the estimate was roughly 4x too high). This affects every function that supports `treeType`: `tpa()`, `diversity()`, `biomass()`, `volume()`, and `fsi()`. As a consequence, `treeType = 'all'` (which includes every tree regardless of status) is no longer equal to `treeType = 'live'` plus `treeType = 'dead'`, since `'all'` still includes the non-standing dead trees that `'dead'` now excludes.
+ Fixed a bug in `carbon()` and `biomass()` where `nPlots_AREA` did not reflect restrictions imposed by `landType` or `areaDomain`, instead always reporting the plot count for the broader unrestricted land base (the same class of bug already fixed in `tpa()` and `area()`; see above). Point estimates and sampling errors were not affected. For `carbon()` specifically, this also caused an `areaDomain` matching no conditions to return a row of `NaN` values instead of a clean empty result, since `carbon()`'s numerator is built by joining onto the same phantom-row-containing condition list; this is now a clean 0-row result, consistent with every other estimation function.
+ Fixed a bug in `biomass()` where `nPlots_TREE` over-counted plots for any `component` (or `byComponent`) request restricted to `STEM`, `STEM_BARK`, `STUMP_BARK`, `BOLE`, `BOLE_BARK`, or `BRANCH`, in states with a meaningful amount of woodland-form forest (e.g. pinyon-juniper woodland in Arizona, Utah, and Colorado). NSVB does not model these components for woodland species (`DRYBIO_STEM`/etc. are `NA`, not 0, for e.g. juniper and pinyon), and while their 0 contribution to `BIO_ACRE`/`BIO_ACRE_SE` was already handled correctly, plots whose only tallied trees were woodland species were still being counted toward `nPlots_TREE`. This inflated `nPlots_TREE` by up to ~3x in woodland-heavy states (e.g. Arizona `BRANCH`: 3137 reported vs. 842 actual contributing plots); point estimates and sampling errors were not affected.
+ Fixed a bug in `volume()` where `nPlots_AREA` did not reflect restrictions imposed by `landType` or `areaDomain`, instead always reporting the plot count for the broader unrestricted land base (the same class of bug already fixed in `tpa()`, `area()`, `carbon()`, and `biomass()`; see above -- `volume()` was the one remaining estimator missing this fix). Point estimates and sampling errors were not affected.
+ Fixed two related bugs in `volume()` where `nPlots_TREE` over-counted plots: (1) trees with no defined bole volume (e.g. dead trees under 5 inches DBH, for which `VOLCFNET` is never computed) were still counted, the same class of bug just fixed in `biomass()` but triggered by tree diameter rather than species; and (2) trees with a defined but exactly-zero net volume (a full cull/defect deduction can legitimately zero out `VOLCFNET`) were counted, where EVALIDator's own definitions require a strictly positive volume to count a tree as contributing. Point estimates and sampling errors were not affected; `nPlots_TREE` was inflated by a few percent in the states checked (e.g. Rhode Island `treeType = 'dead'`: 107 reported vs. 103 actual contributing plots).
+ Fixed a bug in `dwm()` where `nPlots_AREA` did not reflect restrictions imposed by `landType` or `areaDomain`, instead always reporting the plot count for the broader unrestricted land base (the same class of bug already fixed in `tpa()`, `area()`, `carbon()`, `biomass()`, and `volume()`; see above -- `dwm()` was the one remaining estimator missing this fix). Point estimates and sampling errors were not affected.
+ Fixed a bug in `dwm()` where `COND_DWM_CALC` was filtered by `PLT_CN` alone when restricting to the current evaluation, but a single plot can appear in `COND_DWM_CALC` under several different `EVALID`s (consecutive annual panels can each report the same not-yet-remeasured plot as their most recent down woody material data), each with slightly different evaluation/stratum-specific adjustment factors. This caused every down woody material condition to be summed once per matching `EVALID` instead of once, inflating the reported `nPlots_DWM` plot count by roughly 4-5x in the states checked (e.g. Colorado: 17775 reported vs. 3897 actual contributing plots) and, more subtly, adding spurious phantom estimation-unit groups with no effect on the final point estimate or standard error (their area contribution was always `NA` and dropped), but real effect on the plot count. Point estimates and sampling errors were not affected; only `nPlots_DWM` was.
+ Fixed a bug in `dwm()` where `nPlots_DWM` counted every domain-qualifying, DWM-sampled plot regardless of whether it actually had any down woody material of the relevant fuel type, instead of requiring a strictly positive value (matching EVALIDator's own per-attribute definitions, and the same class of fix just made in `volume()`). This was checked and fixed separately for the combined default output (`byFuelType = FALSE`, which requires total FWD + CWD + pile volume across all fuel types to be positive, matching EVALIDator's combined "Total volume of DWM" attribute) and for `byFuelType = TRUE`'s individual fuel-type rows (each of which now requires only its own fuel type's volume -- or biomass, for `DUFF`/`LITTER`, which have no volume equivalent -- to be positive, matching each fuel type's own EVALIDator attribute). Point estimates and sampling errors were not affected.

# rFIA v1.1.4

+ Removed `.dots` argument from all calls to `dplyr::group_by()`, which resulted in an error with the latest version of `dplyr` (see [#54](https://github.com/doserjef/rFIA/issues/54)).  
+ Removed dependency on the `bit64` package. 
+ Removed `N` from the return output of all model fitting functions as this was not always being properly calculated when different filters were applied. Additionally, we updated our recommended approach for calculating confidence intervals and so this value is no longer part of that recommended calculation. 
+ Removed the argument `variance` from all estimation functions, with the exception of `fsi()`. Previous documentation was misleading in that it said valid confidence intervals cannot be constructed from the sampling errors. This is not strictly true. The sampling error is a function of the variance/standard error, and so the sampling error *can* be used to calcualte confidence intervals when manipulated appropriately. See the note in all model fitting functions documentation for further details on how to do this. 
+ Confidence interval calculations provided by `fsi()` were too precise. This has been updated to better reflect the amount of uncertainty in the associated estimates. Confidence intervals are now calculated with the number of plots used to inform the given FSI estimate, not the number of plots within all estimation units that encompass the population of interest. Consider the case where we calculate FSI for an individual species. Because FSI is a measure of change over time, only plots where the species was present at for at least one time point go into informing the FSI estimate. The previous use of all plots, even those without the species of interest, substantially inflated the sample size.  
+ Fixed some minor bugs in `plotFIA()` that errored when including error bars. Also fixed code to remove a warning message in simple time series plots.  
+ Updated the "Estimating Forest Attributes" document, particularly the section on sampling error and how to calculate confidence intervals. 

# rFIA v1.1.3

+ Substantial updates to the `fsi()` function. Some of these functions fixed some common errors that could be encountered under specific circumstances where the function broke, which happened as a result of updates to FIADB since the last time this function underwent a major update. An additional update fixes an important bug where the `scaleBy` function would not always work as was reported. In particular, under certain situations (namely `byPlot = TRUE`) the subsequent calculations of relative density did not use the level-specific intercepts and slopes that were estimated in the regression model, and instead the overall mean was used (i.e., equivalent to if `scaleBy` was not specified). This could lead to sub-optimal accuracy of the relative density calculations, and in subsequent FSI outputs. Apologies for any problems this may have caused.  
+ Updated the `biomass()` function to fix a bug in reported estimates when `component = 'TOTAL'`. There was a mismatch in what was reported between the documentation and the function output. The estimate provided simply added up all biomass across the different components provided by `biomass()`, which did not make much sense since the different components are not mutually exclusive. This is now fixed such that `component = 'TOTAL'` provides biomass estimates equal to the sum of ROOT, STEM, STEM_BARK, BRANCH, and FOLIAGE components. Apologies for the inconvenience this error may have caused.  

# rFIA v1.1.2

+ Updated all estimation functions to allow grouping by variables in the `PLOTGEOM` database table within the `grpBy` argument. Also changed `readFIA()` to by default read in `PLOTGEOM` as one of the common database tables. Thanks to Jacob Fraser for the suggestion [here](https://github.com/doserjef/rFIA/issues/55).
+ [Fixed a bug](https://github.com/doserjef/rFIA/pull/58) with `dtplyr 1.3.2` that led to an error in `areaChange()` 


# rFIA v1.1.1

+ Jeff Doser is the new package maintainer. Please send all inquiries via email to Jeff (jwdoser@ncsu.edu) or post potential bugs on the GitHub development page.  
+ Updated the `fiaRI` object to reflect recent changes in the FIA Database. These changes resulted in the package functions successfully working with the previous version of `fiaRI` but not working for actual user data when pulling data from recent versions of the FIA Database.
+ Updated functionality for working with external spatial (`sf`) objects with the following functions: `tpa()`. Changes in recent versions of the `sf` package led to errors when attempting to return a spatial object. This bug is now fixed.
+ Updated a substantial bug in `area()` and `areaChange()` that resulted in incorrect area (or area change) estimates being reported when specifying `treeDomain` and `grpBy` (when using grouping variables from TREE). In the previous version, the filters were not properly applied, and so area estimates did not adequately represent the filtering conditions and often just provided the same values as if `treeDomain` was not specified. Estimates now provide correct results that are more inline with intuition. For example, if specifying `treeDomain = SPCD == 121` [i.e., longleaf pine], the previous `area()` function would essentially ignore this and return area of all forest plots. Now, `area()` will return the estimate of land area where at least one longleaf pine tree occurs. Further, the estimate of percent area will be the percentage of total land area (which is determined by `landType`) that contains longleaf pine.  
+ Substantial updates to `biomass()`. Previous versions were not compatible with updates in FIADB and the new National Scale Volume and Biomass (NSVB) estimators. The function is now updated and returns biomass and carbon estimates using the NSVB procedure. 
+ Updated `findEVALID()` to return the correct evaluation IDs. Previous versions had an incorrect join that resulted in additional, incorrect EVALIDs being returned for a given set of criteria. This function should only be used by users familiar with FIA and desiring to use FIA data for use outside of `rFIA`, as `rFIA` is built in a way that users do not need to directly interact with EVALIDs. 
+ Updated `dwm()` when `byPlot = TRUE` to set the `YEAR` column equal to the year each plot was measured (`MEASYEAR`), which may differ slightly from its associated inventory year (`INVYR`). This is what all other `rFIA` functions do and what was reported in the manual, but the `YEAR` returned prior to this version was actually the inventory year. 
+ Fixed a bug with `growMort()` that resulted in estimates of mean annual survivor growth and mean annual net change reporting as 0.
+ Fixed a discrepancy with `growMort()` calculation of removals and the description of it in the manual. Removal estimates provided by `growMort()` do NOT include stems that grow beyond the 5-inch diameter threshold and then are subject to harvest or natural mortality before the remeasurement period. In other words, `rFIA` recruitment does not include trees corresponding to FIA growth components of CUT2 and MORTALITY2.  
+ Fixed a typo in the `standStruct()` documentation that incorrectly said the lower diameter for Pole class was set at 11cm while it is in fact set at 12.7cm (5in).  
+ Fixed typo in documentation of `plotFIA()` regarding the error bars produced when `se = TRUE`. These are 95% confidence intervals, not 68% confidence intervals.
+ Added more details to `vegStruct()` on reporting of estimates by canopy layer and growth habit.
+ Updated internal data to now contain the Dec 2024 `REF_SPECIES` table from FIADB, which provides access to the `CARBON_RATIO_LIVE` attribute for using the NSVB species-specific carbon fractions. 
+ Updated all estimation functions to fix a bug that resulted in an error when setting `method = 'EMA'`. 
+ Removed all references to "ECOSUBCD" in the help pages since this column was removed from the PLOT table in FIADB v9.3. 
+ Updated `writeFIA()` to allow users to write database tables by state when only a subset of the table is originally read into R. This currently requires either the PLOT or COND tables to be read in.  
+ Fixed a bug in `plotFIA()` that led to an error in animated plots when `gganimate` was not loaded (note that `gganimate` still needs to be installed).
