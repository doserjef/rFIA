# rFIA v1.1.2.9000

+ Updated the `biomass()` function to fix a bug in reported estimates when `component = 'TOTAL'`. There was a mismatch in what was reported between the documentation and the function output. The esitmate provided simply added up all biomass across the different components provided by `biomass()`, which did not make much sense since the different components are not mutually exclusive. This is now fixed such that `component = 'TOTAL'` provides biomass estimates equal to the sum of ROOT, STEM, STEM_BARK, BRANCH, and FOLIAGE components. Apologies for the inconvience this error may have caused.  

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
