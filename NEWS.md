# rFIA v1.1.0

+ New package maintainer (jwdoser@ncsu.edu).  
+ Updated the `fiaRI` object to reflect recent changes in the FIA Database. These changes resulted in the package functions successfully working with the previous version of `fiaRI` but not working for actual user data when pulling data from recent versions of the FIA Database.
+ Updated functionality for working with external spatial (`sf`) objects with the following functions: `tpa()`. Changes in recent versions of the `sf` package led to errors when attempting to return a spatial object. This bug is now fixed.
+ Updated `dwm()` when `byPlot = TRUE` to set the `YEAR` column equal to the year each plot was measured (`MEASYEAR`), which may differ slightly from its associated inventory year (`INVYR`). This is what all other `rFIA` functions do and what was reported in the manual, but the `YEAR` returned prior to this version was actually the inventory year. 
