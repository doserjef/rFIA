# rFIA v1.1.0

+ Jeff Doser is the new package maintainer (jwdoser@ncsu.edu).  
+ Updated the `fiaRI` object to reflect recent changes in the FIA Database. These changes resulted in the package functions successfully working with the previous version of `fiaRI` but not working for actual user data when pulling data from recent versions of the FIA Database.
+ Updated functionality for working with external spatial (`sf`) objects with the following functions: `tpa()`. Changes in recent versions of the `sf` package led to errors when attempting to return a spatial object. This bug is now fixed.
