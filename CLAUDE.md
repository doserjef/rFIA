# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

rFIA is a CRAN R package for analyzing USFS Forest Inventory and Analysis (FIA) data — design-based
estimators (Bechtold & Patterson 2005) for forest attributes over user-defined space/time domains.
Pure R, no compiled code (no `src/`).

## Design Approach

- All joins and database logic should adhere to the most recent FIA database documentation found [here](https://research.fs.usda.gov/understory/forest-inventory-and-analysis-database-user-guide-nfi) and in the FIA population estimation guide found at `core_references/fia_pop_estimation_user_guide.pdf`.  
- All estimation functions should follow the statistical post-stratified estimators described in `core_references/bechtoldFIA.pdf` and `core_references/westfall2022USDA.pdf`. 
- Estimates must match FIA EVALIDator tool when using the temporally indifferent estimation method.

## Build, test, check

No Makefile or custom scripts — use standard R tooling directly:

- Run tests: `devtools::test()` or `testthat::test_dir("tests/testthat")`
- Full package check: `devtools::check()` or `R CMD build . && R CMD check *.tar.gz`
- CI runs `rcmdcheck::rcmdcheck()` (via `r-lib/actions/check-r-package`) across ubuntu/macOS/windows, plus `covr::codecov()`
- `tests/` is excluded from the built tarball (`.Rbuildignore`), so tests only run from a git checkout, never against an installed package
- Render a single vignette: `./compile <name>` (runs `rmarkdown::render('<name>.Rmd')` and opens the result)

### System dependencies

Tests and checks require system libraries, not just R packages:
- `sf` needs GDAL, GEOS, and PROJ (e.g. `libgdal-dev libgeos-dev libproj-dev proj-bin`)
- `Suggests: R2jags, coda` means JAGS must be installed system-side for any code path touching the
  quantile-regression models in `inst/extdata/*.jag`
- Minimum R version is 4.1.0 — the codebase uses native lambda syntax (`\(x) ...`) throughout

## Documentation: do NOT run `devtools::document()`

Do not use `roxygen2` for package documentation. The man files should be directly edited in the `man/` directory. 


## Code style

- Functions and arguments use **lowerCamelCase** (`getFIA`, `growMort`, `grpBy`, `nCores`), not snake_case
- 2-space indentation, no tabs
- Heavy use of dplyr pipes (`%>%`) and tidy-eval (`rlang::enquo`, `!!!`) inside function bodies
- New functions, arguments, and documentation should follow the standards set by `biomass()`. 

## Architecture pattern

Each exported estimator function (e.g. `R/tpa.R`) is a thin dispatcher: it quotes arguments via
`rlang::enquo`, picks a local or remote iterator, then `lapply`s over a paired `*Starter.R` file
(e.g. `R/tpaStarter.R`) that does the actual per-state/per-iteration computation. This split exists
for every estimator (`areaStarter`, `biomassStarter`, `carbonStarter`, `dwmStarter`, etc.) and is what
lets the same estimator work both in-memory and against a `Remote.FIA.Database` (state-by-state,
memory-conserving). New estimators should follow this same dispatcher + Starter split.

## Release process

This package ships to CRAN. Every user-facing change should add a corresponding bullet to `NEWS.md` 
at the time of the change.  
