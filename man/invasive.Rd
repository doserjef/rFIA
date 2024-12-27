\name{invasive}
\alias{invasive}

\title{Estimate invasive species coverage from FIADB}

\description{
Produces estimates of areal coverage of invasive species from the Forest Inventory and Analysis Database. Estimates can be produced for regions defined within the FIA Database (e.g. counties), at the plot level, or within user-defined areal units. All estimates are returned by species although can be grouped by other variables defined in the FIADB. If multiple reporting years (EVALIDs) are included in the data, estimates will be output as a time series. If multiple states are represented by the data, estimates will be output for the full region (all area combined), unless specified otherwise (e.g. \code{grpBy = STATECD}).}

\usage{
invasive(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, landType = "forest",
         method = 'TI', lambda = 0.5, areaDomain = NULL, totals = FALSE,
         variance = FALSE, byPlot = FALSE, nCores = 1)
}

\arguments{
  \item{db}{\code{FIA.Database} or \code{Remote.FIA.Database} object produced from \code{\link{readFIA}} or \code{\link{getFIA}}. If a \code{Remote.FIA.Database}, data will be read in and processed state-by-state to conserve RAM (see details for an example).}

 \item{grpBy}{variables from PLOT or COND tables to group estimates by (NOT quoted). Multiple grouping variables should be combined with \code{c()}, and grouping will occur heirarchically. For example, to produce seperate estimates for each ownership group within ecoregion subsections, specify \code{c(ECOSUBCD, OWNGRPCD)}.}

  \item{polys}{\code{sp} or \code{sf} Polygon/MultiPolgyon object; Areal units to bin data for estimation. Separate estimates will be produced for region encompassed by each areal unit. FIA plot locations will be reprojected to match projection of \code{polys} object.}

  \item{returnSpatial}{logical; if TRUE, merge population estimates with \code{polys} and return as \code{sf} multipolygon object. When \code{byPlot = TRUE}, return plot-level estimates as \code{sf} spatial points.}

  \item{landType}{character ("forest" or "timber"); Type of land that estimates will be produced for. Timberland is a subset of forestland (default) which has high site potential and non-reserve status (see details).}

  \item{method}{character; design-based estimator to use. One of:  "TI" (temporally indifferent, default), "annual" (annual), "SMA" (simple moving average), "LMA" (linear moving average), or "EMA" (exponential moving average). See \href{https://www.fs.usda.gov/nrs/pubs/jrnl/2020/nrs_2020_stanke_001.pdf}{Stanke et al 2020} for a complete description of these estimators.}

  \item{lambda}{numeric (0,1); if \code{method = 'EMA'}, the decay parameter used to define weighting scheme for annual panels. Low values place higher weight on more recent panels, and vice versa. Specify a vector of values to compute estimates using mulitple wieghting schemes, and use \code{plotFIA} with \code{grp} set to \code{lambda} to produce moving average ribbon plots. See \href{https://www.fs.usda.gov/nrs/pubs/jrnl/2020/nrs_2020_stanke_001.pdf}{Stanke et al 2020} for examples.}

  \item{areaDomain}{logical predicates defined in terms of the variables in PLOT and/or COND tables. Used to define the area for which estimates will be produced (e.g. within 1 mile of improved road: \code{RDDISTCD \%in\% c(1:6)}, Hard maple/basswood forest type: \code{FORTYPCD == 805}. Multiple conditions are combined with \code{&} (and) or \code{|} (or). Only plots within areas where the condition evaluates to TRUE are used in producing estimates. Should NOT be quoted.}

  \item{totals}{logical; if TRUE, return total population estimates (e.g. total area) along with ratio estimates (e.g. mean trees per acre).}

    \item{variance}{logical; if TRUE, return estimated variance (\code{VAR}) and sample size (\code{N}). If FALSE, return 'sampling error' (\code{SE}) as returned by EVALIDator. Note: sampling error cannot be used to construct confidence intervals.}

  \item{byPlot}{logical; if TRUE, returns estimates for individual plot locations instead of population estimates.}

  \item{nCores}{numeric; number of cores to use for parallel implementation. Check available cores using \code{\link{detectCores}}. Default = 1, serial processing.}
}

\details{
\strong{Estimation Details}

Estimation of forest variables follows the procedures documented in Bechtold and Patterson (2005) and \href{https://www.fs.usda.gov/nrs/pubs/jrnl/2020/nrs_2020_stanke_001.pdf}{Stanke et al 2020}.

Specifically, percent areal coverage is computed using a sample-based ratio-of-means estimator of total invasive coverage area / total land area within the domain of interest. Estimates of areal coverage of individual invasive species should NOT be summed to produce estimates of areal coverage by ALL invasive species, as areal coverage by species is not mutually exclusive (multiple species may occur in the same area). Current FIA data collection protocols do not allow for the unbiased estimation of areal coverage by all invasive species.

Users may specify alternatives to the 'Temporally Indifferent' estimator using the \code{method} argument. Alternative design-based estimators include the annual estimator ("ANNUAL"; annual panels, or estimates from plots measured in the same year), simple moving average ("SMA"; combines annual panels with equal weight), linear moving average ("LMA"; combine annual panels with weights that decay \emph{linearly} with time since measurement), and exponential moving average ("EMA"; combine annual panels with weights that decay \emph{exponentially} with time since measurement). The "best" estimator depends entirely on user-objectives, see \href{https://www.fs.usda.gov/nrs/pubs/jrnl/2020/nrs_2020_stanke_001.pdf}{Stanke et al 2020} for a complete description of these estimators and tradeoffs between precision and temporal specificity.

When \code{byPlot = FALSE} (i.e., population estimates are returned), the "YEAR" column in the resulting dataframe indicates the final year of the inventory cycle that estimates are produced for. For example, an estimate of current forest area (e.g., 2018) may draw on data collected from 2008-2018, and "YEAR" will be listed as 2018 (consistent with EVALIDator). However, when \code{byPlot = TRUE} (i.e., plot-level estimates returned), the "YEAR" column denotes the year that each plot was measured (MEASYEAR), which may differ slightly from its associated inventory year (INVYR).

Stratified random sampling techniques are most often employed to compute estimates in recent inventories, although double sampling and simple random sampling may be employed for early inventories. Estimates are adjusted for non-response bias by assuming attributes of non-response plot locations to be equal to the mean of other plots included within thier respective stratum or population.


\strong{Working with "Big Data"}

If FIA data are too large to hold in memory (e.g., R throws the "cannot allocate vector of size ..." errors), use larger-than-RAM options. See documentation of \code{link{readFIA}} for examples of how to set up a \code{Remote.FIA.Database}. As a reference, we have used rFIA's larger-than-RAM methods to estimate forest variables using the entire FIA Database (~50GB) on a standard desktop computer with 16GB of RAM. Check out \href{https://rfia.netlify.app/}{our website} for more details and examples.

Easy, efficient parallelization is implemented with the \code{\link{parallel}} package. Users must only specify the \code{nCores} argument with a value greater than 1 in order to implement parallel processing on their machines. Parallel implementation is achieved using a snow type cluster on any Windows OS, and with multicore forking on any Unix OS (Linux, Mac). Implementing parallel processing may substantially decrease free memory during processing, particularly on Windows OS. Thus, users should be cautious when running in parallel, and consider implementing serial processing for this task if computational resources are limited (\code{nCores = 1}).

\strong{Definition of forestland}

Forest land must have at least 10-percent canopy cover by live tally trees of any size, including land that formerly had such tree cover and that will be naturally or artificially regenerated. Forest land includes transition zones, such as areas between heavily forest and non-forested lands that meet the mimium tree canopy cover and forest areas adjacent to urban and built-up lands. The minimum area for classification of forest land is 1 acre in size and 120 feet wide measured stem-to-stem from the outer-most edge. Roadside, streamside, and shelterbelt strips of trees must have a width of at least 120 feet and continuous length of at least 363 feet to qualify as forest land. Tree-covered areas in agricultural production settings, such as fruit orchards, or tree-covered areas in urban settings, such as city parks, are not considered forest land.

Timber land is a subset of forest land that is producing or is capable of producing crops of industrial wood and not withdrawn from timber utilization by statute or administrative regulation. (Note: Areas qualifying as timberland are capable of producing at least 20 cubic feet per acre per year of industrial wood in natural stands. Currently inaccessible and inoperable areas are NOT included).
}


\value{
Dataframe or sf object (if \code{returnSpatial = TRUE}). If \code{byPlot = TRUE}, values are returned for each plot (proportion of plot in domain of interest; \code{PLOT_STATUS_CD = 1} when forest exists at the plot location). All variables with names ending in \code{SE}, represent the estimate of sampling error (\%) of the variable. When \code{variance = TRUE}, variables ending in \code{VAR} denote the variance of the variable and \code{N} is the total sample size (i.e., including non-zero plots).

\itemize{
      \item{\strong{YEAR}: reporting year associated with estimates}
      \item{\strong{SYMBOL}: unique species ID from NRCS Plant Reference Guide}
      \item{\strong{SCIENTIFIC_NAME}: scientific name of the species}
      \item{\strong{COMMON_NAME}: common name of the species}
      \item{\strong{COVER_PCT}: estimate of percent areal coverage of the species }
      \item{\strong{COVER_AREA}: estimate of areal coverage of the species (acres) }
      \item{\strong{AREA}: estimate of total land area (acres) }
      \item{\strong{nPlots_INV}: number of non-zero plots used to compute invasive coverage estimates}
      \item{\strong{nPlots_AREA}: number of non-zero plots used to compute land area estimates}
      }
}

\references{
rFIA website: \url{https://rfia.netlify.app/}

FIA Database User Guide: \url{https://research.fs.usda.gov/understory/forest-inventory-and-analysis-database-user-guide-nfi}

Bechtold, W.A.; Patterson, P.L., eds. 2005. The Enhanced Forest Inventory and Analysis Program - National Sampling Design and Estimation Procedures. Gen. Tech. Rep. SRS - 80. Asheville, NC: U.S. Department of Agriculture, Forest Service, Southern Research Station. 85 p. \url{https://www.srs.fs.usda.gov/pubs/gtr/gtr_srs080/gtr_srs080.pdf}

Stanke, H., Finley, A. O., Weed, A. S., Walters, B. F., & Domke, G. M. (2020). rFIA: An R package for estimation of forest attributes with the US Forest Inventory and Analysis database. Environmental Modelling & Software, 127, 104664.
}

\author{
Hunter Stanke and Andrew Finley
}

\note{All sampling error estimates (SE) are returned as the "percent coefficient of variation" (standard deviation / mean * 100) for consistency with EVALIDator. IMPORTANT: sampling error cannot be used to construct confidence intervals. Please use \code{variance = TRUE} for that (i.e., return variance and sample size instead of sampling error).}

\examples{

## Load data from the rFIA package
data(fiaRI)
data(countiesRI)

## Most recents subset
fiaRI_mr <- clipFIA(fiaRI)

\donttest{
## Most recent estimates on forest land
invasive(db = fiaRI_mr,
         landType = 'forest')

## Most recent estimates on forest land
invasive(db = fiaRI_mr,
         landType = 'forest',
         byPlot = TRUE)

## Same as above, but implemented in parallel (much quicker)
parallel::detectCores(logical = FALSE) # 4 cores available, we will take 2
invasive(db = fiaRI_mr,
         landType = 'forest',
         nCores = 2)

## Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
invasive(db = fiaRI_mr,
         grpBy = STAND_AGE)

## Estimates on forested mesic sites (all available inventories)
invasive(fiaRI,
         areaDomain = PHYSCLCD \%in\% 21:29) # Mesic Physiographic classes
}
}

