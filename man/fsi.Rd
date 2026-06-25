\name{fsi}
\alias{fsi}


\title{Estimate the Forest Stability Index from the FIADB}


\description{
Estimate annual change in relative live tree density from the FIADB using the Forest Stability Index (FSI). See Stanke et al. 2021 (doi: 10.1038/s41467-020-20678-z) for a complete description of the the Forest Stability Index.   }

\usage{
fsi(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE,
    bySpecies = FALSE, bySizeClass = FALSE,
    landType = "forest", treeType = "live", method = "TI",
    lambda = 0.5, treeDomain = NULL, areaDomain = NULL,
    totals = TRUE, variance = TRUE, byPlot = FALSE,
    useSeries = FALSE, mostRecent = FALSE, scaleBy = NULL,
    betas = NULL, returnBetas = FALSE, nCores = 1)
}

%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{db}{\code{FIA.Database} or \code{Remote.FIA.Database} object produced from \code{readFIA()} or \code{getFIA()}. If a \code{Remote.FIA.Database}, data will be read in and processed state-by-state to conserve RAM (see details for an example).}

\item{grpBy}{variables from PLOT, PLOTGEOM, COND, or TREE tables to group estimates by (NOT quoted). Multiple grouping variables should be combined with \code{c()}, and grouping will occur heirarchically. For example, to produce separate estimates for each ownership group within methods of stand regeneration, specify \code{c(STDORGCD, OWNGRPCD)}.}

  \item{polys}{\code{sp} or \code{sf} Polygon/MultiPolgyon object; Areal units to bin data for estimation. Separate estimates will be produced for region encompassed by each areal unit. FIA plot locations will be reprojected to match projection of \code{polys} object.}

  \item{returnSpatial}{logical; if TRUE, merge population estimates with \code{polys} and return as \code{sf} multipolygon object. When \code{byPlot = TRUE}, return plot-level estimates as \code{sf} spatial points.}

  \item{bySpecies}{logical; if TRUE, returns estimates grouped by species.}

  \item{bySizeClass}{logical; if TRUE, returns estimates grouped by size class (2-inch intervals, see \code{\link{makeClasses}} to compute different size class intervals).}

  \item{landType}{character ('forest' or 'timber'); Type of land which estimates will be produced for. Timberland is a subset of forestland (default) which has high site potential and non-reserve status (see details).}

  \item{treeType}{character ('live' or 'gs'); Type of tree which estimates will be produced for. Live includes all stems greater than 1 in. DBH which are live (leaning less than 45 degrees). GS (growing-stock) includes live stems greater than 5 in. DBH which contain at least one 8 ft merchantable log. }

  \item{method}{character; design-based estimator to use. One of:  "TI" (temporally indifferent, default), "annual" (annual), "SMA" (simple moving average), "LMA" (linear moving average), or "EMA" (exponential moving average). See \href{https://research.fs.usda.gov/treesearch/59521}{Stanke et al 2020} for a complete description of these estimators.}

  \item{lambda}{numeric (0,1); if \code{method = 'EMA'}, the decay parameter used to define weighting scheme for annual panels. Low values place higher weight on more recent panels, and vice versa. Specify a vector of values to compute estimates using mulitple weighting schemes, and use \code{plotFIA()} with \code{grp} set to \code{lambda} to produce moving average ribbon plots. See \href{https://research.fs.usda.gov/treesearch/59521}{Stanke et al 2020} for examples.}

  \item{treeDomain}{logical predicates defined in terms of the variables in PLOT, TREE, and/or COND tables. Used to define the type of trees for which estimates will be produced (e.g. DBH greater than 20 inches: \code{DIA > 20}, Dominant/Co-dominant crowns only: \code{CCLCD \%in\% c(2,3)}). Multiple conditions are combined with \code{&} (and) or \code{|} (or). Only trees where the condition evaluates to TRUE are used in producing estimates. Should NOT be quoted.}

  \item{areaDomain}{logical predicates defined in terms of the variables in PLOT and/or COND tables. Used to define the area for which estimates will be produced (e.g. within 1 mile of improved road: \code{RDDISTCD \%in\% c(1:6)}, Hard maple/basswood forest type: \code{FORTYPCD == 805)}. Multiple conditions are combined with \code{&} (and) or \code{|} (or). Only plots within areas where the condition evaluates to TRUE are used in producing estimates. Should NOT be quoted.}

  \item{totals}{logical; if TRUE, return total population estimates (e.g. total area) along with ratio estimates (e.g. mean trees per acre).}

    \item{variance}{logical; if TRUE, return estimated variance (\code{VAR}) and sample size (\code{N}). If FALSE, return 'sampling error' (\code{SE}) as returned by EVALIDator. Note: sampling error cannot be used to construct confidence intervals.}

  \item{byPlot}{logical; if TRUE, returns estimates for individual plot locations instead of population estimates.}

  \item{useSeries}{logical; If TRUE, use multiple remeasurements to estimate annual change in relative density on each plot, when available.}

  \item{mostRecent}{logical; If TRUE, only return results for the most recent inventory in each state. Only useful when \code{useSeries=TRUE}, as in this case, using \code{clipFIA} to select the most recent inventory will drop all but the most recent remeasurement.}

  \item{scaleBy}{variables from PLOT or COND tables to use as 'random effects' in model of size-density relationships. Multiple variables should be combined with \code{c()}.}

  \item{betas}{data.frame; coefficients of maximum size-density models returned in a previous call to \code{fsi} when \code{returnBetas = TRUE}. See examples.}

  \item{returnBetas}{logical; If true, returns estimated coefficients of maximum size-density models along with results. These coefficients can then be handed to the \code{beta} argument (see above) in subsequent runs. This speeds up processing and ensures the same coefficients are used to model maximum-size density curves between function calls. See Value below for more details. }

  \item{nCores}{numeric; number of cores to use for parallel implementation. Check available cores using \code{\link{detectCores}}. Default = 1, serial processing.}

}
\details{
\strong{Estimation Details}

Estimation of forest variables follows the procedures documented in Bechtold and Patterson (2005) and \href{https://research.fs.usda.gov/treesearch/59521}{Stanke et al 2020}.

Please see Stanke et al. 2021 (doi: 10.1038/s41467-020-20678-z) for a complete description of the Forest Stability Index (FSI). In short, the FSI is a direct measure of temporal change in the relative density of live trees, where relative density is defined as the ratio of observed tree density to maximum potential tree density. Maximum potential tree density is modeled as a power function of average tree size - in the current implementation average tree basal area is used. Users may allow both the "slopes" and intercepts of this power function to vary by classified groups, like forest community type using the \code{scaleBy} argument. Users may return the estimated parameters of maximum size-density models by specifying \code{returnBetas = TRUE}.

Users may specify alternatives to the 'Temporally Indifferent' estimator using the \code{method} argument. Alternative design-based estimators include the annual estimator ("ANNUAL"; annual panels, or estimates from plots measured in the same year), simple moving average ("SMA"; combines annual panels with equal weight), linear moving average ("LMA"; combine annual panels with weights that decay \emph{linearly} with time since measurement), and exponential moving average ("EMA"; combine annual panels with weights that decay \emph{exponentially} with time since measurement). The "best" estimator depends entirely on user-objectives, see \href{https://research.fs.usda.gov/treesearch/59521}{Stanke et al 2020} for a complete description of these estimators and tradeoffs between precision and temporal specificity.

When \code{byPlot = FALSE} (i.e., population estimates are returned), the "YEAR" column in the resulting dataframe indicates the final year of the inventory cycle that estimates are produced for. For example, an estimate of current forest area (e.g., 2018) may draw on data collected from 2008-2018, and "YEAR" will be listed as 2018 (consistent with EVALIDator). However, when \code{byPlot = TRUE} (i.e., plot-level estimates returned), the "YEAR" column denotes the year that each plot was measured (MEASYEAR), which may differ slightly from its associated inventory year (INVYR).

Stratified random sampling techniques are most often employed to compute estimates in recent inventories, although double sampling and simple random sampling may be employed for early inventories. Estimates are adjusted for non-response bias by assuming attributes of non-response plot locations to be equal to the mean of other plots included within their respective stratum or population.



\strong{Working with "Big Data"}

If FIA data are too large to hold in memory (e.g., R throws the "cannot allocate vector of size ..." errors), use larger-than-RAM options. See documentation of \code{link{readFIA}} for examples of how to set up a \code{Remote.FIA.Database}. As a reference, we have used rFIA's larger-than-RAM methods to estimate forest variables using the entire FIA Database (~50GB) on a standard desktop computer with 16GB of RAM. Check out \href{https://doserlab.com/files/rfia/}{our website} for more details and examples.

Easy, efficient parallelization is implemented with the \code{\link{parallel}} package. Users must only specify the \code{nCores} argument with a value greater than 1 in order to implement parallel processing on their machines. Parallel implementation is achieved using a snow type cluster on any Windows OS, and with multicore forking on any Unix OS (Linux, Mac). Implementing parallel processing may substantially decrease free memory during processing, particularly on Windows OS. Thus, users should be cautious when running in parallel, and consider implementing serial processing for this task if computational resources are limited (\code{nCores = 1}).



\strong{Definition of forestland}

Forest land must have at least 10-percent canopy cover by live tally trees of any size, including land that formerly had such tree cover and that will be naturally or artificially regenerated. Forest land includes transition zones, such as areas between heavily forest and non-forested lands that meet the mimium tree canopy cover and forest areas adjacent to urban and built-up lands. The minimum area for classification of forest land is 1 acre in size and 120 feet wide measured stem-to-stem from the outer-most edge. Roadside, streamside, and shelterbelt strips of trees must have a width of at least 120 feet and continuous length of at least 363 feet to qualify as forest land. Tree-covered areas in agricultural production settings, such as fruit orchards, or tree-covered areas in urban settings, such as city parks, are not considered forest land.

Timber land is a subset of forest land that is producing or is capable of producing crops of industrial wood and not withdrawn from timber utilization by statute or administrative regulation. (Note: Areas qualifying as timberland are capable of producing at least 20 cubic feet per acre per year of industrial wood in natural stands. Currently inaccessible and inoperable areas are NOT included).

}


\value{
When \code{returnBetas = TRUE}, a list will be returned. This list will contain a dataframe named "results", containing estimates of the FSI, and another named "betas", containing estimated parameters of the maximum size-density model. When \code{returnBetas = FALSE}, a data.frame corresponding with "results" will be returned.

\strong{Results} Dataframe or sf object (if \code{returnSpatial = TRUE}). If \code{byPlot = TRUE}, values are returned for each plot (\code{PLOT_STATUS_CD = 1} when forest exists at the plot location). All variables with names ending in \code{SE}, represent the estimate of sampling error (\%) of the variable. When \code{variance = TRUE}, variables ending in \code{VAR} denote the variance of the variable and \code{N} is the total sample size (i.e., including non-zero plots).

\itemize{
      \item{\strong{YEAR}: reporting year associated with estimates}
      \item{\strong{FSI}: estimate of forest stability index (i.e., annual change in relative live tree density)}
      \item{\strong{PERC_FSI}: estimate of \% forest stability index (i.e., \% annual change in relative live tree density)}
      \item{\strong{FSI_STATUS}: indication of the forest stability index (i.e., decline, stable, or expand)}
      \item{\strong{FSI_INT}: width of 95\% confidence interval of mean FSI}
      \item{\strong{PREV_RD}: estimate of relative live tree density at initial measurement of all plots (i.e., observed density / maximum potential density)}
      \item{\strong{PREV_RD}: estimate of relative live tree density at final measurement of all plots (i.e., observed density / maximum potential density)}
      \item{\strong{TPA_RATE}: standardized estimate of annual change in TPA (proportionate change)}
      \item{\strong{BA_RATE}: standardized estimate of annual change in BA (proportionate change)}
}


\strong{Betas} Within betas, all variable names ending in "upper" or "lower" represent the upper and lower bounds of the 95\% Bayesian credible interval of their respective variables. All variable names beginning with "fixed" represent the fixed effects in random slope/intercept models (i.e., the global average).

\itemize{
      \item{\strong{grps}: unique identifier associated with the group (i.e., unique combination of variables listed in \code{scaleBy}).}
      \item{\strong{alpha}: posterior median of scaling factor that describes the maximum tree density at average tree basal area of one sq. ft.}
      \item{\strong{rate}: posterior median of negative exponent controlling the decay in maximum tree density with increasing average tree size.}
      \item{\strong{n}: number of observations with the group with an approximately normal diameter distribution and no evidence of recent disturbance.}
}

}

\references{
Stanke, H., Finley, A. O., Domke, G. M., Weed, A. S., & MacFarlane, D. W. (2021). Over half of western United States' most abundant tree species in decline. Nature Communications, 12(1), 451. doi: 10.1038/s41467-020-20678-z

FIA Database User Guide: \url{https://research.fs.usda.gov/understory/forest-inventory-and-analysis-database-user-guide-nfi}

Bechtold, W.A.; Patterson, P.L., eds. 2005. The Enhanced Forest Inventory and Analysis Program - National Sampling Design and Estimation Procedures. Gen. Tech. Rep. SRS - 80. Asheville, NC: U.S. Department of Agriculture, Forest Service, Southern Research Station. 85 p. \url{https://www.srs.fs.usda.gov/pubs/gtr/gtr_srs080/gtr_srs080.pdf}

Stanke, H., Finley, A. O., Weed, A. S., Walters, B. F., & Domke, G. M. (2020). rFIA: An R package for estimation of forest attributes with the US Forest Inventory and Analysis database. Environmental Modelling & Software, 127, 104664.
}

\author{
Jeffrey W. Doser, Hunter Stanke, Andrew Finley
}

\note{All percent sampling error estimates (SE) are returned as the "percent coefficient of variation" (standard deviation / mean * 100) for consistency with EVALIDator. To determine the margin of error for confidence intervals, use the following formula, t * (SE * mean / 100), where t is the corresponding t value for your desired level of confidence (e.g., approximately 2 for a 95\% confidence interval).}

\examples{
\dontshow{
  # Restrict threads for CRAN compliance
  dt_threads <- data.table::getDTthreads()
  data.table::setDTthreads(1)
}
# Load data from the rFIA package
data(fiaRI)
data(countiesRI)

# Most recents subset
fiaRI_mr <- clipFIA(fiaRI)

\donttest{
## Most recent estimates for all live trees in RI
## Allowing maximum size-density relationship to
## vary by forest community type
# fsi(db = fiaRI_mr,
#     scaleBy = FORTYPCD)
# 
# ## Same as above at the plot-level
# fsi(db = fiaRI_mr,
#     scaleBy = FORTYPCD,
#     byPlot = TRUE)
# 
# 
# ## Same as above, but return the estimated coefficients of the
# ## maximum size-density model
# results <- fsi(db = fiaRI_mr,
#                scaleBy = FORTYPCD,
#                returnBetas = TRUE)
# ## Our results are stored in a list, where "results" gives us the
# ## estimates of the FSI, and "betas" gives us the estimated
# ## model coefficients
# results$results # FSI estimates
# results$betas # model coefficients
# 
# 
# ## Estimates for live white pine ( > 12" DBH) on
# ## forested mesic sites (all available inventories)
# ## Here we instead allow maximum size-density relationships
# ## to vary by site productivity class
# fsi(fiaRI_mr,
#     scaleBy = SITECLCD,
#     treeType = 'live',
#     treeDomain = SPCD == 129 & DIA > 12, # Species code for white pine
#     areaDomain = PHYSCLCD \%in\% 21:29) # Mesic Physiographic classes
}
\dontshow{
  # Restore thread setting
  data.table::setDTthreads(dt_threads)
}

}
