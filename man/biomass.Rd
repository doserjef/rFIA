\name{biomass}
\alias{biomass}

\title{Estimate tree biomass and carbon stocks from the FIADB}

\description{
Produces estimates of tree biomass and carbon on a per acre basis from FIA data, along with population estimates for each variable. Estimates can be produced for regions defined within the FIA Database (e.g. counties), at the plot level, or within user-defined areal units. Options to group estimates by species, size class, and other variables defined in the FIADB. If multiple reporting years (EVALIDs) are included in the data, estimates will be output as a time series. If multiple states are represented by the data, estimates will be output for the full region (all area combined), unless specified otherwise (e.g. \code{grpBy = STATECD}).
}


\usage{
biomass(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE,
        bySpecies = FALSE, bySizeClass = FALSE, byComponent = FALSE,
        landType = "forest", treeType = "live", method = "TI",
        lambda = 0.5, treeDomain = NULL, areaDomain = NULL,
        totals = FALSE, byPlot = FALSE, treeList = FALSE, 
        component = "AG", bioMethod = "NSVB", nCores = 1)
}

\arguments{
  \item{db}{\code{FIA.Database} or \code{Remote.FIA.Database} object produced from \code{readFIA()} or \code{getFIA()}. If a \code{Remote.FIA.Database}, data will be read in and processed state-by-state to conserve RAM (see details for an example).}

\item{grpBy}{variables from PLOT, PLOTGEOM, COND, or TREE tables to group estimates by (NOT quoted). Multiple grouping variables should be combined with \code{c()}, and grouping will occur heirarchically. For example, to produce seperate estimates for each ownership group within methods of stand regeneration, specify \code{c(STDORGCD, OWNGRPCD)}.}

  \item{polys}{\code{sp} or \code{sf} Polygon/MultiPolgyon object; Areal units to bin data for estimation. Separate estimates will be produced for region encompassed by each areal unit. FIA plot locations will be reprojected to match projection of \code{polys} object.}

  \item{returnSpatial}{logical; if TRUE, merge population estimates with \code{polys} and return as \code{sf} multipolygon object. When \code{byPlot = TRUE}, return plot-level estimates as \code{sf} spatial points.}

\item{bySpecies}{logical; if TRUE, returns estimates grouped by species.}

  \item{bySizeClass}{logical; if TRUE, returns estimates grouped by size class (2-inch intervals, see \code{makeClasses()} to compute different size class intervals).}

  \item{byComponent}{logical; if TRUE, returns estimates grouped by the following biomass components: stem, stem bark, branches, foliage, stump, stump bark, merchantable bole, merchantable bole bark, sawlog, sawlog bark, and belowground roots.}

  \item{landType}{character ("forest" or "timber"); Type of land that estimates will be produced for. Timberland is a subset of forestland (default) which has high site potential and non-reserve status (see details).}

  \item{treeType}{character ("all", "live", "dead", or "gs"); Type of tree which estimates will be produced for. All includes all stems, live and dead, greater than 1 in. DBH. Live/Dead includes all stems greater than 1 in. DBH which are live (default) or dead (leaning less than 45 degrees), respectively. GS (growing-stock) includes live stems greater than 5 in. DBH which contain at least one 8 ft merchantable log.}

  \item{method}{character; design-based estimator to use. One of:  "TI" (temporally indifferent, default), "annual" (annual), "SMA" (simple moving average), "LMA" (linear moving average), or "EMA" (exponential moving average). See \href{https://research.fs.usda.gov/treesearch/59521}{Stanke et al 2020} for a complete description of these estimators.}

  \item{lambda}{numeric (0,1); if \code{method = 'EMA'}, the decay parameter used to define weighting scheme for annual panels. Low values place higher weight on more recent panels, and vice versa. Specify a vector of values to compute estimates using mulitple wieghting schemes, and use \code{plotFIA()} with \code{grp} set to \code{lambda} to produce moving average ribbon plots. See \href{https://research.fs.usda.gov/treesearch/59521}{Stanke et al 2020} for examples.}

  \item{treeDomain}{logical predicates defined in terms of the variables in PLOT, TREE, and/or COND tables. Used to define the type of trees for which estimates will be produced (e.g. DBH greater than 20 inches: \code{DIA > 20}, Dominant/Co-dominant crowns only: \code{CCLCD \%in\% c(2,3)}. Multiple conditions are combined with \code{&} (and) or \code{|} (or). Only trees where the condition evaluates to TRUE are used in producing estimates. Should NOT be quoted.}

  \item{areaDomain}{logical predicates defined in terms of the variables in PLOT and/or COND tables. Used to define the area for which estimates will be produced (e.g. within 1 mile of improved road: \code{RDDISTCD \%in\% c(1:6)}, Hard maple/basswood forest type: \code{FORTYPCD == 805}. Multiple conditions are combined with \code{&} (and) or \code{|} (or). Only plots within areas where the condition evaluates to TRUE are used in producing estimates. Should NOT be quoted.}

  \item{totals}{logical; if TRUE, return total population estimates (e.g. total area) along with ratio estimates (e.g. mean trees per acre).}

  \item{byPlot}{logical; if TRUE, returns estimates for individual plot locations instead of population estimates.}

  \item{treeList}{logical; if TRUE, returns tree-level summaries intended for subsequent use with \code{customPSE()}.}

  \item{component}{character, combination of: "TOTAL" (sum of all components), "AG" (aboveground components excluding foliage), "STEM" (total stem of timber species from ground line to the tree tip), "STEM_BARK", "BRANCH", (branch/limbs of timber species), "FOLIAGE" (foliage for live trees at least 1.0 inches dbh/drc), "STUMP", "STUMP_BARK", "BOLE" (merchantable bole), "BOLE_BARK", "SAWLOG", "SAWLOG_BARK", "ROOT" (beloground portion of tree including coarse roots with a root diameter at elast 0.1 inch); biomass component to use in estimation. Note that "TOTAL" includes foliage for biomass estimates but does not include foliage for carbon estimates. See Details below for more detailed descriptions of the components.}

  \item{bioMethod}{character; tree-level biomass estimation procedures to use. As of \code{rFIA v1.1.0}, the only biomass estimation procedure supported is the National Scale Volume and Biomass ("NSVB") models. See Estimation Details for references.}

  \item{nCores}{numeric; number of cores to use for parallel implementation. Check available cores using \code{detectCores()}. Default = 1, serial processing.}
}

\details{
\strong{Estimation Details}

Estimation of forest variables follows the procedures documented in Bechtold and Patterson (2005) and \href{https://research.fs.usda.gov/treesearch/59521}{Stanke et al 2020}. Specifically, tree biomass and carbon per acre are computed using a sample-based ratio-of-means estimator of total biomass / total land area within the domain of interest.

A sum of aboveground biomass components, excluding foliage, is estimated by default (\code{component = 'AG'}). However, users may specify unique combinations of biomass components if they wish to do so. For example, to estimate aboveground biomass, including foliage, specify \code{component = c('AG', 'FOLIAGE')} in the call to \code{biomass}. To estimate all biomass components simultaneously (i.e., grouped by copmonent), specify \code{byComponent = TRUE}. All biomass components are computed using the National Scale Volume and Biomass (NSVB) approach adopted by FIA in September 2023. See \href{https://research.fs.usda.gov/treesearch/66998}{Westfall et al. 2024} for more detailed information on NSVB procedures. The following biomass components can be estimated using \code{biomass()} (note the components are not mutually exclusive):  

\itemize{
  \item \emph{TOTAL}: total biomass, which is equivalent to the sum of the following components: "ROOT", "STEM", "STEM_BARK", "BRANCH", and "FOLIAGE". \emph{NOTE}: total carbon estimates do not include estimates of carbon in foliage.   
  \item \emph{AG}: aboveground biomass/carbon, not including foliage. This is equivalent to the sum of the following components: "STEM", "STEM_BARK", and "BRANCH".  
  \item \emph{STEM}: oven-dry biomass/carbon of wood in the total stem of timber species (trees where diameter is measured at breast height) with dbh at least 1.0 inches, from ground line to the tree tip. Calculated for live and standing dead trees.
  \item \emph{STEM_BARK}: oven-dry biomass/carbon of bark in the total stem of timber species with dbh at least 1.0 inches, from ground line to the tree tip. Calculated for live and standing dead trees. 
  \item \emph{BRANCH}: oven-dry biomass/carbon of wood and bark in the branches/limbs of timber species with at least 1.0 inches. This only includes branches; it does not include any portion of the total stem. Calculated for live and standing dead trees. For live trees, this value is reduced for broken tops. For standing dead trees, this value is reduced for broken tops as well as decay. 
    \item \emph{FOLIAGE}: oven-dry biomass of foliage for live trees with dbh/drc at least 1.0 inches. \emph{NOTE}: foliar carbon is not calculated and is instead set to 0.
  \item \emph{STUMP}: oven-dry biomass/carbon of wood in the stump of timber species with dbh at least 5.0 inches. The stump is that portion of the tree from the ground line to the bottom of the merchantable bole (i.e., below 1 foot). Calculated for live and standing dead trees. 
  \item \emph{STUMP_BARK}:oven-dry biomass/carbon of bark in the stump of timber species with dbh at least 5.0 inches. The stump is that portion of the tree from the ground line to the bottom of the merchantable bole (i.e., below 1 foot). Calculated for live and standing dead trees.  
  \item \emph{BOLE}: oven-dry biomass/carbon of wood in the merchantable bole of timber species with dbh at least 5.0 inches, from a 1-foot stump to a minimum 4-inch top diameter. Calculated for live and standing dead trees. 
  \item \emph{BOLE_BARK}: oven-dry biomass/carbon of bark in the merchantable bole of timber species with dbh at least 5.0 inches, from a 1-foot stump to a minimum 4-inch top diameter. Calculated for live and standing dead trees.  
  \item \emph{SAWLOG}: the oven-dry biomass/carbon of wood in the sawlog portion of timber species of sawtimber size from a 1-foot stump to a minimum top diameter or to where the central stem breaks into limbs, all of which are less than the minimum top diamter. Minimum dbh is 9.0 inches for softwoods and 11.0 inches for hardwoods. The minimum top diameter is 7.0 inches for softwoods and 9.0 inches for hardwoods. 
  \item \emph{SAWLOG_BARK}: the oven-dry biomass/carbon of bark in the sawlog portion of timber species of sawtimber size from a 1-foot stump to a minimum top diameter or to where the central stem breaks into limbs, all of which are less than the minimum top diamter. Minimum dbh is 9.0 inches for softwoods and 11.0 inches for hardwoods. The minimum top diameter is 7.0 inches for softwoods and 9.0 inches for hardwoods. 
  \item \emph{ROOT}: oven-dry biomass of the below ground portion of a tree, including coarse roots with a root diameter of at least 0.1 inches. This is a modeled estimate, calculated for live and standing dead trees with dbh/drc at least 1.0 inches. This component, unlike all other compoments, is estimated using the Component Ratio Method (CRM).   
}

Users may specify alternatives to the 'Temporally Indifferent' estimator using the \code{method} argument. Alternative design-based estimators include the annual estimator ("ANNUAL"; annual panels, or estimates from plots measured in the same year), simple moving average ("SMA"; combines annual panels with equal weight), linear moving average ("LMA"; combine annual panels with weights that decay \emph{linearly} with time since measurement), and exponential moving average ("EMA"; combine annual panels with weights that decay \emph{exponentially} with time since measurement). The "best" estimator depends entirely on user-objectives, see \href{https://research.fs.usda.gov/treesearch/59521}{Stanke et al 2020} for a complete description of these estimators and tradeoffs between precision and temporal specificity.

When \code{byPlot = FALSE} (i.e., population estimates are returned), the "YEAR" column in the resulting dataframe indicates the final year of the inventory cycle that estimates are produced for. For example, an estimate of current forest area (e.g., 2018) may draw on data collected from 2008-2018, and "YEAR" will be listed as 2018 (consistent with EVALIDator). However, when \code{byPlot = TRUE} (i.e., plot-level estimates returned), the "YEAR" column denotes the year that each plot was measured (MEASYEAR), which may differ slightly from its associated inventory year (INVYR).

Stratified random sampling techniques are most often employed to compute estimates in recent inventories, although double sampling and simple random sampling may be employed for early inventories. Estimates are adjusted for non-response bias by assuming attributes of non-response plot locations to be equal to the mean of other plots included within thier respective stratum or population.

\strong{Working with "Big Data"}

If FIA data are too large to hold in memory (e.g., R throws the "cannot allocate vector of size ..." errors), use larger-than-RAM options. See documentation of \code{readFIA()} for examples of how to set up a \code{Remote.FIA.Database}. As a reference, we have used rFIA's larger-than-RAM methods to estimate forest variables using the entire FIA Database (~50GB) on a standard desktop computer with 16GB of RAM. Check out \href{https://doserlab.com/files/rfia/}{our website} for more details and examples.

Easy, efficient parallelization is implemented with the \code{\link{parallel}} package. Users must only specify the \code{nCores} argument with a value greater than 1 in order to implement parallel processing on their machines. Parallel implementation is achieved using a snow type cluster on any Windows OS, and with multicore forking on any Unix OS (Linux, Mac). Implementing parallel processing may substantially decrease free memory during processing, particularly on Windows OS. Thus, users should be cautious when running in parallel, and consider implementing serial processing for this task if computational resources are limited (\code{nCores = 1}).

\strong{Definition of forestland}

Forest land must have at least 10-percent canopy cover by live tally trees of any size, including land that formerly had such tree cover and that will be naturally or artificially regenerated. Forest land includes transition zones, such as areas between heavily forest and non-forested lands that meet the mimium tree canopy cover and forest areas adjacent to urban and built-up lands. The minimum area for classification of forest land is 1 acre in size and 120 feet wide measured stem-to-stem from the outer-most edge. Roadside, streamside, and shelterbelt strips of trees must have a width of at least 120 feet and continuous length of at least 363 feet to qualify as forest land. Tree-covered areas in agricultural production settings, such as fruit orchards, or tree-covered areas in urban settings, such as city parks, are not considered forest land.

Timber land is a subset of forest land that is producing or is capable of producing crops of industrial wood and not withdrawn from timber utilization by statute or administrative regulation. (Note: Areas qualifying as timberland are capable of producing at least 20 cubic feet per acre per year of industrial wood in natural stands. Currently inaccessible and inoperable areas are NOT included).

}

\value{
Dataframe or sf object (if \code{returnSpatial = TRUE}). If \code{byPlot = TRUE}, values are returned for each plot (\code{PLOT_STATUS_CD = 1} when forest exists at the plot location). All variables with names ending in \code{SE}, represent the estimate of sampling error (\%) of the variable.

\itemize{
      \item{\strong{YEAR}: reporting year associated with estimates}
      \item{\strong{BIO_ACRE}: estimate of mean tree biomass per acre (short tons/acre)}
      \item{\strong{CARB_ACRE}: estimate of mean tree carbon per acre (short tons/acre)}
      \item{\strong{nPlots_TREE}: number of non-zero plots used to compute biomass and carbon estimates}
      \item{\strong{nPlots_AREA}: number of non-zero plots used to compute land area estimates}
      }
}

\references{
rFIA website: \url{https://doserlab.com/files/rfia/}

FIA Database User Guide: \url{https://research.fs.usda.gov/understory/forest-inventory-and-analysis-database-user-guide-nfi}

Bechtold, W.A.; Patterson, P.L., eds. 2005. The Enhanced Forest Inventory and Analysis Program - National Sampling Design and Estimation Procedures. Gen. Tech. Rep. SRS - 80. Asheville, NC: U.S. Department of Agriculture, Forest Service, Southern Research Station. 85 p. \url{https://www.srs.fs.usda.gov/pubs/gtr/gtr_srs080/gtr_srs080.pdf}

Stanke, H., Finley, A. O., Weed, A. S., Walters, B. F., & Domke, G. M. (2020). rFIA: An R package for estimation of forest attributes with the US Forest Inventory and Analysis database. Environmental Modelling & Software, 127, 104664.

Westfall, James A., Coulston, John W., Gray, Andrew N., Shaw, John D., Radtke, Philip J., Walker, David M., Weiskittel, Aaron R., MacFarlane, David W., Affleck, David L.R., Zhao, Dehai, Temesgen, Hailemariam, Poudel, Krishna P., Frank, Jereme M., Prisley, Stephen P., Wang, Yingfang, Sánchez Meador, Andrew J., Auty, David, Domke, Grant M. 2024. A national-scale tree volume, biomass, and carbon modeling system for the United States. Gen. Tech. Rep. WO-104. Washington, DC: U.S. Department of Agriculture, Forest Service. 37 p. \url{https://research.fs.usda.gov/treesearch/66998}. 
}

\author{
Jeffrey W. Doser, Hunter Stanke, Andrew Finley
}

\note{All percent sampling error estimates (SE) are returned as the "percent coefficient of variation" (standard deviation / mean * 100) for consistency with EVALIDator. To determine the margin of error for confidence intervals, use the following formula, t * (SE * mean / 100), where t is the corresponding t value for your desired level of confidence (e.g., approximately 2 for a 95\% confidence interval). To determine the degrees of freedom for the t value, use \code{nPlots_AREA}. Alternatively, you can use a normal approximation for the confidence interval. The validity of the normal approximation decreases as the sample size decreases.}


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


# Most recent estimates of aboveground biomass (excluding foliage)
# for growing-stock trees on timber land
biomass(db = fiaRI_mr,
        landType = 'timber',
        treeType = 'gs')

# Same as above but include foliage
biomass(db = fiaRI_mr,
        landType = 'timber',
        treeType = 'gs',
        component = c('AG', 'FOLIAGE'))
\donttest{

# Same as above, but at the plot-level
biomass(db = fiaRI_mr,
        landType = 'timber',
        treeType = 'gs',
        component = c('AG', 'FOLIAGE'),
        byPlot = TRUE)

# Belowground (i.e., coarse roots) and stump biomass only
biomass(db = fiaRI_mr,
        component = c('ROOT', 'STUMP'))

# Estimate all biomass components simultaneosly
biomass(db = fiaRI_mr,
        byComponent = TRUE)

# Estimates for live white pine ( > 12" DBH) on forested mesic sites (all available inventories)
biomass(fiaRI_mr,
        treeType = 'live',
        treeDomain = SPCD == 129 & DIA > 12, # Species code for white pine
        areaDomain = PHYSCLCD \%in\% 21:29) # Mesic Physiographic classes

# Most recent estimates grouped by stand age on forest land
# Make a categorical variable which represents stand age (grouped by 10 yr intervals)
fiaRI_mr$COND$STAND_AGE <- makeClasses(fiaRI_mr$COND$STDAGE, interval = 10)
biomass(db = fiaRI_mr,
        grpBy = STAND_AGE)

# Estimates for snags greater than 20 in DBH on forestland for all
#  available inventories (time-series)
biomass(db = fiaRI,
        landType = 'forest',
        treeType = 'dead',
        treeDomain = DIA > 20)

# Most recent estimates for live stems on forest land by species
biomass(db = fiaRI_mr,
        landType = 'forest',
        treeType = 'live',
        bySpecies = TRUE)

# Same as above, but implemented in parallel (much quicker)
# parallel::detectCores(logical = FALSE) # 4 cores available, we will take 2
# biomass(db = fiaRI_mr,
#         landType = 'forest',
#         treeType = 'live',
#         bySpecies = TRUE,
#         nCores = 2)


# Most recent estimates for all stems on forest land grouped by user-defined areal units
ctSF <- biomass(fiaRI_mr,
                polys = countiesRI,
                returnSpatial = TRUE)
plot(ctSF) # Plot multiple variables simultaneously
plotFIA(ctSF, BIO_ACRE) # Plot of aboveground biomass per acre
}
\dontshow{
  # Restore thread setting
  data.table::setDTthreads(dt_threads)
}
}

