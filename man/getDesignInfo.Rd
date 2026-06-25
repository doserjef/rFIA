\name{getDesignInfo}
\alias{getDesignInfo}

\title{Extract design features for FIA inventories}

\description{
Extracts design information for post-stratified FIA inventories, intended to aid the development of alternative model-based estimators of forest variables. Design information is currently limited to estimation unit land area (AREA_USED) and strata weights (proportion of estimation unit represented by each stratum). This is sufficient to acknowledge design features in an aspatial model, however, inclusion probabilities and strata boundaries will be necessary to incorporate spatial predictors.
}

\usage{
getDesignInfo(db,
              type = c("ALL", "CURR", "VOL", "GROW", "MORT",
                       "REMV", "CHNG", "DWM", "REGEN"),
              mostRecent = TRUE,
              evalid = NULL)
}

\arguments{
  \item{db}{\code{FIA.Database} or \code{Remote.FIA.Database} object produced from \code{readFIA()} or \code{getFIA()}. If a \code{Remote.FIA.Database}, data will be read in and processed state-by-state to conserve RAM.}

  \item{type}{character ('ALL', 'CURR', 'VOL', 'GROW', 'MORT', 'REMV', 'CHNG', 'DWM', 'REGEN'). See Reference Population Evaluation Type Description Table (REF_POP_EVAL_TYP_DESCR) in FIADB P2 User Guide (link in references) for descriptions of evaluation types.}

  \item{mostRecent}{logical; if TRUE, returns EVALIDs associated with most recent inventory.}

  \item{evalid}{character; unique value which identifies an inventory year and inventory type for a state. If you would like to subset data for an inventory year other than the most recent, use \code{\link{findEVALID}} to locate this value.}
}

\details{

The FIA database is not limited for use with standard design-based estimators. A plethora of recent work has highlighted the improvements that model-assisted and model-based estimators can provide relative to the standard post-stratified estimators implemented in \code{rFIA}. However, when implementing alternative model-assisted or model-based estimators it is important to accommodate the design information for post-stratified FIA inventories. \code{getDesignInfo()} allows you to quickly extract design-based information for use when implementing alternative estimators with FIA data. See the Stanke et al. 2021 reference below for a paper describing use of \code{rFIA} for implementing model-based estimators to improve small area estimation of forest parameters. 
}

\value{

Generates a data frame with the columns described below, containing the design information associated with one or multiple FIA inventories.

For reference, there are often multiple non-overlapping estimation units within a state, and multiple, non-overlapping, and exhaustive strata within each estimation unit. Estimation unit and strata boundaries vary with inventories (i.e., by reporting year and by inventory type), though multiple inventories may draw from data collected at a single plot visit. Hence, ESTN_UNIT_CN and STRATUM_CN are specific to EVALIDs (or alternatively, the combination of STATE, YEAR, and EVAL_TYP, for all states except Texas). However, a single PLT_CN may be associated with multiple ESTN_UNIT_CNs and STRATUM_CNs.

\itemize{
      \item{\strong{STATECD}: unique identifier for states.}
      \item{\strong{YEAR}: reporting year associated with estimates (END_INVYR from POP_EVAL).}
      \item{\strong{EVAL_TYP}: an identifier describing the type of evaluation. For example, "EXPDWM" represents sampled plots used for down woody material estimates.}
      \item{\strong{EVALID}: unique identifier that represents the population used to produce a type of estimate.}
      \item{\strong{ESTN_UNIT_CN}: unique identifier for estimation unit.}
      \item{\strong{AREA_USED}: area of estimation unit used for population estimation.}
      \item{\strong{STRATUM_CN}: unique identifier for strata.}
      \item{\strong{STRATUM_WGT}: proportion of estimation unit area (AREA_USED) that is occupied by a particular stratum. Defined on the range (0,1].}
      \item{\strong{pltID}: unique identifier for plot \emph{location}.}
      \item{\strong{PLT_CN}: unique identifier for plot \emph{visit}.}
}
}

\references{
FIA Database User Guide: \url{https://research.fs.usda.gov/understory/forest-inventory-and-analysis-database-user-guide-nfi}
}

\author{
Jeffrey W. Doser, Hunter Stanke, Andrew Finley
}

\seealso{\code{findEVALID}}

\examples{
\dontshow{
  # Restrict threads for CRAN compliance
  dt_threads <- data.table::getDTthreads()
  data.table::setDTthreads(1)
}
# Load the Rhode Island subset included w/ rFIA
data(fiaRI)

# Extract the design information associated with the most recent current
# volume inventory in the state (2018 for this subset)
wgts <- getDesignInfo(db = fiaRI, mostRecent = TRUE, type = 'VOL')
\dontshow{
  # Restore thread setting
  data.table::setDTthreads(dt_threads)
}
}

