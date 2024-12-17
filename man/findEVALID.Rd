\name{findEVALID}
\alias{findEVALID}

\title{Find EVALIDs used in the FIADB}
\description{
Lookup Evaluation IDs (EVALIDs) associated with reporting years and evaluation types used in the Forest Inventory and Analysis Database. NOT required to run other \pkg{rFIA} functions. Only use if you are interested in subsetting an \code{FIA.Database} object for a specific reporting year or evaluation type using \code{\link{clipFIA}}.
}

\usage{
findEVALID(db, mostRecent = FALSE, state = NULL, year = NULL, type = NULL)
}

\arguments{
  \item{db}{list; FIA Database object produced from \code{\link{readFIA}}.}

  \item{mostRecent}{logical; if TRUE, returns EVALIDs associated with most recent inventory.}

  \item{state}{character vector containing full names of states of interest (e.g. \code{c('Michigan', 'Minnesota', 'Wisconsin')})}
  \item{year}{numeric vector containing years of interest (e.g. \code{c(2015, 2016, 2017)})}
  \item{type}{character ('ALL', 'CURR', 'VOL', 'GROW', 'MORT', 'REMV', 'CHANGE', 'DWM', 'REGEN'). See Reference Population Evaluation Table Description Type Table in FIADB P2 User Guide (References) for descriptions of evaluation types. }
}

\details{
EVALIDs in the FIA Database are used to reference data points associated with particular inventory years and evaluation types within a state (e.g. 2017 Current Volume in Michigan). They are often extraordinarily confusing for those not familiar for the FIA Database. With this in mind, \pkg{rFIA} has been designed to eliminate users dependence on identifying and specifying appropriate EVALIDs to produce desired estimates, and we therefore do not recommend users attempt to identify EVALIDs independently.

Any \code{state} or \code{year} specified must be present in \code{db} to return associated EVALIDS.
}

\value{ A numeric vector containing the EVALIDs associated with states, years, or evaluation types specified.
}
\references{
FIA Database User Guide: \url{https://research.fs.usda.gov/understory/forest-inventory-and-analysis-database-user-guide-nfi}
}

\author{
Hunter Stanke and Andrew Finley
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{clipFIA}}
}

\examples{
## Lookup all EVALIDs in an FIA.Database object
findEVALID(fiaRI)

## Find the most recent EVALIDs
findEVALID(fiaRI, mostRecent = FALSE)

}

