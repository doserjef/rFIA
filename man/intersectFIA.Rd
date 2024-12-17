\name{intersectFIA}
\alias{intersectFIA}

\title{Intersect FIA data with spatial polygons}


\description{
Performs spatial intersection between FIA data and user-supplied spatial polygons (sp or sf). Polygon attributes appended to PLOT table, and hence can be used as grouping variables in subsequent calls to rFIA estimator functions. Alternative to the \code{polys} argument in rFIA estimator functions.
}

\usage{
intersectFIA(db, polys, nCores = 1)
}


%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{db}{\code{FIA.Database} or \code{Remote.FIA.Database} object produced from \code{\link{readFIA}} or \code{\link{getFIA}}. If a \code{Remote.FIA.Database}, data will be read in and processed state-by-state to conserve RAM (see details for an example).}

  \item{polys}{\code{sp} or \code{sf} Polygon/MultiPolgyon object; Areal units to bin data for estimation. Seperate estimates will be produces for region encompassed by each areal unit. FIA plot locations will be reprojected to match projection of \code{polys} object.}

  \item{nCores}{numeric; number of cores to use for parallel implementation. Check available cores using \code{\link{detectCores}}. Default = 1, serial processing.}
}


\details{
All polygon attributes will be joined onto the PLOT table.

Primarily useful if you intend to make multiple calls to rFIA estimator functions, e.g., you need to call both \code{\link{tpa}} and \code{biomass} and group by spatial polygons in both cases. If using the \code{polys} argument in each function call, spatial intersection will occur mulitple times, and hence be slower than performing the intersection a single time upfront.
}


\value{
FIA.Database or Remote.FIA.Database, depending on specification of \code{db}.
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

\seealso{\code{\link{clipFIA}}}

\examples{

data(fiaRI)
data(countiesRI)

## Perform spatial intersection
db <- intersectFIA(fiaRI,
                   countiesRI)

## Group estimates by variable defined
## in `countiesRI`
tpa(db,
    grpBy = COUNTY)
}
