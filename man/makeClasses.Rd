\name{makeClasses}
\alias{makeClasses}

\title{Convert numeric variables to class intervals (factor)}

\description{
Convert continuous numeric variables to class intervals with output as factor or numeric classes. Simplified implementation of \code{\link[base]{cut}}. Example uses include computing diameter or height classes for summarization with \pkg{rFIA} functions (e.g. \code{\link{tpa}}, \code{biomass}).}

\usage{
makeClasses(x, interval = NULL, lower = NULL, upper = NULL,
            brks = NULL, numLabs = FALSE)
}

\arguments{
  \item{x}{numeric vector to be converted to factor (class intervals).}

  \item{interval}{numeric; interval of desired output classes. e.g. specify \code{x = DIA} and \code{interval = 2} for 2-inch diameter class intervals.}
  \item{lower}{lower bound of output classes, included in lowest class. e.g. [\code{lower}, ...).}
  \item{upper}{upper bound of output classes, NOT included in highest class. e.g. [..., \code{upper}).}
  \item{brks}{numeric vector of desired breakpoints (bounds) of class intervals.}
  \item{numLabs}{logical; if TRUE, return class intervals as numeric vector with values representing the lower bounds of each interval. If FALSE, return factor with labels of form \code{'[b1, b2)'}.}
}


\value{
Factor or integer vector. Factor values represent class intervals with \code{[b1, b2)} notation, values of integer vectors represent the lower bounds of class intervals (e.g. \code{b1}).
}


\author{
Hunter Stanke and Andrew Finley
}


\seealso{
\code{\link{clipFIA}}
}

\examples{
## Load data from the rFIA package
data(fiaRI)

## Compute Diameter Classes on 1-inch intervals for each tree in TREE table ----
# Factor w/ interval labels
makeClasses(fiaRI$TREE$DIA, interval = 1)
# Numeric w/ lower bound of each class as returned value
makeClasses(fiaRI$TREE$DIA, interval = 1, numLabs = TRUE)

## Compute Stand Age Classes on 20 year intervals for each
## condition in COND table ----
# NOTE: Unrecorded stand age recorded as -999, replace negative values with NA
fiaRI$COND$STDAGE[fiaRI$COND$STDAGE < 0] <- NA
makeClasses(fiaRI$COND$STDAGE, interval = 25)

## Compute Stand Stocking Classes (10\%) for all live (ALSTK),
## and growing stock (GSSTK) in COND table ----
makeClasses(fiaRI$COND$ALSTK, interval = 10) # All Live
makeClasses(fiaRI$COND$GSSTK, interval = 10) # Growing Stock

## Compute \% Slope Classes (20\%) for each condition in COND table ----
makeClasses(fiaRI$COND$SLOPE, interval = 20)

}

