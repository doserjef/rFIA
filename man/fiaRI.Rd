\name{fiaRI}
\alias{fiaRI}
\docType{data}
\title{FIADB for Rhode Island 2013 - 2018}

\description{Subset of the Forest Inventory and Analysis Database for the state of Rhode Island. Reporting years range from 2013 - 2018. Specify \code{fiaRI} as the \code{db} argument in any \code{rFIA} function to produce estimates for the state of Rhode Island. NOTE: the fiaRI object was updated in v1.1.0 to reflect changes in the FIA Database that took place since creation of the original object. 

Download other subsets of the FIA Database from the FIA Datamart: \url{https://apps.fs.usda.gov/fia/datamart/datamart.html}. Once downloaded, unzip the directory, and read into R using \code{\link{readFIA}}.
}

\usage{data("fiaRI")}
\format{
---- FIA Database Object -----
Reporting Years:  2013 2014 2015 2016 2017 2018
States:           RHODE ISLAND
Total Plots:      769
Memory Used:      20.1 Mb
Tables:           COND_DWM_CALC COND INVASIVE_SUBPLOT_SPP P2VEG_SUBP_STRUCTURE PLOT POP_ESTN_UNIT POP_EVAL_GRP POP_EVAL_TYP POP_EVAL POP_PLOT_STRATUM_ASSGN POP_STRATUM SEEDLING SUBP_COND_CHNG_MTRX SUBP_COND SUBPLOT SURVEY TREE_GRM_BEGIN TREE_GRM_COMPONENT TREE_GRM_MIDPT TREE
}

\examples{
data(fiaRI)
summary(fiaRI)
print(fiaRI)
}
\keyword{datasets}
