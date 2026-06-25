\name{writeFIA}
\alias{writeFIA}

\title{Write FIA tables to local directory}

\description{
Write \code{FIA.Database} object to local directory as a series of .csv files representing each table. Most useful for writing merged states and temporal/spatial subsets of the database. Once written as .csv, files can be reloaded into R with \code{readFIA()}.
}

\usage{
writeFIA(db, dir, byState = FALSE, nCores = 1, ...)
}

\arguments{
  \item{db}{\code{FIA.Database} object produced from \code{readFIA()} or \code{getFIA()}.}
  \item{dir}{directory where FIA Datatables will be stored.}
  \item{nCores}{numeric; number of cores to use for parallel implementation. Check available cores using \code{\link{detectCores}}. Default = 1, serial processing.}
  \item{byState}{logical; should tables be written out by state? Must be TRUE if planning to load data as an out-of-memory database in the future (see \code{readFIA()}).}
  \item{\dots}{other arguments to pass to \code{data.table::fwrite()}.}
}

\details{
Easy, efficient parallelization is implemented with the \code{\link{parallel}} package. Users must only specify the \code{nCores} argument with a value greater than 1 in order to implement parallel processing on their machines. Parallel implementation is achieved using a snow type cluster on any Windows OS, and with multicore forking on any Unix OS (Linux, Mac). Implementing parallel processing may substantially decrease decrease free memory during processing, particularly on Windows OS. Thus, users should be cautious when running in parallel, and consider implementing serial processing for this task if computational resources are limited (\code{nCores = 1}).
}

\author{
Jeffrey W. Doser, Hunter Stanke, Andrew Finley
}

\seealso{
\code{\link{readFIA}}, \code{\link{getFIA}}
}

\value{
  No return value, called to export \code{FIA.Database} object to a local directory.
}

\examples{
\dontshow{
  # Restrict threads for CRAN compliance
  dt_threads <- data.table::getDTthreads()
  data.table::setDTthreads(1)
}
# Write the 'fiaRI' object to a temporary directory
# Replace temp_dir with the path to your directory (where data will be saved)
temp_dir = tempdir()
writeFIA(fiaRI, dir = temp_dir)
\dontshow{
  # Restore thread setting
  data.table::setDTthreads(dt_threads)
}
}
