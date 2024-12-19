\name{plotFIA}
\alias{plotFIA}
\title{Static and animated plots of FIA summaries}
\description{
Default behavior for non-spatial summaries produces time-series plots, and for spatial summaries (class \code{sf}) produces choropleth maps. For non-spatial summaries, the user may specify the \code{grp} parameter to produce plots with multiple lines, colored by a grouping variable. Additionally, users may specify an x-axis to produce plots other than time series (e.g. BAA (\code{y}) by size class (\code{x}) colored by species (\code{grp})).
}

\usage{
plotFIA(data, y = NULL, grp = NULL, x = NULL, animate = FALSE, facet = FALSE,
        se = FALSE,n.max = NULL, plot.title = NULL, y.lab = NULL, x.lab = NULL,
        legend.title = NULL, legend.labs = waiver(), limits = c(NA, NA),
        color.option = 'viridis', line.color = "gray30", line.width =1,
        min.year = 2005, direction = 1, alpha = .9, transform = "identity",
        text.size = 1, text.font = '', lab.width = 1, legend.height = 1,
        legend.width = 1, device = "png", savePath = NULL, fileName = NULL)
           }

\arguments{
  \item{data}{dataframe, \code{sf} object, or \code{FIA.Database} object; FIA summary produced from other \code{rFIA} functions (e.g. \code{tpa}, \code{biomass}, etc.). Also accepts \code{FIA.Database}, will return map of plot locations.}

  \item{y}{variable contained in \code{data} which will be used as y-axis or to fill polygons (spatial). NOT quoted.}

  \item{grp}{variable contained in \code{data} which will be used as a grouping variable. Not meaningful for spatial summaries. NOT quoted.}

  \item{x}{variable contained in \code{data} which will be used as a x-axis in place of time. If NULL, time-series plot will be produced. Not meaningful for spatial summaries. NOT quoted.}

  \item{animate}{logical; if TRUE, produces temporally animated plots.}

  \item{facet}{logical; if TRUE, produces temporally grouped plots (stationary).}

    \item{se}{logical; if TRUE, plots error bars along with estimates. All error bars represent 68\% confidence.}

  \item{n.max}{numeric; maximum number of groups to plot. If positive, will plot the top \code{n} groups with respect to \code{y}, and if negative, will plot the bottom \code{n}. Not meaningful for spatial summaries.}

  \item{plot.title}{character; plot title.}

  \item{y.lab}{character; y-axis label. Not meaningful for spatial summaries.}

  \item{x.lab}{character; x-axis label. Not meaningful for spatial summaries.}

  \item{legend.title}{character; title for legend.}

  \item{legend.labs}{character; labels for legend values.}

  \item{limits}{numeric vector of length 2; minumum and maximum of continuous scale for legend.}

  \item{color.option}{character; one of: "viridis" (default), "magma", "inferno", "plasma", or "cividis".}

  \item{line.color}{character; color of plotted line (non-spatial) or polygon outline color (spatial).}

  \item{line.width}{numeric; scalar for plotted line width (non-spatial) polygon outline width (spatial). Specify \code{lineWidth = 0} for no outline.}

  \item{min.year}{numeric; earliest year to be included in animation. FIA data is sparse in years prior to 2005 and estimates are unlikely to be available.}

  \item{direction}{numeric; sets the order of colors in the scale. If 1, the default, colors are ordered from darkest to lightest. If -1, the order of colors is reversed.}

  \item{alpha}{numeric; alpha transparency, a number in [0,1], see argument alpha in \code{hsv}.}

  \item{transform}{character; transformations to apply to plotted variable \code{y}. Options include: "asn", "atanh", "boxcox", "exp", "identity", "log", "log10", "log1p", "log2", "logit", "reciprocal", "reverse", "sqrt".}

  \item{text.size}{numeric; scalar for text size (e.g. text.size = 2 would be twice the default size).}

  \item{text.font}{character; font family. Choose from: 'Short', 'Canonical', 'mono', 'Courier', 'sans', 'Helvetica', 'serif', 'Times', 'AvantGarde', 'Bookman', 'Helvetica-Narrow', 'NewCenturySchoolbook', 'Palatino', 'URWGothic',  'URWBookman', 'NimbusMon', 'URWHelvetica', 'NimbusSan', 'NimbusSanCond', 'CenturySch', 'URWPalladio', 'URWTimes', or 'NimbusRom'.}

  \item{lab.width}{numeric; scalar for legend title width. This value controls text wrapping in title.}

  \item{legend.height}{numeric; scalar for legend height.}

  \item{legend.width}{numeric; scalar for legend width.}

  \item{device}{character; device to use for image save. Can either be a device function (e.g. png()), or one of "eps", "ps", "tex" (pictex), "pdf", "jpeg", "tiff", "png", "bmp", "svg" or "wmf" (windows only).}

  \item{savePath}{character; path to save plot to (combined with fileName).}

  \item{fileName}{character; file name to create on disk.}
}


\details{
To produce spatial plots, summaries must be returned as spatial objects (e.g. specify \code{returnSpatial = TRUE} when computing summaries using \code{\link{tpa}}). For animated plots, also requires that multiple reporting years be present in the summary data (animations iterate through time). For a map of plot locations contained in your \code{FIA.Database}, specify the object as the \code{data} argument.

For objects produced with \code{byPlot = TRUE} and \code{returnSpatial = TRUE} (spatial point patterns), a categorical grouping variable can be specified to \code{grp}. Point radii will reflect magnitude of \code{y} and color will reflect categorical groups (\code{grp}).

If \code{animate = FALSE} and multiple reporting years are present in the summary, produces plots of the most recent subset.

Specify \code{savePath} and \code{fileName} to save plots (animations saved as .gif files).


}



\author{
Hunter Stanke and Andrew Finley
}

\examples{
## Load data from the rFIA package
data(fiaRI)
data(countiesRI)

\dontrun{\donttest{

################### SPATIAL PLOTTING #############################
## Compute abundance estimates for live stems in Rhode Island
## for all available inventory years, summarized by counties and
## return a spatial object
tpaRI <- tpa(fiaRI, polys = countiesRI, returnSpatial = TRUE)

## Produce animated plot
plotFIA(tpaRI, y = TPA, animate = TRUE, legend.title = 'Abundance (TPA)')
## With a square root transform
plotFIA(tpaRI, y = TPA, animate = TRUE, legend.title = 'Abundance (TPA)', transform = 'sqrt')

## Same as above, but for static plots (most recent subset from RI)
tpaMR <- tpa(clipFIA(fiaRI), polys = countiesRI, returnSpatial = TRUE)
## Produce animated plot
plotFIA(tpaMR, y = TPA, animate = FALSE, plot.title = 'Abundance (TPA)')


################# NON-SPATIAL PLOTTING #########################
## Same as above, but return a non-spatial object (no spatial grouping)
tpaRI <- tpa(fiaRI)

## Plot TPA over time
plotFIA(tpaRI, TPA)

## BAA over time, grouped by ownership group
tpaRI_own <- tpa(fiaRI, grpBy = OWNGRPCD)
plotFIA(tpaRI_own, y = BAA, grp = OWNGRPCD)

## BAA by size class (not a time series) grouped by species
tpaRI_sc <- tpa(clipFIA(fiaRI), bySpecies = TRUE, bySizeClass = TRUE)
plotFIA(tpaRI_sc, y = BAA, grp = COMMON_NAME, x = sizeClass, n.max = 4)# Only the top 4

}
}
}

