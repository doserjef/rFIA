# Helper for querying the FIADB-API "fullreport" endpoint (the programmatic
# interface behind the EVALIDator web tool) to generate ground-truth reference
# values for rFIA's numeric regression tests.
#
# This script is NOT part of the rFIA package (core_references/ is in
# .Rbuildignore) and is not sourced by tests/testthat/ at run time. It is run
# once, by hand, to produce hard-coded reference values that get pasted into
# test files -- so the test suite itself stays network-free.
#
# API docs: https://apps.fs.usda.gov/fiadb-api/ (see also
# static/pdf/EVALIDator_User_Guide-v2.1.pdf on that host)

library(jsonlite)
library(curl)

#' Query the FIADB-API fullreport endpoint
#'
#' @param wc Eval group code: STATECD + 4-digit year, e.g. 442024 for Rhode
#'   Island's 2024 eval group (see POP_EVAL_GRP.EVAL_GRP in the raw tables).
#' @param snum Numerator ATTRIBUTE_NBR (see EVALIDATOR_POP_ESTIMATE.csv for the
#'   attribute library / SQL definitions).
#' @param sdenom Optional denominator ATTRIBUTE_NBR for a ratio estimate
#'   (e.g. 2 = area of forest land, in acres; 3 = area of timberland).
#' @param strFilter Optional raw SQL WHERE-clause fragment applied to BOTH the
#'   numerator and denominator, e.g. "COND.PHYSCLCD in (21,22,23,24,25,26,27,28,29)".
#'   This is the right tool for areaDomain-style filters, which restrict which
#'   conditions/plots count in the area denominator as well as the tree sum.
#'   Do NOT prefix with "AND" -- empirically that causes the API to return an
#'   "Estimate Error: Parameters resulted in no data being retrieved" even for
#'   filters that should match plenty of data (confirmed against COND.OWNCD/
#'   TREE.SPCD test queries), despite the docs claiming a leading AND is
#'   optional.
#' @param wnum Optional raw SQL WHERE-clause fragment applied to the
#'   numerator ONLY, e.g. "TREE.SPCD = 129". This is the right tool for
#'   treeDomain-style filters, which should not change the area denominator
#'   (matches rFIA's treeDomain semantics -- confirmed: RI white pine TPA via
#'   wnum matches rFIA's tpa(treeDomain = SPCD == 129) to full double
#'   precision, both point estimate and percent SE). Only meaningful when
#'   `sdenom` is also supplied.
#' @param rselected,cselected Row/column grouping. "Total" for a single
#'   population-level estimate with no grouping.
#'
#' @return A named list with either estimate/se/plotCount (non-ratio) or
#'   ratioEstimate/ratioSE/ratioSEPercent/numPlotCount/denPlotCount (ratio).
fetch_evalidator <- function(wc,
                              snum,
                              sdenom = NULL,
                              strFilter = NULL,
                              wnum = NULL,
                              rselected = "Total",
                              cselected = "Total") {
  base <- "https://apps.fs.usda.gov/fiadb-api/fullreport"
  params <- list(
    wc = wc,
    snum = snum,
    rselected = rselected,
    cselected = cselected,
    outputFormat = "NJSON"
  )
  if (!is.null(sdenom)) params$sdenom <- sdenom
  if (!is.null(strFilter)) params$strFilter <- strFilter
  if (!is.null(wnum)) params$wnum <- wnum

  query <- paste(
    vapply(names(params), function(n) {
      paste0(n, "=", curl::curl_escape(as.character(params[[n]])))
    }, character(1)),
    collapse = "&"
  )
  url <- paste0(base, "?", query)

  resp <- curl::curl_fetch_memory(url)
  if (resp$status_code != 200) {
    stop("FIADB-API request failed (HTTP ", resp$status_code, "): ", url)
  }
  txt <- rawToChar(resp$content)
  parsed <- jsonlite::fromJSON(txt, simplifyVector = FALSE)

  if (is.null(parsed$estimates) || length(parsed$estimates) == 0) {
    stop("No estimates returned for query: ", url,
         "\nRaw response head: ", substr(txt, 1, 500))
  }
  est <- parsed$estimates[[1]]

  if (!is.null(est$RATIO_ESTIMATE)) {
    list(
      ratioEstimate = est$RATIO_ESTIMATE,
      ratioSE = est$RATIO_SE,
      ratioSEPercent = est$RATIO_SE_PERCENT,
      numEstimate = est$NUMERATOR_ESTIMATE,
      numPlotCount = est$NUMERATOR_PLOT_COUNT,
      denEstimate = est$DENOMINATOR_ESTIMATE,
      denPlotCount = est$DENOMINATOR_PLOT_COUNT,
      url = url
    )
  } else {
    list(
      estimate = est$ESTIMATE,
      se = est$SE,
      sePercent = est$SE_PERCENT,
      plotCount = est$PLOT_COUNT,
      url = url
    )
  }
}
