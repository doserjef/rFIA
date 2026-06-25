area <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                 byLandType =  FALSE, landType = 'forest', method = 'TI', 
                 lambda = 0.5, treeDomain = NULL, areaDomain = NULL, 
                 totals = TRUE, byPlot = FALSE, condList = FALSE, 
                 nCores = 1) {

  # Defuse user-supplied expressions in grpBy, areaDomain, and treeDomain
  grpBy_quo <- rlang::enquo(grpBy)
  areaDomain <- rlang::enquo(areaDomain)
  treeDomain <- rlang::enquo(treeDomain)

  # Handle iterator if db is remote
  remote <- ifelse(class(db) == 'Remote.FIA.Database', 1, 0)
  # Takes value 1 if not remote. iter for remote is a vector of different state IDs. 
  iter <- remoteIter(db, remote)

  # Check for a most recent subset (simple logical value)
  mr <- checkMR(db, remote)

  # Prep for areal summary (converts polys to sf, converts factors to chrs, 
  # and adds the polyID column giving a unique ID to each areal unit). 
  polys <- arealSumPrep1(polys)

  # Run the main portion of the model
  out <- lapply(X = iter, FUN = areaStarter, db, grpBy_quo = grpBy_quo, 
                polys, returnSpatial, byLandType, landType, method, 
                lambda, treeDomain, areaDomain, totals, byPlot, condList, 
                nCores, remote, mr)

  # Bring the results back
  out <- unlist(out, recursive = FALSE)

  # Throw an error if there are no plots that occur within the polygons of interest
  if (remote) {
    out <- dropStatesOutsidePolys(out)
  }
  # Extract things from the output list
  aEst <- dplyr::bind_rows(out[names(out) == 'aEst'])
  tEst <- dplyr::bind_rows(out[names(out) == 'tEst'])
  grpBy <- out[names(out) == 'grpBy'][[1]]
  grpSyms <- dplyr::syms(grpBy)

  # Summarize population estimates across estimation units ----------------
  if (!byPlot & !condList) {
    # Combine most recent population estimates across states with potentially
    # different reporting schedules (e.g., if 2016 is most recent in MI and 2017 
    # is most recent in WI, combine them and label as 2017. 
    if (mr) {
      tEst <- combineMR(tEst)
      aEst <- combineMR(aEst)
    }

    # Totals and ratios ---------------
    tEst <- tEst %>%
      dplyr::group_by(!!!grpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE)))
    aEst <- aEst %>%
      dplyr::group_by(YEAR) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>%
      dplyr::select(YEAR, fad_mean, fad_var, nPlots.y)

    # Bring them together
    tEst <- tEst %>%
      dplyr::left_join(aEst, by = 'YEAR') %>%
      # Renaming, computing ratios, and SE
      dplyr::mutate(PERC_AREA = fa_mean / fad_mean,
                    AREA_TOTAL = fa_mean,
                    AREA_TOTAL_SE = sqrt(fa_var) / AREA_TOTAL *100,
                    # Ratio variance
                    rVar = ratioVar(fa_mean, fad_mean, fa_var, fad_var, fa_cv),
                    # Convert to percentage
                    PERC_AREA = PERC_AREA * 100,
                    rVar = rVar * (100^2),
                    # Ratio variances
                    # These aren't truly negative values, but come from rounding errors
                    # when PERC_AREA = 100, i.e., estimated variance is 0
                    PERC_AREA_SE = dplyr::case_when(rVar < 0 ~ 0,
                                                    TRUE ~ sqrt(rVar) / PERC_AREA * 100),
                    PERC_AREA_VAR = dplyr::case_when(rVar < 0 ~ 0,
                                                     TRUE ~ rVar),
                    AREA_TOTAL_VAR = fa_var,
                    nPlots_AREA_NUM = nPlots.x,
                    nPlots_AREA_DEN = nPlots.y) %>%
      dplyr::select(!!!grpSyms, PERC_AREA, AREA_TOTAL, PERC_AREA_SE, AREA_TOTAL_SE,
                    PERC_AREA_VAR, AREA_TOTAL_VAR, nPlots_AREA_NUM, nPlots_AREA_DEN)

    # Drop totals unless told not to
    if (!totals) {
      tEst <- tEst[, !stringr::str_detect(names(tEst), '_TOTAL')] 
    }

    # Remove variance columns to avoid confusion
    tEst <- tEst[,!stringr::str_detect(names(tEst), '_VAR')]

  }

  # Pretty output
  tEst <- tEst %>%
    dplyr::ungroup() %>% 
    dplyr::mutate_if(is.factor, as.character) %>% 
    as_tibble()

  # We don't include YEAR in condList output, and NA groups will be important
  # for retaining non-treed forestland
  if (!condList) {
    tEst <- tEst %>% 
      tidyr::drop_na(grpBy[!c(grpBy %in% names(polys))]) %>% 
      dplyr::arrange(YEAR)
  }

  # Prep for spatial plots
  if (returnSpatial & byPlot) {
    grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]
  }

  # For spatial polygons
  if (returnSpatial & !byPlot) {
    sfCol <- attr(polys, 'sf_column')
    sfColSyms <- dplyr::syms(sfCol)
    tEst <- dplyr::left_join(tEst, 
                             as.data.frame(dplyr::select(polys, polyID, !!!sfColSyms)), 
                             by = 'polyID')
  }

  # Convert to sf if spatial
  if (returnSpatial) {
    tEst <- sf::st_sf(tEst)
  }

  return(tEst)

}
