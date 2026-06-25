areaChange <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                       byLandType = FALSE, landType = 'forest', method = 'TI', 
                       lambda = 0.5, treeDomain = NULL, areaDomain = NULL, 
                       byPlot = FALSE, condList = FALSE, chngType = 'net', nCores = 1) {

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

  out <- lapply(X = iter, FUN = areaChangeStarter, db, grpBy_quo = grpBy_quo, 
                polys, returnSpatial, byLandType, landType, method, lambda, 
                treeDomain, areaDomain, byPlot, condList, chngType, 
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

  # Summarize population estimates across estimation units
  if (!byPlot & !condList){

    # Combine most-recent population estimates across states with potentially
    # different reporting schedules, i.e., if 2016 is most recent in MI and 2017 is
    # most recent in WI, combine them and label as 2017
    if (mr) {
      tEst <- combineMR(tEst)
      aEst <- combineMR(aEst)
    }

    # Totals and ratios -------------------------------------------------------
    aEst <- aEst %>%
      dplyr::group_by(!!!grpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>% 
      dplyr::select(!!!grpSyms, prev_mean, prev_var, nPlots.y)


    tEst <- tEst %>%
      dplyr::group_by(!!!grpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>% 
      dplyr::left_join(aEst, by = grpBy) %>%
      dplyr::mutate(AREA_CHNG = ac_mean,
                    PREV_AREA = prev_mean,
                    # Ratios
                    PERC_CHNG = AREA_CHNG / PREV_AREA,
                    # Variances
                    AREA_CHNG_VAR = ac_var,
                    PREV_AREA_VAR = prev_var,
                    PERC_CHNG_VAR = ratioVar(ac_mean, prev_mean, ac_var, prev_var, ac_cv),

                    # Convert to percentage
                    PERC_CHNG = PERC_CHNG * 100,
                    PERC_CHNG_VAR = PERC_CHNG_VAR * (100^2),

                    # Sampling Errors
                    AREA_CHNG_SE = sqrt(ac_var) / abs(ac_mean) * 100,
                    PREV_AREA_SE = sqrt(prev_var) / abs(prev_mean) * 100,
                    PERC_CHNG_SE = sqrt(PERC_CHNG_VAR) / abs(PERC_CHNG) * 100,

                    # Plot counts
                    nPlots_AREA = nPlots.x) %>%
      dplyr::select(!!!grpSyms, PERC_CHNG, AREA_CHNG, PREV_AREA,
                    PERC_CHNG_VAR, AREA_CHNG_VAR, PREV_AREA_VAR,
                    PERC_CHNG_SE, AREA_CHNG_SE, PREV_AREA_SE,
                    nPlots_AREA, N)

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
      # NOTE: the drop_na call is being dropped because this leads to a drop
      #       in certain rows of interest under certain grpBy specifications.
      # tidyr::drop_na(grpBy[!c(grpBy %in% names(polys))]) %>%
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

