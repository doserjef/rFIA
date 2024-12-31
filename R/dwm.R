dwm <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                landType = 'forest', method = 'TI', lambda = 0.5, 
                areaDomain = NULL, totals = FALSE, variance = FALSE, 
                byPlot = FALSE, condList = FALSE, byFuelType = TRUE, 
                nCores = 1) {

  # Defuse user-supplied expressions in grpBy, areaDomain, and treeDomain
  grpBy_quo <- rlang::enquo(grpBy)
  areaDomain <- rlang::enquo(areaDomain)

  # Handle iterator if db is remote
  remote <- ifelse(class(db) == 'Remote.FIA.Database', 1, 0)
  # Takes value 1 if not remote. iter for remote is a vector of different state IDs. 
  iter <- remoteIter(db, remote)

  # Check for a most recent subset (simple logical value)
  mr <- checkMR(db, remote)

  # Prep for areal summary (converts polys to sf, converts factors to chrs, 
  # and adds the polyID column giving a unique ID to each areal unit). 
  polys <- arealSumPrep1(polys)

  # Run the main portion of the function
  out <- lapply(X = iter, FUN = dwmStarter, db, grpBy_quo = grpBy_quo, 
                polys, returnSpatial, landType, method, lambda, areaDomain, 
                byPlot, condList, totals, byFuelType, nCores, remote, mr)

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
  aGrpBy <- out[names(out) == 'aGrpBy'][[1]]
  grpSyms <- dplyr::syms(grpBy)
  aGrpSyms <- dplyr::syms(aGrpBy)

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
    aEst <- aEst %>%
      dplyr::group_by(!!!aGrpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>%
      dplyr::select(!!!aGrpSyms, fa_mean, fa_var, nPlots.y)

    tEst <- tEst %>%
      dplyr::group_by(!!!grpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>%
      dplyr::left_join(aEst, by = aGrpBy) %>%
      dplyr::mutate(VOL_TOTAL = VOL_mean, 
                    BIO_TOTAL = BIO_mean, 
                    CARB_TOTAL = CARB_mean, 
                    AREA_TOTAL = fa_mean, 
                    # Ratios
                    VOL_ACRE = VOL_TOTAL / AREA_TOTAL, 
                    BIO_ACRE = BIO_TOTAL / AREA_TOTAL, 
                    CARB_ACRE = CARB_TOTAL / AREA_TOTAL, 
                    # Variances
                    VOL_TOTAL_VAR = VOL_var,
                    BIO_TOTAL_VAR = BIO_var, 
                    CARB_TOTAL_VAR = CARB_var, 
                    AREA_TOTAL_VAR = fa_var, 
                    VOL_ACRE_VAR = ratioVar(VOL_mean, fa_mean, VOL_var, fa_var, VOL_cv), 
                    BIO_ACRE_VAR = ratioVar(BIO_mean, fa_mean, BIO_var, fa_var, BIO_cv), 
                    CARB_ACRE_VAR = ratioVar(CARB_mean, fa_mean, CARB_var, fa_var, CARB_cv), 
                    # Sampling Errors
                    VOL_TOTAL_SE = sqrt(VOL_var) / VOL_mean * 100,
                    BIO_TOTAL_SE = sqrt(BIO_var) / BIO_mean * 100,
                    CARB_TOTAL_SE = sqrt(CARB_var) / CARB_mean * 100,
                    AREA_TOTAL_SE = sqrt(fa_var) / fa_mean * 100,
                    VOL_ACRE_SE = sqrt(VOL_ACRE_VAR) / VOL_ACRE * 100,
                    BIO_ACRE_SE = sqrt(BIO_ACRE_VAR) / BIO_ACRE * 100,
                    CARB_ACRE_SE = sqrt(CARB_ACRE_VAR) / CARB_ACRE * 100,
                    # Plot counts
                    nPlots_DWM = nPlots.x,
                    nPlots_AREA = nPlots.y,
                    N = P2PNTCNT_EU) %>%
      dplyr::select(!!!grpSyms, VOL_ACRE, BIO_ACRE, CARB_ACRE,
                    VOL_TOTAL, BIO_TOTAL, CARB_TOTAL, AREA_TOTAL,
                    VOL_ACRE_VAR, BIO_ACRE_VAR, CARB_ACRE_VAR,
                    VOL_TOTAL_VAR, BIO_TOTAL_VAR, CARB_TOTAL_VAR, AREA_TOTAL_VAR,
                    VOL_ACRE_SE, BIO_ACRE_SE, CARB_ACRE_SE,
                    VOL_TOTAL_SE, BIO_TOTAL_SE, CARB_TOTAL_SE, AREA_TOTAL_SE,
                    nPlots_DWM, nPlots_AREA, N)

    # Drop totals unless told not to
    if (!totals) {
      tEst <- tEst[, !stringr::str_detect(names(tEst), '_TOTAL')]
    }

    # Select either variance or sampling errors, depending on input
    if (variance) {
      tEst <- tEst[, !stringr::str_detect(names(tEst), '_SE')]
    } else {
      tEst <- tEst[, !stringr::str_detect(names(tEst), '_VAR')]
    }
  }

  # Pretty output
  tEst <- tEst %>%
    dplyr::ungroup() %>% 
    dplyr::mutate_if(is.factor, as.character) %>% 
    as_tibble()

  # We don't include YEAR in treeList output, and NA groups will be important
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
