growMort <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                     bySpecies = FALSE, bySizeClass = FALSE, landType = 'forest', 
                     treeType = 'all', method = 'TI', lambda = 0.5, 
                     stateVar = 'TPA', treeDomain = NULL, areaDomain = NULL, 
                     totals = FALSE, byPlot = FALSE, 
                     treeList = FALSE, nCores = 1) {

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

  # Run the main portion of the function
  out <- lapply(X = iter, FUN = growMortStarter, db, grpBy_quo, 
                polys, returnSpatial, bySpecies, bySizeClass, landType, 
                treeType, method, lambda, stateVar, treeDomain, areaDomain, 
                totals, byPlot, treeList, nCores, remote, mr)

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
  if (!byPlot & !treeList) {
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
      dplyr::mutate(CURR_TOTAL = tPlot_mean,
                    PREV_TOTAL = pPlot_mean,
                    RECR_TOTAL = rPlot_mean,
                    MORT_TOTAL = mPlot_mean,
                    REMV_TOTAL = hPlot_mean,
                    GROW_TOTAL = gPlot_mean,
                    CHNG_TOTAL = cPlot_mean,
                    AREA_TOTAL = fa_mean,

                    # Ratios
                    RECR_TPA = rPlot_mean / fa_mean,
                    MORT_TPA = mPlot_mean / fa_mean,
                    REMV_TPA = hPlot_mean / fa_mean,
                    GROW_TPA = gPlot_mean / fa_mean,
                    CHNG_TPA = cPlot_mean / fa_mean,
                    RECR_PERC = rPlot_mean / pPlot_mean,
                    MORT_PERC = mPlot_mean / pPlot_mean,
                    REMV_PERC = hPlot_mean / pPlot_mean,
                    GROW_PERC = gPlot_mean / pPlot_mean,
                    CHNG_PERC = cPlot_mean / pPlot_mean,

                    # Variances
                    CURR_TOTAL_VAR = tPlot_var,
                    PREV_TOTAL_VAR = pPlot_var,
                    RECR_TOTAL_VAR = rPlot_var,
                    MORT_TOTAL_VAR = mPlot_var,
                    REMV_TOTAL_VAR = hPlot_var,
                    GROW_TOTAL_VAR = gPlot_var,
                    CHNG_TOTAL_VAR = cPlot_var,
                    AREA_TOTAL_VAR = fa_var,
                    RECR_TPA_VAR = ratioVar(rPlot_mean, fa_mean, rPlot_var, fa_var, rPlot_cv),
                    MORT_TPA_VAR = ratioVar(mPlot_mean, fa_mean, mPlot_var, fa_var, mPlot_cv),
                    REMV_TPA_VAR = ratioVar(hPlot_mean, fa_mean, hPlot_var, fa_var, hPlot_cv),
                    GROW_TPA_VAR = ratioVar(gPlot_mean, fa_mean, gPlot_var, fa_var, gPlot_cv),
                    CHNG_TPA_VAR = ratioVar(cPlot_mean, fa_mean, cPlot_var, fa_var, cPlot_cv),
                    RECR_PERC_VAR = ratioVar(rPlot_mean, pPlot_mean, rPlot_var, pPlot_var, rPlot_cv_t),
                    MORT_PERC_VAR = ratioVar(mPlot_mean, pPlot_mean, mPlot_var, pPlot_var, mPlot_cv_t),
                    REMV_PERC_VAR = ratioVar(hPlot_mean, pPlot_mean, hPlot_var, pPlot_var, hPlot_cv_t),
                    GROW_PERC_VAR = ratioVar(gPlot_mean, pPlot_mean, gPlot_var, pPlot_var, gPlot_cv_t),
                    CHNG_PERC_VAR = ratioVar(cPlot_mean, pPlot_mean, cPlot_var, pPlot_var, cPlot_cv_t),

                    # Convert to percentages
                    RECR_PERC = RECR_PERC * 100,
                    MORT_PERC = MORT_PERC * 100,
                    REMV_PERC = REMV_PERC * 100,
                    GROW_PERC = GROW_PERC * 100,
                    CHNG_PERC = CHNG_PERC * 100,
                    RECR_PERC_VAR = RECR_PERC_VAR * 100^2,
                    MORT_PERC_VAR = MORT_PERC_VAR * 100^2,
                    REMV_PERC_VAR = REMV_PERC_VAR * 100^2,
                    GROW_PERC_VAR = GROW_PERC_VAR * 100^2,
                    CHNG_PERC_VAR = CHNG_PERC_VAR * 100^2,

                    # Sampling Errors
                    CURR_TOTAL_SE = sqrt(tPlot_var) / tPlot_mean * 100,
                    PREV_TOTAL_SE = sqrt(pPlot_var) / pPlot_mean * 100,
                    RECR_TOTAL_SE = sqrt(rPlot_var) / rPlot_mean * 100,
                    MORT_TOTAL_SE = sqrt(mPlot_var) / mPlot_mean * 100,
                    REMV_TOTAL_SE = sqrt(hPlot_var) / rPlot_mean * 100,
                    GROW_TOTAL_SE = sqrt(gPlot_var) / gPlot_mean * 100,
                    CHNG_TOTAL_SE = sqrt(cPlot_var) / cPlot_mean * 100,
                    AREA_TOTAL_SE = sqrt(fa_var) / fa_mean * 100,
                    RECR_TPA_SE = sqrt(RECR_TPA_VAR) / RECR_TPA * 100,
                    MORT_TPA_SE = sqrt(MORT_TPA_VAR) / MORT_TPA * 100,
                    REMV_TPA_SE = sqrt(REMV_TPA_VAR) / REMV_TPA * 100,
                    GROW_TPA_SE = sqrt(GROW_TPA_VAR) / GROW_TPA * 100,
                    RECR_PERC_SE = sqrt(RECR_PERC_VAR) / RECR_PERC * 100,
                    MORT_PERC_SE = sqrt(MORT_PERC_VAR) / MORT_PERC * 100,
                    REMV_PERC_SE = sqrt(REMV_PERC_VAR) / REMV_PERC * 100,
                    GROW_PERC_SE = sqrt(GROW_PERC_VAR) / GROW_PERC * 100,
                    CHNG_PERC_SE = sqrt(CHNG_PERC_VAR) / CHNG_PERC * 100,

                    # Plot counts
                    nPlots_TREE = nPlots.x,
                    nPlots_AREA = nPlots.y) %>%
      dplyr::select(!!!grpSyms, RECR_TPA:CHNG_PERC,
                    RECR_TOTAL:CHNG_TOTAL, PREV_TOTAL, CURR_TOTAL, AREA_TOTAL,
                    RECR_TPA_VAR:CHNG_PERC_VAR,
                    RECR_TOTAL_VAR:CHNG_TOTAL_VAR, PREV_TOTAL_VAR, CURR_TOTAL_VAR, AREA_TOTAL_VAR,
                    RECR_TPA_SE:CHNG_PERC_SE,
                    RECR_TOTAL_SE:CHNG_TOTAL_SE, PREV_TOTAL_SE, CURR_TOTAL_SE, AREA_TOTAL_SE,
                    nPlots_TREE, nPlots_AREA, N) %>%
      # Rounding errors can cause GROW_TPA to take an extremely small value instead of zero
      # Make it zero when this happens
      dplyr::mutate(dplyr::across(c(GROW_TPA, GROW_PERC, GROW_TOTAL,
                                    GROW_TPA_VAR, GROW_PERC_VAR, GROW_TOTAL_VAR),
                                  .fns = ~case_when(abs(.x) < 1e-5 ~ 0,
                                                   TRUE ~ .x)))

    # Drop totals unless told not to
    if (!totals) {
      tEst <- tEst[, !stringr::str_detect(names(tEst), '_TOTAL')] 
    }

    # Remove variance columns to avoid confusion
    tEst <- tEst[,!stringr::str_detect(names(tEst), '_VAR')]

  }

  # Modify names if a different state variable was given
  if (stateVar != 'TPA') {
    names(tEst) <- str_replace(names(tEst), 'TPA', paste(stateVar, 'ACRE', sep = '_'))
  }
  names(tEst) <- str_replace(names(tEst), 'BAA_ACRE', 'BAA')

  # Pretty output
  tEst <- tEst %>%
    dplyr::ungroup() %>% 
    dplyr::mutate_if(is.factor, as.character) %>% 
    as_tibble()

  # We don't include YEAR in treeList output, and NA groups will be important
  # for retaining non-treed forestland
  if (!treeList) {
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
