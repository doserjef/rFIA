vitalRates <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                       bySpecies = FALSE, bySizeClass = FALSE, landType = 'forest', 
                       treeType = 'all', method = 'TI', lambda = 0.5, 
                       treeDomain = NULL, areaDomain = NULL, totals = FALSE, 
                       byPlot = FALSE, treeList = FALSE, nCores = 1) {

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
  out <- lapply(X = iter, FUN = vitalRatesStarter, db, 
                grpBy_quo = grpBy_quo, polys, returnSpatial, 
                bySpecies, bySizeClass, landType, treeType, 
                method, lambda, treeDomain, areaDomain, 
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
      dplyr::mutate(TREE_TOTAL = tPlot_mean,
                    DIA_TOTAL = dPlot_mean,
                    BA_TOTAL = bPlot_mean,
                    NETVOL_TOTAL = gPlot_mean,
                    SAWVOL_TOTAL = sPlot_mean,
                    BIO_TOTAL = bioPlot_mean,
                    AREA_TOTAL = fa_mean,

                    # Ratios
                    DIA_GROW = dPlot_mean / tPlot_mean,
                    BA_GROW = bPlot_mean / tPlot_mean,
                    NETVOL_GROW = gPlot_mean / tPlot_mean,
                    SAWVOL_GROW = sPlot_mean / tPlot_mean,
                    BIO_GROW = bioPlot_mean / tPlot_mean,
                    BA_GROW_AC = bPlot_mean / fa_mean,
                    NETVOL_GROW_AC = gPlot_mean / fa_mean,
                    SAWVOL_GROW_AC = sPlot_mean / fa_mean,
                    BIO_GROW_AC = bioPlot_mean / fa_mean,

                    # Variances
                    TREE_TOTAL_VAR = tPlot_var,
                    DIA_TOTAL_VAR = dPlot_var,
                    BA_TOTAL_VAR = bPlot_var,
                    NETVOL_TOTAL_VAR = gPlot_var,
                    SAWVOL_TOTAL_VAR = sPlot_var,
                    BIO_TOTAL_VAR = bioPlot_var,
                    AREA_TOTAL_VAR = fa_var,

                    DIA_GROW_VAR = ratioVar(dPlot_mean, tPlot_mean, dPlot_var, tPlot_var, dPlot_cv_t),
                    BA_GROW_VAR = ratioVar(bPlot_mean, tPlot_mean, bPlot_var, tPlot_var, bPlot_cv_t),
                    NETVOL_GROW_VAR = ratioVar(gPlot_mean, tPlot_mean, gPlot_var, tPlot_var, gPlot_cv_t),
                    SAWVOL_GROW_VAR = ratioVar(sPlot_mean, tPlot_mean, sPlot_var, tPlot_var, sPlot_cv_t),
                    BIO_GROW_VAR = ratioVar(bioPlot_mean, tPlot_mean, bioPlot_var, tPlot_var, bioPlot_cv_t),
                    BA_GROW_AC_VAR = ratioVar(bPlot_mean, fa_mean, bPlot_var, fa_var, bPlot_cv),
                    NETVOL_GROW_AC_VAR = ratioVar(gPlot_mean, fa_mean, gPlot_var, fa_var, gPlot_cv),
                    SAWVOL_GROW_AC_VAR = ratioVar(sPlot_mean, fa_mean, sPlot_var, fa_var, sPlot_cv),
                    BIO_GROW_AC_VAR = ratioVar(bioPlot_mean, fa_mean, bioPlot_var, fa_var, bioPlot_cv),

                    # Sampling Errors
                    TREE_TOTAL_SE = sqrt(tPlot_var) / tPlot_mean * 100,
                    DIA_TOTAL_SE = sqrt(dPlot_var) / abs(dPlot_mean) * 100,
                    BA_TOTAL_SE = sqrt(bPlot_var) / abs(bPlot_mean) * 100,
                    NETVOL_TOTAL_SE = sqrt(gPlot_var) / abs(gPlot_mean) * 100,
                    SAWVOL_TOTAL_SE = sqrt(sPlot_var) / abs(sPlot_mean) * 100,
                    BIO_TOTAL_SE = sqrt(bioPlot_var) / abs(bioPlot_mean) * 100,
                    AREA_TOTAL_SE = sqrt(fa_var) / fa_mean * 100,

                    DIA_GROW_SE = sqrt(DIA_GROW_VAR) / abs(DIA_GROW) * 100,
                    BA_GROW_SE = sqrt(BA_GROW_VAR) / abs(BA_GROW) * 100,
                    NETVOL_GROW_SE = sqrt(NETVOL_GROW_VAR) / abs(NETVOL_GROW) * 100,
                    SAWVOL_GROW_SE = sqrt(SAWVOL_GROW_VAR) / abs(SAWVOL_GROW) * 100,
                    BIO_GROW_SE = sqrt(BIO_GROW_VAR) / abs(BIO_GROW) * 100,
                    BA_GROW_AC_SE = sqrt(BA_GROW_AC_VAR) / abs(BA_GROW_AC) * 100,
                    NETVOL_GROW_AC_SE = sqrt(NETVOL_GROW_AC_VAR) / abs(NETVOL_GROW_AC) * 100,
                    SAWVOL_GROW_AC_SE = sqrt(SAWVOL_GROW_AC_VAR) / abs(SAWVOL_GROW_AC) * 100,
                    BIO_GROW_AC_SE = sqrt(BIO_GROW_AC_VAR) / abs(BIO_GROW_AC) * 100,


                    # Plot counts
                    nPlots_TREE = nPlots.x,
                    nPlots_AREA = nPlots.y) %>%
      dplyr::select(!!!grpSyms,
                    DIA_GROW:BIO_GROW_AC,
                    DIA_TOTAL:BIO_TOTAL, TREE_TOTAL, AREA_TOTAL,
                    DIA_GROW_VAR:BIO_GROW_AC_VAR,
                    DIA_TOTAL_VAR:BIO_TOTAL_VAR, TREE_TOTAL_VAR, AREA_TOTAL_VAR,
                    DIA_GROW_SE:BIO_GROW_AC_SE,
                    DIA_TOTAL_SE:BIO_TOTAL_SE, TREE_TOTAL_SE, AREA_TOTAL_SE,
                    nPlots_TREE, nPlots_AREA)

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
