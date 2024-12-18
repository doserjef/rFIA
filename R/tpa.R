tpa <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                bySpecies = FALSE, bySizeClass = FALSE, landType = 'forest', 
                treeType = 'live', method = 'TI', lambda = 0.5, treeDomain = NULL, 
                areaDomain = NULL, totals = FALSE, variance = FALSE, byPlot = FALSE, 
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
  out <- lapply(X = iter, FUN = tpaStarter, db, grpBy_quo = grpBy_quo, 
                polys, returnSpatial, bySpecies, bySizeClass, landType, 
                treeType, method, lambda, treeDomain, areaDomain, totals, 
                byPlot, treeList, nCores, remote, mr)

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
                    BA_TOTAL = bPlot_mean,
                    AREA_TOTAL = fa_mean,
                    # Ratios
                    TPA = TREE_TOTAL / AREA_TOTAL,
                    BAA = BA_TOTAL / AREA_TOTAL, 
                    # Variances
                    TREE_TOTAL_VAR = tPlot_var, 
                    BA_TOTAL_VAR = bPlot_var,
                    AREA_TOTAL_VAR = fa_var,
                    # TODO: need to look through ratioVar.
                    TPA_VAR = ratioVar(tPlot_mean, fa_mean, tPlot_var, fa_var, tPlot_cv),
                    BAA_VAR = ratioVar(bPlot_mean, fa_mean, bPlot_var, fa_var, bPlot_cv),
                    # Sampling Errors
                    TREE_TOTAL_SE = sqrt(tPlot_var) / tPlot_mean * 100,
                    BA_TOTAL_SE = sqrt(bPlot_var) / bPlot_mean * 100,
                    AREA_TOTAL_SE = sqrt(fa_var) / fa_mean * 100,
                    TPA_SE = sqrt(TPA_VAR) / TPA * 100,
                    BAA_SE = sqrt(BAA_VAR) / BAA * 100,
                    # Plot counts
                    nPlots_TREE = nPlots.x, 
                    nPlots_AREA = nPlots.y,
                    N = P2PNTCNT_EU) %>%
      dplyr::select(!!!grpSyms, TPA, BAA, TREE_TOTAL, BA_TOTAL, AREA_TOTAL, 
                    TPA_VAR, BAA_VAR, TREE_TOTAL_VAR, BA_TOTAL_VAR, AREA_TOTAL_VAR, 
                    TPA_SE, BAA_SE, TREE_TOTAL_SE, BA_TOTAL_SE, AREA_TOTAL_SE, 
                    nPlots_TREE, nPlots_AREA, N)

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
