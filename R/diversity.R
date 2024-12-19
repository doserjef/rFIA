diversity <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                      bySizeClass = FALSE, landType = 'forest', 
                      treeType = 'live', method = 'TI', lambda = 0.5, 
                      stateVar = TPA_UNADJ, grpVar = SPCD, treeDomain = NULL, 
                      areaDomain = NULL, byPlot = FALSE, condList = FALSE,
                      totals = FALSE, variance = FALSE, nCores = 1) {

  # Defuse user-supplied expressions in grpBy, areaDomain, treeDomain, 
  # stateVar, and grpVar
  grpBy_quo <- rlang::enquo(grpBy)
  areaDomain <- rlang::enquo(areaDomain)
  treeDomain <- rlang::enquo(treeDomain)
  stateVar <- rlang::enquo(stateVar)
  grpVar <- rlang::enquo(grpVar)

  # Handle iterator if db is remote
  remote <- ifelse(class(db) == 'Remote.FIA.Database', 1, 0)
  # Takes value 1 if not remote. iter for remote is a vector of different state IDs. 
  iter <- remoteIter(db, remote)

  # Check for a most recent subset
  mr <- checkMR(db, remote)

  # Prep for areal summary (converts polys to sf, converts factors to chrs, 
  # and adds the polyID column giving a unique ID to each areal unit). 
  polys <- arealSumPrep1(polys)

  # Run the main portion of the function
  out <- lapply(X = iter, FUN = diversityStarter, db, grpBy_quo = grpBy_quo, 
                polys, returnSpatial, bySizeClass, landType, treeType, method, 
                lambda, stateVar, grpVar, treeDomain, areaDomain, byPlot, 
                condList, totals, nCores, remote, mr)

  # Bring the results back
  out <- unlist(out, recursive = FALSE)

  # Throw an error if there are no plots that occur within the polygons of interest
  if (remote) {
    out <- dropStatesOutsidePolys(out)
  }
  # Extract things from the output list
  aEst <- dplyr::bind_rows(out[names(out) == 'aEst'])
  tEst <- dplyr::bind_rows(out[names(out) == 'tEst'])
  full <- dplyr::bind_rows(out[names(out) == 'full'])
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
    aEst <- aEst %>% 
      dplyr::group_by(!!!grpSyms) %>% 
      dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>% 
      dplyr::select(!!!grpSyms, fa_mean, fa_var, nPlots.y)

    tEst <- tEst %>%
      dplyr::group_by(!!!grpSyms) %>%
      dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>%
      dplyr::left_join(aEst, by = grpBy) %>%
      dplyr::mutate(H_a = H_mean / fa_mean,
                    Eh_a = Eh_mean / fa_mean,
                    S_a = S_mean / fa_mean,
                    AREA_TOTAL = fa_mean,
                    # Variances
                    AREA_TOTAL_VAR = fa_var,
                    # TODO: need to look through ratioVar
                    H_a_VAR = ratioVar(H_mean, fa_mean, H_var, fa_var, H_cv),
                    Eh_a_VAR = ratioVar(Eh_mean, fa_mean, Eh_var, fa_var, Eh_cv),
                    S_a_VAR = ratioVar(S_mean, fa_mean, S_var, fa_var, S_cv),
                    # Sampling Errors
                    H_a_SE = sqrt(H_a_VAR) / H_a * 100,
                    Eh_a_SE = sqrt(Eh_a_VAR) / Eh_a * 100,
                    S_a_SE = sqrt(S_a_VAR) / S_a * 100,
                    AREA_TOTAL_SE = sqrt(fa_var) / fa_mean * 100,
                    # Plot counts
                    nPlots_TREE = nPlots.x,
                    nPlots_AREA = nPlots.y,
                    N = P2PNTCNT_EU)

    # Up a few spatial scales
    fullGrps <- dplyr::syms(grpBy[!c(grpBy %in% 'lambda')])
    full <- full %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(!!!fullGrps) %>%
      dplyr::summarize(H_g = divIndex(grp, state, index = 'H'),
                       Eh_g = divIndex(grp, state, index = 'Eh'),
                       S_g = divIndex(grp, state, index = 'S')) %>%
      as.data.frame()
    tEst <- tEst %>%
      dplyr::left_join(full, by = grpBy[!c(grpBy %in% 'lambda')]) %>%
      dplyr::mutate(H_b = H_g - H_a,
                    Eh_b = Eh_g - Eh_a,
                    S_b = S_g - S_a) %>%
      dplyr::select(!!!grpSyms,
                    H_a, Eh_a, S_a, # Alpha
                    H_b, Eh_b, S_b, # Beta
                    H_g, Eh_g, S_g, # Gamma
                    AREA_TOTAL,
                    H_a_VAR, Eh_a_VAR, S_a_VAR, AREA_TOTAL_VAR,
                    H_a_SE, Eh_a_SE, S_a_SE, AREA_TOTAL_SE,
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
