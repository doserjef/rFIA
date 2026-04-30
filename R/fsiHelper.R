fsiHelper1 <- function(x, plts, db, grpBy, scaleBy, byPlot) {
  
  # Add scaleBy to grpBy, Does not modify outside environment, just need 
  # scaleBy in here as well
  if (is.null(grpBy)){
    aGrps <- NULL
    grpBy <- scaleBy
  } else {
    aGrps <- grpBy
    grpBy <- unique(c(grpBy, scaleBy))
  }
  
  # Selecting the plots for one county
  db$PLOT <- plts[[x]]
  
  # Disturbance or treatment ever happen on the plot remeasurement? If so
  # we want to cut it before we model the max size-density curve
  disturb <- select(db$PLOT, PLT_CN, pltID) %>%
    dplyr::left_join(select(db$COND, PLT_CN, DSTRBCD1, TRTCD1), by = 'PLT_CN') %>%
    dplyr::mutate(DSTRBCD1 = tidyr::replace_na(DSTRBCD1, 0),
                  TRTCD1 = tidyr::replace_na(TRTCD1, 0)) %>%
    dplyr::filter(DSTRBCD1 > 0) %>%
    # Natural regen is ok
    dplyr::filter(TRTCD1 > 0 & !c(TRTCD1 %in% 40)) %>%
    # These plots were disturbed or treated
    dplyr::distinct(PLT_CN, pltID)
  
  
  # Which grpByNames are in which table? Helps us subset below. 
  # grpBy names in PLOT
  grpP <- names(db$PLOT)[names(db$PLOT) %in% c(grpBy, scaleBy)]
  # grpBy names in COND
  grpC <- names(db$COND)[names(db$COND) %in% c(grpBy, scaleBy) & 
                           !c(names(db$COND) %in% grpP)]
  # grpBy names in TREE                         
  grpT <- names(db$TREE)[names(db$TREE) %in% c(grpBy, scaleBy) & 
                           !c(names(db$TREE) %in% c(grpP, grpC))]
  
  # Makes a unique tree ID.  
  db$TREE$treID <- paste(db$TREE$SUBP, db$TREE$TREE, sep = '_')
  
  # Join tables needed to produce plot-level estimates 
  data <- db$PLOT %>%
    filter(DESIGNCD %in% c(1, 501:505) & PLOT_STATUS_CD == 1 & !is.na(REMPER) & !is.na(PREV_PLT_CN)) %>%
    dplyr::left_join(db$COND, by = c('PLT_CN')) %>%
    dplyr::left_join(db$TREE, by = c('PLT_CN', 'CONDID')) %>%
    dplyr::left_join(select(db$PLOT, c(PLT_CN, sp, DESIGNCD, PLOT_STATUS_CD)), 
                     by = c('PREV_PLT_CN' = 'PLT_CN'), suffix = c('2', '1')) %>%
    dplyr::left_join(select(db$COND, c(PLT_CN, CONDID, landD, aD, COND_STATUS_CD)), 
                     by = c('PREV_PLT_CN' = 'PLT_CN', 'PREVCOND' = 'CONDID'), suffix = c('2', '1')) %>%
    dplyr::left_join(select(db$TREE, c(TRE_CN, all_of(grpT), treID, typeD, tD, TPA_UNADJ, 
                                       BAA, DIA, STATUSCD, SPCD)), 
                     by = c('PREV_TRE_CN' = 'TRE_CN'), suffix = c('2', '1')) %>%
    filter(DESIGNCD1 %in% c(1, 501:505) & PLOT_STATUS_CD1 == 1) %>%
    dplyr::mutate_if(is.factor, as.character)
  
  # Comprehensive tree-level indicator function -- w/ growth accounting
  # landD2 comes from landType
  # aD2 comes from areaDomain
  # tD2 comes from treeDomain
  # typeD2 comes from treeType
  # sp is the spatial filter. 
  data$tDI2 <- data$landD2 * data$aD2 * data$tD2 * data$typeD2 * data$sp2 *
    dplyr::if_else(data$STATUSCD2 == 1, 1, 0)
  
  data$tDI1 <- data$landD1 * data$aD1 * data$tD1 * data$typeD1 * data$sp1 *
    dplyr::if_else(data$STATUSCD1 == 1, 1, 0)
  
  # Comprehensive indicator function -- w/ growth accounting
  data$pDI2 <- data$landD2 * data$aD2 * data$typeD2 * data$sp2 *
    dplyr::if_else(data$STATUSCD2 == 1, 1, 0)
  
  data$pDI1 <- data$landD1 * data$aD1 * data$typeD1 * data$sp1 *
    dplyr::if_else(data$STATUSCD1 == 1, 1, 0)
  
  # Save a copy for area calculations
  aData <- db$PLOT %>%
    dplyr::select(PLT_CN, PREV_PLT_CN, pltID, DESIGNCD, REMPER, STATECD, 
                  MACRO_BREAKPOINT_DIA, INVYR, MEASYEAR, MEASMON, MEASDAY, PLOT_STATUS_CD, 
                  all_of(grpP), sp) %>%
    dplyr::filter(DESIGNCD %in% c(1, 501:505) & PLOT_STATUS_CD == 1) %>%
    dplyr::left_join(select(db$COND, PLT_CN, CONDPROP_UNADJ, PROP_BASIS, COND_STATUS_CD, 
                            CONDID, grpC, aD, landD), by = c('PLT_CN')) %>%
    dplyr::mutate(aDI = landD * aD * sp)
  
  # Calculate mortality and survival based on previous and current attributes
  data <- data %>%
    dplyr::mutate(MORT = dplyr::case_when(STATUSCD1 == 1 & STATUSCD2 == 2 ~ 1,
                                          STATUSCD1 == 1 & STATUSCD2 == 3 ~ 1,
                                          TRUE ~ 0),
                  SURV = dplyr::case_when(STATUSCD1 == 1 & STATUSCD2 == 1 ~ 1, 
                                          TRUE ~ 0)
    )
  
  # Subset the data to only keep what we need for the current calculations. 
  data <- data %>%
    dplyr::select(PLT_CN, PREV_PLT_CN, pltID, TRE_CN, SUBP, CONDID, TREE, CONDPROP_UNADJ,
                  MEASYEAR, MACRO_BREAKPOINT_DIA, PROP_BASIS, grpP[grpP != 'PLOT_STATUS_CD'], grpC,
                  REMPER, PLOT_STATUS_CD1, PLOT_STATUS_CD2, treID1, treID2,
                  one_of(str_c(grpT, 1),str_c(grpT, 2)),
                  tDI1, tDI2, pDI1, pDI2, STATUSCD1, STATUSCD2,
                  DIA1, DIA2, BAA1, BAA2, TPA_UNADJ1, TPA_UNADJ2, SPCD1, SPCD2) %>%
    # Set tree-level t1 BAA and TPA to negative values. 
    dplyr::mutate(BAA1 = -(BAA1),
                  TPA_UNADJ1 = -(TPA_UNADJ1)) %>%
    # Rearrange previous values as observations
    tidyr::pivot_longer(cols = -c(PLT_CN:REMPER),
                        names_to = c(".value", 'ONEORTWO'),
                        names_sep = -1) %>%
    dplyr::mutate(PLOT_BASIS = dplyr::case_when(
      # When DIA is na, adjustment is NA
      is.na(DIA) ~ NA_character_,
      # When DIA is less than 5", use microplot value
      DIA < 5 ~ 'MICR',
      # When DIA is greater than 5", use subplot value
      DIA >= 5 & is.na(MACRO_BREAKPOINT_DIA) ~ 'SUBP',
      DIA >= 5 & DIA < MACRO_BREAKPOINT_DIA ~ 'SUBP',
      DIA >= MACRO_BREAKPOINT_DIA ~ 'MACR'))
  
  # Set NAs in TPA_UNADJ and BAA to 0.   
  data <- data %>%
    dplyr::mutate(TPA_UNADJ = tidyr::replace_na(TPA_UNADJ, replace = 0),
                  BAA = tidyr::replace_na(BAA, replace = 0))
  
  # For use in group_by below 
  scaleSyms <- rlang::syms(scaleBy)
  
  # Extract plot-level information for use with the size-density scaling calculations. 
  # Note that this is "plot-level" information, in the sense that it is each plot
  # remeasurement. 
  t1 <- data %>%
    # No disturbance/treatment plots. 
    dplyr::filter(!c(pltID %in% disturb$pltID)) %>%
    dplyr::distinct(PLT_CN, SUBP, TREE, ONEORTWO, .keep_all = TRUE) %>% 
    dtplyr::lazy_dt() %>% 
    dplyr::filter(STATUSCD == 1) %>% 
    # Filter to only grab the plots within the domain of interest
    dplyr::filter(pDI == 1) %>%  
    dplyr::group_by(!!!scaleSyms, PLT_CN) %>% 
    dplyr::summarize(REMPER = dplyr::first(REMPER), 
                     # Total basal area for the trees at each PLT_CN. 
                     BAA1 = sum(-BAA[ONEORTWO == 1], na.rm = TRUE), 
                     TPA1 = sum(-TPA_UNADJ[ONEORTWO == 1], na.rm = TRUE), 
                     BAA2 = sum(BAA[ONEORTWO == 2], na.rm = TRUE), 
                     TPA2 = sum(TPA_UNADJ[ONEORTWO == 2], na.rm = TRUE), 
                     skew1 = skewness(rep(DIA[ONEORTWO == 1], round(-TPA_UNADJ[ONEORTWO == 1]))), 
                     skew2 = skewness(rep(DIA[ONEORTWO == 2], round(TPA_UNADJ[ONEORTWO == 2]))) 
    ) %>% 
    # Calculate average basal area per tree for t1 and t2
    dplyr::mutate(BA1 = dplyr::if_else(TPA1 != 0, BAA1 / TPA1, 0), 
                  BA2 = dplyr::if_else(TPA2 != 0, BAA2 / TPA2, 0)) %>% 
    # Remove plots with high skewness as those plots should not inform the calculation 
    # of the maximum density curve. 
    dplyr::filter(skew2 >= -1 & skew2 <= 1) %>% 
    as.data.frame()
  
  if (byPlot) {
    grpBy <- c('YEAR', grpBy)
    grpSyms <- rlang::syms(grpBy)
    # t are the tree-level data that are subsequently used for relative density 
    # and FSI calculations. 
    t <- data %>%
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dplyr::distinct(PLT_CN, SUBP, TREE, ONEORTWO, .keep_all = TRUE) %>%
      dplyr::select(all_of(grpBy), PLT_CN, PREV_PLT_CN, PLOT_STATUS_CD, REMPER, SUBP, TREE,
                    ONEORTWO, tDI, TPA_UNADJ, BAA)
    
    a <- NULL
    
  } else {
    
    if (length(aGrps[aGrps %in% names(aData)]) < 1) {
      aGrps <- NULL
    }  else {
      aGrps <- aGrps[aGrps %in% names(aData)]
    }
    
    aSyms <- syms(aGrps)
    
    a <- aData %>%
      # date column
      dplyr::mutate(date = paste(MEASYEAR, MEASMON, MEASDAY, sep = '-'),
                    date = as.Date(date, "%Y-%m-%d")) %>%
      # Adding PROP_BASIS so we can handle adjustment factors at strata level
      dplyr::distinct(PLT_CN, pltID, date, PROP_BASIS, CONDID, !!!aSyms, .keep_all = TRUE) %>%
      dplyr::group_by(PLT_CN, pltID, date, PROP_BASIS, CONDID, !!!aSyms) %>%
      dplyr::summarize(CONDPROP_UNADJ = dplyr::first(CONDPROP_UNADJ * aDI)) %>%
      dplyr::mutate(CONDPROP_UNADJ = tidyr::replace_na(CONDPROP_UNADJ, 0)) %>%
      # Average forested area between min and max date across all remeasurements.
      dplyr::group_by(pltID, PROP_BASIS, .dots = aGrps) %>%
      dplyr::summarize(minDate = min(date, na.rm = TRUE),
                       maxDate = max(date, na.rm = TRUE),
                       amin = sum(CONDPROP_UNADJ[date == minDate], na.rm = TRUE),
                       amax = sum(CONDPROP_UNADJ[date == maxDate], na.rm = TRUE),
                       fa = (amin + amax) / 2) %>%
      dplyr::left_join(dplyr::select(dplyr::ungroup(db$PLOT), PLT_CN, pltID), by = c('pltID'))
    
    # t is the tree level data needed for calculation of relative density and 
    # subsequently FSI. 
    t <- data %>%
      dplyr::distinct(PLT_CN, TRE_CN, ONEORTWO, .keep_all = TRUE) %>%
      dplyr::select(tidyselect::all_of(grpBy), PLT_CN, pltID, PLOT_STATUS_CD, 
                    PLOT_BASIS, MEASYEAR, REMPER, TRE_CN,
                    ONEORTWO, STATUSCD, tDI, TPA_UNADJ, BAA)
  } 
  
  pltOut <- list(t = t, a = a, t1 = t1)
  return(pltOut)
} 

# Calculates population-level summaries given the population information and a 
# set of tree-level data that contains the RD for each plot in each year. 
fsiHelper2 <- function(x, popState, t, a, grpBy, scaleBy, method, useSeries) {
  # Note that this does not modify the outside environment. 
  if (str_to_upper(method) %in% c("SMA", 'EMA', 'LMA', 'ANNUAL')) {
    grpBy <- c(grpBy, 'INVYR')
    popState[[x]]$P2POINTCNT <- popState[[x]]$P2POINTCNT_INVYR
    popState[[x]]$P1POINTCNT <- popState[[x]]$P1POINTCNT_INVYR
    popState[[x]]$P2PNTCNT_EU <- popState[[x]]$P2PNTCNT_EU_INVYR
  }
  
  # Tree estimates + CV  --------------------------------------------------
  aAdj <- a %>%
    # Rejoin with population tables
    dplyr::inner_join(dplyr::select(popState[[x]], -c(STATECD)), by = c('PLT_CN', 'pltID')) %>%
    dplyr::mutate(
      # Determine the area adjustment factor based on the size of the plot.  
      aAdj = dplyr::case_when(
        # When NA, stay NA
        is.na(PROP_BASIS) ~ NA_real_,
        # If the proportion was measured for a macroplot, use the macroplot value
        PROP_BASIS == 'MACR' ~ as.numeric(ADJ_FACTOR_MACR),
        # Otherwise, use the subpplot value
        PROP_BASIS == 'SUBP' ~ ADJ_FACTOR_SUBP),
      fa = fa * aAdj) %>%
    dplyr::ungroup()
  
  # Sometimes less specific area groups
  aGrps <- unique(grpBy[grpBy %in% names(aAdj)])
  
  grpSyms <- rlang::syms(unique(c(grpBy[!c(grpBy %in% c('YEAR', 'INVYR'))], scaleBy)))
  
  # First compute plot-level values of FSI. Note the correspondence in what is 
  # shown here with what is used in fsi() for plot-level calculations.  
  tEst <- t %>%
    dtplyr::lazy_dt() %>%
    dplyr::ungroup() %>%
    # Converting to average tree size (where tree size is tree BA)
    dplyr::mutate(BA = dplyr::if_else(ONEORTWO == 1, -BAA / TPA_UNADJ, BAA / TPA_UNADJ)) %>%
    # Summing within scaleBy. This is necessary in the case where a single forest 
    # plot spans multiple of the categories included in the scaleBy. 
    dplyr::group_by(!!!grpSyms, PLT_CN, pltID, MEASYEAR, REMPER, PLOT_BASIS) %>%
    dplyr::summarize(PREV_RD = -sum(rd[ONEORTWO == 1] * tDI[ONEORTWO == 1], na.rm = TRUE),
                     CURR_RD = sum(rd[ONEORTWO == 2] * tDI[ONEORTWO == 2], na.rm = TRUE),
                     PREV_TPA = -sum(TPA_UNADJ[ONEORTWO == 1] * tDI[ONEORTWO == 1], na.rm = TRUE),
                     PREV_BA = -sum(BA[ONEORTWO == 1] * tDI[ONEORTWO == 1], na.rm = TRUE),
                     CHNG_TPA = sum(TPA_UNADJ * tDI, na.rm = TRUE) / dplyr::first(REMPER),
                     CHNG_BA = sum(BA * tDI, na.rm = TRUE) / dplyr::first(REMPER),
                     # Determine if at least one tree is in the tree domain of interest, 
                     # which thus qualifies the plot for use in the estimation. 
                     plotIn_t = dplyr::if_else(sum(tDI, na.rm = TRUE) > 0, 1, 0)) %>%
    # Summing across scaleBy
    dplyr::group_by(PLT_CN, pltID, MEASYEAR, PLOT_BASIS, REMPER, !!!grpSyms) %>%
    dplyr::summarize(CHNG_TPA = sum(CHNG_TPA, na.rm = TRUE),
                     CHNG_BA = sum(CHNG_BA, na.rm = TRUE),
                     PREV_TPA = sum(PREV_TPA, na.rm = TRUE),
                     PREV_BA = sum(PREV_BA, na.rm = TRUE),
                     PREV_RD = mean(PREV_RD, na.rm = TRUE),
                     CURR_RD = mean(CURR_RD, na.rm = TRUE),
                     plotIn_t = ifelse(sum(plotIn_t, na.rm = TRUE) >  0, 1,0)) %>%
    dplyr::mutate(FSI = (CURR_RD - PREV_RD) / REMPER) %>%
    dplyr::ungroup() %>%
    as.data.frame()
  
  # Now need to handle the case for when we want to use multiple remeasurements
  # to inform an estimate of change. 
  if (useSeries) {
    # Get a unique ID for each remeasurement in the series
    nMeas <- t %>%
      dplyr::distinct(pltID, PLT_CN, MEASYEAR, REMPER) %>%
      dplyr::group_by(pltID) %>%
      dplyr::mutate(n = length(unique(PLT_CN)),
                    series = dplyr::min_rank(MEASYEAR)) %>%
      dplyr::ungroup() %>%
      dplyr::select(pltID, PLT_CN, REMPER, n, series)
    
    # Only if more than one remeasurement available
    if (any(nMeas$n > 1)) {
      
      # Now we loop over the unique values of n
      # Basically have to chunk up the data each time
      # in order to get intermediate estimates
      nRems <- unique(nMeas$n)
      remsList <- list()
      for (i in 1:length(nRems)) {
        # Temporal weights for each plot
        wgts <- nMeas %>%
          dplyr::filter(series <= nRems[i] & n >= nRems[i]) %>%
          dplyr::group_by(pltID) %>%
          # Total remeasurement interval and weights for
          # individual remeasurements
          dplyr::mutate(fullRemp = sum(REMPER, na.rm = TRUE),
                        wgt = REMPER / fullRemp) %>%
          dplyr::ungroup() %>%
          dplyr::select(PLT_CN, n, series, wgt, fullRemp)
        
        dat <- tEst %>%
          dplyr::left_join(wgts, by = c('PLT_CN')) %>%
          dplyr::filter(series <= nRems[i] & n >= nRems[i]) %>%
          dtplyr::lazy_dt() %>%
          dplyr::group_by(pltID, PLOT_BASIS, fullRemp, !!!grpSyms) %>%
          dplyr::summarize(FSI = sum(FSI * wgt, na.rm = TRUE),
                           CHNG_TPA = sum(CHNG_TPA*wgt, na.rm = TRUE),
                           CHNG_BA = sum(CHNG_BA*wgt, na.rm = TRUE),
                           PLT_CN = PLT_CN[which.max(series)],
                           CURR_RD = CURR_RD[which.max(series)],
                           PREV_RD = PREV_RD[which.min(series)],
                           PREV_TPA = PREV_TPA[which.min(series)],
                           PREV_BA = PREV_BA[which.min(series)],
                           plotIn_t = if_else(any(plotIn_t > 0), 1, 0)) %>%
          dplyr::ungroup() %>%
          dplyr::select(-c(pltID)) %>%
          dplyr::rename(REMPER = fullRemp) %>%
          dplyr::ungroup() %>%
          as.data.frame()
        remsList[[i]] <- dat
      }
      # Bring it all back together
      dat <- dplyr::bind_rows(remsList)
      
      # Update columns in tEst
      tEst <- tEst %>%
        dplyr::select(-c(CHNG_TPA:FSI, REMPER)) %>%
        dplyr::left_join(dat, by = c('PLT_CN', 'PLOT_BASIS', grpBy[!c(grpBy %in% c('YEAR', 'INVYR'))]))
    }
  }
  
  grpSyms <- rlang::syms(grpBy)
  
  # Now summarize the plot-level data at the strata level and onward. 
  tEst <- tEst %>%
    dtplyr::lazy_dt() %>%
    # Rejoin with population tables
    dplyr::inner_join(dplyr::select(ungroup(popState[[x]]), -c(STATECD)), by = 'PLT_CN') %>%
    dplyr::ungroup() %>%
    # Add adjustment factors
    dplyr::mutate(tAdj = dplyr::case_when(is.na(PLOT_BASIS) ~ NA_real_,
                                          # If the proportion was measured for a 
                                          # macroplot, use the macroplot value
                                          PLOT_BASIS == 'MACR' ~ as.numeric(ADJ_FACTOR_MACR),
                                          # Otherwise, use the subpplot value
                                          PLOT_BASIS == 'SUBP' ~ as.numeric(ADJ_FACTOR_SUBP),
                                          PLOT_BASIS == 'MICR' ~ as.numeric(ADJ_FACTOR_MICR)),
                  CHNG_TPA = CHNG_TPA * tAdj,
                  CHNG_BA = CHNG_BA * tAdj,
                  CURR_RD = CURR_RD * tAdj,
                  PREV_TPA = PREV_TPA * tAdj,
                  PREV_BA = PREV_BA * tAdj,
                  PREV_RD = PREV_RD * tAdj,
                  REMPER = REMPER * tAdj,
                  FSI = FSI * tAdj) %>%
    # Extra step for variance issues - summing micro, subp, and macr components
    dplyr::group_by(ESTN_UNIT_CN, ESTN_METHOD, STRATUM_CN, PLT_CN, REMPER, !!!grpSyms) %>%
    dplyr::summarize(CHNG_TPA = sum(CHNG_TPA, na.rm = TRUE),
                     CHNG_BA = sum(CHNG_BA, na.rm = TRUE),
                     PREV_TPA = sum(PREV_TPA, na.rm = TRUE),
                     PREV_BA = sum(PREV_BA, na.rm = TRUE),
                     PREV_RD = sum(PREV_RD, na.rm = TRUE),
                     CURR_RD = sum(CURR_RD, na.rm = TRUE),
                     FSI = sum(FSI, na.rm = TRUE),
                     plotIn_t = ifelse(sum(plotIn_t, na.rm = TRUE) >  0, 1,0),
                     nh = dplyr::first(P2POINTCNT),
                     P2PNTCNT_EU = dplyr::first(P2PNTCNT_EU),
                     a = dplyr::first(AREA_USED),
                     w = dplyr::first(STRATUM_WGT)) %>%
    # Add on area
    dplyr::left_join(dplyr::select(aAdj, ESTN_UNIT_CN, STRATUM_CN, PLT_CN, aGrps, fa),
                     by = c('ESTN_UNIT_CN', 'STRATUM_CN', 'PLT_CN', aGrps)) %>%
    # FSI is area adjusted
    dplyr::mutate(si = FSI * fa,
                  ra1 = PREV_RD * fa,
                  ra2 = CURR_RD * fa) %>%
    # Replace NAN w/ zeros
    dplyr::mutate(si = tidyr::replace_na(si, 0),
                  ra1 = tidyr::replace_na(ra1, 0),
                  ra2 = tidyr::replace_na(ra2, 0)) %>%
    dplyr::ungroup() %>%
    # Strata-level
    dplyr::group_by(ESTN_UNIT_CN, ESTN_METHOD, STRATUM_CN, !!!grpSyms) %>%
    dplyr::summarize(nh = dplyr::first(nh),
                     a = dplyr::first(a),
                     w = dplyr::first(w),
                     P2PNTCNT_EU = dplyr::first(P2PNTCNT_EU),
                     ctStrat = sum(CHNG_TPA, na.rm = TRUE),
                     cbStrat = sum(CHNG_BA, na.rm = TRUE),
                     ptStrat = sum(PREV_TPA, na.rm = TRUE),
                     pbStrat = sum(PREV_BA, na.rm = TRUE),
                     siStrat = sum(si, na.rm = TRUE),
                     ra1Strat = sum(ra1, na.rm = TRUE),
                     ra2Strat = sum(ra2, na.rm = TRUE),
                     faStrat = sum(fa, na.rm = TRUE),
                     rempStrat = sum(REMPER, na.rm = TRUE),
                     plotIn_t = sum(plotIn_t, na.rm = TRUE),
                     
                     # ## Strata level variances
                     ctv = sum(CHNG_TPA^2, na.rm = TRUE),
                     cbv = sum(CHNG_BA^2, na.rm = TRUE),
                     ptv = sum(PREV_TPA^2, na.rm = TRUE),
                     pbv = sum(PREV_BA^2, na.rm = TRUE),
                     siv = sum(si^2, na.rm = TRUE),
                     ra1v = sum(ra1^2, na.rm = TRUE),
                     ra2v = sum(ra2^2, na.rm = TRUE),
                     fav = sum(fa^2, na.rm = TRUE),
                     rempv = sum(REMPER^2, na.rm = TRUE),
                     
                     # Strata level covariances
                     cvStrat_ct = sum(CHNG_TPA*PREV_TPA, na.rm = TRUE),
                     cvStrat_cb = sum(CHNG_BA*PREV_BA, na.rm = TRUE),
                     cvStrat_si = sum(si*fa, na.rm = TRUE),
                     cvStrat_ra1 = sum(ra1*fa, na.rm = TRUE),
                     cvStrat_ra2 = sum(ra2*fa, na.rm = TRUE),
                     cvStrat_psi = sum(si*ra1, na.rm = TRUE),
                     cvStrat_remp = sum(REMPER * fa, na.rm = TRUE)) %>%
    dplyr::mutate(ctStrat = ctStrat / nh,
                  cbStrat = cbStrat / nh,
                  ptStrat = ptStrat / nh,
                  pbStrat = pbStrat / nh,
                  siStrat = siStrat / nh,
                  ra1Strat = ra1Strat / nh,
                  ra2Strat = ra2Strat / nh,
                  faStrat = faStrat / nh,
                  rempStrat = rempStrat / nh,
                  adj = nh * (nh-1),
                  ctv = (ctv - (nh*ctStrat^2)) / adj,
                  cbv = (cbv - (nh*cbStrat^2)) / adj,
                  ptv = (ptv - (nh*ptStrat^2)) / adj,
                  pbv = (pbv - (nh*pbStrat^2)) / adj,
                  siv = (siv - (nh*siStrat^2)) / adj,
                  ra1v = (ra1v - (nh*ra1Strat^2)) / adj,
                  ra2v = (ra2v - (nh*ra2Strat^2)) / adj,
                  fav = (fav - (nh*faStrat^2)) / adj,
                  rempv = (rempv - (nh*rempStrat^2)) / adj,
                  
                  cvStrat_ct = (cvStrat_ct - (nh * ctStrat * ptStrat)) / adj,
                  cvStrat_cb = (cvStrat_cb - (nh * cbStrat * pbStrat)) / adj,
                  cvStrat_si = (cvStrat_si - (nh * siStrat * faStrat)) / adj,
                  cvStrat_ra1 = (cvStrat_ra1 - (nh * ra1Strat * faStrat)) / adj,
                  cvStrat_ra2 = (cvStrat_ra2 - (nh * ra2Strat * faStrat)) / adj,
                  cvStrat_psi = (cvStrat_psi - (nh * siStrat * ra1Strat)) / adj,
                  cvStrat_remp = (cvStrat_remp - (nh * rempStrat * faStrat)) / adj) %>%
    as.data.frame() %>%
    
    # Estimation unit
    dplyr::group_by(ESTN_UNIT_CN, .dots = grpBy) %>%
    dplyr::summarize(ctEst = unitMean(ESTN_METHOD, a, nh, w, ctStrat),
                     cbEst = unitMean(ESTN_METHOD, a, nh, w, cbStrat),
                     ptEst = unitMean(ESTN_METHOD, a, nh, w, ptStrat),
                     pbEst = unitMean(ESTN_METHOD, a, nh, w, pbStrat),
                     siEst = unitMean(ESTN_METHOD, a, nh, w, siStrat),
                     ra1Est = unitMean(ESTN_METHOD, a, nh, w, ra1Strat),
                     ra2Est = unitMean(ESTN_METHOD, a, nh, w, ra2Strat),
                     faEst = unitMean(ESTN_METHOD, a, nh, w, faStrat),
                     rempEst = unitMean(ESTN_METHOD, a, nh, w, rempStrat),
                     
                     P2PNTCNT_EU = dplyr::first(P2PNTCNT_EU),
                     nh = dplyr::first(nh),
                     # Estimation of unit variance
                     ctVar = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, ctv, ctStrat, ctEst),
                     cbVar = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, cbv, cbStrat, cbEst),
                     ptVar = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, ptv, ptStrat, ptEst),
                     pbVar = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, pbv, pbStrat, pbEst),
                     siVar = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, siv, siStrat, siEst),
                     ra1Var = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, ra1v, ra1Strat, ra1Est),
                     ra2Var = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, ra2v, ra2Strat, ra2Est),
                     faVar = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, fav, faStrat, faEst),
                     rempVar = unitVarNew(method = 'var', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, rempv, rempStrat, rempEst),
                     
                     # Covariances
                     cvEst_ct = unitVarNew(method = 'cov', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, cvStrat_ct, ctStrat, ctEst, ptStrat, ptEst),
                     cvEst_cb = unitVarNew(method = 'cov', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, cvStrat_cb, cbStrat, cbEst, pbStrat, pbEst),
                     cvEst_si = unitVarNew(method = 'cov', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, cvStrat_si, siStrat, siEst, faStrat, faEst),
                     cvEst_ra1 = unitVarNew(method = 'cov', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, cvStrat_ra1, ra1Strat, ra1Est, faStrat, faEst),
                     cvEst_ra2 = unitVarNew(method = 'cov', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, cvStrat_ra2, ra2Strat, ra2Est, faStrat, faEst),
                     cvEst_psi = unitVarNew(method = 'cov', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, cvStrat_psi, siStrat, siEst, ra1Strat, ra1Est),
                     cvEst_remp = unitVarNew(method = 'cov', ESTN_METHOD, a, nh, dplyr::first(P2PNTCNT_EU), w, cvStrat_remp, rempStrat, rempEst, faStrat, faEst),
                     
                     plotIn_t = sum(plotIn_t, na.rm = TRUE)) %>%
    dplyr::ungroup()
  
  out <- list(tEst = tEst, aEst = NULL)
  
  return(out)
  
}  
