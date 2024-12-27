invasiveStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                            returnSpatial = FALSE, landType = 'forest', 
                            method = 'TI', lambda = 0.5, areaDomain = NULL, 
                            byPlot = FALSE, totals = FALSE, nCores = 1, 
                            remote = NULL, mr) {

  # Read required data and prep the database ------------------------------
  reqTables <- c('PLOT', 'INVASIVE_SUBPLOT_SPP', 'SUBP_COND', 'COND', 'POP_PLOT_STRATUM_ASSGN', 
                 'POP_ESTN_UNIT', 'POP_EVAL', 'POP_STRATUM', 'POP_EVAL_TYP', 
                 'POP_EVAL_GRP')

  # If remote, read in state by state. Otherwise, drop all unnecessary tables.
  db <- readRemoteHelper(x, db, remote, reqTables, nCores)

  # If the object was clipped
  if ('prev' %in% names(db$PLOT)) {
    # Filter to only include the most recent plot measurements 
    db$PLOT <- dplyr::filter(db$PLOT, prev == 0)
  }

  # Handle TX issues: we only keep inventory years that are present in 
  # BOTH EAST AND WEST TX
  db <- handleTX(db)

  # Check some of the inputs ----------------------------------------------
  # polys -----------------------------
  if (!is.null(polys) & 
      dplyr::first(class(polys)) %in% 
      c('sf', 'SpatialPolygons', 'SpatialPolygonsDataFame') == FALSE) {
    stop("polys must be a spatial polygons object of class sp or sf.")
  }
  # landType --------------------------
  if (landType %in% c('timber', 'forest') == FALSE) {
    stop('landType must be one of: "forest" or "timber".')
  }
  # db required tables ----------------
  if (any(reqTables %in% names(db) == FALSE)) {
    missT <- reqTables[reqTables %in% names(db) == FALSE]
    stop(paste('Tables', paste(as.character(missT), collapse = ', '),
               'not found in object db.'))
  }
  # method ----------------------------
  if (stringr::str_to_upper(method) %in% c('TI', 'SMA', 'LMA', 'EMA', 
                                           'ANNUAL') == FALSE) {
    warning(paste('Method', method, 'unknown. Defaulting to Temporally Indifferent (TI).'))
  }

  # Other basic variable prep ---------------------------------------------
  # Reduce our samle right off the bat, only plots sampled for invasives
  # TODO: add to notes
  db$PLOT <- db$PLOT %>%
    dplyr::filter(INVASIVE_SAMPLING_STATUS_CD %in% 1:2)

  # Get a plot CN and a new pltID that gives a unique ID to each plot
  # PLT_CN is UNITCD, STATECD, COUNTYCD, PLOT, and INVYR
  db$PLOT <- db$PLOT %>% 
    dplyr::mutate(PLT_CN = CN, 
                  pltID = stringr::str_c(UNITCD, STATECD, COUNTYCD, PLOT, sep = '_'))

  # Convert grpBy to character
  grpBy <- grpByToChar(db, grpBy_quo)

  # Unique plot ID through time (pltID)
  if (byPlot) {
    grpBy <- c('pltID', grpBy)
  }

  # Intersect plots with polygons if polygons are given
  if (!is.null(polys)) {
    # Add shapefile names to grpBy
    # Determine the name of the sf column. This assumes the user has not 
    # manually changed the "sf_column" attribute. By default, this is the 
    # same as the sf column. This is needed, because there is no single
    # name for the sf column. It is usually geom or geometry, but that
    # is not always the case.
    sfCol <- attr(polys, 'sf_column')
    grpBy <- c(grpBy, names(polys)[names(polys) != sfCol])

    # Do the intersection
    db <- arealSumPrep2(db, grpBy, polys, nCores, remote)

    # If there's nothing there, skip the state
    if (is.null(db)) return('no plots in polys')
  }

  # Update grpBy if returning spatial points.
  if (byPlot & returnSpatial) {
    grpBy <- c(grpBy, 'LON', 'LAT')
  }

  # Add on species names
  grpBy <- c(grpBy, 'SYMBOL', 'SCIENTIFIC_NAME', 'COMMON_NAME')

  # Adding invasive names to INV
  suppressWarnings({
    db$INVASIVE_SUBPLOT_SPP <- db$INVASIVE_SUBPLOT_SPP %>%
      dplyr::left_join(intData$REF_PLANT_DICTIONARY, by = c('VEG_SPCD' = 'SYMBOL')) %>%
      dplyr::mutate(SYMBOL = VEG_SPCD) %>%
      dplyr::mutate_if(is.factor, as.character)
  })

  # Build a domain indicator for each observation (1 or 0) ----------------
  # Land type
  db$COND$landD <- landTypeDomain(landType, db$COND$COND_STATUS_CD, 
                                  db$COND$SITECLCD, db$COND$RESERVCD)

  # Spatial boundary (determine which of the plots fall within the polygons
  # supplied in polys)
  if (!is.null(polys)) {
    db$PLOT$sp <- ifelse(!is.na(db$PLOT$polyID), 1, 0)
  } else {
    # If no polys, use all plots.
    db$PLOT$sp <- 1
  }

  # User defined domain indicator for area (ex. specific forest type)
  db <- udAreaDomain(db, areaDomain)

  # Handle population tables ----------------------------------------------
  # Filtering out all inventories that are not relevant to the current
  # estimation type. If using estimator other than TI, handle the differences in
  # P2POINTCNT and in assigning YEAR column (YEAR = END_INVYR if method = 'TI')
  # TODO: need to go through handlePops, and also understand the differences in 
  #       the evalTypes
  pops <- handlePops(db, evalType = c('CURR'), method, mr, pltList = db$PLOT$PLT_CN)

  # A lot of states do their stratification in such a way that makes it impossible
  # to estimate variance of annual panels with the post-stratified estimator. That is,
  # the number of plots within a panel within a stratum is less than 2. When this happens
  # merge strata so that all have at least two observations.
  # TODO: need to go through mergeSmallStrata
  if (stringr::str_to_upper(method) %in% c('SMA', 'LMA', 'EMA', 'ANNUAL') & !byPlot) {
    pops <- mergeSmallStrata(db, pops)
  }

  # Prep the tree list ----------------------------------------------------
  # Narrow the tables to the necessary variables. 
  # Which grpByNames are in which table? Helps us subset below. 
  # grpBy names in PLOT
  grpP <- names(db$PLOT)[names(db$PLOT) %in% grpBy]
  # grpBy names in COND
  grpC <- names(db$COND)[names(db$COND) %in% grpBy & 
                         !c(names(db$COND) %in% grpP)]

  # PLOT ------------------------------
  db$PLOT <- db$PLOT %>%
    # Dropping irrelevant rows and columns
    dplyr::select(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA, INVYR, MEASYEAR,
                  PLOT_STATUS_CD, dplyr::all_of(grpP), sp, COUNTYCD) %>%
    # Drop non-forested plots, and those otherwise outside our domain of interest.
    # TODO: add PLOT_STATUS_CD to the iPad descriptions. A code that describes the
    #       sampling status of the plot. (1 = sampled, at least one accessible
    #       forest land condtion; 2 = sampled, no accessible forest land condition;
    #       3 = non-sampled).
    dplyr::filter(PLOT_STATUS_CD == 1 & sp == 1) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # COND ------------------------------
  db$COND <- db$COND %>%
    dplyr::select(PLT_CN, CONDPROP_UNADJ, PROP_BASIS, COND_STATUS_CD, CONDID,
                  dplyr::all_of(grpC), aD, landD) %>%
    # Drop non-forested plots, and those otherwise outside our domain of interest
    dplyr::filter(aD == 1 & landD == 1) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # SUBP_COND -------------------------
  db$SUBP_COND <- db$SUBP_COND %>%
    dplyr::select(PLT_CN, SUBP, CONDID, SUBPCOND_PROP) %>%
    # Drop visis nots used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # INVASIVE_SUBPLOT_SPP --------------
  db$INVASIVE_SUBPLOT_SPP <- db$INVASIVE_SUBPLOT_SPP %>%
    dplyr::select(PLT_CN, COVER_PCT, VEG_SPCD, SUBP, CONDID, 
                  SYMBOL, SCIENTIFIC_NAME, COMMON_NAME) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # Separate area grouping names from tree grouping names
  if (!is.null(polys)) {
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND), names(polys))]
  } else {
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND))]
  }

  # Full condition list
  data <- db$PLOT %>% 
    dplyr::left_join(db$COND, by = c('PLT_CN')) %>% 
    dplyr::left_join(db$SUBP_COND, by = c('PLT_CN', 'CONDID')) %>%
    dplyr::left_join(db$INVASIVE_SUBPLOT_SPP, by = c('PLT_CN', 'CONDID', 'SUBP'))

  # Comprehensive indicator function
  data$aDI <- data$landD * data$aD * data$sp

  # Plot-level summaries --------------------------------------------------
  if (byPlot) {
    grpBy <- c('YEAR', grpBy)
    # Create a list of symbols for the grpBy statements
    grpSyms <- rlang::syms(grpBy)
    aGrpSyms <- rlang::syms(aGrpBy)

    # Plot-level estimates
    # Area
    a <- data %>% 
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      # Convert to a lazy data table for quicker analysis.
      dtplyr::lazy_dt() %>% 
      # Note the use of !!! to inject aGrpSyms back into an evaluation context.
      dplyr::group_by(PLT_CN, !!!aGrpSyms) %>%
      # Calculate proportion of area in the plot that meets the current area domain 
      dplyr::summarize(PROP_FOREST = sum(CONDPROP_UNADJ * aDI, na.rm = TRUE)) %>%
      # Convert to data frame
      as.data.frame()

    # Areal invasive coverage
    t <- data %>% 
      # Set the YEAR to the measurement year for plot-level estimates. 
      dplyr::rename(YEAR = MEASYEAR) %>% 
      dplyr::filter(!is.na(SYMBOL)) %>%
      dplyr::distinct(PLT_CN, CONDID, SUBP, VEG_SPCD, .keep_all = TRUE) %>% 
      dtplyr::lazy_dt() %>% 
      dplyr::group_by(!!!grpSyms, PLT_CN, SUBP) %>%  
      dplyr::summarize(cover = sum(COVER_PCT / 100 * SUBPCOND_PROP * aDI, na.rm = TRUE)) %>% 
      dplyr::ungroup() %>%
      dplyr::group_by(PLT_CN, !!!grpSyms) %>%
      dplyr::summarize(PROP_INV_COVER = mean(cover, na.rm = TRUE)) %>%
      dplyr::ungroup() %>%
      as.data.frame() %>% 
      dplyr::left_join(a, by = c('PLT_CN', aGrpBy)) %>% 
      dplyr::distinct()

    # Make it spatial if the user wants it.
    if (returnSpatial) {
      t <- t %>% 
        dplyr::filter(!is.na(LAT) & !is.na(LON)) %>% 
        sf::st_as_sf(coords = c('LON', 'LAT'), 
                     crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
      grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]
    }

    out <- list(tEst = t, grpBy = grpBy, aGrpBy = aGrpBy)

  } else {
    # Population estimation -----------------------------------------------
    # Create a list of symbols for the grpBy statements
    aGrpSyms <- syms(aGrpBy)

    # Condition list
    a <- data %>% 
      # Will be lots of trees here, so CONDPROP is listed multiple times, the 
      # distinct is needed to just get those distinct ones. 
      # TODO: add PROP_BASIS to iPad notes.
      # Adding PROP_BASIS so we can handle adjustment factors at strata level.
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>% 
      # TODO: add CONDPROP_UNADJ to iPad notes
      dplyr::mutate(fa = CONDPROP_UNADJ * aDI) %>% 
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!aGrpSyms, fa)

    # Create list of symols for the grpBy statements
    grpSyms <- syms(grpBy)
    # Tree list
    t <- data %>% 
      dplyr::distinct(PLT_CN, CONDID, SUBP, VEG_SPCD, .keep_all = TRUE) %>% 
      dtplyr::lazy_dt() %>% 
      dplyr::group_by(!!!grpSyms, PLT_CN, PROP_BASIS, CONDID) %>%
      dplyr::summarize(cover = sum(COVER_PCT / 100 * SUBPCOND_PROP * aDI * CONDPROP_UNADJ, 
                                   na.rm = TRUE) / 4) %>%
      dplyr::ungroup() %>%
      dplyr::rename(AREA_BASIS = PROP_BASIS) %>%
      as.data.frame()

    # Sum variable(s) up to plot-level and adjust for non-response
    # TODO: need to go through sumToPlot
    tPlt <- sumToPlot(t, pops, grpBy)
    aPlt <- sumToPlot(a, pops, aGrpBy)

    # Add YEAR to groups
    grpBy <- c('YEAR', grpBy)
    aGrpBy <- c('YEAR', aGrpBy)

    # Sum variable(s) up to strata then estimation unit level
    # TODO: need to go through sumToEU
    eu.sums <- sumToEU(db, tPlt, aPlt, pops, grpBy, aGrpBy, method)
    tEst <- eu.sums$x
    aEst <- eu.sums$y
   
    out <- list(tEst = tEst, aEst = aEst, grpBy = grpBy, aGrpBy = aGrpBy) 

  }

  return(out)

}
