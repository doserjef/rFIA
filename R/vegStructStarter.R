vegStructStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                             returnSpatial = FALSE, landType = 'forest', 
                             method = 'TI', lambda = 0.5, areaDomain = NULL, 
                             byPlot = FALSE, totals = FALSE, nCores = 1,
                             remote = NULL, mr) {

  # Read required data and prep the database ------------------------------
  reqTables <- c('PLOT', 'P2VEG_SUBP_STRUCTURE', 'SUBP_COND', 'COND', 
                 'POP_PLOT_STRATUM_ASSGN', 'POP_ESTN_UNIT', 'POP_EVAL', 
                 'POP_STRATUM', 'POP_EVAL_TYP', 'POP_EVAL_GRP', 'PLOTGEOM')

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
  # Join PLOT with PLOTGEOM to allow plot-level geographic attributes to be used
  # in grpBy statements
  # First need to get rid of other columns in PLOT in PLOTGEOM. 
  db$PLOTGEOM <- db$PLOTGEOM %>%
    dplyr::select(-STATECD, -INVYR, -UNITCD, -COUNTYCD, -PLOT, -LAT, -LON, 
                  -dplyr::starts_with('CREATED'), -dplyr::starts_with('MODIFIED'))
  db$PLOT <- db$PLOT %>%
    dplyr::left_join(db$PLOTGEOM, by = 'CN')

  # Get a plot CN and a new pltID that gives a unique ID to each plot
  # PLT_CN is UNITCD, STATECD, COUNTYCD, PLOT, and INVYR
  db$PLOT <- db$PLOT %>% 
    dplyr::mutate(PLT_CN = CN, 
                  pltID = stringr::str_c(UNITCD, STATECD, COUNTYCD, PLOT, sep = '_'))

  # Reduce the sample right off the bat to only use plots sampled for veg
  # 0 are plots that are not part of P2 veg sample.
  db$PLOT <- db$PLOT %>%
    dplyr::filter(P2VEG_SAMPLING_STATUS_CD %in% 1:2)

  # Convert grpBy to character
  grpBy <- grpByToChar(db[!c(names(db) %in% c('TREE'))], grpBy_quo)

  # Unique plot ID through time (pltID)
  if (byPlot) {
    grpBy <- c('pltID', grpBy)
  }

  # Add variables to grouping
  grpBy <- c(grpBy, 'LAYER', 'GROWTH_HABIT')

  # Adding names of id columns for layer and growth habit
  db$P2VEG_SUBP_STRUCTURE <- db$P2VEG_SUBP_STRUCTURE %>%
    dplyr::mutate(LAYER = dplyr::case_when(is.na(LAYER) ~ NA_character_,
                                           LAYER == 1 ~ '0 to 2.0 feet',
                                           LAYER == 2 ~ '2.1 to 6.0 feet',
                                           LAYER == 3 ~ '6.1 to 16.0 feet',
                                           LAYER == 4 ~ 'Greater than 16 feet',
                                           LAYER == 5 ~ 'Aerial: all layers'),
                  # Full P2VEG_SUBP_STRUCTURE.GROWTH_HABIT_CD domain per the
                  # FIADB User Guide (Database Description, ch. 4.3.10),
                  # including the region-specific codes DS (Interior West,
                  # RSCD 22: dead pinyon-species shrubs) and the PNWRS-only
                  # AL/MO/SL/SS/ST codes (RSCD 26/27). Any code missing here
                  # would silently drop that record's real cover data, since
                  # GROWTH_HABIT is part of grpBy and the final output step
                  # drops NA groups (see vegStruct.md).
                  GROWTH_HABIT = dplyr::case_when(is.na(GROWTH_HABIT_CD) ~ NA_character_,
                                                  GROWTH_HABIT_CD == 'TT' ~ 'Tally tree',
                                                  GROWTH_HABIT_CD == 'NT' ~ 'Non-tally tree',
                                                  GROWTH_HABIT_CD == 'SH' ~ 'Shrubs/vines',
                                                  GROWTH_HABIT_CD == 'FB' ~ 'Forbs',
                                                  GROWTH_HABIT_CD == 'GR' ~ 'Graminoids',
                                                  GROWTH_HABIT_CD == 'DS' ~ 'Dead pinyon species shrubs',
                                                  GROWTH_HABIT_CD == 'AL' ~ 'All vegetation',
                                                  GROWTH_HABIT_CD == 'MO' ~ 'Moss/bryophytes',
                                                  GROWTH_HABIT_CD == 'SL' ~ 'Bare soil',
                                                  GROWTH_HABIT_CD == 'SS' ~ 'Newly sprouted shrub cover',
                                                  GROWTH_HABIT_CD == 'ST' ~ 'Seedlings'))

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
  pops <- handlePops(db, evalType = c('CURR'), method, mr)

  # A lot of states do their stratification in such a way that makes it impossible
  # to estimate variance of annual panels with the post-stratified estimator. That is,
  # the number of plots within a panel within a stratum is less than 2. When this happens
  # merge strata so that all have at least two observations.
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
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # P2VEG_SUBP_STRUCTURE --------------
  db$P2VEG_SUBP_STRUCTURE <- db$P2VEG_SUBP_STRUCTURE %>%
    dplyr::select(PLT_CN, COVER_PCT, SUBP, CONDID, LAYER, GROWTH_HABIT) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # Separate area grouping names from tree grouping names
  if (!is.null(polys)){
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND), names(polys))]
  } else {
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND))]
  }

  # Full condition list
  data <- db$PLOT %>% 
    dplyr::left_join(db$COND, by = c('PLT_CN')) %>% 
    left_join(db$SUBP_COND, by = c('PLT_CN', 'CONDID')) %>%
    left_join(db$P2VEG_SUBP_STRUCTURE, by = c("PLT_CN", "CONDID", 'SUBP'))

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

    # Areal coverage 
    t <- data %>%
      # Set the YEAR to the measurement year for plot-level estimates.
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dplyr::distinct(PLT_CN, CONDID, SUBP, LAYER, GROWTH_HABIT, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(!!!grpSyms, PLT_CN, SUBP) %>%
      dplyr::summarize(cover = sum(COVER_PCT/100 * SUBPCOND_PROP * aDI, na.rm = TRUE)) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(PLT_CN, !!!grpSyms) %>%
      # A LAYER/GROWTH_HABIT combination not recorded on every subplot (the
      # normal case for patchy vegetation) only has rows here for the
      # subplots it *was* recorded on -- mean(cover, na.rm = TRUE) would
      # treat the other subplots as missing to exclude from the average
      # rather than real 0-cover observations to include, inflating
      # PROP_COVER by up to 4x (same bug already fixed in invasive()'s
      # byPlot branch; see vegStruct.md). Divide by a fixed 4 instead,
      # matching the population-estimate branch's own formula below.
      dplyr::summarize(PROP_COVER = sum(cover, na.rm = TRUE) / 4) %>%
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

    out <- list(tEst = t, aEst = NULL, grpBy = grpBy, aGrpBy = aGrpBy)

  } else {
    # Population estimation -----------------------------------------------
    # Create a list of symbols for the grpBy statements
    grpSyms <- rlang::syms(grpBy)
    aGrpSyms <- rlang::syms(aGrpBy)

    # Condition list
    a <- data %>%
      # Will be lots of trees here, so CONDPROP is listed multiple times, the
      # distinct is needed to just get those distinct ones.
      # Adding PROP_BASIS so we can handle adjustment factors at strata level.
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      # Plots whose conditions were all dropped by the land type/areaDomain
      # filter upstream (db$COND) survive this left_join as a CONDID = NA
      # row (same phantom-row pattern as tpa()/area()/biomass()/seedling()/
      # standStruct()/diversity()/etc). Drop them so nPlots_AREA doesn't
      # count them as contributing plots.
      dplyr::filter(!is.na(CONDID)) %>%
      dplyr::mutate(fa = CONDPROP_UNADJ * aDI) %>%
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!aGrpSyms, fa)

    # Tree list
    t <- data %>%
      dplyr::distinct(PLT_CN, CONDID, SUBP, LAYER, GROWTH_HABIT, .keep_all = TRUE) %>%
      # A plot with no conditions passing the land type/areaDomain filter
      # would otherwise still survive as a CONDID = NA row with a real-
      # looking cover = 0 (from sum(NA, na.rm = TRUE) / 4), rather than a
      # genuinely empty result (same class of bug fixed in
      # standStruct()/diversity(); see vegStruct.md).
      dplyr::filter(!is.na(CONDID)) %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(!!!grpSyms, PLT_CN, PROP_BASIS, CONDID, CONDPROP_UNADJ) %>%
      dplyr::summarize(cover = sum(COVER_PCT/100 * SUBPCOND_PROP * aDI * CONDPROP_UNADJ, 
                                   na.rm = TRUE) / 4) %>%
      dplyr::ungroup() %>%
      dplyr::rename(AREA_BASIS = PROP_BASIS) %>%
      as.data.frame()

    # Sum variable(s) up to plot-level and adjust for non-response
    tPlt <- sumToPlot(t, pops, grpBy)
    aPlt <- sumToPlot(a, pops, aGrpBy)

    # Add YEAR to groups
    grpBy <- c('YEAR', grpBy)
    aGrpBy <- c('YEAR', aGrpBy)

    # Sum variable(s) up to strata then estimation unit level
    eu.sums <- sumToEU(db, tPlt, aPlt, pops, grpBy, aGrpBy, method, lambda)
    tEst <- eu.sums$x
    aEst <- eu.sums$y

    out <- list(tEst = tEst, aEst = aEst, grpBy = grpBy, aGrpBy = aGrpBy)
  } # Population or plot-level estimation
  return(out)
}
