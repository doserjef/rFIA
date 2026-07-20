areaStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                        returnSpatial = FALSE, byLandType = FALSE, 
                        landType = 'forest', method = 'TI', lambda = 0.5, 
                        treeDomain = NULL, areaDomain = NULL, totals = FALSE, 
                        byPlot = FALSE, condList = FALSE, nCores = 1, 
                        remote, mr) {

  # Read required data and prep the database ------------------------------
  reqTables <- c('PLOT', 'TREE', 'COND', 'POP_PLOT_STRATUM_ASSGN',
                 'POP_ESTN_UNIT', 'POP_EVAL', 'POP_STRATUM', 'POP_EVAL_TYP',
                 'POP_EVAL_GRP', 'PLOTGEOM')

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
  if (landType %in% c('timber', 'forest', 'water', 'non-forest', 'all') == FALSE) {
    stop('landType must be one of: "forest", "timber", "non-forest", "water" or "all".')
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

  # Convert grpBy to character
  grpBy <- grpByToChar(db, grpBy_quo)

  # Unique plot ID through time (pltID)
  if (byPlot | condList) {
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

  # Build a domain indicator for each observation (1 or 0) ----------------
  # Land type
  db$COND$landD <- landTypeDomain(landType, db$COND$COND_STATUS_CD,
                                  db$COND$SITECLCD, db$COND$RESERVCD)
  # The 'all' branch of landTypeDomain() sets landD = 1 unconditionally, which
  # incorrectly counts nonsampled conditions (COND_STATUS_CD == 5, e.g.
  # hazardous or denied-access plots) as land area. Excluding them here (rather
  # than inside the shared landTypeDomain() utility) keeps the fix scoped to
  # area(), since areaChange() also calls landTypeDomain() and has not yet
  # been validated for landType = 'all'.
  if (tolower(landType) == 'all') {
    db$COND$landD[db$COND$COND_STATUS_CD == 5] <- 0
  }

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

  # User defined domain indicator for trees (ex. trees > 20 ft tall)
  db <- udTreeDomain(db, treeDomain)

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

  # Handle canned groups --------------------------------------------------
  # Make a new column that describes the land type and hold in COND
  if (byLandType) {
    grpBy <- c(grpBy, 'landType')
    db$COND <- db$COND %>%
      dplyr::mutate(landType = dplyr::case_when(
        COND_STATUS_CD == 1 & SITECLCD %in% c(1:6) & RESERVCD == 0 ~ 'Timber',
        COND_STATUS_CD == 1 ~ 'Non-Timber Forest',
        COND_STATUS_CD == 2 ~ 'Non-Forest',
        COND_STATUS_CD == 3 | COND_STATUS_CD == 4 ~ 'Water'),
        landD = 1) # Reset the land basis to all
    db$COND <- db$COND[!is.na(db$COND$landType),]
  }

  # Prep the database before heading to estimators ------------------------
  # Narrow the tables to the necessary variables.
  # Which grpByNames are in which table? Helps us subset below.
  # grpBy names in PLOT
  grpP <- names(db$PLOT)[names(db$PLOT) %in% grpBy]
  # grpBy names in COND
  grpC <- names(db$COND)[names(db$COND) %in% grpBy &
                         !c(names(db$COND) %in% grpP)]
  # grpBy names in TREE
  grpT <- names(db$TREE)[names(db$TREE) %in% grpBy &
                         !c(names(db$TREE) %in% c(grpP, grpC))]

  # PLOT ------------------------------
  db$PLOT <- db$PLOT %>%
    # Dropping irrelevant rows and columns
    dplyr::select(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA, INVYR, MEASYEAR,
                  PLOT_STATUS_CD, dplyr::all_of(grpP), sp, COUNTYCD) %>%
    # Drop plots outside of area of interest. Note this doesn't filter on
    # PLOT_STATUS_CD == 1, since area() supports landType values other than
    # 'forest' (e.g. 'water', 'non-forest', 'all'), and land-type restriction
    # is already handled via landD/aD at the COND level below.
    dplyr::filter(sp == 1) %>%
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

  # TREE ------------------------------
  db$TREE <- db$TREE %>%
    dplyr::select(PLT_CN, CONDID, DIA, SPCD, TPA_UNADJ, SUBP, TREE,
        dplyr::all_of(grpT), tD) %>%
  # Need to keep all trees for now, otherwise areas aren't right.
    # dplyr::filter(!is.na(DIA) & TPA_UNADJ > 0 & tD == 1) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% db$PLOT$PLT_CN)

  # Was treeDomain NULL? If so, replace NAs with 1
  treeD <- ifelse(mean(db$TREE$tD, na.rm = TRUE) == 1, 1, 0)

  # Summarize variables of interest to plot level -------------------------
  grpSyms <- dplyr::syms(grpBy)

  # Only joining tables necessary to produce plot level estimates,
  # adjusted for non-response
  data <- db$PLOT %>%
    dplyr::left_join(db$COND, by = c('PLT_CN')) %>%
    dplyr::left_join(db$TREE, by = c('PLT_CN', 'CONDID')) %>%
    dtplyr::lazy_dt() %>%
    dplyr::mutate(tD = tidyr::replace_na(tD, treeD)) %>%
    dplyr::group_by(PLT_CN, PROP_BASIS, CONDID, !!!grpSyms) %>%
    dplyr::mutate(tD = ifelse(sum(tD, na.rm = TRUE) > 0, 1, 0)) %>%
    dplyr::ungroup() %>%
    as.data.frame()


  # Comprehensive indicator function
  data$aDI <- data$landD * data$aD * data$sp * data$tD
  data$pDI <- data$landD * data$aD

  if (byPlot & !condList) {

    grpBy <- c('YEAR', grpBy, 'PLOT_STATUS_CD')
    grpSyms <- dplyr::syms(grpBy)

    tEst <- data %>%
      dplyr::mutate(YEAR = INVYR) %>%
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(!!!grpSyms, PLT_CN, CONDID) %>%
      dplyr::summarize(CONDPROP_UNADJ = data.table::first(CONDPROP_UNADJ),
                       aDI = data.table::first(aDI)) %>%
      dplyr::group_by(!!!grpSyms, PLT_CN) %>%
      dplyr::summarize(PROP_FOREST = sum(CONDPROP_UNADJ * aDI, na.rm = TRUE)) %>%
      as.data.frame()

    # Make it spatial
    if (returnSpatial){
      tEst <- tEst %>%
        dplyr::filter(!is.na(LAT) & !is.na(LON)) %>%
        sf::st_as_sf(coords = c('LON', 'LAT'),
                     crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
      grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]

    }

    out <- list(tEst = tEst, aEst = NULL, grpBy = grpBy)


  } else {

    grpSyms <- syms(grpBy)

    # Plot-level estimates
    t <- data %>%
      # Will be lots of trees here, so CONDPROP listed multiple times
      # Adding PROP_BASIS so we can handle adjustment factors at stratum level
      dplyr::distinct(PLT_CN, CONDID, !!!grpSyms, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      dplyr::mutate(fa = CONDPROP_UNADJ * aDI) %>%
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!grpSyms, fa, aDI) %>%
      as.data.frame()

    # Total land area in areaDomain and landType, for proportions
    a <- data %>%
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      # Plots whose conditions were all dropped by the land type/areaDomain
      # filter upstream (db$COND) survive this left_join as a CONDID = NA row.
      # They correctly contribute 0 area (via na.rm = TRUE downstream), but
      # left unfiltered here their PLT_CN would still be counted in
      # nPlots_AREA_DEN.
      dplyr::filter(!is.na(CONDID)) %>%
      dtplyr::lazy_dt() %>%
      dplyr::mutate(fad = CONDPROP_UNADJ * pDI) %>%
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, fad) %>%
      as.data.frame()

    # Filter to only contain plots with treeDomain = 1 so sample size
    # calculations for numerator are correct.
    t <- t %>%
      dplyr::filter(aDI == 1)

    # Return a tree/condition list ready to be handed to `customPSE`
    if (condList) {

      tEst <- t %>%
        dplyr::mutate(EVAL_TYP = 'CURR') %>%
        dplyr::select(PLT_CN, EVAL_TYP, AREA_BASIS, !!!grpSyms, CONDID,
                      PROP_FOREST = fa)

      out <- list(tEst = tEst, aEst = NULL, grpBy = grpBy)

    # Otherwise, proceed to population estimation
    } else {

      # Sum variable(s) up to plot-level and adjust for non-response
      tPlt <- sumToPlot(t, pops, grpBy)
      aPlt <- sumToPlot(a, pops, NULL)

      # Adding YEAR to groups
      grpBy <- c('YEAR', grpBy)
      aGrpBy <- c('YEAR')


      # Sum variable(s) up to strata then estimation unit level
      eu.sums <- sumToEU(db, tPlt, aPlt, pops, grpBy, aGrpBy, method, lambda)
      tEst <- eu.sums$x
      aEst <- eu.sums$y

      out <- list(tEst = tEst, aEst = aEst, grpBy = grpBy)
    }
  }
  return(out)
}
