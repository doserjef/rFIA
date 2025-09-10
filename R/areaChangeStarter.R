areaChangeStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                              returnSpatial = FALSE, byLandType = FALSE, 
                              landType = 'forest', 
                              method = 'TI', lambda = 0.5, treeDomain = NULL, 
                              areaDomain = NULL, byPlot = FALSE, 
                              condList = FALSE, chngType = 'net', nCores = 1, 
                              remote, mr) {

  # Read required data, prep the database -------------------------------------
  reqTables <- c('PLOT', 'TREE', 'COND', 'SUBP_COND_CHNG_MTRX',
                 'POP_PLOT_STRATUM_ASSGN',
                 'POP_ESTN_UNIT', 'POP_EVAL',
                 'POP_STRATUM', 'POP_EVAL_TYP', 'POP_EVAL_GRP', 'PLOTGEOM')

  # If remote, read in state by state. Otherwise, drop all unnecessary tables.
  db <- readRemoteHelper(x, db, remote, reqTables, nCores)


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
  # treeType --------------------------
  if (chngType %in% c('net', 'component') == FALSE) {
    stop('chngType must be one of: "net", "component".')
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
  pops <- handlePops(db, evalType = c('CHNG'), method, mr)

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

  # Only the necessary plots for EVAL of interest
  # We want the plots in the inventory and their previous measurements
  keepThese <- distinct(pops, PLT_CN) %>% 
    left_join(select(db$PLOT, PLT_CN, PREV_PLT_CN), by = 'PLT_CN')

  # PLOT ------------------------------
  db$PLOT <- db$PLOT %>%
    dplyr::select(c(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA,
                    INVYR, MEASYEAR, PLOT_STATUS_CD,
                    dplyr::all_of(grpP), sp, COUNTYCD,
                    PREV_PLT_CN, REMPER)) %>%
    # Drop plots outside of area of interest. Note this differs from other
    # functions, where there is also a filter on PLOT_STATUS_CD == 1, but 
    # here we want plots that were forested and non-forested for 
    # diversion and reversion.
    dplyr::filter(sp == 1) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% !!c(keepThese$PLT_CN, keepThese$PREV_PLT_CN))

  # COND ------------------------------
  db$COND <- db$COND %>%
    dplyr::select(c(PLT_CN, CONDPROP_UNADJ, PROP_BASIS,
                    COND_STATUS_CD, CONDID,
                    dplyr::all_of(grpC), aD, landD)) %>%
    # Note the below is commented out because need to keep all conditions 
    # since conditions can change from one time point to the next.
    # Drop non-forested plots, and those otherwise outside our domain of interest
    # dplyr::filter(aD == 1 & landD == 1) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))

  # TREE ------------------------------
  db$TREE <- db$TREE %>%
    dplyr::select(PLT_CN, CONDID, DIA, SPCD, TPA_UNADJ, SUBP, TREE,
                  dplyr::all_of(grpT), tD) %>%
    # Hold onto all trees for now, otherwise things get messed up later on. 
    # dplyr::filter(!is.na(DIA) & TPA_UNADJ > 0 & tD == 1) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))

  # Was treeDomain NULL? If so, replace NAs with 1
  treeD <- ifelse(mean(db$TREE$tD, na.rm = TRUE) == 1, 1, 0)

  # SUBP_COND_CHNG_MTRX ---------------
  db$SUBP_COND_CHNG_MTRX <- dplyr::select(db$SUBP_COND_CHNG_MTRX,
                                          c(PLT_CN, SUBP, CONDID, PREVCOND,
                                            SUBPTYP, SUBPTYP_PROP_CHNG)) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))

  # Determine current and previous groups
  grp1 <- if (length(grpBy) > 0 ) { paste0(grpBy, '1') } else { character(0) }
  grp2 <- if (length(grpBy) > 0 ) { paste0(grpBy, '2') } else { character(0) }
  grp1Syms <- dplyr::syms(grp1)
  grp2Syms <- dplyr::syms(grp2)


  # Was treeDomain NULL? If so, replace NAs w/ 1 below
  # This will be used later, to prevent drops of non-treed forestland in
  # component change
  db$TREE <- db$TREE %>%
    dtplyr::lazy_dt() %>%
    dplyr::group_by(PLT_CN, CONDID) %>%
    dplyr::summarize(tD = sum(tD)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(tD = dplyr::case_when(tD > 0 ~ 1,
                                        TRUE ~ 0)) %>%
    as.data.frame()

  # Full condition list ---------------------------------------------------
  data <- db$PLOT %>%
    dtplyr::lazy_dt() %>%
    dplyr::filter(PLT_CN %in% keepThese$PLT_CN) %>%
    dplyr::left_join(db$COND, by = c('PLT_CN')) %>%
    dplyr::left_join(db$TREE, by = c('PLT_CN', 'CONDID')) %>%
    dplyr::filter(!is.na(PROP_BASIS) & !is.na(CONDPROP_UNADJ)) %>%
    dplyr::left_join(db$SUBP_COND_CHNG_MTRX, by = c('PLT_CN', 'CONDID')) %>%
    dplyr::left_join(dplyr::select(db$PLOT, PLT_CN, sp, dplyr::all_of(grpP)), 
                     by = c('PREV_PLT_CN' = 'PLT_CN'), suffix = c('2', '1')) %>%
    dplyr::left_join(dplyr::select(db$COND, PLT_CN, landD, dplyr::all_of(grpC), 
                     aD, CONDPROP_UNADJ, CONDID), 
                     by = c('PREV_PLT_CN' = 'PLT_CN', 'PREVCOND' = 'CONDID'), 
                     suffix = c('2', '1')) %>%
    dplyr::left_join(db$TREE, by = c('PREV_PLT_CN' = 'PLT_CN', 'PREVCOND' = 'CONDID'), 
                     suffix = c('2', '1')) %>%
    dplyr::rename(CONDID1 = PREVCOND,
                  CONDID2 = CONDID) %>%
    # Don't want to drop non-treed forestland
    # TODO: this might lead to some problems when including any filters
    dplyr::mutate(tD1 = tidyr::replace_na(tD1, treeD),
                  tD2 = tidyr::replace_na(tD2, treeD)) %>%
    # Drop all microplot proportions
    dplyr::filter(SUBPTYP != 2) %>%
    # Check if macroplot is available
    dplyr::group_by(PLT_CN) %>%
    dplyr::mutate(macro = ifelse(3 %in% unique(SUBPTYP), 1, 0)) %>%
    dplyr::ungroup() %>%
    # If so, drop the subplot entries
    dplyr::mutate(dropThese = dplyr::case_when(macro == 1 & SUBPTYP == 1 ~ 0,
                                               TRUE ~ 1)) %>%
    dplyr::filter(dropThese == 1) %>%
    dplyr::select(-c(dropThese)) %>%
    as.data.frame()

  # Set up growth accounting for net change -------------------------------
  if (chngType == 'net') {
    data <- data %>%
      # Clean up domain indicator and drop unnecessary cols
      dplyr::mutate(tDI1 = landD1 * aD1 * sp1 * tD1,
                    tDI2 = landD2 * aD2 * sp2 * tD2) %>%
      dplyr::select(PLT_CN, REMPER, SUBP, MEASYEAR, PLOT_STATUS_CD, PROP_BASIS,
                    tDI1, tDI2, dplyr::all_of(grp1), dplyr::all_of(grp2),
                    CONDID1, CONDID2,
                    SUBPTYP_PROP_CHNG) %>%
      tidyr::pivot_longer(cols = tDI1:CONDID2,
                   names_to = c(".value", 'ONEORTWO'),
                   names_sep = -1) %>%
      # Adjust for subplot, negate previous values, and apply domain indicator
      dplyr::mutate(PREV_CONDPROP = dplyr::case_when(ONEORTWO == 1 ~ SUBPTYP_PROP_CHNG * tDI / 4,
                                                     TRUE ~ 0),
                    CONDPROP_CHNG = dplyr::case_when(
                      ONEORTWO == 1 ~ -SUBPTYP_PROP_CHNG * tDI / 4 / REMPER,
                      TRUE ~ SUBPTYP_PROP_CHNG * tDI / 4 / REMPER))


    # Change by component
  } else {
    data <- data %>%
      dtplyr::lazy_dt() %>%
      dplyr::mutate(STATUS1 = dplyr::case_when(
        landD1 & stringr::str_to_lower(landType) == 'forest' ~ 'Forest',
        landD1 & stringr::str_to_lower(landType) == 'timber' ~ 'Timber',
        !landD1 & stringr::str_to_lower(landType) == 'forest' ~ 'Non-forest',
        !landD1 & stringr::str_to_lower(landType) == 'timber' ~ 'Non-timber')) %>%
      dplyr::mutate(STATUS2 = dplyr::case_when(
        landD2 & stringr::str_to_lower(landType) == 'forest' ~ 'Forest',
        landD2 & stringr::str_to_lower(landType) == 'timber' ~ 'Timber',
        !landD2 & stringr::str_to_lower(landType) == 'forest' ~ 'Non-forest',
        !landD2 & stringr::str_to_lower(landType) == 'timber' ~ 'Non-timber')) %>%
      # Clean up domain indicator and drop unnecessary cols
      dplyr::mutate(TREE_DOMAIN1 = tD1,
                    TREE_DOMAIN2 = tD2,
                    AREA_DOMAIN1 = landD1 * aD1 * sp1,
                    AREA_DOMAIN2 = landD2 * aD2 * sp2) %>%
      dplyr::select(PLT_CN, REMPER, SUBP, MEASYEAR, PLOT_STATUS_CD, PROP_BASIS,
                    STATUS1, STATUS2,
                    TREE_DOMAIN1, TREE_DOMAIN2, AREA_DOMAIN1, AREA_DOMAIN2,
                    dplyr::all_of(grp1), dplyr::all_of(grp2),
                    CONDID1, CONDID2,
                    SUBPTYP_PROP_CHNG) %>%
      # Did the groups change? If so, track them and drop the rest
      dplyr::mutate(chng = dplyr::case_when(
        paste(TREE_DOMAIN1, AREA_DOMAIN1, !!!grp1Syms) != 
        paste(TREE_DOMAIN2, AREA_DOMAIN2, !!!grp2Syms) ~ 1,
        TRUE ~ 0)) %>%
      # Adjust for subplot, negate previous values, and apply domain indicator
      dplyr::mutate(tDI1 = TREE_DOMAIN1 * AREA_DOMAIN1,
                    tDI2 = TREE_DOMAIN2 * AREA_DOMAIN2,
                    tDI = ifelse(tDI1 + tDI2 > 0, 1, 0), # for nPlots
                    PREV_CONDPROP = SUBPTYP_PROP_CHNG * tDI / 4,
                    CONDPROP_CHNG = dplyr::case_when(chng == 1 ~ SUBPTYP_PROP_CHNG / 4 / REMPER,
                                                     TRUE ~ 0)) %>%
      # Need this to trick sumToPlots
      dplyr::mutate(CONDID = stringr::str_c(CONDID1, CONDID2)) %>%
      as.data.frame()

    # Update grpBy
    grpBy = sort(c(c('STATUS1', grp1), c('STATUS2', grp2)))

    if (!rlang::quo_is_null(treeDomain)) grpBy <- c('TREE_DOMAIN1', 'TREE_DOMAIN2', grpBy)
    if (!rlang::quo_is_null(areaDomain)) grpBy <- c('AREA_DOMAIN1', 'AREA_DOMAIN2', grpBy)

  }

 # Plot-level summaries ------------------------------------------------------
  if (byPlot & !condList) {

    grpBy <- c('YEAR', grpBy)
    grpSyms <- dplyr::syms(grpBy)

    t <- data %>%
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(!!!grpSyms, PLT_CN, REMPER) %>%
      dplyr::summarize(PROP_CHNG = sum(CONDPROP_CHNG, na.rm = TRUE),
                       PREV_PROP_FOREST = sum(PREV_CONDPROP, na.rm = TRUE)) %>%
      as.data.frame()

    # Make it spatial
    if (returnSpatial) {
      t <- t %>%
        dplyr::filter(!is.na(LAT) & !is.na(LON)) %>%
        sf::st_as_sf(coords = c('LON', 'LAT'),
                     crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
      grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]
    }

    out <- list(tEst = t, aEst = NULL, grpBy = grpBy)

  } else {

    grpSyms <- dplyr::syms(grpBy)
    t <- data %>%
      dtplyr::lazy_dt() %>%
      dplyr::mutate(ac = CONDPROP_CHNG,
                    prev = PREV_CONDPROP) %>%
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!grpSyms, ac, prev, tDI) %>%
      as.data.frame()

    # Filter to only contain plots with treeDomain == 1 so sample size
    # calculations for numerator are correct, and so the actual values 
    # themselves are correct
    t <- t %>% 
      dplyr::filter(tDI == 1) %>% 
      select(-tDI)

    if (condList) {

      tEst <- t %>%
        dtplyr::lazy_dt() %>%
        dplyr::mutate(EVAL_TYP = 'CHNG') %>%
        dplyr::group_by(PLT_CN, EVAL_TYP, AREA_BASIS, !!!grpSyms, CONDID) %>%
        dplyr::summarize(PROP_CHNG = sum(ac, na.rm = TRUE),
                         PREV_PROP_FOREST = sum(prev, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        as.data.frame()

      out <- list(tEst = tEst, aEst = NULL, grpBy = grpBy)

    } else {
      # Sum variable(s) up to plot-level and adjust for non-response
      tPlt <- sumToPlot(dplyr::select(t, -c(prev)), pops, grpBy)
      aPlt <- sumToPlot(dplyr::select(t, -c(ac)), pops, grpBy)

      # Adding YEAR to groups
      grpBy <- c('YEAR', grpBy)

      # Sum variable(s) up to strata then estimation unit level
      eu.sums <- sumToEU(db, tPlt, aPlt, pops, grpBy, grpBy, method, lambda)
      tEst <- eu.sums$x
      aEst <- eu.sums$y

      out <- list(tEst = tEst, aEst = aEst, grpBy = grpBy)
    }
  }
  return(out)
}
