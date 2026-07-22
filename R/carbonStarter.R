carbonStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                          returnSpatial = FALSE, byPool = TRUE, 
                          byComponent = FALSE, landType = 'forest', 
                          method = 'TI', lambda = 0.5, areaDomain = NULL, 
                          totals = FALSE, byPlot = FALSE, condList = FALSE, 
                          nCores = 1, remote, mr) {

  # Read required data and prep the database ------------------------------
  reqTables <- c('PLOT', 'TREE', 'COND', 'POP_PLOT_STRATUM_ASSGN',
                 'POP_ESTN_UNIT', 'POP_EVAL',
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
  if (landType %in% c('timber', 'forest', 'all') == FALSE) {
    stop('landType must be one of: "forest", "timber", or "all".')
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
  grpBy <- grpByToChar(db[names(db) != 'TREE'], grpBy_quo)

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

  # Handle population tables ----------------------------------------------
  # Filtering out all inventories that are not relevant to the current 
  # estimation type. If using estimator other than TI, handle the differences in 
  # P2POINTCNT and in assigning YEAR column (YEAR = END_INVYR if method = 'TI')
  pops <- handlePops(db, evalType = c('VOL'), method, mr)

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
                  dplyr::all_of(grpC), aD, landD, 
                  CARBON_DOWN_DEAD, CARBON_LITTER, CARBON_SOIL_ORG, 
                  CARBON_UNDERSTORY_AG, CARBON_UNDERSTORY_BG) %>% 
    # Drop non-forested plots, and those otherwise outside our domain of interest
    dplyr::filter(aD == 1 & landD == 1) %>% 
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # TREE ------------------------------
  db$TREE <- db$TREE %>%
    dplyr::select(PLT_CN, CONDID, DIA, SPCD, TPA_UNADJ, SUBP, TREE, STATUSCD, 
                  CARBON_AG, CARBON_BG) %>%
    # Drop plots outside our domain of interest
    dplyr::filter(!is.na(DIA) & TPA_UNADJ > 0) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% db$PLOT$PLT_CN)

  # Full tree list
  data <- db$PLOT %>% 
    dplyr::left_join(db$COND, by = c('PLT_CN')) %>% 
    dplyr::left_join(db$TREE, by = c('PLT_CN', 'CONDID')) %>%
    dplyr::mutate(live = dplyr::case_when(STATUSCD == 1 ~ 1, 
                                          is.na(DIA) ~ NA_real_, 
                                          TRUE ~ 0), 
                  dead = dplyr::case_when(STATUSCD == 2 ~ 1, 
                                          is.na(DIA) ~ NA_real_,
                                          TRUE ~ 0))

  # Comprehensive indicator function
  data$aDI <- data$landD * data$aD * data$sp

  # Plot-level summaries --------------------------------------------------
  if (byPlot & !condList) {
    grpBy <- c('YEAR', grpBy)
    # Create a list of symbols for the grpBy statements
    grpSyms <- rlang::syms(grpBy)

    # Plot-level estimates
    # Area
    a <- data %>%
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      # Convert to a lazy data table for quicker analysis.
      dtplyr::lazy_dt() %>%
      # Note the use of !!! to inject aGrpSyms back into an evaluation context.
      dplyr::group_by(PLT_CN, !!!grpSyms) %>%
      dplyr::summarize(AG_UNDER_LIVE = sum(CONDPROP_UNADJ * CARBON_UNDERSTORY_AG * aDI, na.rm = TRUE),
                       BG_UNDER_LIVE = sum(CONDPROP_UNADJ * CARBON_UNDERSTORY_BG * aDI, na.rm = TRUE),
                       DOWN_DEAD = sum(CONDPROP_UNADJ * CARBON_DOWN_DEAD * aDI, na.rm = TRUE), 
                       LITTER = sum(CONDPROP_UNADJ * CARBON_LITTER * aDI, na.rm = TRUE), 
                       SOIL_ORG = sum(CONDPROP_UNADJ * CARBON_SOIL_ORG * aDI, na.rm = TRUE), 
                       PROP_FOREST = sum(CONDPROP_UNADJ * aDI, na.rm = TRUE)) %>%
      # Convert to data frame
      as.data.frame()

    # Tree-level
    t <- data %>%
      # Set the YEAR to the measurement year for plot-level estimates.
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dplyr::distinct(PLT_CN, SUBP, TREE, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(!!!grpSyms, PLT_CN) %>%
      dplyr::summarize(AG_OVER_LIVE = sum(CARBON_AG * TPA_UNADJ * live * aDI, na.rm = TRUE) / 2000,
                       BG_OVER_LIVE = sum(CARBON_BG * TPA_UNADJ * live * aDI, na.rm = TRUE) / 2000,
                       AG_OVER_DEAD = sum(CARBON_AG * TPA_UNADJ * dead * aDI, na.rm = TRUE) / 2000,
                       BG_OVER_DEAD = sum(CARBON_BG * TPA_UNADJ * dead * aDI, na.rm = TRUE) / 2000) %>%
      dplyr::mutate(STAND_DEAD = AG_OVER_DEAD + BG_OVER_DEAD) %>%
      dplyr::select(-c(AG_OVER_DEAD, BG_OVER_DEAD)) %>%
      as.data.frame()

    # Join back together
    t <- a %>%
      dplyr::left_join(t, by = c('PLT_CN', grpBy))

    # Convert to long format, where rows are ecosystem components
    t <- t %>%
      tidyr::pivot_longer(cols = -c(PLT_CN, !!!grpSyms, PROP_FOREST), 
                          names_to = 'COMPONENT', 
                          values_to = 'CARB_ACRE') %>%
      dtplyr::lazy_dt() %>%
      dplyr::mutate(POOL = dplyr::case_when(COMPONENT %in% c('AG_UNDER_LIVE', 'AG_OVER_LIVE') ~ 'AG_LIVE',
                                            COMPONENT %in% c('BG_UNDER_LIVE', 'BG_OVER_LIVE') ~ 'BG_LIVE',
                                            COMPONENT %in% c('STAND_DEAD', 'DOWN_DEAD') ~ 'DEAD_WOOD',
                                            TRUE ~ COMPONENT)) %>%
      # Convert to metric tonnes per acre
      dplyr::mutate(CARB_ACRE = CARB_ACRE * 0.90718474) %>%
      as.data.frame()

      # Add either component or pool to grpBy, depending on user input
      if (byComponent) {
        grpBy <- c(grpBy, 'COMPONENT')
      }
      if (byPool) {
        grpBy <- c(grpBy, 'POOL')
      }
      grpSyms <- rlang::syms(grpBy)

      # Summarize across COMPONENTS, if necessary
      if (!byComponent) {
        t <- t %>%
          dtplyr::lazy_dt() %>%
          dplyr::group_by(PLT_CN, !!!grpSyms, PROP_FOREST) %>%
          dplyr::summarise(CARB_ACRE = sum(CARB_ACRE, na.rm = TRUE)) %>%
          dplyr::ungroup() %>%
          dplyr::relocate(PROP_FOREST, .after = CARB_ACRE) %>%
          as.data.frame()
      } else {
        t <- t %>%
          dplyr::select(PLT_CN, !!!grpSyms, CARB_ACRE, PROP_FOREST)
      }

    # Make it spatial if the user wants it.
    if (returnSpatial) {
      t <- t %>%
        dplyr::filter(!is.na(LAT) & !is.na(LON)) %>%
        sf::st_as_sf(coords = c('LON', 'LAT'),
                     crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
      grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]
    }

    out <- list(tEst = t, grpBy = grpBy, aGrpBy = NULL)

  } else {
    # Population estimation or prep for it (condList) ---------------------
    # Create a list of symbols for the grpBy statements
    grpSyms <- dplyr::syms(grpBy)

    # Condition list
    a <- data %>%
      dplyr::mutate(YEAR = MEASYEAR) %>%
      # Will be lots of trees here, so CONDPROP is listed multiple times, the
      # distinct is needed to just get those distinct ones.
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      # Plots whose conditions were all dropped by the land type/areaDomain
      # filter upstream (db$COND) survive this left_join as a CONDID = NA row.
      # They correctly contribute 0 area (via na.rm = TRUE downstream), but
      # left unfiltered here their PLT_CN would still be counted in
      # nPlots_AREA. Drop them, mirroring the `!is.na(TREE_BASIS)` filter
      # used for the tree list below.
      dplyr::filter(!is.na(CONDID)) %>%
      dtplyr::lazy_dt() %>%
      dplyr::mutate(AG_UNDER_LIVE = CONDPROP_UNADJ * CARBON_UNDERSTORY_AG * aDI,
                    BG_UNDER_LIVE = CONDPROP_UNADJ * CARBON_UNDERSTORY_BG * aDI,
                    DOWN_DEAD = CONDPROP_UNADJ * CARBON_DOWN_DEAD * aDI,
                    LITTER = CONDPROP_UNADJ * CARBON_LITTER * aDI,
                    SOIL_ORG = CONDPROP_UNADJ * CARBON_SOIL_ORG * aDI,
                    PROP_FOREST = CONDPROP_UNADJ * aDI) %>%
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!grpSyms, 
                    AG_UNDER_LIVE:PROP_FOREST) %>%
      as.data.frame()

    # Tree list
    t <- data %>%
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dplyr::distinct(PLT_CN, SUBP, TREE, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      dplyr::mutate(AG_OVER_LIVE = CARBON_AG * TPA_UNADJ * live * aDI / 2000,
                    BG_OVER_LIVE = CARBON_BG * TPA_UNADJ * live * aDI / 2000,
                    AG_OVER_DEAD = CARBON_AG * TPA_UNADJ * dead * aDI / 2000,
                    BG_OVER_DEAD = CARBON_BG * TPA_UNADJ * dead * aDI / 2000) %>%
      # Need a code that tells us where the tree was measured
      # (macroplot, microplot, subplot)
      dplyr::mutate(
        TREE_BASIS = dplyr::case_when(
          # When DIA is NA, adjustment is NA
          is.na(DIA) ~ NA_character_,
          # When DIA is less than 5", use microplot value
          DIA < 5 ~ 'MICR',
          # Use SUBP when macroplot breakpoint DIA is non-positive
          MACRO_BREAKPOINT_DIA <= 0 ~ 'SUBP',
          # When DIA is greater than 5", use subplot value, as long
          # as diameter is less than the minimum dia on macroplots
          DIA >= 5 & is.na(MACRO_BREAKPOINT_DIA) ~ 'SUBP',
          DIA >= 5 & DIA < MACRO_BREAKPOINT_DIA ~ 'SUBP',
          # Use macroplot if DIA >= the min dbh measured on macroplots
          DIA >= MACRO_BREAKPOINT_DIA ~ 'MACR'
        )
      ) %>%
      dplyr::filter(!is.na(TREE_BASIS)) %>%
      dplyr::select(PLT_CN, CONDID, TREE_BASIS, !!!grpSyms, AG_OVER_LIVE, 
                    BG_OVER_LIVE, AG_OVER_DEAD, BG_OVER_DEAD) %>%
      as.data.frame()

    # Return a tree/condition list ready to be handed to `customPSE()`
    if (condList) {
      t <- t %>%
        dtplyr::lazy_dt() %>%
        dplyr::group_by(PLT_CN, CONDID, !!!grpSyms) %>%
        dplyr::summarize(dplyr::across(AG_OVER_LIVE:BG_OVER_DEAD, \(x) sum(x, na.rm = TRUE))) %>%
        dplyr::ungroup() %>%
        as.data.frame()

      # Join with a and specify STAND_DEAD
      t <- a %>%
        dplyr::left_join(t, by = c('PLT_CN', grpBy, 'CONDID')) %>%
        dplyr::mutate(STAND_DEAD = AG_OVER_DEAD + BG_OVER_DEAD) %>%
        dplyr::select(-c(AG_OVER_DEAD, BG_OVER_DEAD))

      # Convert to long format, where rows are ecosystem components
      t <- t %>%
        tidyr::pivot_longer(cols = -c(PLT_CN, AREA_BASIS, CONDID, PROP_FOREST, !!!grpSyms),
                            names_to = 'COMPONENT',
                            values_to = 'CARB_ACRE') %>%
        dtplyr::lazy_dt() %>%
        dplyr::mutate(POOL = dplyr::case_when(COMPONENT %in% c('AG_UNDER_LIVE', 'AG_OVER_LIVE') ~ 'AG_LIVE',
                                              COMPONENT %in% c('BG_UNDER_LIVE', 'BG_OVER_LIVE') ~ 'BG_LIVE',
                                              COMPONENT %in% c('STAND_DEAD', 'DOWN_DEAD') ~ 'DEAD_WOOD',
                                              TRUE ~ COMPONENT)) %>%
        dplyr::mutate(CARB_ACRE = CARB_ACRE * 0.90718474) %>%
        as.data.frame()
      

      # Add either component or pool to grpBy
      if (byComponent) {
        grpBy <- c(grpBy, 'POOL', 'COMPONENT') 
      }
      if (byPool & !byComponent) {
        grpBy <- c(grpBy, 'POOL')
      }
      grpSyms <- dplyr::syms(grpBy)
      aGrpBy <- grpBy[!c(grpBy %in% c('POOL', 'COMPONENT'))]
      aGrpSyms <- dplyr::syms(aGrpBy)

      # Summarize across COMPONENTS, if necessary
      if (!byComponent) {
        t <- t %>%
          dtplyr::lazy_dt() %>%
          dplyr::group_by(PLT_CN, CONDID, !!!grpSyms, PROP_FOREST, AREA_BASIS) %>%
          dplyr::summarise(CARB_ACRE = sum(CARB_ACRE, na.rm = TRUE)) %>%
          dplyr::ungroup() %>%
          as.data.frame()
      }

      # Reorder variable names
      t <- t %>%
        dplyr::mutate(EVAL_TYP = 'VOL') %>%
        dplyr::select(PLT_CN, EVAL_TYP, AREA_BASIS,
                      CONDID, !!!grpSyms, CARB_ACRE,
                      PROP_FOREST)

      out <- list(tEst = t, aEst = NULL, grpBy = grpBy, aGrpBy = aGrpBy)

    } else {

      # If a tree list is not desired, let's move to population estimation.
      # Sum variable(s) up to plot-level and adjust for non-response
      tPlt <- sumToPlot(t, pops, grpBy)
      aPlt <- sumToPlot(a, pops, grpBy)
      tPlt <- aPlt %>%
        dplyr::select(-c(PROP_FOREST)) %>%
        dplyr::left_join(tPlt, by = c('ESTN_UNIT_CN', 'STRATUM_CN', 'PLT_CN', grpBy))

      # Add in the correct snag estimate
      tPlt <- tPlt %>%
        dplyr::mutate(STAND_DEAD = AG_OVER_DEAD + BG_OVER_DEAD) %>%
        dplyr::select(-c(AG_OVER_DEAD, BG_OVER_DEAD))

      # Convert to long format, where rows are ecosystem components
      tPlt <- tPlt %>%
        tidyr::pivot_longer(cols = -c(ESTN_UNIT_CN, STRATUM_CN, PLT_CN, !!!grpSyms),
                            names_to = 'COMPONENT',
                            values_to = 'cPlot') %>%
        dtplyr::lazy_dt() %>%
        dplyr::mutate(POOL = dplyr::case_when(COMPONENT %in% c('AG_UNDER_LIVE', 'AG_OVER_LIVE') ~ 'AG_LIVE',
                                              COMPONENT %in% c('BG_UNDER_LIVE', 'BG_OVER_LIVE') ~ 'BG_LIVE',
                                              COMPONENT %in% c('STAND_DEAD', 'DOWN_DEAD') ~ 'DEAD_WOOD',
                                              TRUE ~ COMPONENT)) %>%
        dplyr::mutate(cPlot = cPlot * 0.90718474) %>%
        as.data.frame()

      # Add either component or pool to grpBy
      if (byComponent) {
        grpBy <- c(grpBy, 'POOL', 'COMPONENT') 
      }
      if (byPool & !byComponent) {
        grpBy <- c(grpBy, 'POOL')
      }
      grpSyms <- rlang::syms(grpBy)
      aGrpBy <- grpBy[!c(grpBy %in% c('POOL', 'COMPONENT'))]
      aGrpSyms <- rlang::syms(aGrpBy)

      # Summarize across COMPONENTS, if necessary
      if (!byComponent) {
        tPlt <- tPlt %>%
          dtplyr::lazy_dt() %>%
          dplyr::group_by(ESTN_UNIT_CN, STRATUM_CN, PLT_CN, !!!grpSyms) %>%
          dplyr::summarise(cPlot = sum(cPlot, na.rm = TRUE)) %>%
          dplyr::ungroup() %>%
          as.data.frame()
      }

      # Drop all variables except forested area from a
      aPlt <- aPlt %>%
        dplyr::select(ESTN_UNIT_CN, STRATUM_CN, PLT_CN,
                      !!!aGrpSyms, fa = PROP_FOREST)

      # Add YEAR to groups
      grpBy <- c('YEAR', grpBy)
      aGrpBy <- c('YEAR', aGrpBy)

      # Sum variable(s) up to strata then estimation unit level
      eu.sums <- sumToEU(db, tPlt, aPlt, pops, grpBy, aGrpBy, method, lambda)
      tEst <- eu.sums$x
      aEst <- eu.sums$y

      out <- list(tEst = tEst, aEst = aEst, grpBy = grpBy, aGrpBy = aGrpBy)
    } # population or condition-level estimation
  } # Population or plot-level estimation

  return(out)
} # carbonStarter
