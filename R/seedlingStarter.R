seedlingStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                            returnSpatial = FALSE, bySpecies = FALSE, 
                            landType = 'forest', method = 'TI', lambda = 0.5, 
                            treeDomain = NULL, areaDomain = NULL, 
                            totals = FALSE, byPlot = FALSE, treeList = FALSE, 
                            nCores = 1, remote, mr) {

  # Read required data and prep the database ------------------------------
  # Note the requirement of SEEDLING instead of TREE
  reqTables <- c('PLOT', 'SEEDLING', 'COND', 'POP_PLOT_STRATUM_ASSGN', 
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
  # Get a plot CN and a new pltID that gives a unique ID to each plot
  # PLT_CN is UNITCD, STATECD, COUNTYCD, PLOT, and INVYR
  db$PLOT <- db$PLOT %>% 
    dplyr::mutate(PLT_CN = CN, 
                  pltID = stringr::str_c(UNITCD, STATECD, COUNTYCD, PLOT, sep = '_'))

  # Convert grpBy to character
  grpBy <- grpByToChar(db, grpBy_quo)

  # Unique plot ID through time (pltID)
  if (byPlot | treeList) {
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
  db <- udSeedDomain(db, treeDomain)

  # Handle population tables ----------------------------------------------
  # Filtering out all inventories that are not relevant to the current 
  # estimation type. If using estimator other than TI, handle the differences in 
  # P2POINTCNT and in assigning YEAR column (YEAR = END_INVYR if method = 'TI')
  # TODO: need to go through handlePops
  pops <- handlePops(db, evalType = c('VOL'), method, mr)

  # A lot of states do their stratification in such a way that makes it impossible
  # to estimate variance of annual panels with the post-stratified estimator. That is, 
  # the number of plots within a panel within a stratum is less than 2. When this happens
  # merge strata so that all have at least two observations.
  # TODO: need to go through mergeSmallStrata
  if (stringr::str_to_upper(method) %in% c('SMA', 'LMA', 'EMA', 'ANNUAL') & !byPlot) {
    pops <- mergeSmallStrata(db, pops)
  }

  # Handle canned groups --------------------------------------------------
  # Add species to groups
  # Note that intData is an internal data object. 
  if (bySpecies) {
    db$SEEDLING <- db$SEEDLING %>%
      dplyr::left_join(dplyr::select(intData$REF_SPECIES_2018, 
                                     c('SPCD', 'COMMON_NAME', 'GENUS', 'SPECIES')), 
                       by = 'SPCD') %>%
      dplyr::mutate(SCIENTIFIC_NAME = stringr::str_c(GENUS, SPECIES, sep = ' ')) %>% 
      dplyr::mutate_if(is.factor, character)
    grpBy <- c(grpBy, 'SPCD', 'COMMON_NAME', 'SCIENTIFIC_NAME')
  }

  # Prep the tree list ----------------------------------------------------
  # Narrow the tables to the necessary variables.
  # Which grpByNames are in which table? Helps us subset below.
  # grpBy names in PLOT
  grpP <- names(db$PLOT)[names(db$PLOT) %in% grpBy]
  # grpBy names in COND
  grpC <- names(db$COND)[names(db$COND) %in% grpBy &
                         !c(names(db$COND) %in% grpP)]
  # grpBy names in SEEDLING
  grpT <- names(db$SEEDLING)[names(db$SEEDLING) %in% grpBy &
                         !c(names(db$SEEDLING) %in% c(grpP, grpC))]

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

  # SEEDLING --------------------------
  # TODO: add TREECOUNT_CALC to iPad descriptions
  db$SEEDLING <- db$SEEDLING %>%
    dplyr::select(PLT_CN, CONDID, SPCD, TPA_UNADJ, SUBP, TREECOUNT_CALC,
                  dplyr::all_of(grpT), tD) %>%
    # Drop plots outside our domain of interest
    dplyr::filter(TPA_UNADJ > 0 & tD == 1) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% db$PLOT$PLT_CN)

  # Separate area grouping names from tree grouping names
  if (!is.null(polys)) {
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND), names(polys))]
  } else {
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND))]
  }

  # Full tree list
  data <- db$PLOT %>%
    dplyr::left_join(db$COND, by = c('PLT_CN')) %>%
    dplyr::left_join(db$SEEDLING, by = c('PLT_CN', 'CONDID'))

  # Comprehensive indicator function
  data$aDI <- data$landD * data$aD * data$sp
  data$tDI <- data$landD * data$aD * data$tD * data$sp

  # Plot-level summaries --------------------------------------------------
  if (byPlot & !treeList) {
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

    t <- data %>% 
      # Set the YEAR to the measurement year for plot-level estimates. 
      dplyr::mutate(YEAR = MEASYEAR) %>% 
      dplyr::distinct(PLT_CN, SUBP, SPCD, .keep_all = TRUE) %>% 
      dtplyr::lazy_dt() %>% 
      dplyr::group_by(!!!grpSyms, PLT_CN) %>%  
      dplyr::summarize(TPA = sum(TPA_UNADJ * tDI, na.rm = TRUE)) %>% 
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
    # Population estimation or prep for it (treeList) ---------------------
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
      dplyr::distinct(PLT_CN, SUBP, SPCD, .keep_all = TRUE) %>% 
      dplyr::mutate(tPlot = TPA_UNADJ * tDI, 
                    TREE_BASIS = 'MICR') %>% 
      dplyr::select(PLT_CN, TREE_BASIS, SPCD, !!!grpSyms, tPlot) %>% 
      as.data.frame()

    # Return a tree/condition list ready to be handed to `customPSE()`
    if (treeList) {
      tEst <- a %>% 
        dplyr::left_join(t, by = c('PLT_CN', aGrpBy)) %>% 
        # customPSE requires these be present, so simply set to NA for seedling
        dplyr::mutate(SUBP = NA, 
                      TREE = NA, 
                      PROP_FOREST = fa, 
                      EVAL_TYP = 'VOL') %>% 
        dplyr::group_by(PLT_CN, EVAL_TYP, TREE_BASIS, AREA_BASIS, 
                        !!!grpSyms, CONDID, SUBP, TREE, PROP_FOREST) %>%
        dplyr::summarize(TPA = sum(tPlot, na.rm = TRUE)) %>%
        dplyr::ungroup()

      out <- list(tEst = tEst, aEst = NULL, grpBy = grpBy, aGrpBy = aGrpBy)

    } else {
      # If a tree list is not desired, let's move to population estimation.
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
    } # population or tree-level estimation
  } # population or plot-level estimation

  return(out)

} # seedlingStarter
