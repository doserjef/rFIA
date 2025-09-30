biomassStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                           returnSpatial = FALSE, bySpecies = FALSE, 
                           bySizeClass = FALSE, byComponent = FALSE, 
                           landType = 'forest', treeType = 'live', 
                           method = 'TI', lambda = 0.5, treeDomain = NULL, 
                           areaDomain = NULL, totals = FALSE, byPlot = FALSE, 
                           treeList = FALSE, component = 'AG', bioMethod = 'NSVB',
                           nCores = 1, remote, mr) {

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
  if (landType %in% c('timber', 'forest') == FALSE) {
    stop('landType must be one of: "forest" or "timber".')
  }
  # treeType --------------------------
  if (treeType %in% c('live', 'dead', 'gs', 'all') == FALSE) {
    stop('treeType must be one of: "live", "dead", "gs", or "all".')
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
  if (sum(stringr::str_to_upper(component) %in% 
      c('AG', 'STEM', 'STEM_BARK', 'BRANCH', 'FOLIAGE', 
        'STUMP', 'STUMP_BARK', 'BOLE', 'BOLE_BARK', 'SAWLOG', 
        'SAWLOG_BARK', 'ROOT', 'TOTAL')) != length(component)) {
    stop('Unknown component. Must be a combination of: "AG", "STEM", "STEM_BARK", "BRANCH", "FOLIAGE", "STUMP", "STUMP_BARK", "BOLE", "BOLE_BARK", "SAWLOG", "SAWLOG_BARK", "ROOT". Alternatively, use "TOTAL" for a total biomass, and set "byComponent=TRUE" to estimate all components simultaneously.')
  }

  # Biomass method warnings
  bioMethod <- stringr::str_to_upper(bioMethod)
  if (length(bioMethod) > 1) {
    stop('Only one "bioMethod" allowed at this time. Please set this to "NSVB" (National Scale Volume Biomass models).')
  }
  if (!c(bioMethod %in% c('CRM', 'JENKINS', 'NSVB'))) {
    stop('"bioMethod" not recognized. Please set this to "NSVB" (National Scale Volume Biomass models).')
  }
  if (bioMethod == 'CRM') {
    message('The Component Ratio Method (CRM) is no longer used by FIA for aboveground biomass estimation as of Sep 2023.\nThe current method is NSVB. If using FIADB tables from prior to Sep 2023, biomass()\nwill give estimates based on CRM when bioMethod = "NSVB"') 
    bioMethod = 'NSVB'
  }
  if (bioMethod == 'JENKINS') {
    stop('As of v1.1.0, rFIA::biomass() now only supports biomass estimation using the National Scale Volume Biomass (NSVB) models. Please set bioMethod = "NSVB"') 
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
  if (byPlot | treeList) {
    grpBy <- c('pltID', grpBy)
  }

  # When component TOTAL, replace with component names. 
  # Note that you only grab the components that are needed 
  # to sum to total biomass. 
  component <- stringr::str_to_upper(unique(component))
  if ('TOTAL' %in% component | byComponent) {
    component <- c('STEM', 'STEM_BARK', 'BRANCH', 'FOLIAGE', 
                   'ROOT')
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

  # Join with REF_SPECIES (in intData) to get species-level carbon fractions.
  db$TREE <- db$TREE %>%
    dplyr::left_join(dplyr::select(intData$REF_SPECIES_DEC_2024, 
                                   c('SPCD', 'COMMON_NAME', 'GENUS', 'SPECIES', 
                                     'CARBON_RATIO_LIVE')), 
                     by = 'SPCD') %>%
    dplyr::mutate(SCIENTIFIC_NAME = paste(GENUS, SPECIES, sep = ' ')) %>% 
    dplyr::mutate_if(is.factor, character)

  # Build a domain indicator for each observation (1 or 0) ----------------
  # Land type
  db$COND$landD <- landTypeDomain(landType, db$COND$COND_STATUS_CD, 
                                  db$COND$SITECLCD, db$COND$RESERVCD)
  # Tree type
  db$TREE$typeD <- treeTypeDomain(treeType, db$TREE$STATUSCD, db$TREE$DIA, 
                                  db$TREE$TREECLCD)

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

  # User defined domain indicator for tree (ex. trees > 20 ft tall)
  db <- udTreeDomain(db, treeDomain)

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

  # Handle canned groups --------------------------------------------------
  # Add species to groups
  # Note that intData is an internal data object. 
  if (bySpecies) {
    # Add species names to grpBy. Note the data was already connected
    # to REF_SPECIES previously, which is used to pull in the species-specific
    # carbon fractions regardless of reporting by species.
    grpBy <- c(grpBy, 'SPCD', 'COMMON_NAME', 'SCIENTIFIC_NAME')
  }

  # Break into size classes
  if (bySizeClass) {
    grpBy <- c(grpBy, 'sizeClass')
    db$TREE$sizeClass <- makeClasses(db$TREE$DIA, interval = 2, numLabs = TRUE)
    db$TREE <- db$TREE[!is.na(db$TREE$sizeClass), ]
  }

  # Add component to grpBy
  if (byComponent) {
    grpBy <- c(grpBy, 'COMPONENT')
  }

  # Prep the tree list ----------------------------------------------------
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

  # TREE ------------------------------
  db$TREE <- db$TREE %>%
    dplyr::select(PLT_CN, CONDID, DIA, SPCD, TPA_UNADJ, SUBP, TREE, CARBON_RATIO_LIVE, 
                  dplyr::all_of(grpT), tD, typeD, DRYBIO_STEM, DRYBIO_STEM_BARK, 
                  DRYBIO_BRANCH, DRYBIO_FOLIAGE, DRYBIO_STUMP, DRYBIO_STUMP_BARK, 
                  DRYBIO_BOLE, DRYBIO_BOLE_BARK, DRYBIO_SAWLOG, DRYBIO_SAWLOG_BARK, 
                  DRYBIO_ROOT = DRYBIO_BG, DRYBIO_AG) %>% 
    # Drop plots outside our domain of interest
    dplyr::filter(!is.na(DIA) & TPA_UNADJ > 0 & tD == 1 & typeD == 1) %>%
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
    dplyr::left_join(db$TREE, by = c('PLT_CN', 'CONDID'))

  # Comprehensive indicator function
  data$aDI <- data$landD * data$aD * data$sp
  data$tDI <- data$landD * data$aD * data$tD * data$typeD * data$sp

  # Convert to long format, where biomass component is the observation (multiple per tree). 
  # Also generate carbon column here. 
  data <- data %>%
    tidyr::pivot_longer(cols = DRYBIO_STEM:DRYBIO_AG,
                        names_to = c(".value", 'COMPONENT'),
                        names_sep = 7) %>%
    dplyr::rename(DRYBIO = DRYBIO_) %>%
    dplyr::mutate(CARBON = ifelse(COMPONENT != 'FOLIAGE', 
                                  DRYBIO * CARBON_RATIO_LIVE, 
                                  NA)) %>%
    dplyr::filter(COMPONENT %in% component)

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

    # Biomass and carbon for each plot
    t <- data %>% 
      # Set the YEAR to the measurement year for plot-level estimates. 
      dplyr::mutate(YEAR = MEASYEAR) %>% 
      dplyr::distinct(PLT_CN, SUBP, TREE, COMPONENT, .keep_all = TRUE) %>% 
      dtplyr::lazy_dt() %>% 
      dplyr::group_by(!!!grpSyms, PLT_CN) %>%  
      # 2000 is to convert from pounds/acre to short tons/acre
      dplyr::summarize(BIO_ACRE = sum(DRYBIO * TPA_UNADJ * tDI, na.rm = TRUE) / 2000, 
                       CARB_ACRE = sum(CARBON * TPA_UNADJ * tDI, na.rm = TRUE) / 2000) %>% 
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
    aGrpSyms <- rlang::syms(aGrpBy)

    # Condition list
    a <- data %>% 
      # Will be lots of trees here, so CONDPROP is listed multiple times, the 
      # distinct is needed to just get those distinct ones. 
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>% 
      dplyr::mutate(fa = CONDPROP_UNADJ * aDI) %>% 
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!aGrpSyms, fa)

    # Create list of symols for the grpBy statements
    grpSyms <- rlang::syms(grpBy)
    # Tree list
    t <- data %>% 
      dplyr::distinct(PLT_CN, SUBP, TREE, COMPONENT, .keep_all = TRUE) %>% 
      dplyr::mutate(bPlot = DRYBIO * TPA_UNADJ * tDI / 2000, 
                    cPlot = CARBON * TPA_UNADJ * tDI / 2000) %>% 
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
      dplyr::select(PLT_CN, TREE_BASIS, SUBP, TREE, !!!grpSyms, bPlot, cPlot) %>% 
      as.data.frame()

    # Return a tree/condition list ready to be handed to `customPSE()`
    if (treeList) {
      tEst <- a %>% 
        dplyr::left_join(t, by = c('PLT_CN', aGrpBy)) %>% 
        dtplyr::lazy_dt() %>%
        dplyr::mutate(EVAL_TYP = 'VOL') %>% 
        # Summarize over components so output isn't confusing to end user
        dplyr::group_by(PLT_CN, EVAL_TYP, TREE_BASIS, AREA_BASIS, 
                        !!!grpSyms, CONDID, SUBP, TREE, fa) %>%
        dplyr::summarize(bPlot = sum(bPlot, na.rm = TRUE), 
                         cPlot = sum(cPlot, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        dplyr::select(PLT_CN, EVAL_TYP, TREE_BASIS, AREA_BASIS, 
                      !!!grpSyms, CONDID, SUBP, TREE, 
                      BIO_ACRE = bPlot, CARB_ACRE = cPlot, PROP_FOREST = fa) %>%
        as.data.frame()

      out <- list(tEst = tEst, aEst = NULL, grpBy = grpBy, aGrpBy = aGrpBy)

    } else {
      # If a tree list is not desired, let's move to population estimation.
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
    } # population or tree-level estimation
  } # Population or plot-level estimation

  return(out)


}
