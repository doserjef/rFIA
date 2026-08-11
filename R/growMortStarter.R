growMortStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                            returnSpatial = FALSE, bySpecies = FALSE, 
                            bySizeClass = FALSE, landType = 'forest', 
                            treeType = 'all', method = 'TI', lambda = 0.5, 
                            stateVar = 'TPA', treeDomain = NULL, 
                            areaDomain = NULL, totals = FALSE, byPlot = FALSE,
                            treeList = FALSE, nCores = 1, remote, mr) {

  # Read required data and prep the database ------------------------------
  # If SUBP_COND_CHNG_MTRX is not provided, estimates will not use growth 
  # accounting method. If it is, growth accounting method is used.
  if ('SUBP_COND_CHNG_MTRX' %in% names(db)) {
    reqTables <- c('PLOT', 'COND', 'TREE', 'TREE_GRM_COMPONENT', 'TREE_GRM_MIDPT',
                   'TREE_GRM_BEGIN', 'SUBP_COND_CHNG_MTRX',
                   'POP_PLOT_STRATUM_ASSGN', 'POP_ESTN_UNIT', 'POP_EVAL',
                   'POP_STRATUM', 'POP_EVAL_TYP', 'POP_EVAL_GRP', 'PLOTGEOM')
  } else {
    reqTables <- c('PLOT', 'COND', 'TREE', 'TREE_GRM_COMPONENT', 'TREE_GRM_MIDPT',
                   'TREE_GRM_BEGIN', 'POP_PLOT_STRATUM_ASSGN', 'POP_ESTN_UNIT', 'POP_EVAL',
                   'POP_STRATUM', 'POP_EVAL_TYP', 'POP_EVAL_GRP', 'PLOTGEOM')
  }

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
  if (landType %in% c('timber', 'forest') == FALSE) {
    stop('landType must be one of: "forest" or "timber".')
  }
  # treeType --------------------------
  if (treeType %in% c('gs', 'all') == FALSE) {
    stop('treeType must be one of: "gs", or "all".')
  }
  # db required tables ----------------
  if (any(reqTables[!c(reqTables %in% 'SUBP_COND_CHNG_MTRX')] %in% names(db) == FALSE)){
    missT <- reqTables[reqTables %in% names(db) == FALSE]
    stop(paste('Tables', paste (as.character(missT), collapse = ', '),
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
  # Also rename for later
  db$TREE <- db$TREE %>%
    dplyr::mutate(TRE_CN = CN) 

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

  # Join with REF_SPECIES (stored in package's internal data) to get 
  # species-level carbon fractions.
  db$TREE <- db$TREE %>%
    dplyr::left_join(dplyr::select(intData$REF_SPECIES_DEC_2024, 
                                   c('SPCD', 'COMMON_NAME', 'GENUS', 'SPECIES', 
                                     'CARBON_RATIO_LIVE')), 
                     by = 'SPCD') %>%
    dplyr::mutate(SCIENTIFIC_NAME = paste(GENUS, SPECIES, sep = ' ')) %>% 
    dplyr::mutate_if(is.factor, character)
  db$TREE_GRM_MIDPT <- db$TREE_GRM_MIDPT %>%
    dplyr::left_join(dplyr::select(intData$REF_SPECIES_DEC_2024, 
                                   c('SPCD', 'COMMON_NAME', 'GENUS', 'SPECIES', 
                                     'CARBON_RATIO_LIVE')), 
                     by = 'SPCD') %>%
    dplyr::mutate(SCIENTIFIC_NAME = paste(GENUS, SPECIES, sep = ' ')) %>% 
    dplyr::mutate_if(is.factor, character)
  db$TREE_GRM_BEGIN <- db$TREE_GRM_BEGIN %>%
    dplyr::left_join(dplyr::select(intData$REF_SPECIES_DEC_2024, 
                                   c('SPCD', 'COMMON_NAME', 'GENUS', 'SPECIES', 
                                     'CARBON_RATIO_LIVE')), 
                     by = 'SPCD') %>%
    dplyr::mutate(SCIENTIFIC_NAME = paste(GENUS, SPECIES, sep = ' ')) %>% 
    dplyr::mutate_if(is.factor, character)


  # Handle the state variable, only applying to the midpoint table for consistency
  if (stringr::str_to_upper(stateVar) == 'TPA'){
    db$TREE_GRM_MIDPT$state <- 1
    db$TREE_GRM_BEGIN$state <- 1
    db$TREE$state_recr <- 1
  } else if (stringr::str_to_upper(stateVar) == 'BAA'){
    db$TREE_GRM_MIDPT$state <- basalArea(db$TREE_GRM_MIDPT$DIA)
    db$TREE_GRM_BEGIN$state <- basalArea(db$TREE_GRM_BEGIN$DIA)
    db$TREE$state_recr <- basalArea(db$TREE$DIA)
  } else if (stringr::str_to_upper(stateVar) == 'SAWVOL'){
    db$TREE_GRM_MIDPT$state <- db$TREE_GRM_MIDPT$VOLCSNET
    db$TREE_GRM_BEGIN$state <- db$TREE_GRM_BEGIN$VOLCSNET
    db$TREE$state_recr <- db$TREE$VOLCSNET
  } else if (stringr::str_to_upper(stateVar) == 'SAWVOL_BF'){
    db$TREE_GRM_MIDPT$state <- db$TREE_GRM_MIDPT$VOLBFNET
    db$TREE_GRM_BEGIN$state <- db$TREE_GRM_BEGIN$VOLBFNET
    db$TREE$state_recr <- db$TREE$VOLBFNET
  } else if (stringr::str_to_upper(stateVar) == 'NETVOL'){
    db$TREE_GRM_MIDPT$state <- db$TREE_GRM_MIDPT$VOLCFNET
    db$TREE_GRM_BEGIN$state <- db$TREE_GRM_BEGIN$VOLCFNET
    db$TREE$state_recr <- db$TREE$VOLCFNET
  } else if (stringr::str_to_upper(stateVar) == 'SNDVOL'){
    db$TREE_GRM_MIDPT$state <- db$TREE_GRM_MIDPT$VOLCFSND
    db$TREE_GRM_BEGIN$state <- db$TREE_GRM_BEGIN$VOLCFSND
    db$TREE$state_recr <- db$TREE$VOLCFSND
  } else if (stringr::str_to_upper(stateVar) == 'BIO_AG'){
    # 2000 is to convert from pounds to short tons, matching biomass()/carbon()
    db$TREE_GRM_MIDPT$state <- db$TREE_GRM_MIDPT$DRYBIO_AG / 2000
    db$TREE_GRM_BEGIN$state <- db$TREE_GRM_BEGIN$DRYBIO_AG / 2000
    db$TREE$state_recr <- db$TREE$DRYBIO_AG / 2000
  } else if (stringr::str_to_upper(stateVar) == 'BIO_BG'){
    db$TREE_GRM_MIDPT$state <- db$TREE_GRM_MIDPT$DRYBIO_BG / 2000
    db$TREE_GRM_BEGIN$state <- db$TREE_GRM_BEGIN$DRYBIO_BG / 2000
    db$TREE$state_recr <- db$TREE$DRYBIO_BG / 2000
  } else if (stringr::str_to_upper(stateVar) == 'BIO'){
    db$TREE_GRM_MIDPT$state <- (db$TREE_GRM_MIDPT$DRYBIO_BG + db$TREE_GRM_MIDPT$DRYBIO_AG) / 2000
    db$TREE_GRM_BEGIN$state <- (db$TREE_GRM_BEGIN$DRYBIO_BG + db$TREE_GRM_BEGIN$DRYBIO_AG) / 2000
    db$TREE$state_recr <- (db$TREE$DRYBIO_BG + db$TREE$DRYBIO_AG) / 2000
  } else if (stringr::str_to_upper(stateVar) == 'CARB_AG'){
    db$TREE_GRM_MIDPT$state <- db$TREE_GRM_MIDPT$DRYBIO_AG * db$TREE_GRM_MIDPT$CARBON_RATIO_LIVE / 2000
    db$TREE_GRM_BEGIN$state <- db$TREE_GRM_BEGIN$DRYBIO_AG * db$TREE_GRM_BEGIN$CARBON_RATIO_LIVE / 2000
    db$TREE$state_recr <- db$TREE$DRYBIO_AG * db$TREE$CARBON_RATIO_LIVE / 2000
  } else if (stringr::str_to_upper(stateVar) == 'CARB_BG'){
    db$TREE_GRM_MIDPT$state <- db$TREE_GRM_MIDPT$DRYBIO_BG * db$TREE_GRM_MIDPT$CARBON_RATIO_LIVE / 2000
    db$TREE_GRM_BEGIN$state <- db$TREE_GRM_BEGIN$DRYBIO_BG * db$TREE_GRM_BEGIN$CARBON_RATIO_LIVE / 2000
    db$TREE$state_recr <- db$TREE$DRYBIO_BG * db$TREE$CARBON_RATIO_LIVE / 2000
  } else if (stringr::str_to_upper(stateVar) == 'CARB'){
    db$TREE_GRM_MIDPT$state <- (db$TREE_GRM_MIDPT$DRYBIO_AG + db$TREE_GRM_MIDPT$DRYBIO_BG) *
                               db$TREE_GRM_MIDPT$CARBON_RATIO_LIVE / 2000
    db$TREE_GRM_BEGIN$state <- (db$TREE_GRM_BEGIN$DRYBIO_AG + db$TREE_GRM_BEGIN$DRYBIO_BG) *
                               db$TREE_GRM_BEGIN$CARBON_RATIO_LIVE / 2000
    db$TREE$state_recr <- (db$TREE$DRYBIO_AG + db$TREE$DRYBIO_BG) *
                          db$TREE$CARBON_RATIO_LIVE / 2000
  } else {
    stop(paste0('Method not known for stateVar: ', stateVar, '. Please choose one of: TPA, BAA, SAWVOL, SAWVOL_BF, NETVOL, BIO_AG, BIO_BG, BIO, CARB_AG, CARB_BG, or CARB.' ))
  }

  # Build a domain indicator for each observation (1 or 0) ----------------
  # Land type and tree type combined
  db <- typeDomain_grow(db, treeType, landType, type = 'gm', stateVar)

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
  pops <- handlePops(db, evalType = c('GROW', 'MORT', 'REMV'), method, mr)

  # Generate a list that tells us if a plot is ever associated with a growth 
  # accounting inventory. If so, we will be more strict with domain definitions.
  plt.ga <- pops %>%
    dtplyr::lazy_dt() %>%
    dplyr::mutate(ga = case_when(GROWTH_ACCT == 'Y' ~ 1,
                                 TRUE ~ 0)) %>%
    dplyr::group_by(PLT_CN) %>%
    dplyr::summarise(ga = sum(ga, na.rm = TRUE)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(ga = case_when(ga > 0 ~ 1, TRUE ~ 0)) %>%
    as.data.frame()

  # A lot of states do their stratification in such a way that makes it impossible
  # to estimate variance of annual panels with the post-stratified estimator. That is,
  # the number of plots within a panel within a stratum is less than 2. When this happens
  # merge strata so that all have at least two observations.
  if (stringr::str_to_upper(method) %in% c('SMA', 'LMA', 'EMA', 'ANNUAL') & !byPlot) {
    pops <- mergeSmallStrata(db, pops)
  }

  ## Canned groups -------------------------------------------------------------
  ## Add species to groups
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
    dplyr::select(c(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA,
                    INVYR, MEASYEAR, PLOT_STATUS_CD,
                    dplyr::all_of(grpP), sp, COUNTYCD,
                    PREV_PLT_CN, REMPER)) %>%
    # Drop non-forested plots, and those otherwise outside our domain of interest. 
    # Note that this will result in keeping all plots that were forested during 
    # at least one time point.
    dplyr::filter(PLOT_STATUS_CD == 1 & sp == 1) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # COND ------------------------------
  db$COND <- db$COND %>%
    dplyr::select(c(PLT_CN, CONDPROP_UNADJ, PROP_BASIS,
                    COND_STATUS_CD, CONDID,
                    dplyr::all_of(grpC), aD, landD)) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))
  
  # TREE ------------------------------
  db$TREE <- db$TREE %>%
    dplyr::select(c(PLT_CN, CONDID, PREVCOND, TRE_CN,
                    PREV_TRE_CN, SUBP, TREE, dplyr::all_of(grpT), tD,
                    typeD, state_recr, TPA_UNADJ,
                    STATUSCD, DIA)) %>%
    # Drop plots outside our domain of interest
    dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))

  # TREE_GRM_COMPONENT ----------------
  db$TREE_GRM_COMPONENT <- db$TREE_GRM_COMPONENT %>%
    dplyr::select(c(PLT_CN, TRE_CN, SUBPTYP_GRM, TPAGROW_UNADJ, TPARECR_UNADJ,
                    TPAREMV_UNADJ, TPAMORT_UNADJ, COMPONENT)) %>%
    # Alternative would be to do COMPONENT != 'NOT USED'
    dplyr::filter(TPAGROW_UNADJ > 0 | TPARECR_UNADJ > 0 | TPAREMV_UNADJ > 0 | TPAMORT_UNADJ > 0) %>%
    # Drop visits not used in our eval of interest. This used to be the plt filter
    # dplyr::filter(PLT_CN %in% db$PLOT$PLT_CN) %>%
    dplyr::filter(TRE_CN %in% db$TREE$TRE_CN) %>%
    # Convert NAs in TPAMORT_UNADJ and TPAREMV_UNADJ to 0s. This corresponds to 
    # situations where the given tree is not MORT or REMV, but there is some sort of 
    # GRM record for it (i.e., GROW or RECR). This is needed for calculations later on.
    dplyr::mutate(TPAREMV_UNADJ = ifelse(is.na(TPAREMV_UNADJ), 0, TPAREMV_UNADJ), 
                  TPAMORT_UNADJ = ifelse(is.na(TPAMORT_UNADJ), 0, TPAMORT_UNADJ)) %>%
    dplyr::select(-c(PLT_CN))

  # TREE_GRM_MIDPT --------------------
  db$TREE_GRM_MIDPT <- db$TREE_GRM_MIDPT %>%
    select(c(TRE_CN, DIA, state)) %>%
    filter(TRE_CN %in% db$TREE$TRE_CN)

  # TREE_GRM_BEGIN --------------------
  db$TREE_GRM_BEGIN <- db$TREE_GRM_BEGIN %>%
    dplyr::select(c(TRE_CN, DIA, state)) %>%
    dplyr::filter(TRE_CN %in% db$TREE$TRE_CN)

  # SUBP_COND_CHNG_MTRX ---------------
  if ('SUBP_COND_CHNG_MTRX' %in% names(db)) {
    db$SUBP_COND_CHNG_MTRX <- dplyr::select(db$SUBP_COND_CHNG_MTRX, PLT_CN, PREV_PLT_CN,
                                     SUBPTYP, SUBPTYP_PROP_CHNG, PREVCOND, CONDID) %>%
      dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))
  }

  # Separate area grouping names from tree grouping names
  if (!is.null(polys)){
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND), names(polys))]
  } else {
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND))]
  }

  # Full tree list. Note that this consists of trees that were alive only in T1
  # (mortality and cut), trees only in the sample in T2 (ingrowth), and trees 
  # alive in both (survivor)
  data <- db$PLOT %>%
    dtplyr::lazy_dt() %>%
    dplyr::select(c(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA, INVYR,
                    MEASYEAR, PLOT_STATUS_CD, PREV_PLT_CN, REMPER,
                    dplyr::all_of(grpP), sp)) %>%
    # Link to COND                    
    dplyr::left_join(dplyr::select(db$COND, c(PLT_CN, CONDPROP_UNADJ, PROP_BASIS,
                                              COND_STATUS_CD, CONDID, dplyr::all_of(grpC),
                                              aD, landD)),
                     by = c('PLT_CN')) %>%
    # Link to TREE
    dplyr::left_join(dplyr::select(db$TREE, c(PLT_CN, CONDID, PREVCOND, TRE_CN,
                                              PREV_TRE_CN, SUBP, TREE, dplyr::all_of(grpT),
                                              tD, typeD, state_recr, TPA_UNADJ, STATUSCD, DIA)),
                     by = c('PLT_CN', 'CONDID')) %>%
    # Link to TREE_GRM_COMPONENT
    dplyr::left_join(dplyr::select(db$TREE_GRM_COMPONENT, c(TRE_CN, SUBPTYP_GRM,
                                                            TPAGROW_UNADJ, TPARECR_UNADJ,
                                                            TPAREMV_UNADJ, TPAMORT_UNADJ,
                                                            COMPONENT)),
                     by = c('TRE_CN')) %>%
    # Link to TREE_GRM_MIDPT
    dplyr::left_join(dplyr::select(db$TREE_GRM_MIDPT, c(TRE_CN, DIA, state)),
                     by = c('TRE_CN'), suffix = c('', '.mid')) %>%
    # Join with TREE_GRM_BEGIN
    dplyr::left_join(dplyr::select(db$TREE_GRM_BEGIN, c(TRE_CN, DIA, state)), 
                     by = c('TRE_CN'), suffix = c('', '.beg')) %>%
    # Link previous plot measurement to PLOT
    dplyr::left_join(dplyr::select(db$PLOT, c(PLT_CN, dplyr::all_of(grpP), sp)),
                     by = c('PREV_PLT_CN' = 'PLT_CN'), suffix = c('', '.prev')) %>%
    # Link previous cond to COND
    dplyr::left_join(dplyr::select(db$COND, c(PLT_CN, CONDID, landD, aD, dplyr::all_of(grpC),
                                              COND_STATUS_CD)),
                     by = c('PREV_PLT_CN' = 'PLT_CN', 'PREVCOND' = 'CONDID'),
                     suffix = c('', '.prev')) %>%
    # Link previous tree measurement to TREE
    dplyr::left_join(dplyr::select(db$TREE, c(TRE_CN, dplyr::all_of(grpT),
                                              typeD, tD, DIA, STATUSCD, TPA_UNADJ, 
                                              state.prev = state_recr)),
                     by = c('PREV_TRE_CN' = 'TRE_CN'), suffix = c('', '.prev')) %>%
    # Reset the TPA*_UNADJ to the specific state of interest. Fixing the column names
    # is dealt with later.
    # Replace state.prev with state.beg if state.beg is not NA.
    # state.beg comes from the TREE_GRM_BEGIN. The reason they differ is 
    # because of changes made to an original measurement upon remeasurement of 
    # the tree (e.g., fix an obvious error, change species identification).
    dplyr::mutate(state.prev = ifelse(!is.na(state.beg), state.beg, state.prev)) %>% 
    dplyr::mutate(TPAREMV_UNADJ = TPAREMV_UNADJ * state,
                  TPAMORT_UNADJ = TPAMORT_UNADJ * state,
                  TPARECR_UNADJ = TPARECR_UNADJ * state_recr / REMPER,
                  # State recruit is the state variable adjustment for ALL TREES at T2,
                  # So we can estimate live TPA at t2 (t1 unavailable w/out growth accounting) with:
                  TPA_UNADJ = TPAGROW_UNADJ * state_recr *
                              ifelse(COMPONENT %in% c('SURVIVOR', 'INGROWTH'), 1, 0),
                  # Previous-period (T1) contribution. Survivors are valued at
                  # their begin measurement, as before. CUT1/DIVERSION1/
                  # MORTALITY1 trees (present at T1 but gone by T2) are ALSO
                  # valued at their begin measurement here, matching
                  # EVALIDator's growth-accounting SQL (confirmed via
                  # attribute 202's VBA_SUMFROMWHERE): net growth subtracts
                  # each such tree's begin value, and (for CUT/DIVERSION only)
                  # separately adds back its midpoint value via hPlot in tPlot
                  # below, crediting growth up to the point of removal.
                  TPA_UNADJ.prev = TPAGROW_UNADJ * state.prev *
                                   ifelse(COMPONENT %in% c('SURVIVOR', 'CUT1',
                                                            'DIVERSION1', 'MORTALITY1'),
                                          1, 0),
                  ) %>%
    # Add our indicator of whether or not a plot is ever associated with a
    # growth accounting inventory
    dplyr::left_join(plt.ga, by = 'PLT_CN') %>%
    # Set up our domain indicators accordingly
    dplyr::mutate(aChng = dplyr::case_when(ga == 1 & COND_STATUS_CD == 1 & 
                                           COND_STATUS_CD.prev == 1 & !is.null(CONDPROP_UNADJ) ~ 1,
                                           ga == 0 ~ 1,
                                           TRUE ~ 0),
                  tChng = dplyr::case_when(ga == 1 & COND_STATUS_CD == 1 & COND_STATUS_CD.prev == 1 ~ 1,
                                           ga == 0 ~ 1,
                                           TRUE ~ 0),
                  landD.prev = dplyr::case_when(ga == 1 & landD == 1 & landD.prev == 1 ~ 1,
                                                ga == 0 & is.na(landD) & is.na(landD.prev) ~ 0,
                                                ga == 0 & is.na(landD.prev) ~ landD,
                                                ga == 0 ~ landD.prev,
                                                TRUE ~ 0)) %>%
    # If previous attributes are unavailable for trees, default to current
    dplyr::mutate(tD.prev = dplyr::case_when(is.na(tD.prev) ~ tD, TRUE ~ tD.prev),
                  typeD.prev = dplyr::case_when(is.na(typeD.prev) ~ typeD, TRUE ~ typeD.prev),
                  aD.prev = dplyr::case_when(is.na(aD.prev) ~ aD, TRUE ~ aD.prev),
                  sp.prev = dplyr::case_when(is.na(sp.prev) ~ sp, TRUE ~ sp.prev)) %>%
    # Comprehensive domain indicators. Note aD (not aD.prev) is used here to
    # match the area list's own aDI, which restricts the current condition
    # only -- EVALIDator's strFilter mechanism restricts both the numerator
    # and denominator using the current condition consistently.
    dplyr::mutate(tDI = landD.prev * aD * tD.prev * typeD.prev * sp.prev * tChng,
                  tDI_r = landD * aD * tD * typeD * sp * tChng, # All previous attributes NA for recruitment
                  aDI = landD * aD * sp * aChng) %>%
    as.data.frame()
    
  if ('SUBP_COND_CHNG_MTRX' %in% names(db)) {
    # Doing area separately now for growth accounting plots 
    aData <- dplyr::select(db$PLOT, c(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA,
                                      INVYR, MEASYEAR, PLOT_STATUS_CD, PREV_PLT_CN,
                                      REMPER, dplyr::all_of(grpP), sp)) %>%
      dplyr::left_join(dplyr::select(db$SUBP_COND_CHNG_MTRX, PLT_CN, PREV_PLT_CN,
                                     SUBPTYP, SUBPTYP_PROP_CHNG, PREVCOND, CONDID),
                       by = c('PLT_CN', 'PREV_PLT_CN')) %>%
      dplyr::left_join(dplyr::select(db$COND, c(PLT_CN, CONDPROP_UNADJ, PROP_BASIS,
                                                COND_STATUS_CD, CONDID, dplyr::all_of(grpC),
                                                aD, landD)),
                       by = c('PLT_CN', 'CONDID')) %>%
      dplyr::left_join(dplyr::select(db$COND, c(PLT_CN, PROP_BASIS, COND_STATUS_CD,
                                                CONDID, dplyr::all_of(grpC), aD, landD)),
                       by = c('PREV_PLT_CN' = 'PLT_CN', 'PREVCOND' = 'CONDID'), 
                       suffix = c('', '.prev')) %>%
      dplyr::mutate(aChng = dplyr::if_else(COND_STATUS_CD == 1 &
                                           COND_STATUS_CD.prev == 1 &
                                           !is.null(CONDPROP_UNADJ) &
                                           ((SUBPTYP == 1 & PROP_BASIS == 'SUBP') |
                                              (SUBPTYP == 3 & PROP_BASIS == 'MACR')),
                                           1, 0),
                    SUBPTYP_PROP_CHNG = SUBPTYP_PROP_CHNG * .25)

    aData$landD <- ifelse(aData$landD == 1 & aData$landD.prev == 1, 1, 0)
    aData$aDI_ga <- aData$landD * aData$aD * aData$sp * aData$aChng

  }


  # Plot-level summaries --------------------------------------------------
  if (byPlot & !treeList) {
    grpBy <- c('YEAR', grpBy)
    # Create a list of symbols for the grpBy statements
    grpSyms <- rlang::syms(grpBy)
    aGrpSyms <- rlang::syms(aGrpBy)

    if ('SUBP_COND_CHNG_MTRX' %in% names(db)) {
      # Forested land area w growth accounting
      a.ga <- aData %>%
        dtplyr::lazy_dt() %>%
        dplyr::group_by(PLT_CN, !!!aGrpSyms) %>%
        dplyr::summarize(fa_ga = sum(SUBPTYP_PROP_CHNG * aDI_ga, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        as.data.frame()

      a <- data %>%
        dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
        dtplyr::lazy_dt() %>%
        dplyr::group_by(PLT_CN, !!!aGrpSyms) %>%
        dplyr::summarize(fa = sum(CONDPROP_UNADJ * aDI, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        dplyr::left_join(dplyr::select(a.ga, PLT_CN, !!!aGrpSyms, fa_ga),
                         by = c('PLT_CN', aGrpBy)) %>%
        dplyr::left_join(plt.ga, by = 'PLT_CN') %>%
        dplyr::mutate(PROP_FOREST = case_when(ga == 1 ~ fa_ga,
                                              TRUE ~ fa)) %>%
        dplyr::select(PLT_CN, !!!aGrpSyms, PROP_FOREST) %>%
        as.data.frame()

    } else {
      # Forested land area w/out growth accounting
      a <- data %>%
        dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
        dtplyr::lazy_dt() %>%
        dplyr::group_by(PLT_CN, !!!aGrpSyms) %>%
        dplyr::summarize(PROP_FOREST = sum(CONDPROP_UNADJ * aDI, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        as.data.frame()
    }

    t <- data %>%
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dplyr::distinct(PLT_CN, TRE_CN, COMPONENT, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      # Compute estimates at plot level
      dplyr::group_by(!!!grpSyms, PLT_CN, REMPER) %>%
      dplyr::summarise(RECR_TPA = sum(TPARECR_UNADJ * tDI_r, na.rm = TRUE),
                       MORT_TPA = sum(TPAMORT_UNADJ * tDI, na.rm = TRUE),
                       REMV_TPA = sum(TPAREMV_UNADJ * tDI, na.rm = TRUE),
                       CURR_TPA = sum(TPA_UNADJ * tDI, na.rm = TRUE),
                       PREV_TPA = sum(TPA_UNADJ.prev * tDI, na.rm = TRUE)) %>%
      # Add back harvested trees' midpoint value (an annual rate, converted to
      # a period total), crediting growth up to the point of removal -- see
      # the matching comment on tPlot/pPlot in the population-estimation
      # branch above. No corresponding correction for MORT_TPA: mortality
      # trees get no midpoint credit, only the begin-value subtraction
      # already folded into PREV_TPA via TPA_UNADJ.prev.
      dplyr::mutate(CURR_TPA = CURR_TPA + (REMV_TPA * REMPER)) %>%
      dplyr::ungroup() %>%
      dplyr::mutate(CHNG_TPA = (CURR_TPA - PREV_TPA) / REMPER,
                    GROW_TPA = CHNG_TPA - RECR_TPA + MORT_TPA + REMV_TPA,
                    RECR_PERC = RECR_TPA / PREV_TPA * 100,
                    MORT_PERC = MORT_TPA / PREV_TPA * 100,
                    REMV_PERC = REMV_TPA / PREV_TPA * 100,
                    GROW_PERC = GROW_TPA / PREV_TPA * 100,
                    CHNG_PERC = CHNG_TPA / PREV_TPA * 100) %>%
      dplyr::select(PLT_CN, !!!grpSyms, REMPER, RECR_TPA:REMV_TPA, GROW_TPA, 
                    CHNG_TPA, RECR_PERC:CHNG_PERC, PREV_TPA, CURR_TPA) %>%
      as.data.frame() %>%
      # Rounding errors will generate tiny values, make them zero
      dplyr::mutate(GROW_TPA = dplyr::case_when(abs(GROW_TPA) < 1e-5 ~ 0,
                                                TRUE ~ GROW_TPA)) %>%
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

    # Forested area
    a <- data %>%
      # Will be lots of trees here, so CONDPROP listed multiple times
      # Adding PROP_BASIS so we can handle adjustment factors at strata level
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      dplyr::mutate(fa = CONDPROP_UNADJ * aDI) %>%
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!aGrpSyms, fa)

    if ('SUBP_COND_CHNG_MTRX' %in% names(db)) {
      # Plot-level estimates -- growth accounting
      a_ga <- aData %>%
        dtplyr::lazy_dt() %>%
        dplyr::filter(!is.na(PROP_BASIS)) %>%
        dplyr::group_by(PLT_CN, PROP_BASIS, CONDID, !!!aGrpSyms) %>%
        dplyr::summarize(fa_ga = sum(SUBPTYP_PROP_CHNG * aDI_ga, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        as.data.frame()

      # TODO: testing
      # test <- a %>%
      #   dplyr::left_join(dplyr::select(a_ga, PLT_CN, AREA_BASIS = PROP_BASIS, 
      #                                  CONDID, !!!aGrpSyms, fa_ga),
      #                    by = c('PLT_CN', 'AREA_BASIS', 'CONDID', aGrpBy)) %>%
      #   dplyr::left_join(plt.ga, by = 'PLT_CN') %>%
      #   dplyr::mutate(fa = case_when(ga == 1 ~ fa_ga,
      #                                TRUE ~ fa)) %>%
      #   dplyr::select(PLT_CN, AREA_BASIS, CONDID, !!!aGrpSyms, fa) %>%
      #   dplyr::filter(fa > 0)
      a <- a %>%
        dplyr::left_join(dplyr::select(a_ga, PLT_CN, AREA_BASIS = PROP_BASIS, 
                                       CONDID, !!!aGrpSyms, fa_ga),
                         by = c('PLT_CN', 'AREA_BASIS', 'CONDID', aGrpBy)) %>%
        dplyr::left_join(plt.ga, by = 'PLT_CN') %>%
        dplyr::mutate(fa = case_when(ga == 1 ~ fa_ga,
                                     TRUE ~ fa)) %>%
        dplyr::select(PLT_CN, AREA_BASIS, CONDID, !!!aGrpSyms, fa) %>%
        dplyr::filter(fa > 0)
    }

    grpSyms <- syms(grpBy)
    # Tree list
    t <- data %>%
      dplyr::distinct(PLT_CN, TRE_CN, .keep_all = TRUE) %>%
      # dtplyr::lazy_dt() %>%
      dplyr::filter(!is.na(SUBPTYP_GRM)) %>%
      dplyr::filter(tDI > 0 | tDI_r > 0) %>%
      # Compute estimates at plot level. Every term is coalesced to 0 (rather
      # than left as NA) at the row level: a state variable can be undefined
      # for a given tree/period for reasons unrelated to that row's actual
      # component (e.g. board-foot volume is only defined above the
      # sawtimber-size threshold), and without this, one column's NA (dropped
      # later via sum(..., na.rm = TRUE)) would silently misalign with
      # another column's non-NA value for the very same row, breaking the
      # CHNG = GROW + RECR - MORT - REMV identity at the aggregate level even
      # though each row's own arithmetic is internally consistent.
      dplyr::mutate(# Recruitment
                    rPlot = dplyr::coalesce(TPARECR_UNADJ * tDI_r, 0),
                    # Mortality
                    mPlot = dplyr::coalesce(TPAMORT_UNADJ * tDI, 0),
                    # Harvested
                    hPlot = dplyr::coalesce(TPAREMV_UNADJ * tDI, 0),
                    # T2 trees. hPlot*REMPER adds back harvested trees' midpoint
                    # value (converted from an annual rate to a period total),
                    # crediting their growth up to the point of removal --
                    # matching EVALIDator's growth-accounting SQL, which values
                    # CUT/DIVERSION components at their midpoint measurement on
                    # the "ending" side (see TPA_UNADJ.prev's comment above).
                    tPlot = dplyr::coalesce(TPA_UNADJ * tDI, 0) + (hPlot * REMPER),
                    # T1 trees. Survivors and CUT1/DIVERSION1/MORTALITY1 trees
                    # (present at T1, gone by T2) are valued at their begin
                    # measurement via TPA_UNADJ.prev; no separate mortality/
                    # removal correction is needed here.
                    pPlot = dplyr::coalesce(TPA_UNADJ.prev * tDI, 0),
                    # Change
                    cPlot = (tPlot - pPlot) / REMPER,
                    # Growth
                    gPlot = cPlot - rPlot + mPlot + hPlot) %>%
      dplyr::mutate(TREE_BASIS = case_when(SUBPTYP_GRM == 0 ~ NA_character_,
                                           SUBPTYP_GRM == 1 ~ 'SUBP',
                                           SUBPTYP_GRM == 2 ~ 'MICR',
                                           SUBPTYP_GRM == 3 ~ 'MACR')) %>%
      as.data.frame() %>%
      dplyr::select(PLT_CN, TREE_BASIS, SUBP, TREE, !!!grpSyms, rPlot:gPlot)

    if (treeList) {
      tEst <- a %>%
        dplyr::left_join(t, by = c('PLT_CN', aGrpBy)) %>%
        dplyr::mutate(EVAL_TYP = list(c('GROW', 'MORT', 'REMV'))) %>%
        dplyr::select(PLT_CN, EVAL_TYP, TREE_BASIS, AREA_BASIS,
                      !!!grpSyms, CONDID, SUBP, TREE,
                      RECR_TPA = rPlot,
                      MORT_TPA = mPlot,
                      REMV_TPA = hPlot,
                      GROW_TPA = gPlot,
                      CHNG_TPA = cPlot,
                      CURR_TPA = tPlot,
                      PREV_TPA = pPlot,
                      PROP_FOREST = fa)

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

      # Have to repeat this with tree totals as the denominator
      eu.sums <- sumToEU(db, dplyr::select(tPlt, -c(tPlot, pPlot)), 
                         dplyr::select(tPlt, -c(rPlot, mPlot, hPlot, gPlot, cPlot, tPlot)), 
                         pops, grpBy, grpBy, method, lambda)
      ttEst <- eu.sums$x %>%
        dplyr::select(ESTN_UNIT_CN, all_of(grpBy),
                      rPlot_cv_t = rPlot_cv, mPlot_cv_t = mPlot_cv, hPlot_cv_t = hPlot_cv,
                      gPlot_cv_t = gPlot_cv, cPlot_cv_t = cPlot_cv)
      tEst <- dplyr::left_join(tEst, ttEst, by = c('ESTN_UNIT_CN', grpBy))

      # Event-specific plot counts (growMort.Rd documents nPlots_RECR/
      # nPlots_MORT/nPlots_REMV/nPlots_GROW alongside nPlots_TREE, which
      # counts plots contributing to *any* event -- RECR/MORT/REMV/
      # survivor-growth combined). Re-uses sumToEU's existing strata/EU
      # plot-count machinery (nPlots.x, an unweighted count of distinct
      # PLT_CN, additive across strata and estimation units) rather than
      # reimplementing it, applied to a version of tPlt restricted to the
      # one relevant column and filtered to plots with a non-zero RECR/MORT/
      # REMV/GROW contribution respectively. tPlt itself never carries YEAR
      # (dropped by sumToPlot's final summarize); sumToEU adds it back
      # internally via its join to `pops`, so only the grpBy columns tPlt
      # actually has are selected here, while the full (YEAR-including)
      # grpBy is still passed as sumToEU's x.grpBy, matching the pattern
      # already used for tEst/ttEst above.
      tPltGrpBy <- grpBy[grpBy != 'YEAR']
      rEu <- sumToEU(db, dplyr::select(tPlt, ESTN_UNIT_CN, STRATUM_CN, PLT_CN, all_of(tPltGrpBy), rPlot) %>%
                        dplyr::filter(rPlot != 0),
                     NULL, pops, grpBy, NULL, method, lambda)
      mEu <- sumToEU(db, dplyr::select(tPlt, ESTN_UNIT_CN, STRATUM_CN, PLT_CN, all_of(tPltGrpBy), mPlot) %>%
                        dplyr::filter(mPlot != 0),
                     NULL, pops, grpBy, NULL, method, lambda)
      hEu <- sumToEU(db, dplyr::select(tPlt, ESTN_UNIT_CN, STRATUM_CN, PLT_CN, all_of(tPltGrpBy), hPlot) %>%
                        dplyr::filter(hPlot != 0),
                     NULL, pops, grpBy, NULL, method, lambda)
      # gPlot is a differenced quantity (cPlot - rPlot + mPlot + hPlot), so a
      # plot with no real survivor growth can still land a hair off zero from
      # floating-point noise -- same reasoning as the GROW_TPA rounding fix
      # below (abs(.x) < 1e-5 treated as zero), applied here to the filter
      # instead of after the fact.
      gEu <- sumToEU(db, dplyr::select(tPlt, ESTN_UNIT_CN, STRATUM_CN, PLT_CN, all_of(tPltGrpBy), gPlot) %>%
                        dplyr::filter(abs(gPlot) >= 1e-5),
                     NULL, pops, grpBy, NULL, method, lambda)
      ePlt <- rEu$x %>%
        dplyr::select(ESTN_UNIT_CN, all_of(grpBy), nPlots_RECR = nPlots.x) %>%
        dplyr::full_join(dplyr::select(mEu$x, ESTN_UNIT_CN, all_of(grpBy), nPlots_MORT = nPlots.x),
                         by = c('ESTN_UNIT_CN', grpBy)) %>%
        dplyr::full_join(dplyr::select(hEu$x, ESTN_UNIT_CN, all_of(grpBy), nPlots_REMV = nPlots.x),
                         by = c('ESTN_UNIT_CN', grpBy)) %>%
        dplyr::full_join(dplyr::select(gEu$x, ESTN_UNIT_CN, all_of(grpBy), nPlots_GROW = nPlots.x),
                         by = c('ESTN_UNIT_CN', grpBy))
      tEst <- dplyr::left_join(tEst, ePlt, by = c('ESTN_UNIT_CN', grpBy))

      out <- list(tEst = tEst, aEst = aEst, grpBy = grpBy, aGrpBy = aGrpBy)

    }
  }
  return(out)
}
