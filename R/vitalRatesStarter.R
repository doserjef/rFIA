vitalRatesStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                              returnSpatial = FALSE, bySpecies = FALSE, 
                              bySizeClass = FALSE, landType = 'forest', 
                              treeType = 'all', method = 'TI', lambda = 0.5, 
                              treeDomain = NULL, areaDomain = NULL, 
                              totals = FALSE, byPlot = FALSE, treeList = FALSE, 
                              nCores = 1, remote, mr) {

  # Read required data and prep the database ------------------------------
  reqTables <- c('PLOT', 'TREE', 'COND', 'TREE_GRM_COMPONENT', 'TREE_GRM_MIDPT', 
                 'TREE_GRM_BEGIN', 'SUBP_COND_CHNG_MTRX', 'POP_PLOT_STRATUM_ASSGN', 
                 'POP_ESTN_UNIT', 'POP_EVAL', 'POP_STRATUM', 'POP_EVAL_TYP', 
                 'POP_EVAL_GRP')

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
  if (treeType %in% c('live', 'gs', 'all') == FALSE) {
    stop('treeType must be one of: "live", "gs", or "all".')
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
  db$TREE <- db[['TREE']] %>% 
    dplyr::mutate(TRE_CN = CN)

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
  # Land type and tree type combined
  db <- typeDomain_grow(db, treeType, landType, type = 'vr')

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
  pops <- handlePops(db, evalType = c('GROW'), method, mr)

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
    db$TREE <- db$TREE %>%
      dplyr::left_join(dplyr::select(intData$REF_SPECIES_DEC_2024, 
                                     c('SPCD', 'COMMON_NAME', 'GENUS', 'SPECIES')), 
                       by = 'SPCD') %>%
      dplyr::mutate(SCIENTIFIC_NAME = paste(GENUS, SPECIES, sep = ' ')) %>% 
      dplyr::mutate_if(is.factor, as.character)
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
    # Dropping irrelevant rows and columns
    dplyr::select(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA, INVYR, MEASYEAR, 
                  PLOT_STATUS_CD, dplyr::all_of(grpP), sp, COUNTYCD, PREV_PLT_CN, 
                  REMPER) %>% 
    # Drop non-forested plots, and those otherwise outside our domain of interest.
    dplyr::filter(PLOT_STATUS_CD == 1 & sp == 1) %>% 
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)
  # COND ------------------------------
  db$COND <- db$COND %>% 
    dplyr::select(PLT_CN, CONDPROP_UNADJ, PROP_BASIS, COND_STATUS_CD, CONDID, 
                  dplyr::all_of(grpC), aD, landD) %>% 
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))

  # TREE ------------------------------
  db$TREE <- db$TREE %>%
    dplyr::select(PLT_CN, CONDID, PREVCOND, TRE_CN, PREV_TRE_CN, SUBP, TREE, 
                  dplyr::all_of(grpT), tD, typeD, DIA, DRYBIO_AG, VOLCFNET, 
                  VOLBFNET, STATUSCD) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))

  # TREE_GRM_COMPONENT ----------------
  # Note these column names are created in typeDomain_grow to assign the 
  # correct value to TPAGROW_UNADJ, TPAREMV_UNADJ, TPAMORT_UNADJ, COMPONENT
  db$TREE_GRM_COMPONENT <- db$TREE_GRM_COMPONENT %>%
    dplyr::select(c(TRE_CN, SUBPTYP_GRM, TPAGROW_UNADJ,
                    TPAREMV_UNADJ, TPAMORT_UNADJ, COMPONENT)) %>%
    dplyr::filter(TRE_CN %in% db$TREE$TRE_CN)
  # TREE_GRM_MIDPT --------------------
  db$TREE_GRM_MIDPT <- db$TREE_GRM_MIDPT %>%
    dplyr::select(c(TRE_CN, DIA, VOLCFNET, VOLBFNET, DRYBIO_AG)) %>%
    dplyr::filter(TRE_CN %in% db$TREE$TRE_CN)
  # TREE_GRM_BEGIN --------------------
  db$TREE_GRM_BEGIN <- db$TREE_GRM_BEGIN %>%
    dplyr::select(c(TRE_CN, DIA, VOLCFNET, VOLBFNET, DRYBIO_AG)) %>%
    dplyr::filter(TRE_CN %in% db$TREE$TRE_CN)
  # SUBP_COND_CHNG_MTRX ---------------
  db$SUBP_COND_CHNG_MTRX <- db$SUBP_COND_CHNG_MTRX %>%
    dplyr::select(PLT_CN, PREV_PLT_CN, SUBPTYP, SUBPTYP_PROP_CHNG, PREVCOND, CONDID) %>%
    dplyr::filter(PLT_CN %in% c(db$PLOT$PLT_CN, db$PLOT$PREV_PLT_CN))

  # Separate area grouping names from tree grouping names
  if (!is.null(polys)) {
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND), names(polys))]
  } else {
    aGrpBy <- grpBy[grpBy %in% c(names(db$PLOT), names(db$COND))]
  }

  # Full tree list
  data <- db$PLOT %>% 
    dtplyr::lazy_dt() %>%
    # Grab needed columns in PLOT
    dplyr::select(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA, INVYR, 
                  MEASYEAR, PLOT_STATUS_CD, PREV_PLT_CN, REMPER, 
                  dplyr::all_of(grpP), sp) %>%
    # Join with COND 
    dplyr::left_join(dplyr::select(db$COND, c(PLT_CN, CONDPROP_UNADJ, PROP_BASIS, 
                                              COND_STATUS_CD, CONDID, 
                                              dplyr::all_of(grpC), aD, landD)), 
                     by = 'PLT_CN') %>%
    # Join with TREE
    dplyr::left_join(dplyr::select(db$TREE, c(PLT_CN, CONDID, PREVCOND, TRE_CN, 
                                              PREV_TRE_CN, SUBP, TREE, 
                                              dplyr::all_of(grpT), tD, typeD, DIA, 
                                              DRYBIO_AG, VOLCFNET, VOLBFNET, 
                                              STATUSCD)), 
                     by = c('PLT_CN', 'CONDID')) %>%
    # Join with TREE_GRM_COMPONENT
    dplyr::left_join(dplyr::select(db$TREE_GRM_COMPONENT, c(TRE_CN, SUBPTYP_GRM, 
                                                            TPAGROW_UNADJ, TPAREMV_UNADJ, 
                                                            TPAMORT_UNADJ, COMPONENT)), 
                     by = c('TRE_CN')) %>%
    # Join with TREE_GRM_MIDPT
    dplyr::left_join(dplyr::select(db$TREE_GRM_MIDPT, c(TRE_CN, DIA, VOLCFNET, 
                                                        VOLBFNET, DRYBIO_AG)), 
                     by = c('TRE_CN'), suffix = c('', '.mid')) %>%
    # Join with TREE_GRM_BEGIN
    dplyr::left_join(dplyr::select(db$TREE_GRM_BEGIN, c(TRE_CN, DIA, VOLCFNET, 
                                                        VOLBFNET, DRYBIO_AG)), 
                     by = c('TRE_CN'), suffix = c('', '.beg')) %>%
    # Join PREV_PLT_CN with corresponding information in PLOT
    dplyr::left_join(dplyr::select(db$PLOT, c(PLT_CN, dplyr::all_of(grpP), sp)), 
                     by = c('PREV_PLT_CN' = 'PLT_CN'), 
                     suffix = c('', '.prev')) %>%
    # Join COND with previous condition
    dplyr::left_join(dplyr::select(db$COND, c(PLT_CN, CONDID, landD, aD, 
                                              dplyr::all_of(grpC), COND_STATUS_CD)), 
                     by = c('PREV_PLT_CN' = 'PLT_CN', 'PREVCOND' = 'CONDID'), 
                     suffix = c('', '.prev')) %>%
    # Join TREE with previous tree measurement 
    dplyr::left_join(dplyr::select(db$TREE, c(TRE_CN, dplyr::all_of(grpT), typeD, 
                                              tD, DIA, DRYBIO_AG, VOLCFNET, VOLBFNET, 
                                              STATUSCD)), 
                     by = c('PREV_TRE_CN' = 'TRE_CN'), suffix = c('', '.prev')) %>%
    # Some domain indicators
    dplyr::mutate(aChng = dplyr::case_when(COND_STATUS_CD == 1 & 
                                           COND_STATUS_CD.prev == 1 & 
                                           !is.null(CONDPROP_UNADJ) ~ 1, 
                                           TRUE ~ 0), 
                  tChng = dplyr::case_when(COND_STATUS_CD == 1 & 
                                           COND_STATUS_CD.prev == 1 ~ 1, 
                                           TRUE ~ 0), 
                  landD.prev = dplyr::case_when(landD == 1 & landD.prev == 1 ~ 1, 
                                                TRUE ~ 0), 
                  status = case_when(COMPONENT == 'SURVIVOR' ~ 1, 
                                     TRUE ~ 0)) %>%
    # If previous attributes are unavailable for trees, default to current
    dplyr::mutate(tD.prev = dplyr::case_when(is.na(tD.prev) ~ tD, TRUE ~ tD.prev), 
                  typeD.prev = dplyr::case_when(is.na(typeD.prev) ~ typeD, TRUE ~ typeD.prev), 
                  aD.prev = dplyr::case_when(is.na(aD.prev) ~ aD, TRUE ~ aD.prev), 
                  sp.prev = dplyr::case_when(is.na(sp.prev) ~ sp, TRUE ~ sp.prev)) %>%
    # Comprehensive domain indicators
    dplyr::mutate(tDI = landD.prev * aD.prev * tD.prev * typeD.prev * sp.prev * tChng, 
                  aDI = landD.prev * aD * sp * aChng) %>%
    as.data.frame() %>%
    distinct()

  # Adjust tree domain indicator if only considering live trees
  if (tolower(treeType) == 'live') {
    data$tDI <- data$tDI * data$status
  }
                                                     
  # Modify attributes depending on the component (mortality uses midpoint)
  data <- data %>%
    dplyr::mutate(DIA2 = vrAttHelper(DIA, DIA.prev, DIA.mid, DIA.beg, COMPONENT, REMPER, 2) * tDI, 
                  DIA1 = vrAttHelper(DIA, DIA.prev, DIA.mid, DIA.beg, COMPONENT, REMPER, 1) * tDI, 
                  BA2 = vrAttHelper(basalArea(DIA), basalArea(DIA.prev), basalArea(DIA.mid), 
                                    basalArea(DIA.beg), COMPONENT, REMPER, 2) * tDI, 
                  BA1 = vrAttHelper(basalArea(DIA), basalArea(DIA.prev), 
                                    basalArea(DIA.mid), basalArea(DIA.beg), 
                                    COMPONENT, REMPER, 1) * tDI,
                  VOLCFNET2 = vrAttHelper(VOLCFNET, VOLCFNET.prev, VOLCFNET.mid, 
                                          VOLCFNET.beg, COMPONENT, REMPER, 2) * tDI,
                  VOLCFNET1 = vrAttHelper(VOLCFNET, VOLCFNET.prev, VOLCFNET.mid, 
                                          VOLCFNET.beg, COMPONENT, REMPER, 1) * tDI,
                  VOLBFNET2 = vrAttHelper(VOLBFNET, VOLBFNET.prev, VOLBFNET.mid, 
                                          VOLBFNET.beg, COMPONENT, REMPER, 2) * tDI,
                  VOLBFNET1 = vrAttHelper(VOLBFNET, VOLBFNET.prev, VOLBFNET.mid, 
                                          VOLBFNET.beg, COMPONENT, REMPER, 1) * tDI,
                  DRYBIO_AG2 = vrAttHelper(DRYBIO_AG, DRYBIO_AG.prev, DRYBIO_AG.mid, 
                                           DRYBIO_AG.beg, COMPONENT, REMPER, 2) * tDI,
                  DRYBIO_AG1 = vrAttHelper(DRYBIO_AG, DRYBIO_AG.prev, DRYBIO_AG.mid, 
                                           DRYBIO_AG.beg, COMPONENT, REMPER, 1) * tDI) %>%
    dplyr::select(-c(DIA.mid, VOLCFNET.mid, VOLBFNET.mid, DRYBIO_AG.mid,
                     DIA.prev, VOLCFNET.prev, VOLBFNET.prev, DRYBIO_AG.prev,
                     DIA.beg, VOLCFNET.beg, VOLBFNET.beg, DRYBIO_AG.beg,
                     DIA, VOLCFNET, VOLBFNET, DRYBIO_AG))
                  
  # Only grab what's needed
  data <- data %>%
    dplyr::select(PLT_CN, TRE_CN, SUBP, CONDID, TREE, tDI, grpP, grpC, grpT, 
                  TPAGROW_UNADJ, PROP_BASIS, SUBPTYP_GRM, PLOT_STATUS_CD, 
                  DIA2, DIA1, BA2, BA1, DRYBIO_AG2, DRYBIO_AG1, VOLCFNET2, 
                  VOLCFNET1, VOLBFNET2, VOLBFNET1, MEASYEAR) %>%
    # Rearrange previous values as observations
    tidyr::pivot_longer(cols = DIA2:VOLBFNET1, 
                        names_to = c('.value', 'ONEORTWO'), 
                        names_sep = -1)

  # Doing area separately now for growth accounting plots
  aData <- db$PLOT %>%
    # Get necessary columns from PLOT
    dplyr::select(c(PLT_CN, STATECD, MACRO_BREAKPOINT_DIA, INVYR, MEASYEAR,
                    PLOT_STATUS_CD, PREV_PLT_CN, REMPER, dplyr::all_of(grpP), sp)) %>%
    # join with SUBP_COND_CHNG_MTRX
    dplyr::left_join(dplyr::select(db$SUBP_COND_CHNG_MTRX, PLT_CN, PREV_PLT_CN,
                                   SUBPTYP, SUBPTYP_PROP_CHNG, PREVCOND, CONDID),
                     by = c('PLT_CN', 'PREV_PLT_CN')) %>%
    # Join with COND
    dplyr::left_join(dplyr::select(db$COND, c(PLT_CN, CONDPROP_UNADJ, PROP_BASIS,
                                              COND_STATUS_CD, CONDID, dplyr::all_of(grpC),
                                              aD, landD)),
                     by = c('PLT_CN', 'CONDID')) %>%
    # Join COND with PREV_PLT_CN and PREVCOND
    dplyr::left_join(dplyr::select(db$COND, c(PLT_CN, PROP_BASIS, COND_STATUS_CD,
                                              CONDID, dplyr::all_of(grpC), aD, landD)),
                     by = c('PREV_PLT_CN' = 'PLT_CN', 'PREVCOND' = 'CONDID'), suffix = c('', '.prev')) %>%
    dplyr::mutate(aChng = dplyr::case_when(COND_STATUS_CD == 1 & 
                                           COND_STATUS_CD.prev == 1 & 
                                           !is.null(CONDPROP_UNADJ) & 
                                           SUBPTYP == 1 ~ 1 ,
                                           TRUE ~ 0),
                  # Multiply by .25 since doing woring at plot level
                  SUBPTYP_PROP_CHNG = SUBPTYP_PROP_CHNG * .25) %>%
    ## Comprehensive domain indicator
    dplyr::mutate(aDI = landD * landD.prev * aD * sp * aChng)

  # Plot-level summaries --------------------------------------------------
  if (byPlot & !treeList) {
    grpBy <- c('YEAR', grpBy)
    # Create a list of symbols for the grpBy statements
    grpSyms <- rlang::syms(grpBy)
    aGrpSyms <- rlang::syms(aGrpBy)

    # Plot-level estimates
    # Area
    a <- aData %>% 
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      # Convert to a lazy data table for quicker analysis.
      dtplyr::lazy_dt() %>% 
      # Note the use of !!! to inject aGrpSyms back into an evaluation context.
      dplyr::group_by(PLT_CN, !!!aGrpSyms) %>%
      # Calculate proportion of area in the plot that meets the current area domain 
      dplyr::summarize(PROP_FOREST = sum(SUBPTYP_PROP_CHNG * aDI, na.rm = TRUE)) %>%
      # Convert to data frame
      as.data.frame()

    t <- data %>%
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dplyr::distinct(PLT_CN, SUBP, TREE, ONEORTWO, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(!!!grpSyms, PLT_CN) %>%
      # Note only previous plots used in t calculation
      dplyr::summarize(t = sum(TPAGROW_UNADJ[ONEORTWO == 1] * tDI[ONEORTWO == 1], na.rm = TRUE), 
                       d = sum(DIA * TPAGROW_UNADJ * tDI, na.rm = TRUE), 
                       ba = sum(BA * TPAGROW_UNADJ * tDI, na.rm = TRUE), 
                       vol = sum(VOLCFNET * TPAGROW_UNADJ * tDI, na.rm = TRUE),
                       svol = sum(VOLBFNET * TPAGROW_UNADJ * tDI, na.rm = TRUE) / 1000,
                       bio = sum(DRYBIO_AG * TPAGROW_UNADJ * tDI, na.rm = TRUE)) %>%
      dplyr::mutate(DIA_GROW = d / t,
                    BA_GROW = ba / t,
                    NETVOL_GROW = vol / t,
                    SAWVOL_GROW = svol / t,
                    BIO_GROW = bio / t,
                    BAA_GROW = ba,
                    NETVOL_GROW_AC = vol,
                    SAWVOL_GROW_AC = svol,
                    BIO_GROW_AC = bio,
                    PREV_TPA = t) %>%
      dplyr::select(-c(t:bio)) %>%
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
  # Population estimation or prep for it (treeList) -----------------------
    # Create a list of symbols for the grpBy statements
    aGrpSyms <- rlang::syms(aGrpBy)

    # Condition list
    a <- aData %>% 
      # Will be lots of trees here, so CONDPROP is listed multiple times, the 
      # distinct is needed to just get those distinct ones. 
      # Adding PROP_BASIS so we can handle adjustment factors at strata level.
      dplyr::mutate(fa = SUBPTYP_PROP_CHNG * aDI) %>% 
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!aGrpSyms, fa)


    grpSyms <- syms(grpBy)
    # Tree list
    t <- data %>%
      dplyr::distinct(PLT_CN, SUBP, TREE, ONEORTWO, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      dplyr::mutate(tPlot = dplyr::case_when(ONEORTWO == 1 ~ TPAGROW_UNADJ * tDI,
                                             TRUE ~ 0), # Previous only
                    dPlot = DIA * TPAGROW_UNADJ * tDI,
                    bPlot = BA * TPAGROW_UNADJ * tDI,
                    gPlot = VOLCFNET * TPAGROW_UNADJ * tDI,
                    sPlot = VOLBFNET * TPAGROW_UNADJ * tDI / 1000,
                    bioPlot = DRYBIO_AG * TPAGROW_UNADJ * tDI / 2000) %>%
      # Need a code that tells us where the tree was measured
      # macroplot, microplot, subplot
      dplyr::mutate(TREE_BASIS = case_when(SUBPTYP_GRM == 0 ~ NA_character_,
                                           SUBPTYP_GRM == 1 ~ 'SUBP',
                                           SUBPTYP_GRM == 2 ~ 'MICR',
                                           SUBPTYP_GRM == 3 ~ 'MACR')) %>%
      dplyr::filter(!is.na(TREE_BASIS)) %>%
      dplyr::select(PLT_CN, TREE_BASIS, SUBP, TREE, ONEORTWO, !!!grpSyms, tPlot:bioPlot) %>%
      as.data.frame()

      # Return a tree list ready to be handed to customPSE()
    if (treeList) {

      tEst <- a %>%
        dtplyr::lazy_dt() %>%
        dplyr::group_by(PLT_CN, CONDID, !!!aGrpSyms, AREA_BASIS) %>%
        dplyr::summarize(fa = sum(fa, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        as.data.frame() %>%
        dplyr::left_join(t, by = c('PLT_CN', aGrpBy)) %>%
        dplyr::mutate(EVAL_TYP = 'GROW') %>%
        dplyr::select(PLT_CN, EVAL_TYP, TREE_BASIS, AREA_BASIS,
                      !!!grpSyms, CONDID, SUBP, TREE, ONEORTWO,
                      DIA_GROW = dPlot,
                      BAA_GROW = bPlot,
                      NETVOL_GROW = gPlot,
                      SAWVOL_GROW = sPlot,
                      BIO_GROW = bioPlot,
                      PREV_TPA = tPlot,
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
      eu.sums <- sumToEU(db, dplyr::select(tPlt, -c(tPlot)), dplyr::select(tPlt, -c(dPlot:bioPlot)), 
                         pops, grpBy, grpBy, method, lambda)
      ttEst <- eu.sums$x %>%
        dplyr::select(ESTN_UNIT_CN, all_of(grpBy), dPlot_cv_t = dPlot_cv,
                      gPlot_cv_t = gPlot_cv, bPlot_cv_t = bPlot_cv,
                      sPlot_cv_t = sPlot_cv, bioPlot_cv_t = bioPlot_cv)
      tEst <- dplyr::left_join(tEst, ttEst, by = c('ESTN_UNIT_CN', grpBy))

      out <- list(tEst = tEst, aEst = aEst, grpBy = grpBy, aGrpBy = aGrpBy)
    }
  }
}
