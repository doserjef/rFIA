dwmStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                       returnSpatial = FALSE, landType = 'forest', 
                       method = 'TI', lambda = 0.5, areaDomain = NULL, 
                       byPlot = FALSE, condList = FALSE, totals = FALSE, 
                       byFuelType = TRUE, nCores = 1, remote, mr) {

  # Read required data and prep the database ------------------------------
  reqTables <- c('PLOT', 'COND_DWM_CALC', 'COND', 'POP_PLOT_STRATUM_ASSGN',
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

  # Some better names
  db$COND_DWM_CALC <- db[['COND_DWM_CALC']] %>%
    dplyr::mutate(DWM_CN = CN)
  db$COND <- db[['COND']] %>%
    dplyr::mutate(COND_CN = CN)

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

  # Handle population tables ----------------------------------------------
  # Filtering out all inventories that are not relevant to the current 
  # estimation type. If using estimator other than TI, handle the differences in 
  # P2POINTCNT and in assigning YEAR column (YEAR = END_INVYR if method = 'TI')
  # TODO: need to go through handlePops
  pops <- handlePops(db, evalType = c('DWM'), method, mr)

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
    dplyr::select(PLT_CN, pltID, STATECD, MACRO_BREAKPOINT_DIA, INVYR, MEASYEAR, 
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

  # COND_DWM_CALC ---------------------
  db$COND_DWM_CALC <- db$COND_DWM_CALC %>%
    dplyr::select(-c(STATECD, COUNTYCD, UNITCD, INVYR, MEASYEAR, PLOT, EVALID)) %>%
    # Drop visits not used in our eval of interest
    dplyr::filter(PLT_CN %in% pops$PLT_CN)

  # Full condition list
  data <- db$PLOT %>%
    dplyr::left_join(db$COND, by = c('PLT_CN')) %>%
    dplyr::left_join(db$COND_DWM_CALC, by = c('PLT_CN', 'CONDID'))

  # Comprehensive indicator function
  data$aDI <- data$landD * data$aD * data$sp

  # Plot-level summaries --------------------------------------------------
  if (byPlot & !condList) {
    grpBy <- c('YEAR', grpBy)
    # Create a list of symbols for the grpBy statements
    grpSyms <- rlang::syms(grpBy)

    # Plot-level estimates
    # DWM estimates
    t <- data %>%
      dplyr::mutate(YEAR = MEASYEAR) %>%
      dplyr::distinct(PLT_CN, CONDID, .keep_all = TRUE) %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(!!!grpSyms, PLT_CN) %>%
      dplyr::summarize(VOL_1HR = sum(FWD_SM_VOLCF_ADJ * aDI, na.rm = TRUE), 
                       VOL_10HR = sum(FWD_MD_VOLCF_ADJ * aDI, na.rm = TRUE), 
                       VOL_100HR = sum(FWD_LG_VOLCF_ADJ * aDI, na.rm = TRUE), 
                       VOL_1000HR = sum(CWD_VOLCF_ADJ * aDI, na.rm = TRUE), 
                       VOL_PILE = sum(PILE_VOLCF_ADJ * aDI, na.rm = TRUE), 
                       # 2000 converts from pounds to short tons
                       BIO_DUFF = sum(DUFF_BIOMASS * aDI / 2000, na.rm = TRUE), 
                       BIO_LITTER = sum(LITTER_BIOMASS * aDI / 2000, na.rm = TRUE), 
                       BIO_1HR = sum(FWD_SM_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE), 
                       BIO_10HR = sum(FWD_MD_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE), 
                       BIO_100HR = sum(FWD_LG_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE),
                       BIO_1000HR = sum(CWD_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE), 
                       BIO_PILE = sum(PILE_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE), 
                       CARB_DUFF = sum(DUFF_CARBON * aDI / 2000, na.rm = TRUE), 
                       CARB_LITTER = sum(LITTER_CARBON * aDI / 2000, na.rm = TRUE), 
                       CARB_1HR = sum(FWD_SM_CARBON_ADJ * aDI / 2000, na.rm = TRUE), 
                       CARB_10HR = sum(FWD_MD_CARBON_ADJ * aDI / 2000, na.rm = TRUE), 
                       CARB_100HR = sum(FWD_LG_CARBON_ADJ * aDI / 2000, na.rm = TRUE), 
                       CARB_1000HR = sum(CWD_CARBON_ADJ * aDI / 2000, na.rm = TRUE), 
                       CARB_PILE = sum(PILE_CARBON_ADJ * aDI / 2000, na.rm = TRUE), 
                       PROP_FOREST = sum(CONDPROP_UNADJ * aDI, na.rm = TRUE)
                       ) %>%
      dplyr::ungroup() %>%
      as.data.frame() %>%
      tidyr::pivot_longer(cols = -c(PLT_CN, !!!grpSyms, PROP_FOREST), 
                          names_to = c('.value', 'FUEL_TYPE'), 
                          names_sep = '_') %>%
      dplyr::rename(VOL_ACRE = VOL, 
                    BIO_ACRE = BIO, 
                    CARB_ACRE = CARB) %>%
      dplyr::relocate(PROP_FOREST, .after = CARB_ACRE) %>%
      dplyr::mutate(FUEL_TYPE = factor(FUEL_TYPE, levels = c('DUFF', 'LITTER', 
                                                             '1HR', '10HR', '100HR', 
                                                             '1000HR', 'PILE')))

    # If by fuel type, add to grpBy
    if (byFuelType) {
      grpBy <- c(grpBy, 'FUEL_TYPE')
      t <- t %>%
        dplyr::arrange(PLT_CN, !!!grpSyms, FUEL_TYPE)
    } else {
      # Otherwise, summarize over fuel types for totals
      t <- t %>%
        dplyr::ungroup() %>%
        dtplyr::lazy_dt() %>%
        dplyr::select(-c(FUEL_TYPE)) %>%
        dplyr::group_by(PLT_CN, !!!grpSyms) %>%
        dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>% 
        dplyr::ungroup() %>%
        as.data.frame()
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
    grpSyms <- rlang::syms(grpBy)

    # Condition list
    a <- data %>% 
      dplyr::select(PLT_CN, PROP_BASIS, CONDID, CONDPROP_UNADJ, aDI, !!!grpSyms) %>%
      # Adding PROP_BASIS so we can handle adjustment factors at strata level.
      dplyr::distinct() %>% 
      # TODO: add CONDPROP_UNADJ to iPad notes
      dplyr::mutate(fa = CONDPROP_UNADJ * aDI) %>% 
      dplyr::select(PLT_CN, AREA_BASIS = PROP_BASIS, CONDID, !!!grpSyms, fa)

    # If returning a condition list ready to be handed to customPSE
    if (condList) {

      tPlt <- data %>%
        dplyr::distinct(PLT_CN, CONDID, COND_STATUS_CD, .keep_all = TRUE) %>%
        dtplyr::lazy_dt() %>%
        dplyr::mutate(VOL_1HR = FWD_SM_VOLCF_ADJ * aDI, 
                      VOL_10HR = FWD_MD_VOLCF_ADJ * aDI, 
                      VOL_100HR = FWD_LG_VOLCF_ADJ * aDI,
                      VOL_1000HR = CWD_VOLCF_ADJ * aDI,
                      VOL_PILE = PILE_VOLCF_ADJ * aDI,
                      BIO_DUFF = DUFF_BIOMASS* aDI / 2000,
                      BIO_LITTER = LITTER_BIOMASS * aDI / 2000,
                      BIO_1HR = FWD_SM_DRYBIO_ADJ * aDI / 2000,
                      BIO_10HR = FWD_MD_DRYBIO_ADJ * aDI / 2000,
                      BIO_100HR = FWD_LG_DRYBIO_ADJ * aDI / 2000,
                      BIO_1000HR = CWD_DRYBIO_ADJ * aDI / 2000,
                      BIO_PILE = PILE_DRYBIO_ADJ * aDI / 2000,
                      CARB_DUFF = DUFF_CARBON* aDI / 2000,
                      CARB_LITTER = LITTER_CARBON * aDI / 2000,
                      CARB_1HR = FWD_SM_CARBON_ADJ * aDI / 2000,
                      CARB_10HR = FWD_MD_CARBON_ADJ * aDI / 2000,
                      CARB_100HR = FWD_LG_CARBON_ADJ * aDI / 2000,
                      CARB_1000HR = CWD_CARBON_ADJ * aDI / 2000,
                      CARB_PILE = PILE_CARBON_ADJ * aDI / 2000) %>%
        dplyr::select(PLT_CN, CONDID, !!!grpSyms, VOL_1HR:CARB_PILE) %>%
        as.data.frame() %>%
        tidyr::pivot_longer(cols = -c(PLT_CN, CONDID, !!!grpSyms),
                            names_to = c('.value', 'FUEL_TYPE'),
                            names_sep = '_') %>%
        dplyr::mutate(FUEL_TYPE = factor(FUEL_TYPE, levels = c('DUFF', 'LITTER',
                                                               '1HR', '10HR', '100HR',
                                                               '1000HR', 'PILE'))) %>%
        dplyr::left_join(a, by = c('PLT_CN', 'CONDID', grpBy)) %>%
        dplyr::rename(PROP_FOREST = fa)

      aGrpBy <- grpBy
      # If by fuel type, add to grpBy
      if (byFuelType) {
        grpBy <- c(grpBy, 'FUEL_TYPE')
        grpSyms <- rlang::syms(grpBy)
      } else {
        # Otherwise, summarize over fuel types for totals
        tPlt <- tPlt %>%
          dtplyr::lazy_dt() %>%
          dplyr::select(-c(FUEL_TYPE)) %>%
          dplyr::group_by(PLT_CN, CONDID, !!!grpSyms, AREA_BASIS, PROP_FOREST) %>%
          dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>%
          dplyr::ungroup() %>%
          as.data.frame()
      }
      # Reorder some columns
      tPlt <- tPlt %>%
        dplyr::mutate(EVAL_TYP = 'DWM') %>%
        dplyr::select(PLT_CN, EVAL_TYP, AREA_BASIS, !!!grpSyms, CONDID, 
                      VOL_ACRE = VOL, BIO_ACRE = BIO, CARB_ACRE = CARB, 
                      PROP_FOREST)

    } else {
      # If a condition list is not desired, let's move to population estimation.

      # Sum area variable up to plot-level and adjust for non-response. Note you 
      # don't have to do this for tPlt here, since all DWM variables have already 
      # been adjusted for non-response in COND_DWM_CALC.
      aPlt <- sumToPlot(a, pops, grpBy)

      # All DWM variables have already been adjusted for non-response, so we can 
      # just sum them up here instead of sending to sumToPlot
      tPlt <- data %>%
        dplyr::distinct(STRATUM_CN, PLT_CN, CONDID, COND_STATUS_CD, .keep_all = TRUE) %>%
        dtplyr::lazy_dt() %>%
        dplyr::group_by(STRATUM_CN, PLT_CN, !!!grpSyms) %>%
        dplyr::summarize(VOL_1HR = sum(FWD_SM_VOLCF_ADJ * aDI, na.rm = TRUE),
                         VOL_10HR = sum(FWD_MD_VOLCF_ADJ * aDI, na.rm = TRUE),
                         VOL_100HR = sum(FWD_LG_VOLCF_ADJ * aDI, na.rm = TRUE),
                         VOL_1000HR = sum(CWD_VOLCF_ADJ * aDI, na.rm = TRUE),
                         VOL_PILE = sum(PILE_VOLCF_ADJ * aDI, na.rm = TRUE),
                         BIO_DUFF = sum(DUFF_BIOMASS* aDI / 2000, na.rm = TRUE),
                         BIO_LITTER = sum(LITTER_BIOMASS * aDI / 2000, na.rm = TRUE),
                         BIO_1HR = sum(FWD_SM_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE),
                         BIO_10HR = sum(FWD_MD_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE),
                         BIO_100HR = sum(FWD_LG_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE),
                         BIO_1000HR = sum(CWD_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE),
                         BIO_PILE = sum(PILE_DRYBIO_ADJ * aDI / 2000, na.rm = TRUE),
                         CARB_DUFF = sum(DUFF_CARBON* aDI / 2000, na.rm = TRUE),
                         CARB_LITTER = sum(LITTER_CARBON * aDI / 2000, na.rm = TRUE),
                         CARB_1HR = sum(FWD_SM_CARBON_ADJ * aDI / 2000, na.rm = TRUE),
                         CARB_10HR = sum(FWD_MD_CARBON_ADJ * aDI / 2000, na.rm = TRUE),
                         CARB_100HR = sum(FWD_LG_CARBON_ADJ * aDI / 2000, na.rm = TRUE),
                         CARB_1000HR = sum(CWD_CARBON_ADJ * aDI / 2000, na.rm = TRUE),
                         CARB_PILE = sum(PILE_CARBON_ADJ * aDI / 2000, na.rm = TRUE)) %>%
        dplyr::ungroup() %>%
        dplyr::left_join(dplyr::distinct(dplyr::select(pops, STRATUM_CN, ESTN_UNIT_CN)), 
                         by = 'STRATUM_CN') %>%
        as.data.frame() %>%
        tidyr::pivot_longer(cols = -c(ESTN_UNIT_CN, STRATUM_CN, PLT_CN, !!!grpSyms),
                            names_to = c('.value', 'FUEL_TYPE'),
                            names_sep = '_') %>%
        dplyr::rename(VOL = VOL,
                      BIO = BIO,
                      CARB = CARB) %>%
        dplyr::mutate(FUEL_TYPE = factor(FUEL_TYPE, levels = c('DUFF', 'LITTER',
                                                               '1HR', '10HR', '100HR',
                                                               '1000HR', 'PILE')))
      aGrpBy <- grpBy
      # If by fuel type, add to grpBy
      if (byFuelType) {
        grpBy <- c(grpBy, 'FUEL_TYPE')
      } else {
        # Otherwise, summarize over fuel types for totals
        tPlt <- tPlt %>%
          dtplyr::lazy_dt() %>%
          dplyr::select(-c(FUEL_TYPE)) %>%
          dplyr::group_by(ESTN_UNIT_CN, STRATUM_CN, PLT_CN, !!!grpSyms) %>%
          dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>%
          dplyr::ungroup() %>%
          as.data.frame()
      }

      # Adding YEAR to groups
      grpBy <- c('YEAR', grpBy)
      aGrpBy <- c('YEAR', aGrpBy)

      # Sum variables up to strata then estimation unit level
      eu.sums <- sumToEU(db, tPlt, aPlt, pops, grpBy, aGrpBy, method)
      tEst <- eu.sums$x
      aEst <- eu.sums$y

      out <- list(tEst = tEst, aEst = aEst, grpBy = grpBy, aGrpBy = aGrpBy)
    }
  }
  return(out)
}
