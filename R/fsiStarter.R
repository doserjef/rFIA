fsiStarter <- function(x, db, grpBy_quo = NULL, scaleBy_quo = NULL, polys = NULL,
                       returnSpatial = FALSE, bySpecies = FALSE, bySizeClass = FALSE,
                       landType = 'forest', treeType = 'live', method = 'TI',
                       lambda = .5, treeDomain = NULL, areaDomain = NULL,
                       totals = FALSE, byPlot = FALSE, useSeries = FALSE,
                       mostRecent = FALSE, nCores = 1, remote, mr){


  # Read required data, prep the database -------------------------------------
  reqTables <- c('PLOT', 'TREE', 'COND', 'POP_PLOT_STRATUM_ASSGN',
                 'POP_ESTN_UNIT', 'POP_EVAL', 'POP_STRATUM', 'POP_EVAL_TYP', 
                 'POP_EVAL_GRP', 'PLOTGEOM')

  # If remote, read in state by state. Otherwise, drop all unnecessary tables
  db <- readRemoteHelper(x, db, remote, reqTables, nCores)

  # Handle TX issues: we only keep inventory years that are present in 
  # BOTH EAST AND WEST TX
  db <- handleTX(db)

  # Check some of the inputs ----------------------------------------------
  # polys -----------------------------
  if (!is.null(polys) &
      first(class(polys)) %in%
      c('sf', 'SpatialPolygons', 'SpatialPolygonsDataFrame') == FALSE){
    stop('polys must be spatial polygons object of class sp or sf. ')
  }
  # landType --------------------------
  if (landType %in% c('timber', 'forest') == FALSE){
    stop('landType must be one of: "forest" or "timber".')
  }
  # treeType --------------------------
  if (treeType %in% c('live', 'dead', 'gs', 'all') == FALSE){
    stop('treeType must be one of: "live", "dead", "gs", or "all".')
  }
  # db required tables ----------------
  if (any(reqTables %in% names(db) == FALSE)){
    missT <- reqTables[reqTables %in% names(db) == FALSE]
    stop(paste('Tables', paste (as.character(missT), collapse = ', '),
               'not found in object db.'))
  }
  # method ----------------------------
  if (str_to_upper(method) %in% c('TI', 'SMA', 'LMA', 'EMA', 'ANNUAL') == FALSE) {
    warning(paste('Method', method,
                  'unknown. Defaulting to Temporally Indifferent (TI).'))
  }


  # Other basic variable prep ---------------------------------------------
  # Join PLOT with PLOTGEOM to allow plot-level geographic attributes to be used
  # in grpBy statements
  db$PLOTGEOM <- db$PLOTGEOM %>%
    dplyr::select(-STATECD, -INVYR, -UNITCD, -COUNTYCD, -PLOT, -LAT, -LON, 
                  -dplyr::starts_with('CREATED'), -dplyr::starts_with('MODIFIED'))
  db$PLOT <- db$PLOT %>%
    dplyr::left_join(db$PLOTGEOM, by = 'CN')

  # Get a plot CN and a new pltID that gives a unique ID to each plot
  # PLT_CN is UNITCD, STATECD, COUNTYCD, PLOT, and INVYR
  db$PLOT <- db$PLOT %>%
    mutate(PLT_CN = CN,
           pltID = paste(UNITCD, STATECD, COUNTYCD, PLOT, sep = '_'))
  db$TREE <- db$TREE %>%
    mutate(TRE_CN = CN)

  # Convert grpBy to character
  grpBy <- grpByToChar(db, grpBy_quo)

  # Convert scaleBy to character
  scaleBy <- grpByToChar(db[names(db) %in% 'TREE' == FALSE], scaleBy_quo)

  # Save original grpBy for pretty return with spatial objects
  grpByOrig <- grpBy

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
    # sfCol <- attr(polys, 'sf_column')
    # grpBy <- c(grpBy, names(polys)[names(polys) != sfCol])
    grpBy = c(grpBy, 'polyID')

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
  # Tree type
  db$TREE$typeD <- treeTypeDomain(treeType, db$TREE$STATUSCD,
                                  db$TREE$DIA, db$TREE$TREECLCD)

  # Spatial boundary
  if (!is.null(polys)){
    db$PLOT$sp <- ifelse(!is.na(db$PLOT$polyID), 1, 0)
  } else {
    db$PLOT$sp <- 1
  }

  # User defined domain indicator for area (ex. specific forest type)
  db <- udAreaDomain(db, areaDomain)

  # User defined domain indicator for tree (ex. trees > 20 ft tall)
  db <- udTreeDomain(db, treeDomain)

  # Handle population tables --------------------------------------------------
  # We only want inventory/ population info from t2 plots, but we need the plot tree cond data
  # for t1 and t2
  remPlts <- db$PLOT %>%
    select(PLT_CN, PREV_PLT_CN, DESIGNCD, REMPER, PLOT_STATUS_CD) %>%
    # Has to have a remeasurement, be in the current sample, and of the national design
    filter(!is.na(REMPER) & !is.na(PREV_PLT_CN) & PLOT_STATUS_CD != 3 & DESIGNCD %in% c(1, 501:505)) %>%
    left_join(select(db$PLOT, PLT_CN, DESIGNCD, PLOT_STATUS_CD), by = c('PREV_PLT_CN' = 'PLT_CN'), suffix = c('', '.prev')) %>%
    # Past remasurement must be in the previous sample and of national design
    filter(PLOT_STATUS_CD.prev != 3 & DESIGNCD.prev %in% c(1, 501:505))

  # Filtering out all inventories that are not relevant to the current estimation
  # type. If using estimator other than TI, handle the differences in P2POINTCNT
  # and in assigning YEAR column (YEAR = END_INVYR if method = 'TI')
  # TODO: test this to see if handlePops works instead of handlePops_old. 
  # TODO: left off here. 
  pops <- handlePops_old(db, evalType = c('EXPVOL'), method, mr, pltList = remPlts$PLT_CN)

  ## A lot of states do their stratification in such a way that makes it impossible
  ## to estimate variance of annual panels w/ post-stratified estimator. That is,
  ## the number of plots within a panel within an stratum is less than 2. When
  ## this happens, merge strata so that all have at least two obs
  if (str_to_upper(method) != 'TI') {
    pops <- mergeSmallStrata(db, pops)
  }


  ## If we opt to use multiple remeasurements to estimate change, we can't use
  ## clipFIA to merge most recent inventories. Instead, we'll have to subset the
  ## most recent inventories in the population tables, and combine at the end
  if (useSeries & mostRecent & str_to_upper(method) != 'ANNUAL') {

    ## Pull the most recent YEAR from each state - already filtered EVALTYP above
    pops <- pops %>%
      group_by(STATECD) %>%
      filter(YEAR == max(YEAR, na.rm = TRUE)) %>%
      ungroup()

    ## Trick rFIA into doing the merge, even though db wasn't clipped
    mr = TRUE
  }




  ## Canned groups -------------------------------------------------------------
  ## Add species to groups
  if (bySpecies) {
    db$TREE <- db$TREE %>%
      left_join(select(intData$REF_SPECIES_2018,
                       c('SPCD','COMMON_NAME', 'GENUS', 'SPECIES')), by = 'SPCD') %>%
      mutate(SCIENTIFIC_NAME = paste(GENUS, SPECIES, sep = ' ')) %>%
      mutate_if(is.factor,
                as.character)
    grpBy <- c(grpBy, 'SPCD', 'COMMON_NAME', 'SCIENTIFIC_NAME')
  }

  ## Break into size classes
  if (bySizeClass){
    grpBy <- c(grpBy, 'sizeClass')
    db$TREE$sizeClass <- makeClasses(db$TREE$DIA, interval = 2, numLabs = TRUE)
    db$TREE <- db$TREE[!is.na(db$TREE$sizeClass),]
  }





  ## Slim down the database for we hand it off to the estimators ---------------
  ## Reduces memory requirements and speeds up processing ----------------------

  ## Only the necessary plots for EVAL of interest
  remPltList <- unique(c(remPlts$PLT_CN, remPlts$PREV_PLT_CN))
  db$PLOT <- filter(db$PLOT, PLT_CN %in% remPltList)
  db$COND <- filter(db$COND, PLT_CN %in% remPltList)
  db$TREE <- filter(db$TREE, PLT_CN %in% remPltList)

  ## Tree basal area per acre
  db$TREE <- db$TREE %>%
    mutate(BAA = basalArea(DIA) * TPA_UNADJ)

  ## Which grpByNames are in which table? Helps us subset below
  grpP <- names(db$PLOT)[names(db$PLOT) %in% c(grpBy, scaleBy)]
  grpC <- names(db$COND)[names(db$COND) %in% c(grpBy, scaleBy) & names(db$COND) %in% grpP == FALSE]
  grpT <- names(db$TREE)[names(db$TREE) %in% c(grpBy, scaleBy) & names(db$TREE) %in% c(grpP, grpC) == FALSE]

  ### Only joining tables necessary to produce plot level estimates, adjusted for non-response
  db$PLOT <- select(db$PLOT, c('PLT_CN', pltID, 'REMPER', 'DESIGNCD', 'STATECD', 'MACRO_BREAKPOINT_DIA', 'INVYR',
                               'MEASYEAR', 'MEASMON', 'MEASDAY', 'PLOT_STATUS_CD', PREV_PLT_CN, all_of(grpP), 'sp'))
  db$COND <- select(db$COND, c('PLT_CN', 'CONDPROP_UNADJ', 'PROP_BASIS', 'COND_STATUS_CD', 'CONDID', all_of(grpC), 'aD', 'landD',
                               DSTRBCD1, DSTRBCD2, DSTRBCD3, TRTCD1, TRTCD2, TRTCD3))
  db$TREE <- select(db$TREE, c('PLT_CN', 'TRE_CN', 'CONDID', 'DIA', 'TPA_UNADJ', 'BAA', 'SUBP', 'TREE', all_of(grpT), 'tD', 'typeD',
                               PREVCOND, PREV_TRE_CN, STATUSCD, SPCD))





  ## Compute plot-level summaries ----------------------------------------------
  ## An iterator for plot-level summaries
  plts <- split(db$PLOT, as.factor(paste(db$PLOT$COUNTYCD, db$PLOT$STATECD, sep = '_')))
  suppressWarnings({
    ## Compute estimates in parallel -- Clusters in windows, forking otherwise
    if (Sys.info()['sysname'] == 'Windows'){
      cl <- makeCluster(nCores)
      clusterEvalQ(cl, {
        library(dplyr)
        library(stringr)
        library(rFIA)
        library(tidyr)
      })
      out <- parLapply(cl, X = names(plts), fun = fsiHelper1, plts,
                       db[names(db) %in% c('COND', 'TREE')],
                       grpBy, scaleBy, byPlot)
      #stopCluster(cl) # Keep the cluster active for the next run
    } else { # Unix systems
      out <- mclapply(names(plts), FUN = fsiHelper1, plts,
                      db[names(db) %in% c('COND', 'TREE')],
                      grpBy, scaleBy, byPlot, mc.cores = nCores)
    }
  })


  ## back to dataframes
  out <- unlist(out, recursive = FALSE)
  t <- bind_rows(out[names(out) == 't'])
  t1 <- bind_rows(out[names(out) == 't1'])
  a <- bind_rows(out[names(out) == 'a'])


  out <- list(t = t, t1 = t1, a = a, grpBy = grpBy, scaleBy = scaleBy,
              grpByOrig = grpByOrig, pops = pops, mr = mr)

}

