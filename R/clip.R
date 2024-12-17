# Spatiotemporal clip of FIADB
# TODO: needs assessment.
clipFIA <- function(db,
                    mostRecent = TRUE,
                    mask = NULL,
                    matchEval = FALSE,
                    evalid = NULL,
                    designCD = NULL,
                    nCores = 1) {

  ## Some warnings
  if (!is(db, 'FIA.Database')) {
    ## Add clipFIA arguments to remote database
    if (is(db, 'Remote.FIA.Database')) {

      db$mostRecent = mostRecent
      ## Lists won't accept NULL,
      ## but this is a hack
      db['mask'] = list(mask)

      db['evalid'] = list(evalid)

      db['designCD'] = list(designCD)

      db['matchEval'] = list(matchEval)

      return(db)
    }
    stop('db must be of class "FIA.Database". Use readFIA() to load your FIA data.')
  }
  if (!is.null(mask) & first(class(mask)) %in% c('sf', 'SpatialPolygons', 'SpatialPolygonsDataFrame') == FALSE){
    stop('mask must be spatial polygons object of class sp or sf. ')
  }

  ################### ADD UNIQUE ID TO PLOTS #############################
  db$PLOT <- db$PLOT %>%
    mutate(PLT_CN = CN) %>%
    #mutate(date = ymd(paste(MEASYEAR, MEASMON, MEASDAY, sep = '-'))) %>%
    mutate(pltID = paste(UNITCD, STATECD, COUNTYCD, PLOT, sep = '_')) # Make a unique ID for each plot, irrespective of time

  if (!is.null(designCD)) db$PLOT <- filter(db$PLOT, DESIGNCD == any(as.integer(designCD)))


  ######### IF USER SPECIES EVALID (OR MOST RECENT), EXTRACT APPROPRIATE PLOTS ##########
  if (!is.null(evalid)){
    if (mostRecent) warning('Returning subset by EVALID only. For most recent subset, specify `evalid = NULL` and `mostRecent = TRUE`.')

    # Join appropriate tables and filter out specified EVALIDs
    tempData <- select(db$PLOT, CN, PREV_PLT_CN, STATECD) %>%
      mutate(PLT_CN = CN) %>%
      left_join(select(db$POP_PLOT_STRATUM_ASSGN, c('PLT_CN', 'EVALID')), by = 'PLT_CN') %>%
      #inner_join(select(POP_ESTN_UNIT, c('EVAL_GRP_CN', 'EVALID')), by = 'EVAL_GRP_CN') %>%
      filter(EVALID %in% evalid)


    ## If no EVALIDs are found, then stop, and tell us what's going on
    if ( nrow(tempData) < 1 ) {

      ## First off, what states are we dealing with? This will be really useful to
      ## know for a remote, multi-state DB
      states <- intData$stateNames %>%
        filter(STATECD %in% unique(db$PLOT$STATECD)) %>%
        .$STATEAB

      ## Throw an error
      if (length(states) > 2){
        states <- paste0(paste(states[1:(length(states)-1)], collapse = ', '), paste0(', or ', states[length(states)]))
      } else if (length(states) > 1) {
        states <- paste0(paste(states, collapse = ' or '))
      }
      stop(paste0('None of specified EVALIDs are affiliated with ', states, '. To resolve, either add EVALIDs for these states in the list provided to `clipFIA`, or drop these states from the call to `readFIA` using the `states` argument.'))
    }

    ## If in memory and some states missing from evalid list, warn us
    if ( !all(unique(db$PLOT$STATECD) %in% unique(tempData$STATECD)) ) {

      ## Who's missing?
      states <- intData$stateNames %>%
        filter(STATECD %in% unique(db$PLOT$STATECD)) %>%
        filter(!c(STATECD %in% unique(tempData$STATECD))) %>%
        .$STATEAB

      ## Throw an error
      if (length(states) > 2){
        states <- paste0(paste(states[1:(length(states)-1)], collapse = ', '), paste0(', or ', states[length(states)]))
      } else if (length(states) > 1) {
        states <- paste0(paste(states, collapse = ' or '))
      }
      warning(paste0('None of specified EVALIDs are affiliated with ', states, '. Estimates will not include data from these states.'))
    }

    # Extract plots which relate to specified EVALID (previous for change estimation)
    PPLOT <- db$PLOT[db$PLOT$CN %in% tempData$PREV_PLT_CN,]
    if (nrow(PPLOT) < 1) PPLOT <- NULL
    db$PLOT <- db$PLOT[db$PLOT$CN %in% tempData$PLT_CN,]


  }

  ## Locate the most recent EVALID and subset plots
  if (mostRecent & is.null(evalid)){
    suppressWarnings({
      tempData <- select(db$PLOT, CN, PREV_PLT_CN) %>%
        mutate(PLT_CN = CN) %>%
        left_join(select(db$POP_PLOT_STRATUM_ASSGN, c('STRATUM_CN', 'PLT_CN', 'EVALID')), by = c('PLT_CN'))


      ## Most recent evals by
      mrids <- findEVALID(db, mostRecent = TRUE)

      tempData <- tempData %>%
        filter(EVALID %in% mrids)
    })

    # Write out evalids so that we don't have to repeat above later
    evalid <- unique(tempData$EVALID)

    # # Extract appropriate PLOTS
    # PPLOT <- db$PLOT[db$PLOT$CN %in% tempData$PREV_PLT_CN,]
    # if (nrow(PPLOT) < 1) PPLOT <- NULL
    # db$PLOT <- db$PLOT[db$PLOT$CN %in% tempData$PLT_CN,]
    # #test = db$PLOT <- db$PLOT[db$PLOT$CN %in% tempData$PLT_CN,]

    # Extract appropriate PLOTS
    PPLOT <- db$PLOT[db$PLOT$CN %in% tempData$PREV_PLT_CN,]
    if (nrow(PPLOT) < 1) {
      PPLOT <- NULL
    } else {
      ## Cutting duplicates
      PPLOT <- PPLOT %>%
        filter(!(PLT_CN %in% tempData$PLT_CN))
    }
    db$PLOT <- db$PLOT[db$PLOT$CN %in% tempData$PLT_CN,]
    #test = db$PLOT <- db$PLOT[db$PLOT$CN %in% tempData$PLT_CN,]


    ## If not most recent, but still want matching evals, go for it.
  } else if (matchEval){
    ### Keeping only years where all states represented are reported for
    if (length(unique(db$POP_EVAL$STATECD)) > 1){
      # Counting number of states measured by year, remove years which don't include all states
      numStates <- db$POP_EVAL %>%
        group_by(END_INVYR, STATECD) %>%
        summarize() %>%
        group_by(END_INVYR) %>%
        summarize(n = n()) %>%
        filter(n == length(unique(db$POP_EVAL$STATECD)))

      db$POP_EVAL <- db$POP_EVAL %>%
        filter(END_INVYR %in% numStates$END_INVYR)
    }

  }


  ##########################  SPATIAL-TEMPORAL INTERSECTION ######################################
  ## Oringally did this based solely on PLT_CN, but this method will not work when we try to compute variance
  ##  estimates for a region, because estimates are computed at ESTN_UNIT level. Now the spatial functionality
  ##  of clipFIA is primarily intended to reduce the amount of data which must be held in RAM. As long as all
  ##  plot data is avaliable for any intersecting ESTN_UNIT, we can cut back on memory requirements and still
  ##  produce valid estimates
  ##
  ## We also want to make modifications (additions) to the population tables here, as folks may want to use
  ## an alternative estimator in subsequent estimation. Any modifications to design info needs to happen
  ## before we slim down the plot lists, so handling that here.
  if (!is.null(mask)){

    ## Make additions to pops tables before spatial intersection
    if (all(c('POP_EVAL', 'POP_ESTN_UNIT', 'POP_STRATUM', 'POP_PLOT_STRATUM_ASSGN') %in% names(db))) {
      ### The population tables
      pops <- select(db$POP_EVAL, c('EVALID', 'ESTN_METHOD', 'CN')) %>%
        rename(EVAL_CN = CN) %>%
        left_join(select(db$POP_ESTN_UNIT, c('CN', 'EVAL_CN')), by = c('EVAL_CN')) %>%
        rename(ESTN_UNIT_CN = CN) %>%
        left_join(select(db$POP_STRATUM, c('ESTN_UNIT_CN', 'P2POINTCNT', 'CN')), by = c('ESTN_UNIT_CN')) %>%
        rename(STRATUM_CN = CN) %>%
        distinct(EVALID, ESTN_UNIT_CN, STRATUM_CN, .keep_all = TRUE) %>%
        left_join(select(db$POP_PLOT_STRATUM_ASSGN, c('STRATUM_CN', 'PLT_CN', 'INVYR', 'STATECD')), by = 'STRATUM_CN') %>%
        distinct(EVALID, ESTN_UNIT_CN, STRATUM_CN, PLT_CN, .keep_all = TRUE) %>%
        ungroup() %>%
        ## Count of plots by Stratum/INVYR
        group_by(EVALID, ESTN_UNIT_CN, STRATUM_CN, INVYR) %>%
        mutate(P2POINTCNT_INVYR = n()) %>%
        ## By unit / INVYR
        group_by(EVALID, ESTN_UNIT_CN, INVYR) %>%
        mutate(p2eu_INVYR = n(),
               nStrata_INVYR = length(unique(STRATUM_CN))) %>%
        ## By unit for entire cycle
        group_by(EVALID, ESTN_UNIT_CN) %>%
        mutate(p2eu = n(),
               nStrata = length(unique(STRATUM_CN))) %>%
        ungroup() %>%
        filter(!is.na(PLT_CN))

      ## Append TI variables on estn unit
      db$POP_ESTN_UNIT <- db$POP_ESTN_UNIT %>%
        left_join(distinct(select(pops, ESTN_UNIT_CN, p2eu, nStrata)), by = c('CN' = 'ESTN_UNIT_CN'))

      ## Append annual variables on PLOT_ASSGN
      db$POP_PLOT_STRATUM_ASSGN <- db$POP_PLOT_STRATUM_ASSGN %>%
        left_join(distinct(select(pops, PLT_CN, STRATUM_CN, p2eu_INVYR, nStrata_INVYR, P2POINTCNT_INVYR)), by = c('PLT_CN', 'STRATUM_CN'))
    }

    # Convert polygons to an sf object
    mask <- mask %>%
      as('sf') %>%
      mutate(polyID = 1:nrow(mask))
    ## Make plot data spatial, projected same as polygon layer
    pltSF <- select(db$PLOT, c('LON', 'LAT', pltID)) %>%
      filter(!is.na(LAT) & !is.na(LON)) %>%
      distinct(pltID, .keep_all = TRUE) %>%
      st_as_sf(coords = c('LON', 'LAT'))
    st_crs(pltSF) <- 4326
    pltSF <- st_transform(pltSF, crs = st_crs(mask))

    ## Split up polys
    polyList <- split(mask, as.factor(mask$polyID))
    suppressWarnings({suppressMessages({
      ## Compute estimates in parallel -- Clusters in windows, forking otherwise
      if (Sys.info()['sysname'] == 'Windows'){
        cl <- makeCluster(nCores)
        clusterEvalQ(cl, {
          library(dplyr)
          library(stringr)
          library(rFIA)
          library(sf)
        })
        out <- parLapply(cl, X = names(polyList), fun = areal_par, pltSF, polyList)
        #stopCluster(cl) # Keep the cluster active for the next run
      } else { # Unix systems
        out <- mclapply(names(polyList), FUN = areal_par, pltSF, polyList, mc.cores = nCores)
      }
    })})

    pltSF <- bind_rows(out) %>%
      left_join(select(db$PLOT, PLT_CN, PREV_PLT_CN, pltID), by = 'pltID')

    if (length(unique(pltSF$pltID)) < 1){
      stop('No plots in db overlap with mask')
    }
    #if(mostRecent == FALSE & is.null(evalid)) PPLOT <- filter(db$PLOT, db$PLOT$PLT_CN %in% pltSF$PREV_PLT_CN)
    if(mostRecent == FALSE & is.null(evalid)) {
      PPLOT <- NULL
    } else {
      PPLOT <- filter(PPLOT, pltID %in% pltSF$pltID)
    }
    db$PLOT <- filter(db$PLOT, db$PLOT$PLT_CN %in% pltSF$PLT_CN)

    #  ###OLD
    #  # Convert polygons to an sf object
    #  mask <- mask %>%
    #    as('sf')
    #
    #  ## Make plot data spatial, projected same as polygon layer
    #  pltSF <- select(db$PLOT, c('PLT_CN', 'LON', 'LAT'))
    #  coordinates(pltSF) <- ~LON+LAT
    #  proj4string(pltSF) <- '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    #  pltSF <- as(pltSF, 'sf') %>%
    #    st_transform(crs = st_crs(mask)$proj4string)
    #
    #  # Intersect plot with polygons
    #  mask$polyID <- 1:nrow(mask)
    #  suppressMessages({suppressWarnings({
    #    pltSF <- st_intersection(pltSF, mask) %>%
    #      as.data.frame() %>%
    #      select(-c('geometry')) # removes artifact of SF object
    #  })})
    #
    #  # Identify the estimation units to which plots within the mask belong to
    #  estUnits <- pltSF %>%
    #    inner_join(select(db$POP_PLOT_STRATUM_ASSGN, c('PLT_CN', 'STRATUM_CN')), by = 'PLT_CN') %>%
    #    inner_join(select(db$POP_STRATUM, c('CN', 'ESTN_UNIT_CN')), by = c('STRATUM_CN' = 'CN')) %>%
    #    group_by(ESTN_UNIT_CN) %>%
    #    summarize()
    #
    #  # Identify all the plots which fall inside the above estimation units
    #  plts <- select(db$PLOT, PLT_CN, PREV_PLT_CN) %>%
    #    inner_join(select(db$POP_PLOT_STRATUM_ASSGN, c('PLT_CN', 'STRATUM_CN')), by = 'PLT_CN') %>%
    #    inner_join(select(db$POP_STRATUM, c('CN', 'ESTN_UNIT_CN')), by = c('STRATUM_CN' = 'CN')) %>%
    #    filter(ESTN_UNIT_CN %in% estUnits$ESTN_UNIT_CN) %>%
    #    group_by(PLT_CN, PREV_PLT_CN) %>%
    #    summarize()
    #
    #  # Clip out the above plots from the full database, will reduce size by a shit pile
    #  PPLOT <- filter(db$PLOT, db$PLOT$PLT_CN %in% plts$PREV_PLT_CN)
    #  db$PLOT <- filter(db$PLOT, db$PLOT$PLT_CN %in% plts$PLT_CN)
    #
    # ## IF ozone is specified, do a seperate intersection (PLOTs not colocated w/ veg PLOTs)
    # if (!is.null(db$OZONE_PLOT)) {
    #   # Seperate spatial object
    #   ozoneSP <- db$OZONE_PLOT
    #
    #   ## Make PLOT data spatial, projected same as mask layer
    #   coordinates(ozoneSP) <- ~LON+LAT
    #   proj4string(ozoneSP) <- '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
    #   ozoneSP <- spTransform(ozoneSP, CRSobj = proj4string(mask))
    #
    #   ## Spatial query (keep only PLOTs that fall within shell)
    #   db$OZONE_PLOT <- ozoneSP[mask,]
    #
    #   ## Add coordinates back in to dataframe
    #   coords <- st_coordinates(db$OZONE_PLOT)
    #   db$OZONE_PLOT <- db$OZONE_PLOT %>%
    #     data.frame() %>%
    #     mutate(LAT = coords[,2]) %>%
    #     mutate(LON = coords[,1])
    # }
  }

  ## IF no spatial or temporal clip was specified, make PPLOT NULL
  if (mostRecent == FALSE & is.null(evalid) & is.null(mask)) PPLOT <- NULL

  #################  APPLY  QUERY TO REMAINING TABLES  ###########################
  ## User gives object names and not list object
  otherTables <- db

  if (is.list(otherTables)){
    tableNames <- names(otherTables)
    clippedData <- list()

    # Query for all tables directly related to PLOT
    for (i in 1:length(otherTables)){
      # Pull the object with the name listed from the global environment
      table <- otherTables[[i]]
      name <- tableNames[i]

      if (name == "PLOT"){
        db$PLOT$prev = 0
        #if(nrow(PPLOT) > 0) PPLOT$prev = 1
        if(!is.null(PPLOT) > 0) PPLOT$prev = 1
        clippedData[['PLOT']] <- rbind(db$PLOT, PPLOT)

      } else if (!is.null(db$OZONE_PLOT) & name == 'OZONE_PLOT'){
        clippedData[['OZONE_PLOT']] <- db$OZONE_PLOT

      } else if (name == 'TREE' | name == 'COND'){ # Need previous attributes
        clipTable <- table[table$PLT_CN %in% c(db$PLOT$CN, PPLOT$PLT_CN),]
        clippedData[[name]] <- clipTable

      } else if ('PLT_CN' %in% colnames(table) & str_detect(name, 'OZONE') == FALSE){
        clipTable <- table[table$PLT_CN %in% db$PLOT$CN,]
        clippedData[[name]] <- clipTable

      } else if ('PLT_CN' %in% colnames(table) & !is.null(db$OZONE_PLOT) & str_detect(name, 'OZONE')){
        clipTable <- table[table$PLT_CN %in% db$OZONE_PLOT$CN,]
        clippedData[[name]] <- clipTable

      } else if ('CTY_CN' %in% colnames(table)){
        clipTable <- table[table$CTY_CN %in% db$PLOT$CTY_CN,]
        clippedData[[name]] <- clipTable

      } else if('SRV_CN' %in% colnames(table)){
        clipTable <- table[table$SRV_CN %in% db$PLOT$SRV_CN,]
        clippedData[[name]] <- clipTable

      } else if(name == 'PLOTGEOM'){
        clipTable <- table[table$CN %in% db$PLOT$CN,]
        clippedData[[name]] <- clipTable

      } else if(name == 'SNAP'){
        clipTable <- table[table$CN %in% db$PLOT$CN,]
        clippedData[[name]] <- clipTable

      } else{ # IF it doesn't connect in a way described above, return the whole thing
        clippedData[[name]] <- table
      }
    }


    # Cascade spatial query for all other tables (indirectly related to PLOT)
    if ("TREE_REGIONAL_BIOMASS" %in% tableNames & 'TREE' %in% names(clippedData)){
      clippedData[["TREE_REGIONAL_BIOMASS"]] <- db$TREE_REGIONAL_BIOMASS %>%
        filter(TRE_CN %in% clippedData$TREE$CN)
    }
    # Deal with the POP Tables
    if(!is.null(evalid) | mostRecent){ # User specified EVALIDs
      # Links only to POP_EVAL
      if ('POP_EVAL' %in% tableNames){
        clippedData[['POP_EVAL']] <- db$POP_EVAL %>%
          filter(EVALID %in% evalid)
        if("POP_EVAL_GRP" %in% tableNames){
          clippedData[["POP_EVAL_GRP"]] <- db$POP_EVAL_GRP %>%
            filter(CN %in% clippedData$POP_EVAL$EVAL_GRP_CN)
        }
        if("POP_EVAL_ATTRIBUTE" %in% tableNames){
          clippedData[["POP_EVAL_ATTRIBUTE"]] <- db$POP_EVAL_ATTRIBUTE %>%
            filter(EVAL_CN %in% clippedData$POP_EVAL$CN)
        }
        if('POP_EVAL_TYP' %in% tableNames){
          clippedData[['POP_EVAL_TYP']] <- db$POP_EVAL_TYP %>%
            filter(EVAL_CN %in% clippedData$POP_EVAL$CN)
        }
      }
      # Links to both POP_EVAL & PLOT
      if ('POP_ESTN_UNIT' %in% tableNames){
        clippedData[['POP_ESTN_UNIT']] <- db$POP_ESTN_UNIT %>%
          filter(EVALID %in% evalid)
      }
      if ('POP_STRATUM' %in% tableNames){
        clippedData[['POP_STRATUM']] <- db$POP_STRATUM %>%
          filter(EVALID %in% evalid)
      }
      if ("POP_PLOT_STRATUM_ASSGN" %in% names(clippedData)){ # Should already be in here becuase contains PLT_CN
        clippedData[["POP_PLOT_STRATUM_ASSGN"]] <- clippedData[["POP_PLOT_STRATUM_ASSGN"]] %>%
          filter(EVALID %in% evalid)
      }

      # User did not specify evalids or most recent
    } else {
      if ("POP_PLOT_STRATUM_ASSGN_STRATUM_ASSGN" %in% names(clippedData) & 'POP_STRATUM' %in% tableNames){
        # Estimation units
        units <- otherTables$POP_STRATUM %>%
          filter(CN %in% clippedData$POP_PLOT_STRATUM_ASSGN$STRATUM_CN) %>%
          select(ESTN_UNIT_CN) %>%
          distinct(ESTN_UNIT_CN)
        # Returns all strata for an estimation unit in which plot falls
        clippedData[['POP_STRATUM']] <- otherTables$POP_STRATUM %>%
          filter(ESTN_UNIT_CN %in% units$ESTN_UNIT_CN)
        if ("POP_ESTN_UNIT" %in% tableNames){
          clippedData[['POP_ESTN_UNIT']] <- db$POP_ESTN_UNIT %>%
            filter(CN %in% units$ESTN_UNIT_CN)
        }
      }
      if ("POP_PLOT_STRATUM_ASSGN" %in% names(clippedData) & 'POP_EVAL' %in% tableNames){
        clippedData[['POP_EVAL']] <- db$POP_EVAL %>%
          filter(EVALID %in% clippedData$POP_PLOT_STRATUM_ASSGN$EVALID)
        if("POP_EVAL_GRP" %in% tableNames){
          clippedData[["POP_EVAL_GRP"]] <- db$POP_EVAL_GRP %>%
            filter(CN %in% clippedData$POP_EVAL$EVAL_GRP_CN)
        }
        if("POP_EVAL_ATTRIBUTE" %in% tableNames){
          clippedData[["POP_EVAL_ATTRIBUTE"]] <- db$POP_EVAL_ATTRIBUTE %>%
            filter(EVAL_CN %in% clippedData$POP_EVAL$CN)
        }
        if('POP_EVAL_TYP' %in% tableNames){
          clippedData[['POP_EVAL_TYP']] <- db$POP_EVAL_TYP %>%
            filter(EVAL_CN %in% clippedData$POP_EVAL$CN)
        }
      }
    }
    # User didn't specify other tables, skip above and just return plot
  } else if (is.null(otherTables) & is.null(db$OZONE_PLOT)){
    clippedData <- db$PLOT
  } else if (is.null(otherTables) & !is.null(db$OZONE_PLOT)){
    clippedData <- list(db$PLOT, db$OZONE_PLOT)
  }

  ## Add flags so estimator functions can adapt to modified inputs
  if (mostRecent) clippedData$mostRecent <- TRUE
  if(!is.null(mask)) clippedData$mask <- TRUE

  class(clippedData) <- 'FIA.Database'
  return(clippedData)
}
