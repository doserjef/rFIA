# Startup message ---------------------
.onAttach <- function(lib, pkg) {
  if(interactive() || getOption("verbose"))
    packageStartupMessage(sprintf("Package %s (%s) loaded. Check out our website at https://doserlab.com/files/rfia/.\nType citation(\"%s\") for examples of how to cite rFIA.\n", pkg,
                                  packageDescription(pkg)$Version, pkg))
}

# Iterator if db is remote
remoteIter <- function(db, remote){
  if (remote){

    iter <- db$states

  # In memory
  } else {
    # Some warnings
    if (!is(db, 'FIA.Database')) {
      stop('db must be of class "FIA.Database". Use readFIA() to load your FIA data.')
    }

    # an iterator for remote
    iter <- 1
  }

  return(iter)
}

# Check most recent and handle remote dbs
checkMR <- function(db, remote){
  if (remote){
    if ('mostRecent' %in% names(db)){
      mr = db$mostRecent # logical
    } else {
      mr = FALSE
    }
    # In-memory
  } else {
    if ('mostRecent' %in% names(db)){
      mr = TRUE
    } else {
      mr = FALSE
    }
  }
  return(mr)
}

# Prep database for areal summary
# Converts polygons to sf (in case sp supplied), converts factors to characters,
# and adds a unique ID for each poly.
arealSumPrep1 <- function(polys){

  if(!is.null(polys)) {
    # Convert polygons to an sf object
    polys <- polys %>%
      as('sf')%>%
      dplyr::mutate_if(is.factor,
                       as.character)
    # A unique ID in the polyID column. 
    polys$polyID <- 1:nrow(polys)
  }

  return(polys)
}

# Do the spatial intersection of plots w/ polgyons
arealSumPrep2 <- function(db, grpBy, polys, nCores, remote) {

  # Make plot data spatial, projected same as polygon layer
  pltSF <- dplyr::select(db$PLOT, c('LON', 'LAT', pltID)) %>%
    dplyr::filter(!is.na(LAT) & !is.na(LON)) %>%
    dplyr::distinct(pltID, .keep_all = TRUE) %>%
    sf::st_as_sf(coords = c('LON', 'LAT'))
  sf::st_crs(pltSF) <- 4326
  pltSF <- pltSF %>%
    sf::st_transform(crs = sf::st_crs(polys))

  # Crop the polygon to the bounding box of the FIA plots first
  suppressWarnings({suppressMessages({
    polys <- sf::st_crop(polys, sf::st_bbox(pltSF))
  })})

  # Split up polys
  polyList <- split(polys, as.factor(polys$polyID))
  suppressWarnings({suppressMessages({
    # Compute estimates in parallel -- Clusters in windows, forking otherwise
    if (Sys.info()['sysname'] == 'Windows') {
      cl <- parallel::makeCluster(nCores)
      parallel::clusterEvalQ(cl, {
        library(dplyr)
        library(stringr)
        library(rFIA)
        library(sf)
      })
      out <- parallel::parLapply(cl, X = names(polyList), fun = areal_par, pltSF, polyList)
      #stopCluster(cl) # Keep the cluster active for the next run
    } else { # Unix systems
      out <- parallel::mclapply(names(polyList), FUN = areal_par, pltSF, polyList, mc.cores = nCores)
    }
  })})
  pltSF <- dplyr::bind_rows(out)

  # A warning
  if (length(unique(pltSF$pltID)) < 1){
    if (!remote) {
      stop('No plots in db overlap with polys.')
    } else {
      return()
    }
  }

  # Add polygon names to PLOT
  db$PLOT <- db$PLOT %>%
    dplyr::select(-c(any_of(names(polys)))) %>%
    dplyr::left_join(pltSF, by = 'pltID')

  return(db)

}

# If a state missing a ROI completely, skip it when remote
# Primary purpose is to throw an error when no overlap is present
dropStatesOutsidePolys <- function(x) {
  x <- x[x != 'no plots in polys']

  if (length(x) < 1) {
    stop('No plots in db overlap with polys.')
  }

  return(x)

}

# Helper function to read remote database by state within estimator functions
readRemoteHelper <- function(x, db, remote, reqTables, nCores) {
  if (remote) {
    # Store the original parameters here
    params <- db

    # Read in one state at a time
    db <- readFIA(dir = db$dir,
                  con = db$con,
                  schema = db$schema,
                  common = db$common,
                  tables = reqTables,
                  states = x, # x is the vector of state names
                  nCores = nCores)

    # If a clip was specified, run it now
    if ('mostRecent' %in% names(params)){
      db <- clipFIA(db, mostRecent = params$mostRecent,
                    mask = params$mask, matchEval = params$matchEval,
                    evalid = params$evalid, designCD = params$designCD,
                    nCores = nCores)
    }

  } else {
    # Check to make sure required tables are here.
    if (sum(reqTables %in% names(db)) < length(reqTables)) {
      missing.tables <- reqTables[!c(reqTables %in% names(db))]
      stop(paste(paste (as.character(missing.tables), collapse = ', '), 'tables not found in object `db`.'))
    }
    # Really only want the required tables
    db <- db[names(db) %in% reqTables]
  }

  return(db)
}


# Convert grpBy from quos to character and make sure columns exist
grpByToChar <- function(db, grpBy_quo){

  # If tree table exists
  if ('TREE' %in% names(db)){
    # Probably cheating, but it works
    if (dplyr::quo_name(grpBy_quo) != 'NULL'){
      # Have to join tables to run select with this object type
      plt_quo <- dplyr::filter(db$PLOT, !is.na(PLT_CN))
      # We want a unique error message here to tell us when columns are not present in data
      d_quo <- tryCatch(
        error = function(cnd) {
          return(0)
        },
        plt_quo[10,] %>% # Just the first row
          dplyr::left_join(dplyr::select(db$COND, PLT_CN, names(db$COND)[names(db$COND) %in% names(db$PLOT) == FALSE]), by = 'PLT_CN') %>%
          dplyr::inner_join(dplyr::select(db$TREE, PLT_CN, names(db$TREE)[names(db$TREE) %in% c(names(db$PLOT), names(db$COND)) == FALSE]), by = 'PLT_CN') %>%
          dplyr::select(!!grpBy_quo)
      )

      # If column doesnt exist, just returns 0, not a dataframe
      if (is.null(nrow(d_quo))){
        grpName <- dplyr::quo_name(grpBy_quo)
        stop(paste('Columns', grpName, 'not found in PLOT, TREE, or COND tables. Did you accidentally quote the variables names? e.g. use grpBy = ECOSUBCD (correct) instead of grpBy = "ECOSUBCD". ', collapse = ', '))
      } else {
        # Convert to character
        grpBy <- names(d_quo)
      }
    } else {
      grpBy <- NULL
    }
  } else if ('SEEDLING' %in% names(db)) {
    # Probably cheating, but it works
    if (dplyr::quo_name(grpBy_quo) != 'NULL'){
      # Have to join tables to run select with this object type
      plt_quo <- dplyr::filter(db$PLOT, !is.na(PLT_CN))
      # We want a unique error message here to tell us when columns are not present in data
      d_quo <- tryCatch(
        error = function(cnd) {
          return(0)
        },
        plt_quo[10,] %>% # Just the first row
          dplyr::left_join(dplyr::select(db$COND, PLT_CN, names(db$COND)[names(db$COND) %in% names(db$PLOT) == FALSE]), by = 'PLT_CN') %>%
          dplyr::inner_join(dplyr::select(db$SEEDLING, PLT_CN, names(db$SEEDLING)[names(db$SEEDLING) %in% c(names(db$PLOT), names(db$COND)) == FALSE]), by = 'PLT_CN') %>%
          dplyr::select(!!grpBy_quo)
      )

      # If column doesnt exist, just returns 0, not a dataframe
      if (is.null(nrow(d_quo))){
        grpName <- dplyr::quo_name(grpBy_quo)
        stop(paste('Columns', grpName, 'not found in PLOT, SEEDLING, or COND tables. Did you accidentally quote the variables names? e.g. use grpBy = ECOSUBCD (correct) instead of grpBy = "ECOSUBCD". ', collapse = ', '))
      } else {
        # Convert to character
        grpBy <- names(d_quo)
      }
    } else {
      grpBy <- NULL
    }
  } else { # COND
    # Probably cheating, but it works
    if (dplyr::quo_name(grpBy_quo) != 'NULL'){
      # Have to join tables to run select with this object type
      plt_quo <- dplyr::filter(db$PLOT, !is.na(PLT_CN))
      # We want a unique error message here to tell us when columns are not present in data
      d_quo <- tryCatch(
        error = function(cnd) {
          return(0)
        },
        plt_quo[10,] %>% # Just the first row
          dplyr::left_join(dplyr::select(db$COND, PLT_CN, names(db$COND)[names(db$COND) %in% names(db$PLOT) == FALSE]), by = 'PLT_CN') %>%
          dplyr::select(!!grpBy_quo)
      )

      # If column doesnt exist, just returns 0, not a dataframe
      if (is.null(nrow(d_quo))){
        grpName <- dplyr::quo_name(grpBy_quo)
        stop(paste('Columns', grpName, 'not found in PLOT or COND tables. Did you accidentally quote the variables names? e.g. use grpBy = ECOSUBCD (correct) instead of grpBy = "ECOSUBCD". ', collapse = ', '))
      } else {
        # Convert to character
        grpBy <- names(d_quo)
      }
    } else {
      grpBy <- NULL
    }
  }

  return(grpBy)

}

# Drop all inventories that are specific to east or west tx. Only retain evals
# that span the entire state
handleTX <- function(db){

  if (any(db$POP_EVAL$STATECD %in% 48)){

    badIDS <- db$POP_PLOT_STRATUM_ASSGN %>%
      dplyr::filter(STATECD %in% 48) %>%
      dplyr::distinct(EVALID, UNITCD) %>%
      dplyr::group_by(EVALID) %>%
      dplyr::summarise(n = dplyr::n()) %>%
      dplyr::ungroup() %>%
      dplyr::filter(n != 7) ## i.e., only keep EVALIDs w/out all 7 units

    db$POP_EVAL <- dplyr::filter(db$POP_EVAL, !c(EVALID %in% badIDS$EVALID))

  }

  return(db)
}

# WY fixed this in July 2024, but keeping this in to avoid any problems with any users using
# out-of-date db versions.
# As of Apr 2021, WY labels 2018 and 2019 inventories as 2020. This breaks rFIA,
# so we manually reset these labels to their appropriate values here
handleWY <- function(db){
  if ('POP_EVAL' %in% names(db)) {
    db$POP_EVAL <- db$POP_EVAL %>%
      dplyr::mutate(END_INVYR = dplyr::case_when(EVALID %in% c(561800, 561801, 561803, 561807) ~ as.numeric(2018),
                                                 EVALID %in% c(561900, 561901, 561903, 561907, 561909, 561910) ~ as.numeric(2019),
                                                 TRUE ~ as.numeric(END_INVYR)))
  }
  return(db)
}

# Land type domain indicator
landTypeDomain <- function(landType, COND_STATUS_CD, SITECLCD, RESERVCD) {
  if (tolower(landType) == 'forest'){
    landD <- ifelse(COND_STATUS_CD == 1, 1, 0)
  } else if (tolower(landType) == 'timber'){
    landD <- ifelse(COND_STATUS_CD == 1 & SITECLCD %in% c(1, 2, 3, 4, 5, 6) & RESERVCD == 0, 1, 0)
  } else if (tolower(landType) == 'non-forest'){
    landD <- ifelse(COND_STATUS_CD == 2, 1, 0)
  } else if (tolower(landType) == 'water'){
    landD <- ifelse(COND_STATUS_CD == 3 | COND_STATUS_CD == 4, 1, 0)
  } else if (tolower(landType) == 'census water'){
    landD <- ifelse(COND_STATUS_CD == 3, 1, 0)
  } else if (tolower(landType) == 'non-census water'){
    landD <- ifelse(COND_STATUS_CD == 4, 1, 0)
  } else if (tolower(landType) == 'all') {
    landD <- 1
  }

  return(landD)
}

# Tree type domain indicator
treeTypeDomain <- function(treeType, STATUSCD, DIA, TREECLCD) {
  if (tolower(treeType) == 'live'){
    typeD <- data.table::fifelse(STATUSCD == 1, 1, 0)
  } else if (tolower(treeType) == 'dead'){
    typeD <- data.table::fifelse(STATUSCD == 2, 1, 0)
  } else if (tolower(treeType) == 'gs'){
    typeD <- data.table::fifelse(STATUSCD == 1 & DIA >= 5 & TREECLCD == 2, 1, 0)
  } else if (tolower(treeType) == 'all'){
    typeD <- 1
  }
  return(typeD)
}

typeDomain_grow <- function(db, treeType, landType, type, stateVar = NULL) {

  if (type == 'vr'){
    if (tolower(landType) == 'forest'){
      # Accessible forest land
      db$COND$landD <- ifelse(db$COND$COND_STATUS_CD == 1, 1, 0)
      # Tree Type domain indicator
      if (tolower(treeType) %in% c('live', 'all')){
        db$TREE$typeD <- 1
        # Rename some variables in grm. Note that vitalRates only is applicable for 
        # estimating growth in trees with DIA > 5in. 
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_AL_FOREST,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_AL_FOREST,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_AL_FOREST,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_AL_FOREST,
                                               COMPONENT = SUBP_COMPONENT_AL_FOREST)

      } else if (tolower(treeType) == 'gs'){
        db$TREE$typeD <- 1 
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_GS_FOREST,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_GS_FOREST,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_GS_FOREST,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_GS_FOREST,
                                               COMPONENT = SUBP_COMPONENT_GS_FOREST)
      }
    } else if (tolower(landType) == 'timber'){
      db$COND$landD <- ifelse(db$COND$COND_STATUS_CD == 1 & db$COND$SITECLCD %in% c(1, 2, 3, 4, 5, 6) & db$COND$RESERVCD == 0, 1, 0)
      # Tree Type domain indicator
      if (tolower(treeType)  %in% c('live', 'all')){
        db$TREE$typeD <- 1
        # Rename some variables in grm. note that vitalRates only is applicable for 
        # estimating growth in trees with DIA > 5in
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_AL_TIMBER,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_AL_TIMBER,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_AL_TIMBER,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_AL_TIMBER,
                                               COMPONENT = SUBP_COMPONENT_AL_TIMBER)

      } else if (tolower(treeType) == 'gs'){
        db$TREE$typeD <- 1
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_GS_TIMBER,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_GS_TIMBER,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_GS_TIMBER,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_GS_TIMBER,
                                               COMPONENT = SUBP_COMPONENT_GS_TIMBER)
      }
    }

  } else if (type == 'gm') {
    # Build domain indicator function which is 1 if observation meets criteria, and 0 otherwise
    # Land type domain indicator
    if (tolower(landType) == 'forest'){
      db$COND$landD <- ifelse(db$COND$COND_STATUS_CD == 1, 1, 0)

      # Tree Type domain indicator
      if (toupper(stateVar) %in% c('SAWVOL', 'SAWVOL_BF')) {
        db$TREE$typeD <- 1
        # Rename some variables in grm
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_SL_FOREST,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_SL_FOREST,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_SL_FOREST,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_SL_FOREST,
                                               COMPONENT = SUBP_COMPONENT_SL_FOREST) %>%
          dplyr::mutate(TPARECR_UNADJ = dplyr::case_when(
            is.na(COMPONENT) ~ NA_real_,
            # Note that currently defining removals to NOT include CUT2 and MORTALITY2.
            # COMPONENT %in% c('INGROWTH', 'CUT2', 'MORTALITY2') ~ TPAGROW_UNADJ,
            COMPONENT %in% c('INGROWTH') ~ TPAGROW_UNADJ,
            TRUE ~ 0))

      } else if (tolower(treeType) == 'all'){
        db$TREE$typeD <- 1
        # Rename some variables in grm
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_AL_FOREST,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_AL_FOREST,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_AL_FOREST,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_AL_FOREST,
                                               COMPONENT = SUBP_COMPONENT_AL_FOREST) %>%
          dplyr::mutate(TPARECR_UNADJ = dplyr::case_when(
            is.na(COMPONENT) ~ NA_real_,
            # Note that currently defining removals to NOT include CUT2 and MORTALITY2.
            # COMPONENT %in% c('INGROWTH', 'CUT2', 'MORTALITY2') ~ TPAGROW_UNADJ,
            COMPONENT %in% c('INGROWTH') ~ TPAGROW_UNADJ,
            TRUE ~ 0))

      } else if (tolower(treeType) == 'gs') {
        db$TREE$typeD <- 1
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_GS_FOREST,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_GS_FOREST,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_GS_FOREST,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_GS_FOREST,
                                               COMPONENT = SUBP_COMPONENT_GS_FOREST)%>%
          dplyr::mutate(TPARECR_UNADJ = dplyr::case_when(
            is.na(COMPONENT) ~ NA_real_,
            # Note that currently defining removals to NOT include CUT2 and MORTALITY2.
            # COMPONENT %in% c('INGROWTH', 'CUT2', 'MORTALITY2') ~ TPAGROW_UNADJ,
            COMPONENT %in% c('INGROWTH') ~ TPAGROW_UNADJ,
            TRUE ~ 0))
      }

    } else if (tolower(landType) == 'timber'){
      db$COND$landD <- ifelse(db$COND$COND_STATUS_CD == 1 & db$COND$SITECLCD %in% c(1, 2, 3, 4, 5, 6) & db$COND$RESERVCD == 0, 1, 0)

      # A general fix for land domain issues on timberland, i.e., trees on non-timberland being included
      db$TREE <- db$TREE %>%
        dplyr::left_join(dplyr::select(db$COND, PLT_CN, CONDID, landD), by = c('PLT_CN', 'CONDID')) %>%
        dplyr::mutate(typeD = landD) %>%
        dplyr::select(-c('landD'))

      # Tree Type domain indicator
      if (toupper(stateVar) %in% c('SAWVOL', 'SAWVOL_BF')) {
        # Rename some variables in grm
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_SL_TIMBER,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_SL_TIMBER,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_SL_TIMBER,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_SL_TIMBER,
                                               COMPONENT = SUBP_COMPONENT_SL_TIMBER) %>%
          dplyr::mutate(TPARECR_UNADJ = dplyr::case_when(
            is.na(COMPONENT) ~ NA_real_,
            # Note that currently defining removals to NOT include CUT2 and MORTALITY2.
            # COMPONENT %in% c('INGROWTH', 'CUT2', 'MORTALITY2') ~ TPAGROW_UNADJ,
            COMPONENT %in% c('INGROWTH') ~ TPAGROW_UNADJ,
            TRUE ~ 0))


      } else if (tolower(treeType) == 'all'){
        # Rename some variables in grm
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_AL_TIMBER,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_AL_TIMBER,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_AL_TIMBER,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_AL_TIMBER,
                                               COMPONENT = SUBP_COMPONENT_AL_TIMBER)%>%
          dplyr::mutate(TPARECR_UNADJ = dplyr::case_when(
            is.na(COMPONENT) ~ NA_real_,
            # Note that currently defining removals to NOT include CUT2 and MORTALITY2.
            # COMPONENT %in% c('INGROWTH', 'CUT2', 'MORTALITY2') ~ TPAGROW_UNADJ,
            COMPONENT %in% c('INGROWTH') ~ TPAGROW_UNADJ,
            TRUE ~ 0))

      } else if (tolower(treeType) == 'gs'){
        db$TREE_GRM_COMPONENT <- dplyr::rename(db$TREE_GRM_COMPONENT,
                                               TPAMORT_UNADJ = SUBP_TPAMORT_UNADJ_GS_TIMBER,
                                               TPAREMV_UNADJ = SUBP_TPAREMV_UNADJ_GS_TIMBER,
                                               TPAGROW_UNADJ = SUBP_TPAGROW_UNADJ_GS_TIMBER,
                                               SUBPTYP_GRM = SUBP_SUBPTYP_GRM_GS_TIMBER,
                                               COMPONENT = SUBP_COMPONENT_GS_TIMBER )%>%
          dplyr::mutate(TPARECR_UNADJ = dplyr::case_when(
            is.na(COMPONENT) ~ NA_real_,
            # Note that currently defining removals to NOT include CUT2 and MORTALITY2.
            # COMPONENT %in% c('INGROWTH', 'CUT2', 'MORTALITY2') ~ TPAGROW_UNADJ,
            COMPONENT %in% c('INGROWTH') ~ TPAGROW_UNADJ,
            TRUE ~ 0))

      }
    }
  }

  return(db)
}

# Build domain indicator for UD area domain
udAreaDomain <- function(db, areaDomain) {

  # Only evaluate if areaDomain isn't null
  if (!rlang::quo_is_null(areaDomain)) {
    # We'll join up PLOT and COND, and evaluate in the context of the joined table
    plt <- db$PLOT %>% 
      dplyr::filter(PLOT_STATUS_CD == 1)
    cnd <- db$COND %>% 
      dplyr::filter(COND_STATUS_CD == 1)


    pcEval <- dplyr::left_join(plt, 
      dplyr::select(cnd, -c('STATECD', 'UNITCD', 'COUNTYCD', 'INVYR', 'PLOT')), by = 'PLT_CN')
    pcEval$aD <- rlang::eval_tidy(areaDomain, pcEval) # LOGICAL, THIS IS THE DOMAIN INDICATOR
    if(!is.null(pcEval$aD)) pcEval$aD[is.na(pcEval$aD)] <- 0 # Make NAs 0s. Causes bugs otherwise
    if(is.null(pcEval$aD)) pcEval$aD <- 1 # IF NULL IS GIVEN, THEN ALL VALUES TRUE
    pcEval$aD <- as.numeric(pcEval$aD)
    # Any conditions involving variables in the PLOT table will be extended to the
    # domain indicator in the condition table, no need to "upscale" domain indicator
    # to the PLOT table
    db$COND <- db$COND %>%
      dplyr::left_join(dplyr::select(pcEval, c('PLT_CN', 'CONDID', 'aD')), by = c('PLT_CN', 'CONDID'))
  } else {
    db$COND$aD <- 1
  }

  return(db)
}

# Build domain indicator for UD area domain
udTreeDomain <- function(db, treeDomain) {

  if (!rlang::quo_is_null(treeDomain)) {
    tD <- rlang::eval_tidy(treeDomain, db$TREE) # LOGICAL, THIS IS THE DOMAIN INDICATOR
    if(!is.null(tD)) tD[is.na(tD)] <- 0 # Make NAs 0s. Causes bugs otherwise
    if(is.null(tD)) tD <- 1 # IF NULL IS GIVEN, THEN ALL VALUES TRUE
    db$TREE$tD <- as.numeric(tD)
  } else {
    db$TREE$tD <- 1
  }

  return(db)
}

# Build domain indicator for UD area domain
udSeedDomain <- function(db, treeDomain) {

  tD <- rlang::eval_tidy(treeDomain, db$SEEDLING) # LOGICAL, THIS IS THE DOMAIN INDICATOR
  if(!is.null(tD)) tD[is.na(tD)] <- 0 # Make NAs 0s. Causes bugs otherwise
  if(is.null(tD)) tD <- 1 # IF NULL IS GIVEN, THEN ALL VALUES TRUE
  db$SEEDLING$tD <- as.numeric(tD)

  return(db)
}

# Handle population tables
handlePops <- function(db, evalType, method, mr, pltList = NULL, ga = FALSE) {

  # Reset so we don't break design info
  class(db) <- 'FIA.Database'

  # Pull all design information for the relevant inventories
  pops <- getDesignInfo(db,
                        type = evalType,
                        mostRecent = FALSE) # Will have been carried out already

  # If a most-recent subset, make sure that we don't get two reporting years in
  # western states
  if (mr) {
    pops <- pops %>%
      dplyr::group_by(STATECD) %>%
      dplyr::filter(YEAR == max(YEAR, na.rm = TRUE)) %>%
      dplyr::ungroup()
  }

  # Dropping non-sampled plots for P3 variables, or dropping plots that don't 
  # have a remeasurement. 
  if (!is.null(pltList)) {
    pops <- pops %>%
      dplyr::filter(pops$PLT_CN %in% pltList)
  }

  # P2POINTCNT column is NOT consistent for annual estimates, plots
  # within individual strata and est units are related to different INVYRs

  # Only run if this wasn't already done in clipFIA
  # Calculate the number of P2 plots within each stratum and INVYR to get the number within 
  # a given stratum. Then also sum across stratum, INVYR, and ESTN_UNIT to get the number
  # within a given est unit and INVYR. 
  if (!any(c('P2PNTCNT_EU_INVYR', 'P2POINTCNT_INVYR') %in% names(pops)) & method != 'TI') {
    pops <- pops %>%
      dplyr::left_join(select(db$PLOT, PLT_CN, INVYR), by = 'PLT_CN') %>%
      dtplyr::lazy_dt() %>%
      # Count of plots by Stratum/INVYR
      dplyr::group_by(EVALID, ESTN_UNIT_CN, STRATUM_CN, INVYR) %>%
      dplyr::mutate(P2POINTCNT_INVYR = dplyr::n()) %>%
      dplyr::ungroup() %>%
      # By unit / INVYR
      dplyr::group_by(EVALID, ESTN_UNIT_CN, INVYR) %>%
      dplyr::mutate(P2PNTCNT_EU_INVYR = dplyr::n()) %>%
      dplyr::ungroup() %>%
      as.data.frame() %>%
      # Also get P1 values by estimation unit and strata. 
      dplyr::left_join(dplyr::select(db$POP_ESTN_UNIT, ESTN_UNIT_CN = CN, P1PNTCNT_EU), by = 'ESTN_UNIT_CN') %>%
      dplyr::left_join(dplyr::select(db$POP_STRATUM, STRATUM_CN = CN, P1POINTCNT), by = 'STRATUM_CN')
  }


  # Recode a few of the estimation methods to make things easier below
  pops <- pops %>%
    dplyr::mutate(ESTN_METHOD = dplyr::case_when(ESTN_METHOD %in% c("Post-Stratification", "Stratified random sampling") ~ 'strat',
                                                 ESTN_METHOD %in% c('Simple random sampling', 'Subsampling units of unequal size') ~ 'simple',
                                                 TRUE ~ 'double'))

  return(pops)

}

# When something other than temporally indifferent is used, we may need to merge small strata
# There is no great way to go about this, but very important we do it for variance issues.
# So, what we will do is:
# (1) Identify strata/INVYR pairs with less than 2 ground plots
# (2) For each of those pairs, identify their most similar neighbor based on fuzzy string 
#     matching of STRATUM Descriptions. Neighbors (other strata) must be from the same 
#     estimation unit and measured in the same year (INVYR).
# Important -- we are effectively allowing for stratification to vary across 
# panels within an inventory cycle. Also, we are adjusting strata weights by 
# INVYR within the cycle, i.e., the same stratum may be weighted differently in 
# different years within the same cycle.

mergeSmallStrata <- function(db, pops) {

  ## Make a unique ID for stratum / year pairs
  pops$stratID <- paste(pops$STRATUM_CN, pops$INVYR, sep = '_')
  # We'll allow strata weights to vary by INVYR w/ same strata because not all
  # strata are sampled in a given year
  pops$P1POINTCNT_INVYR <- pops$P1POINTCNT

  if (any(c('nStrata', 'nStrata_INVYR') %in% names(pops))) {
    # Stratum year pairs
    stratYr <- pops %>%
      left_join(select(db$POP_STRATUM, CN, STRATUM_DESCR), by = c('STRATUM_CN' = 'CN')) %>%
      distinct(STATECD, ESTN_UNIT_CN, STRATUM_CN, STRATUM_DESCR, INVYR,
               P2POINTCNT, P1POINTCNT, P2POINTCNT_INVYR, P1POINTCNT_INVYR, stratID,
               nStrata, nStrata_INVYR) %>%
      # If buffer is present in the name, then the stratum has a different intensity
      # than other strata in the same estimation unit (PNW only).
      # Only combine buffer w/ buffer
      mutate(buff = str_detect(STRATUM_DESCR, 'buff') & STATECD %in% c(53, 41, 6)) %>%
      mutate(wrong = P2POINTCNT_INVYR < 2) %>%
      arrange(P2POINTCNT_INVYR)
  } else {
    ## Stratum year pairs
    stratYr <- pops %>%
      left_join(select(db$POP_STRATUM, CN, STRATUM_DESCR), by = c('STRATUM_CN' = 'CN')) %>%
      distinct(STATECD, ESTN_UNIT_CN, STRATUM_CN, STRATUM_DESCR, INVYR,
               P2POINTCNT, P1POINTCNT, P2POINTCNT_INVYR, P1POINTCNT_INVYR, stratID) %>%
      # If buffer is present in the name, then the stratum has a different intensity
      # than other strata in the same estimation unit (PNW only).
      # Only combine buffer w/ buffer
      mutate(buff = str_detect(STRATUM_DESCR, 'buff') & STATECD %in% c(53, 41, 6)) %>%
      mutate(wrong = P2POINTCNT_INVYR < 2) %>%
      group_by(ESTN_UNIT_CN, INVYR) %>%
      mutate(nStrata_INVYR = length(unique(STRATUM_CN))) %>%
      group_by(ESTN_UNIT_CN) %>%
      mutate(nStrata = length(unique(STRATUM_CN))) %>%
      ungroup() %>%
      arrange(P2POINTCNT_INVYR)
  }


  # Check if any fail
  warnMe <- c()

  # If any are too small, i.e., only one plot --> do some merging
  if (sum(stratYr$wrong, na.rm = TRUE) > 0){

    for (i in stratYr$stratID[stratYr$wrong == 1]) {

      # Subset the row
      dat <- filter(stratYr, stratID == i)


      # Use fuzzy string matching if any other strata are available to merge with
      if (dat$nStrata_INVYR > 1) {
        # Find its nearest neighbor of those in the same estimation
        # unit and INVYR
        neighbors <- stratYr %>%
          filter(ESTN_UNIT_CN == dat$ESTN_UNIT_CN) %>%
          filter(INVYR == dat$INVYR) %>%
          filter(stratID != i)

        if (nrow(neighbors) < 1) {
          warnMe <- c(warnMe, TRUE)

        } else {
          warnMe <- c(warnMe, FALSE)

          # Find the most similar neighbor in terms of stratum description
          msn <- adist(dat$STRATUM_DESCR, neighbors$STRATUM_DESCR)
          msnID <- neighbors$stratID[which.min(msn)]


          # In pops, we want to update all rows of the giving and receiving strata
          # Where giving gets a change in STRATUM_CN, P1POINTCNT, and P2POINTCNT

          # Giving stratum ----------------------------------------------------
          pops[pops$stratID == i, 'STRATUM_CN'] <- unique(pops[pops$stratID == msnID, 'STRATUM_CN'])
          pops[pops$stratID == i, 'P1POINTCNT_INVYR'] <- unique(pops[pops$stratID == msnID, 'P1POINTCNT_INVYR']) + dat$P1POINTCNT_INVYR
          pops[pops$stratID == i, 'P2POINTCNT_INVYR'] <- unique(pops[pops$stratID == msnID, 'P2POINTCNT_INVYR']) + dat$P2POINTCNT_INVYR

          # Receiving stratum -------------------------------------------------
          pops[pops$stratID == msnID, 'P1POINTCNT_INVYR'] <- unique(pops[pops$stratID == msnID, 'P1POINTCNT_INVYR']) + dat$P1POINTCNT_INVYR
          pops[pops$stratID == msnID, 'P2POINTCNT_INVYR'] <- unique(pops[pops$stratID == msnID, 'P2POINTCNT_INVYR']) + dat$P2POINTCNT_INVYR

        }




        # If the small stratum is the only one available in the estimation unit in a given year,
        # merge the INVYR within the same stratum
      } else {

        # No other strata measured in the same year, so merge years instead
        neighbors <- stratYr %>%
          filter(STRATUM_CN == dat$STRATUM_CN) %>%
          filter(stratID != i)

        if (nrow(neighbors) > 0) {
          warnMe <- c(warnMe, FALSE)

          # Find the most similar neighbor in terms of INVYR
          msn <- abs(neighbors$INVYR - (dat$INVYR + .01))
          msnID <- neighbors$stratID[which.min(msn)]

          # Giving stratum ----------------------------------------------------
          pops[pops$stratID == i, 'P2POINTCNT_INVYR'] <- unique(pops[pops$stratID == msnID, 'P2POINTCNT_INVYR']) + dat$P2POINTCNT_INVYR
          pops[pops$stratID == i, 'INVYR'] <- unique(pops[pops$stratID == msnID, 'INVYR'])


          # Receiving stratum -------------------------------------------------
          pops[pops$stratID == msnID, 'P2POINTCNT_INVYR'] <- unique(pops[pops$stratID == msnID, 'P2POINTCNT_INVYR']) + dat$P2POINTCNT_INVYR

        } else {
          warnMe <- c(warnMe, TRUE)
        }
      }
    }

    # Keeps it from repeating above
    if (any(warnMe)) warning('Bad stratification, i.e., strata too small to compute variance of annual panels. If you are only interested in totals and/or ratio estimates, disregard this. However, if interested in variance (e.g., for confidence intervals) try using method = "TI".')


    # Update adjustment factors in estimation unit
    pops <- pops %>%
      distinct(EVALID, ESTN_UNIT_CN, STRATUM_CN, INVYR, PLT_CN, .keep_all = TRUE) %>%
      # Fix adjustment factors
      group_by(STRATUM_CN) %>%
      mutate(ADJ_FACTOR_MICR = mean(ADJ_FACTOR_MICR, na.rm = TRUE),
             ADJ_FACTOR_SUBP = mean(ADJ_FACTOR_SUBP, na.rm = TRUE),
             ADJ_FACTOR_MACR = mean(ADJ_FACTOR_MACR)) %>%
      ungroup()
  }


  # Whether any strata are small are not, we will almost surely need to adjust
  # strata weights for some INVYRs. Not all stratum are measured in each panel
  # within a cycle, and when this happens, we need to weight the observed strata
  # higher. If we don't, strata weights will not sum to 1 in some years and we
  # will grossly underestimate means/totals in some cases
  # Adjust stratum weights when not all strata are sampled in an INVYR
  stratYr <- pops %>%
    distinct(ESTN_UNIT_CN, STRATUM_CN, INVYR,
             P1POINTCNT_INVYR, P1PNTCNT_EU,
             P2POINTCNT_INVYR) %>%
    # If multiple stratum were merged onto one, choose the maximum value (i.e.,
    # the result of the last iteration)
    group_by(ESTN_UNIT_CN, STRATUM_CN, INVYR) %>%
    mutate(P1POINTCNT_INVYR = max(P1POINTCNT_INVYR),
           P2POINTCNT_INVYR = max(P2POINTCNT_INVYR)) %>%
    distinct() %>%
    # Note this is grouped by ESTN_UNIT_CN and STRATUM_CN since the last group_by
    # will drop INVYR. 
    mutate(stratWgt = P1POINTCNT_INVYR / P1PNTCNT_EU) %>%
    group_by(ESTN_UNIT_CN, INVYR) %>%
    mutate(propSampled = sum(stratWgt, na.rm = TRUE),
           stratWgt_INVYR = stratWgt / propSampled,
           P1POINTCNT_INVYR = P1PNTCNT_EU * stratWgt_INVYR,
           P2PNTCNT_EU_INVYR = sum(P2POINTCNT_INVYR, na.rm = TRUE)) %>%
    ungroup() %>%
    select(STRATUM_CN, INVYR, P1POINTCNT_INVYR, P2POINTCNT_INVYR, P2PNTCNT_EU_INVYR)

  pops <- pops %>%
    select(-c(P1POINTCNT_INVYR, stratID, P2POINTCNT_INVYR, P2PNTCNT_EU_INVYR)) %>%
    left_join(stratYr, by = c('STRATUM_CN', 'INVYR')) %>%
    distinct(EVALID, ESTN_UNIT_CN, STRATUM_CN, PLT_CN, .keep_all = TRUE)


  return(pops)

}

# For annual estimator, we use the most recent stratification for all years.
# Otherwise we won't be able to compute the covariance between panels, because
# stratum boundaries and assignments differ from year to year.
# TODO:
annualStrataHelper <- function(db, pops) {


  ## Remove at end
  pops <- handlePops(db, evalType = c('EXPVOL'), method, mr)



  ## Add unique plot identifier
  pops <- dplyr::left_join(pops, dplyr::select(db$PLOT, PLT_CN, pltID, MEASYEAR), by = 'PLT_CN')


  ## Keep only design info from the most recent inventory,
  ## then join the full plot list back on
  pops <- pops %>%
    dplyr::group_by(STATECD, EVAL_TYP) %>%
    dplyr::filter(YEAR == max(YEAR, na.rm = TRUE)) %>%
    dplyr::ungroup() %>%
    dplyr::select(-c(PLT_CN, MEASYEAR)) %>%
    ## Add full plot list
    dplyr::left_join(
      dplyr::distinct(
        dplyr::select(pops, PLT_CN, pltID, MEASYEAR)
      ), by = 'pltID') %>%
    ## Add eu/ strat descriptions
    dplyr::left_join(dplyr::select(db$POP_STRATUM, CN, STRATUM_DESCR), by = c('STRATUM_CN' = 'CN')) %>%
    dplyr::left_join(dplyr::select(db$POP_ESTN_UNIT, CN, ESTN_UNIT_DESCR), by = c('ESTN_UNIT_CN' = 'CN')) %>%
    ## Override this for annual only
    dplyr::mutate(INVYR = MEASYEAR) %>%
    ## update annual stratum point counts to use MEASYEAR
    dplyr::group_by(INVYR, ESTN_UNIT_CN, STRATUM_CN) %>%
    dplyr::mutate(P2POINTCNT_INVYR = length(unique(PLT_CN))) %>%
    dplyr::ungroup()

  ## Drop any years where estimation units are not sampled
  keep.these <- pops %>%
    dplyr::group_by(INVYR) %>%
    dplyr::mutate(full.area = dplyr::case_when(all(unique(pops$ESTN_UNIT_CN) %in% ESTN_UNIT_CN) ~ 1,
                                               TRUE ~ 0)) %>%
    dplyr::filter(full.area == 1) %>%
    dplyr::ungroup() %>%
    dplyr::distinct(INVYR)
  pops <- dplyr::filter(pops, INVYR %in% keep.these$INVYR)


  ## A list of all estimation/unit strata by year
  stratYr <- pops %>%
    dplyr::distinct(STATECD, INVYR,
                    ESTN_UNIT_CN, AREA_USED, P2PNTCNT_EU, ESTN_UNIT_DESCR, P1PNTCNT_EU,
                    STRATUM_CN, STRATUM_DESCR, P1POINTCNT, P2POINTCNT, P2POINTCNT_INVYR) %>%
    ## update annual eu point counts to use MEASYEAR
    dplyr::group_by(INVYR, ESTN_UNIT_CN) %>%
    dplyr::mutate(P2PNTCNT_EU_INVYR = sum(P2POINTCNT_INVYR)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(bad.eu = P2PNTCNT_EU_INVYR < 2) %>%
    dplyr::mutate(bad.strat = P2POINTCNT_INVYR < 2)


  ## Check if any fail
  warnMe <- c()

  ## If any estimation units are too small, i.e., only one plot --> do some merging
  if (sum(stratYr$bad.eu, na.rm = TRUE) > 0){

    for ( i in unique(stratYr$ESTN_UNIT_CN[stratYr$bad.eu == 1]) ) {

      ## Subset the row
      dat <- dplyr::filter(pops, ESTN_UNIT_CN == i) %>%
        dplyr::distinct(ESTN_UNIT_CN, ESTN_UNIT_DESCR, AREA_USED, P1PNTCNT_EU, P2PNTCNT_EU)



      ## Use fuzzy string matching to merge estimation units
      ## This is not ideal, but until FIA provides an objective
      ## means to determine sampling intensity within estimation
      ## units and/or stratum, this is the best we can do.
      neighbors <- pops %>%
        dplyr::filter(ESTN_UNIT_CN != dat$ESTN_UNIT_CN) %>%
        dplyr::distinct(ESTN_UNIT_CN, ESTN_UNIT_DESCR)

      if (nrow(neighbors) < 1) {
        warnMe <- c(warnMe, TRUE)

      } else {
        warnMe <- c(warnMe, FALSE)

        ## Find the most similar neighbor in terms of stratum description
        msn <- adist(dat$ESTN_UNIT_DESCR, neighbors$ESTN_UNIT_DESCR)
        msn <- dplyr::filter(pops, ESTN_UNIT_CN == neighbors$ESTN_UNIT_CN[which.min(msn)]) %>%
          dplyr::distinct(ESTN_UNIT_CN, P2PNTCNT_EU, AREA_USED, P1PNTCNT_EU)


        ## Recode the estimation units to carry out the merge
        pops <- pops %>%
          dplyr::mutate(AREA_USED = dplyr::case_when(
            ESTN_UNIT_CN == dat$ESTN_UNIT_CN ~ AREA_USED + msn$AREA_USED,
            ESTN_UNIT_CN == msn$ESTN_UNIT_CN ~ AREA_USED + dat$AREA_USED,
            TRUE ~ AREA_USED
          )) %>%
          dplyr::mutate(P2PNTCNT_EU = dplyr::case_when(
            ESTN_UNIT_CN == dat$ESTN_UNIT_CN ~ P2PNTCNT_EU + msn$P2PNTCNT_EU,
            ESTN_UNIT_CN == msn$ESTN_UNIT_CN ~ P2PNTCNT_EU + dat$P2PNTCNT_EU,
            TRUE ~ P2PNTCNT_EU
          )) %>%
          dplyr::mutate(P1PNTCNT_EU = dplyr::case_when(
            ESTN_UNIT_CN == dat$ESTN_UNIT_CN ~ P1PNTCNT_EU + msn$P1PNTCNT_EU,
            ESTN_UNIT_CN == msn$ESTN_UNIT_CN ~ P1PNTCNT_EU + dat$P1PNTCNT_EU,
            TRUE ~ P1PNTCNT_EU
          )) %>%
          dplyr::mutate(ESTN_UNIT_CN = dplyr::case_when(
            ESTN_UNIT_CN == dat$ESTN_UNIT_CN ~ msn$ESTN_UNIT_CN,
            TRUE ~ ESTN_UNIT_CN
          ))

      }
    } # Close for loop
  }



  ## If any strata are too small, i.e., only one plot --> do some merging
  if (sum(stratYr$bad.strat, na.rm = TRUE) > 0){

    for ( i in unique(stratYr$STRATUM_CN[stratYr$bad.strat == 1] )) {

      ## Previous update may fix future small strata, if so, leave it
      if (i %in% unique(pops$STRATUM_CN)) {

        ## Subset the row
        dat <- dplyr::filter(pops, STRATUM_CN == i) %>%
          dplyr::distinct(ESTN_UNIT_CN, STRATUM_CN, STRATUM_DESCR, P2POINTCNT, P1POINTCNT)



        ## Use fuzzy string matching to merge estimation units
        ## This is not ideal, but until FIA provides an objective
        ## means to determine sampling intensity within estimation
        ## units and/or stratum, this is the best we can do.
        neighbors <- pops %>%
          dplyr::distinct(ESTN_UNIT_CN, STRATUM_CN, STRATUM_DESCR, P2POINTCNT, P1POINTCNT) %>%
          dplyr::filter(ESTN_UNIT_CN == dat$ESTN_UNIT_CN) %>%
          dplyr::filter(STRATUM_CN != dat$STRATUM_CN)

        if (nrow(neighbors) < 1) {
          warnMe <- c(warnMe, TRUE)

        } else {
          warnMe <- c(warnMe, FALSE)

          ## Find the most similar neighbor in terms of stratum description
          msn <- adist(dat$STRATUM_DESCR, neighbors$STRATUM_DESCR)
          msn <- dplyr::filter(pops, STRATUM_CN == neighbors$STRATUM_CN[which.min(msn)]) %>%
            dplyr::distinct(ESTN_UNIT_CN, STRATUM_CN, STRATUM_DESCR, P2POINTCNT, P1POINTCNT)

          ## If we have a tie, random selection
          if (nrow(msn) > 1) {
            msn <- dplyr::sample_n(msn, size = 1)
          }


          ## Recode the estimation units to carry out the merge
          pops <- pops %>%
            dplyr::mutate(P2POINTCNT = dplyr::case_when(
              STRATUM_CN == dat$STRATUM_CN ~ P2POINTCNT + msn$P2POINTCNT,
              STRATUM_CN == msn$STRATUM_CN ~ P2POINTCNT + dat$P2POINTCNT,
              TRUE ~ P2POINTCNT
            )) %>%
            dplyr::mutate(P1POINTCNT = dplyr::case_when(
              STRATUM_CN == dat$STRATUM_CN ~ P1POINTCNT + msn$P1POINTCNT,
              STRATUM_CN == msn$STRATUM_CN ~ P1POINTCNT + dat$P1POINTCNT,
              TRUE ~ P1POINTCNT
            )) %>%
            dplyr::mutate(STRATUM_CN = dplyr::case_when(
              STRATUM_CN == dat$STRATUM_CN ~ msn$STRATUM_CN,
              TRUE ~ STRATUM_CN
            ))

        }
      }

    } # Close for loop
  }



  ## Keeps it from repeating above
  if (any(warnMe)) warning('Bad stratification, i.e., strata too small to compute variance of annual panels. If you are only interested in totals and/or ratio estimates, disregard this. However, if interested in variance (e.g., for confidence intervals) try using method = "TI".')


  ## Update adjustment factors in estimation unit
  pops <- pops %>%
    dplyr::distinct(EVALID, ESTN_UNIT_CN, STRATUM_CN, INVYR, PLT_CN, .keep_all = TRUE) %>%
    ## Fix adjustment factors
    dplyr::group_by(STRATUM_CN) %>%
    dplyr::mutate(ADJ_FACTOR_MICR = mean(ADJ_FACTOR_MICR, na.rm = TRUE),
                  ADJ_FACTOR_SUBP = mean(ADJ_FACTOR_SUBP, na.rm = TRUE),
                  ADJ_FACTOR_MACR = mean(ADJ_FACTOR_MACR)) %>%
    dplyr::ungroup() %>%
    ## Update stratum annual point counts
    dplyr::group_by(INVYR, ESTN_UNIT_CN, STRATUM_CN) %>%
    dplyr::mutate(P2POINTCNT_INVYR = length(unique(PLT_CN))) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(P1POINTCNT_INVYR = P1POINTCNT)


  ## Update eu annual point counts
  euP2 <- pops %>%
    dplyr::distinct(STATECD, INVYR,
                    ESTN_UNIT_CN, AREA_USED, P2PNTCNT_EU, ESTN_UNIT_DESCR, P1PNTCNT_EU,
                    STRATUM_CN, STRATUM_DESCR, P1POINTCNT, P2POINTCNT, P2POINTCNT_INVYR) %>%
    ## update annual eu point counts to use MEASYEAR
    dplyr::group_by(INVYR, ESTN_UNIT_CN) %>%
    dplyr::summarize(P2PNTCNT_EU_INVYR = sum(P2POINTCNT_INVYR)) %>%
    dplyr::ungroup()
  pops <- pops %>%
    dplyr::select(-c(P2PNTCNT_EU_INVYR)) %>%
    left_join(euP2, by = c('INVYR', 'ESTN_UNIT_CN'))


  return(pops)
}


# TODO: 
## Moving average weights
maWeights <- function(pops, method, lambda){

  ### ---- SIMPLE MOVING AVERAGE
  if (stringr::str_to_upper(method) == 'SMA'){
    ## Assuming a uniform weighting scheme
    wgts <- pops %>%
      dplyr::group_by(YEAR, STATECD) %>%
      dplyr::summarize(wgt = 1 / length(unique(INVYR)))

    #### ----- Linear MOVING AVERAGE
  } else if (stringr::str_to_upper(method) == 'LMA'){

    wgts <- pops %>%
      dplyr::distinct(YEAR, STATECD, INVYR, .keep_all = TRUE) %>%
      dplyr::arrange(YEAR, STATECD, INVYR) %>%
      dplyr::group_by(as.factor(YEAR), as.factor(STATECD)) %>%
      dplyr::mutate(rank = dplyr::min_rank(INVYR),
                    nsum = sum(1:dplyr::n())) %>%
      dplyr::ungroup() %>%
      dplyr::mutate(wgt = rank / nsum) %>%
      dplyr::ungroup() %>%
      dplyr::select(YEAR, STATECD, INVYR, wgt)


    #### ----- EXPONENTIAL MOVING AVERAGE
  } else if (stringr::str_to_upper(method) == 'EMA'){
    wgts <- pops %>%
      dplyr::distinct(YEAR, STATECD, INVYR, .keep_all = TRUE) %>%
      dplyr::arrange(YEAR, STATECD, INVYR) %>%
      dplyr::group_by(as.factor(YEAR), as.factor(STATECD)) %>%
      dplyr::mutate(rank = dplyr::min_rank(INVYR))


    if (length(lambda) < 2){
      ## Want sum of weighitng functions
      neu <- wgts %>%
        dplyr::mutate(l = lambda) %>%
        dplyr::group_by(YEAR, STATECD) %>%
        dplyr::summarize(l = 1 - dplyr::first(lambda),
                         sumwgt = sum(l*(1-l)^(1-rank), na.rm = TRUE))

      ## Rejoining and computing wgts
      wgts <- wgts %>%
        dplyr::left_join(neu, by = c('YEAR', 'STATECD')) %>%
        dplyr::mutate(wgt = l*(1-l)^(1-rank) / sumwgt) %>%
        dplyr::ungroup() %>%
        dplyr::select(YEAR, STATECD, INVYR, wgt)
    } else {
      ## Duplicate weights for each level of lambda
      yrWgts <- list()
      for (i in 1:length(unique(lambda))) {
        yrWgts[[i]] <- dplyr::mutate(wgts, lambda = lambda[i])
      }
      wgts <- dplyr::bind_rows(yrWgts)
      ## Want sum of weighitng functions
      neu <- wgts %>%
        dplyr::group_by(lambda, YEAR, STATECD) %>%
        dplyr::summarize(l = 1 - dplyr::first(lambda),
                         sumwgt = sum(l*(1-l)^(1-rank), na.rm = TRUE))

      ## Rejoining and computing wgts
      wgts <- wgts %>%
        dplyr::left_join(neu, by = c('lambda', 'YEAR', 'STATECD')) %>%
        dplyr::mutate(wgt = l*(1-l)^(1-rank) / sumwgt) %>%
        dplyr::ungroup() %>%
        dplyr::select(lambda, YEAR, STATECD, INVYR, wgt)
    }
  }

  return(wgts)
}


# Combine most-recent population estimates across states with potentially
# different reporting schedules, e.g., if 2016 is most recent in MI and 2017 is
# most recent in WI, combine them and label as 2017
combineMR <- function(x){
  out <- x %>%
    dplyr::ungroup() %>%
    dplyr::mutate(YEAR = max(YEAR, na.rm = TRUE))
  return(out)
}

# TODO: should just be able to delete this. 
# Make implicit NA explicit for spatial summaries
prettyNamesSF <- function (tOut, polys, byPlot, grpBy, grpByOrig, tNames, returnSpatial) {

  # Return a spatial object
  if (!is.null(polys) & byPlot == FALSE) {
    ## NO IMPLICIT NA
    nospGrp <- unique(grpBy[grpBy %in% c('SPCD', 'SYMBOL', 'COMMON_NAME', 'SCIENTIFIC_NAME') == FALSE])
    nospSym <- dplyr::syms(nospGrp)
    tOut <- tidyr::complete(tOut, !!!nospSym)
    # If species, we don't want unique combos of variables related to same species
    # but we do want NAs in polys where species are present
    if (length(nospGrp) < length(grpBy)){
      spGrp <- unique(grpBy[grpBy %in% c('SPCD', 'SYMBOL', 'COMMON_NAME', 'SCIENTIFIC_NAME')])
      spSym <- dplyr::syms(spGrp)
      tOut <- tidyr::complete(tOut, tidyr::nesting(!!!nospSym))
    }

    suppressMessages({suppressWarnings({
      tOut <- dplyr::left_join(tOut, polys) %>%
        dplyr::select(c('YEAR', grpByOrig, tNames, names(polys))) %>%
        dplyr::filter(!is.na(polyID))})})

    # Makes it horrible to work with as a dataframe
    if (returnSpatial == FALSE) tOut <- dplyr::select(tOut, -c(geometry))
  } else if (!is.null(polys) & byPlot){
    polys <- as.data.frame(polys)
    tOut <- dplyr::left_join(tOut, dplyr::select(polys, -c(geometry)), by = 'polyID')
  }

  return(tOut)
}

# TODO: 
## Choose annual panels to return
filterAnnual <- function(x, grpBy, pltsVar, ESTN_UNIT) {

  ## Have to handle statecd carefully in grp by
  if ('STATECD' %in% grpBy) {
    grpBy <- grpBy[!c(grpBy %in% 'STATECD')]
    x <- x %>%
      dplyr::ungroup() %>%
      dplyr::select(-c(STATECD))
  }
  pltquo <- rlang::enquo(pltsVar)
  x <- x %>%
    dplyr::left_join(dplyr::distinct(dplyr::select(ESTN_UNIT, CN, STATECD)), by = c('ESTN_UNIT_CN' = 'CN')) %>%
    dplyr::mutate(nplts = !!pltquo) %>%
    dplyr::group_by(STATECD, INVYR, .dots = grpBy[!c(grpBy %in% 'STATECD')]) %>%
    dplyr::summarize(dplyr::across(dplyr::everything(), \(x) sum(x, na.rm = TRUE))) %>% 
    ## Keep these
    dplyr::group_by(STATECD, INVYR, .dots = grpBy[!c(grpBy %in% 'STATECD')]) %>%
    dplyr::mutate(keep = ifelse(INVYR %in% YEAR,
                                ifelse(YEAR == INVYR, 1, 0), ## When TRUE
                                ifelse(nplts == max(nplts, na.rm = TRUE), 1, 0))) %>% ## When INVYR not in YEAR, keep estimates from the inventory where panel has the most plots
    dplyr::ungroup() %>%
    dplyr::filter(keep == 1) %>%
    ## If there are multiple reporting years where a panel has the same number of plots
    ## then the estimate will be way too big, we fix this by taking the first row from each output group
    ## If the above worked it will have no effect. If the above failed, it will save our ass.
    dplyr::mutate(YEAR = INVYR) %>%
    dplyr::group_by(STATECD, .dots = grpBy[!c(grpBy %in% 'STATECD')]) %>%
    dplyr::summarize(dplyr::across(.cols = dplyr::everything(), dplyr::first)) %>%
    dplyr::ungroup()

  return(x)
}




# Estimate skewness in a distribution of values
skewness <- function(x, na.rm = TRUE){

  # Cut any NA
  if (na.rm) x <- x[!is.na(x)]

  # Sample size
  n <- length(x)

  # Estimate the skewness
  skew <- (sum((x-mean(x))^3)/n)/(sum((x-mean(x))^2)/n)^(3/2)

  return(skew)
}

# TODO: 
projectPnts <- function(x, y, slope = NULL, yint = NULL){
  if (is.null(slope)){
    P = data.frame(xOrig = x, yOrig = y)
    P$x <- (P$yOrig+P$xOrig) / 2
    P$y <- P$x
  } else {
    P = data.frame(x, y)
    P$m <- slope
    P$n <- yint
    ## Perp Points
    P$x1 = P$x + -slope
    P$y1 = P$y + 1
    ## Perp Line
    P$m1 = (P$y1-P$y)/(P$x1-P$x)
    P$n1 = P$y - P$m1*P$x
    ## Line intersection
    P$x=(P$n1-P$n)/(P$m-P$m1)
    P$y=P$m*P$x+P$n
  }
  return(P)
}

# TODO: 
projectPoints <- function(x, y, slope = 1, yint = 0, returnPoint = TRUE){
  ## Solve for 1:1 line by default

  ## So where does y = mx and y = -1/m * x + b converge
  perp_slope <-  - 1 / slope
  ## Solve for c given x and y
  perp_int <- -perp_slope*x + y

  ## Set equations equal to each other on y
  ## -1/m*x + b = mx
  xproj <- (perp_int - yint) / (slope + -perp_slope)
  yproj <- slope * xproj + yint

  if (returnPoint){
    out <- data.frame(x = xproj, y = yproj)
  } else {
    out <- sqrt((xproj^2) + (yproj^2))
    out <- dplyr::if_else(xproj < 0, -out, out)
  }
  return(out)
}

# TODO: 
#### SHANNON'S EVENESS INDEX (H)
#
# speciesObs: vector of observations (species or unique ID)
#
# Returns evenness score (numeric)
divIndex <- function(SPCD, TPA, index) {
  # Shannon's Index
  if(index == 'H'){
    species <- unique(SPCD[TPA > 0 & !is.na(TPA)])
    total <- sum(TPA, na.rm = TRUE)

    p <- c() # Empty vector to hold proportions
    for (i in 1:length(species)){
      p[i] <- sum(TPA[SPCD == species[i]], na.rm = TRUE) / total
    }
    value <- -sum(p*log(p), na.rm = TRUE)
  }
  if(index == 'Eh'){
    species <- unique(SPCD[TPA > 0 & !is.na(TPA)])
    total <- sum(TPA, na.rm = TRUE)
    p <- c() # Empty vector to hold proportions
    for (i in 1:length(species)){
      p[i] <- sum(TPA[SPCD == species[i]], na.rm = TRUE) / total
    }
    S <- length(unique(SPCD[TPA > 0 & !is.na(TPA)]))
    if(S == 0) S <- NA
    value <- -sum(p*log(p), na.rm = TRUE) / S
  }
  # Richness
  if(index == 'S'){
    value = length(unique(SPCD[TPA > 0 & !is.na(TPA)])) ## Assumes equal probabilty of detection, not true because of nested sampling design
  }
  # Berger–Parker dominance
  if(index == 'BP'){
    species <- unique(SPCD[TPA > 0])
    total <- sum(TPA, na.rm = TRUE)
    p <- c() # Empty vector to hold proportions
    for (i in 1:length(species)){
      p[i] <- sum(TPA[SPCD == species[i]], na.rm = TRUE) / total
    }
    value <- max(p)
  }
  return(value)
}

areal_par <- function(x, pltSF, polys) {
  pltSF <- sf::st_intersection(pltSF, polys[[x]]) %>%
    as.data.frame() %>%
    dplyr::select(-c('geometry')) # removes artifact of SF object
}

# Exponenetially weighted moving average
ema <- function(x, yrs, var = FALSE){
  l <- 2 / (1 + dplyr::first(yrs))
  wgts <- c()
  for (i in 1:length(x)) wgts[i] <- l^(i-1)*(1-l)

  if (var){
    #out <- sum(wgts^2 * x,na.rm = TRUE)
    out <- wgts^2 * x
  } else {
    #out <- sum(wgts * x,na.rm = TRUE)
    out <- wgts * x
  }

  return(out)
}


# Basal Area Function (returns sq units of diameter)
basalArea <- function(diameter, DIA_MID = NULL){
  # This allows us to retain negative values for change functions.
  ba <- diameter * abs(diameter) * .005454

  return(ba)
}

# Classification of Stand Structural Stage 
# Classifies stand structural stage as pole, mature, late-successional, or mosaic
#  based on relative basal area of live canopy trees within pole, mature & large classes
#  diameter: stem DBH (inches) (DIA)
#  crownClass: canopy position of stem, suppressed and open grown excluded (CCLCD)
structHelper <- function(dia, crownClass){

  # Exclude suppressed (5) and open grown (1) stems from analysis
  dia = dia[crownClass %in% c(2,3,4)]

  # Total basal area within plot
  totalBA = sum(basalArea(dia[dia >= 5]), na.rm = TRUE)

  # Calculate proportion of stems in each size class by basal area
  pole = sum(basalArea(dia[dia >= 5 & dia < 10.23622]), na.rm = TRUE) / totalBA
  mature = sum(basalArea(dia[dia >= 10.23622 & dia < 18.11024]), na.rm = TRUE) / totalBA
  large = sum(basalArea(dia[dia >=18.11024]), na.rm = TRUE) / totalBA

  # Series of conditionals to identify stand structural stage based on basal
  #   area proportions in each size class
  if(is.nan(pole) | is.nan(mature) | is.nan(large)){
    stage = 'mosaic'
  } else if ( ((pole + mature) > .67) &  (pole > mature)){
    stage = 'pole'
  } else if(((pole + mature) > .67) &  (pole < mature)){
    stage = 'mature'
  } else if(((mature + large) > .67) & (mature > large)){
    stage = 'mature'
  } else if(((mature + large) > .67) & (mature < large)){
    stage = 'late'
  } else{
    stage = 'mosaic'
  }

  return(as.factor(stage))
}


# TODO:
# Prop basis helper
adjHelper <- function(DIA, MACRO_BREAKPOINT_DIA, ADJ_FACTOR_MICR, ADJ_FACTOR_SUBP, ADJ_FACTOR_MACR){
  # IF it doesnt exist make it massive
  MACRO_BREAKPOINT_DIA[is.na(MACRO_BREAKPOINT_DIA)] <- 10000
  # Replace DIA with adjustment factors
  adj <- DIA
  adj[is.na(DIA)] <- ADJ_FACTOR_SUBP[is.na(DIA)]
  adj[DIA < 5 & !is.na(DIA)] <- ADJ_FACTOR_MICR[DIA < 5 & !is.na(DIA)]
  adj[DIA >= 5 & DIA < MACRO_BREAKPOINT_DIA & !is.na(DIA)] <- ADJ_FACTOR_SUBP[DIA >= 5 & DIA < MACRO_BREAKPOINT_DIA & !is.na(DIA)]
  adj[DIA >= MACRO_BREAKPOINT_DIA & !is.na(DIA)] <- ADJ_FACTOR_MACR[DIA >= MACRO_BREAKPOINT_DIA & !is.na(DIA)]

  return(adj)

}

# TODO: 
# GRM adjustment helper
grmAdj <- function(subtyp, adjMicr, adjSubp, adjMacr) {

  data <- data.frame(typ = as.numeric(subtyp), micr = as.numeric(adjMicr), subp =as.numeric(adjSubp), macr =as.numeric(adjMacr))

  data <- data %>%
    dplyr::mutate(adj = dplyr::case_when(
      typ == 0 ~ 0,
      typ == 1 ~ subp,
      typ == 2 ~ micr,
      typ == 3 ~ macr,
    ))

  return(data$adj)
}

# TODO: 
# Helper function to compute variance for estimation units (manages different estimation methods)
unitVarDT <- function(method, ESTN_METHOD, a, nh, w, v, stratMean, stratMean1 = NULL){
  unitM <- unitMean(ESTN_METHOD, a, nh, w, stratMean)
  unitM1 <- unitMean(ESTN_METHOD, a, nh, w, stratMean1)
  if(method == 'var'){
    uv = ifelse(first(ESTN_METHOD) == 'strat',
                ((first(a)^2)/sum(nh)) * (sum(w*nh*v) + sum((1-w)*(nh/sum(nh))*v)),
                ifelse(first(ESTN_METHOD) == 'double',
                       (first(a)^2) * (sum(((nh-1)/(sum(nh)-1))*(nh/sum(nh))*v) + ((1/(sum(nh)-1))*sum((nh/sum(nh))*(stratMean - (unitM/first(a)))^2))),
                       sum(v))) # Stratified random case
  } else { # Compute covariance
    cv = ifelse(first(ESTN_METHOD) == 'strat',
                ((first(a)^2)/sum(nh)) * (sum(w*nh*v) + sum((1-w)*(nh/sum(nh))*v)),
                ifelse(first(ESTN_METHOD) == 'double',
                       (first(a)^2) * (sum(((nh-1)/(sum(nh)-1))*(nh/sum(nh))*v) + ((1/(sum(nh)-1))*sum((nh/sum(nh))*(stratMean - unitM) * (stratMean1 - (unitM1/first(a)))))),
                       sum(v))) # Stratified random case (additive covariance)
  }
}

# TODO: 
unitVar <- function(method, ESTN_METHOD, a, nh, w, v, stratMean, unitM, stratMean1 = NULL, unitM1 = NULL){
  if(method == 'var'){
    uv = ifelse(dplyr::first(ESTN_METHOD) == 'strat',
                ((dplyr::first(a)^2)/sum(nh)) * (sum(w*nh*v) + sum((1-w)*(nh/sum(nh))*v)),
                ifelse(dplyr::first(ESTN_METHOD) == 'double',
                       (dplyr::first(a)^2) * (sum(((nh-1)/(sum(nh)-1))*(nh/sum(nh))*v) + ((1/(sum(nh)-1))*sum((nh/sum(nh))*(stratMean - (unitM/dplyr::first(a)))^2))),
                       sum(v))) # Stratified random case
  } else {
    cv = ifelse(dplyr::first(ESTN_METHOD) == 'strat',
                ((dplyr::first(a)^2)/sum(nh)) * (sum(w*nh*v) + sum((1-w)*(nh/sum(nh))*v)),
                ifelse(dplyr::first(ESTN_METHOD) == 'double',
                       (dplyr::first(a)^2) * (sum(((nh-1)/(sum(nh)-1))*(nh/sum(nh))*v) + ((1/(sum(nh)-1))*sum((nh/sum(nh))*(stratMean - unitM) * (stratMean1 - (unitM1/dplyr::first(a)))))),
                       sum(v))) # Stratified random case (additive covariance)
  }
}

# TODO: 
unitVarNew <- function(method, ESTN_METHOD, a, nh, n, w, v, stratMean, unitM, stratMean1 = NULL, unitM1 = NULL){
  if(method == 'var'){
    uv = ifelse(dplyr::first(ESTN_METHOD) == 'strat',
                ((dplyr::first(a)^2)/n) * (sum(w*nh*v, na.rm = TRUE) + sum((1-w)*(nh/n)*v, na.rm = TRUE)),
                ifelse(dplyr::first(ESTN_METHOD) == 'double',
                       (dplyr::first(a)^2) * (sum(((nh-1)/(n-1))*(nh/n)*v, na.rm = TRUE) + ((1/(n-1))*sum((nh/n)*(stratMean - (unitM/dplyr::first(a)))^2, na.rm = TRUE))),
                       sum(v, na.rm = TRUE))) # Stratified random case
  } else {
    cv = ifelse(dplyr::first(ESTN_METHOD) == 'strat',
                ((dplyr::first(a)^2)/n) * (sum(w*nh*v, na.rm = TRUE) + sum((1-w)*(nh/n)*v, na.rm = TRUE)),
                ifelse(dplyr::first(ESTN_METHOD) == 'double',
                       (dplyr::first(a)^2) * (sum(((nh-1)/(n-1))*(nh/n)*v, na.rm = TRUE) + ((1/(n-1))*sum((nh/n)*(stratMean - unitM) * (stratMean1 - (unitM1/dplyr::first(a))), na.rm = TRUE))),
                       sum(v))) # Stratified random case (additive covariance)
  }
}

# Compute ratio variances at the estimation unit level
rVar <- function(x, y, xVar, yVar, xyCov){
  ## Ratio
  r <- y / x
  ## Ratio variance
  rv <- (1 / x^2) * (yVar + (r^2 * xVar) - (2 * r * xyCov))

  return(rv)
}

# TODO: 
# Helper function to compute variance for estimation units (manages different estimation methods)
unitMean <- function(ESTN_METHOD, a, nh, w, stratMean){
  um = ifelse(dplyr::first(ESTN_METHOD) == 'strat',
              sum(stratMean * w, na.rm = TRUE) * dplyr::first(a),
              ifelse(dplyr::first(ESTN_METHOD) == 'double',
                     sum(stratMean * (nh / sum(nh)), na.rm = TRUE) * dplyr::first(a),
                     mean(stratMean, na.rm = TRUE) * dplyr::first(a))) # Simple random case
}

# Replace current attributes with midpoint attributes depending on component
vrAttHelper <- function(attribute, attribute.prev, attribute.mid, attribute.beg, component, remper, oneortwo) {

  # ONLY WORKS FOR ATTRIBUTES DEFINED IN TRE_MIDPNT and TRE_BEGIN
  at <- dplyr::case_when(
    oneortwo == 2 ~ dplyr::case_when(
      stringr::str_detect(component, c('SURVIVOR|INGROWTH|REVERSION')) ~ attribute / remper,
      stringr::str_detect(component, c('CUT|DIVERSION')) ~ attribute.mid / remper),
    oneortwo == 1 ~ dplyr::case_when(
      stringr::str_detect(component, c('SURVIVOR|CUT1|DIVERSION1|MORTALITY1')) ~ dplyr::case_when(
        !is.na(attribute.beg) ~ - attribute.beg / remper,
        TRUE ~ - attribute.prev / remper)))

  return(at)
}

# TODO: 
ratioVar <- function(x, y, x.var, y.var, cv) {
  r.var <- (1 / (y^2)) * (x.var + ((x/y)^2 * y.var) - (2 * (x/y) * cv) )
  # Sometimes rounding errors in covariance estimate cause slightly negative
  # variance estimates for ratios, when this is the case, report 0
  r.var <- dplyr::case_when(r.var < 0 ~ 0,
                            TRUE ~ r.var)
  return(r.var)
}

# TODO: 
sumToPlot <- function(x,
                      pops,
                      grpBy) {

  ## Convert to syms so we can use in dplyr functions
  grp.syms <- syms(grpBy)

  ## Sum x variable(s) up to plot-level
  if ('TREE_BASIS' %in% names(x)) {

    ## Allows us to use across in summary functions
    x.vars <- syms(names(x)[!c(names(x) %in% c('PLT_CN', 'TREE_BASIS',
                                               'CONDID', 'SUBP', 'TREE', 'ONEORTWO',
                                               grpBy))])

    ## Sum up to plot
    x <- x %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(PLT_CN, TREE_BASIS, !!!grp.syms) %>%
      dplyr::summarize(dplyr::across(.cols = c(!!!x.vars), \(x) sum(x, na.rm = TRUE))) %>% 
      dplyr::ungroup() %>%

      ## Join population tables w/ adjustment factors for non-response
      dplyr::inner_join(dplyr::select(pops, ESTN_UNIT_CN, STRATUM_CN, PLT_CN,
                                      YEAR, dplyr::any_of('INVYR'),
                                      ADJ_FACTOR_MICR, ADJ_FACTOR_SUBP,
                                      ADJ_FACTOR_MACR), by = 'PLT_CN') %>%

      # Add adjustment factors
      dplyr::mutate(adj = dplyr::case_when(is.na(TREE_BASIS) ~ NA_real_,
                                            ## If the proportion was measured for a macroplot,
                                            ## use the macroplot value
                                            TREE_BASIS == 'MACR' ~ as.numeric(ADJ_FACTOR_MACR),
                                            ## Otherwise, use the subpplot value
                                            TREE_BASIS == 'SUBP' ~ as.numeric(ADJ_FACTOR_SUBP),
                                            TREE_BASIS == 'MICR' ~ as.numeric(ADJ_FACTOR_MICR))) %>%
      ## Apply adjustment
      dplyr::mutate(across(c(!!!x.vars), .fns = ~ .x * adj)) %>%
      dplyr::distinct() %>% # If multiple EVAL_TYP, drop those that we can
      ## Sum across micro, subp, macro
      dplyr::group_by(ESTN_UNIT_CN, STRATUM_CN, PLT_CN, !!!grp.syms) %>%
      dplyr::summarize(dplyr::across(.cols = c(!!!x.vars), \(x) sum(x, na.rm = TRUE))) %>% 
      as.data.frame()

  } else {

    ## If not tree variables, then condition variables
    x.vars <- syms(names(x)[!c(names(x) %in% c('PLT_CN', 'CONDID',
                                               'AREA_BASIS',
                                               grpBy))])
    # Sum up to plot
    x <- x %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(PLT_CN, AREA_BASIS, !!!grp.syms) %>%
      dplyr::summarize(dplyr::across(.cols = c(!!!x.vars), \(x) sum(x, na.rm = TRUE))) %>% 
      dplyr::ungroup() %>%

      ## Join population tables w/ adjustment factors for non-response
      dplyr::inner_join(dplyr::select(pops, EVALID, ESTN_UNIT_CN, STRATUM_CN, PLT_CN,
                                      YEAR, any_of('INVYR'),
                                      ADJ_FACTOR_MICR, ADJ_FACTOR_SUBP,
                                      ADJ_FACTOR_MACR), by = 'PLT_CN') %>%

      # Add adjustment factors
      dplyr::mutate(adj = dplyr::case_when(
        ## When NA, stay NA
        is.na(AREA_BASIS) ~ NA_real_,
        ## If the proportion was measured for a macroplot,
        ## use the macroplot value
        AREA_BASIS == 'MACR' ~ as.numeric(ADJ_FACTOR_MACR),
        ## Otherwise, use the subplot value
        AREA_BASIS == 'SUBP' ~ ADJ_FACTOR_SUBP)) %>%
      ## Apply adjustment
      dplyr::mutate(across(c(!!!x.vars), .fns = ~ .x * adj)) %>%
      dplyr::distinct() %>% # If multiple EVAL_TYP, drop those that we can
      ## Sum across micro, subp, macro
      dplyr::group_by(ESTN_UNIT_CN, STRATUM_CN, PLT_CN, !!!grp.syms) %>%
      dplyr::summarize(dplyr::across(.cols = c(!!!x.vars), \(x) sum(x, na.rm = TRUE))) %>% 
      dplyr::ungroup() %>%
      as.data.frame()

  }

  return(x)
}

# TODO: 
sumToEU <- function(db,
                    x,
                    y = NULL,
                    pops,
                    x.grpBy,
                    y.grpBy = NULL,
                    method,
                    lambda = NULL) {

  ## If not temporally indifferent, group by INVYR as well
  if (!c(stringr::str_to_upper(method) %in% 'TI')) {
    # We'll remove this after summary to estimation unit
    x.grpBy <- c(x.grpBy, 'INVYR')
    y.grpBy <- c(y.grpBy, 'INVYR')
    pops$P2POINTCNT <- pops$P2POINTCNT_INVYR
    pops$P1POINTCNT <- pops$P1POINTCNT_INVYR
    pops$STRATUM_WGT <- pops$P1POINTCNT_INVYR / pops$P1PNTCNT_EU
    pops$P2PNTCNT_EU <- pops$P2PNTCNT_EU_INVYR

    ## Add INVYR to plot tables
    x <- dplyr::left_join(x, dplyr::select(db$PLOT, PLT_CN, INVYR), by = 'PLT_CN')
    if (!is.null(y)) y <- dplyr::left_join(y, dplyr::select(db$PLOT, PLT_CN, INVYR), by = 'PLT_CN')

    ## Only join by INVYR when using panels
    panel.join.vars <- 'INVYR'
  } else {
    panel.join.vars <- NULL
  }



  ## Convert to syms for dplyr
  x.grp.syms <- dplyr::syms(x.grpBy)
  y.grp.syms <- dplyr::syms(y.grpBy)

  ## Variables names for scoping
  x.var <- names(x)[!c(names(x) %in% c('ESTN_UNIT_CN', 'STRATUM_CN', 'PLT_CN', 'INVYR', x.grpBy))]
  x.var.syms <- dplyr::syms(x.var)
  x.var.m.syms <- dplyr::syms(paste0(x.var, '_mean'))
  x.var.v.syms <- dplyr::syms(paste0(x.var, '_var'))
  x.var.c.syms <- dplyr::syms(paste0(x.var, '_cv'))
  if (!is.null(y)) {
    y.var <- names(y)[!c(names(y) %in% c('ESTN_UNIT_CN', 'STRATUM_CN', 'PLT_CN', 'INVYR', y.grpBy))]
    y.var.syms <- dplyr::sym(y.var)
    y.var.m.syms <- dplyr::sym(paste0(y.var, '_mean'))
    y.var.v.syms <- dplyr::sym(paste0(y.var, '_var'))

  }

  ## Relevant design info at each level
  pops.sub <- pops %>%
    dplyr::select(YEAR, dplyr::any_of('INVYR'), ESTN_UNIT_CN, AREA_USED,
                  P2PNTCNT_EU, STRATUM_CN, STRATUM_WGT, P2POINTCNT) %>%
    dplyr::distinct()

  ## If y is specified
  if (!is.null(y)) {

    ## Strata means, variances for denominator
    yStrat <- y %>%
      dtplyr::lazy_dt() %>%
      dplyr::left_join(pops.sub, by = c('ESTN_UNIT_CN', 'STRATUM_CN', panel.join.vars)) %>%
      dplyr::group_by(ESTN_UNIT_CN, AREA_USED,
                      P2PNTCNT_EU, STRATUM_CN, STRATUM_WGT,
                      P2POINTCNT, !!!y.grp.syms) %>%
      dplyr::summarize(dplyr::across(c(!!y.var.syms),
                              .fns = list(
                                mean = ~ sum(.x / P2POINTCNT, na.rm = TRUE),
                                var = ~ (sum(.x^2, na.rm = TRUE) - (P2POINTCNT * (sum(.x / P2POINTCNT, na.rm = TRUE)^2))) / (P2POINTCNT * (P2POINTCNT - 1))
                              )),
                       nPlots.y = length(unique(PLT_CN))) %>%
      dplyr::ungroup() %>%
      as.data.frame()

    ## EU means, variances for denominator
    yEU <- yStrat %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(ESTN_UNIT_CN, AREA_USED, P2PNTCNT_EU, !!!y.grp.syms) %>%
      dplyr::summarise(dplyr::across(c(!!y.var.m.syms),
                              .fns = ~ AREA_USED * sum(.x * STRATUM_WGT, na.rm = TRUE)
                              ),
                       dplyr::across(c(!!y.var.v.syms),
                              .fns = ~ (AREA_USED^2 / P2PNTCNT_EU) * (sum(.x * STRATUM_WGT * P2POINTCNT, na.rm = TRUE) + sum(.x * (1-STRATUM_WGT) * (P2POINTCNT / P2PNTCNT_EU), na.rm = TRUE))
                              ),
                       nPlots.y = sum(nPlots.y, na.rm = TRUE)) %>%
      dplyr::ungroup() %>%
      as.data.frame()


    ## Strata means, variances for numerator
    xStrat <- x %>%
      dtplyr::lazy_dt() %>%
      dplyr::left_join(y, by = c('ESTN_UNIT_CN', 'STRATUM_CN', 'PLT_CN', panel.join.vars, y.grpBy[!c(y.grpBy %in% c('YEAR', 'INVYR'))])) %>%
      dplyr::left_join(dplyr::select(yStrat, ESTN_UNIT_CN, STRATUM_CN, !!!y.grp.syms, !!y.var.m.syms), by = c('ESTN_UNIT_CN', 'STRATUM_CN', panel.join.vars, y.grpBy[!c(y.grpBy %in% c('YEAR', 'INVYR'))])) %>%
      dplyr::left_join(dplyr::select(pops.sub, -c(any_of(c('YEAR')))), by = c('ESTN_UNIT_CN', 'STRATUM_CN', panel.join.vars)) %>%
      dplyr::group_by(ESTN_UNIT_CN, AREA_USED,
                      P2PNTCNT_EU, STRATUM_CN, STRATUM_WGT,
                      P2POINTCNT, !!!x.grp.syms, !!y.var.m.syms) %>%
      dplyr::summarize(dplyr::across(c(!!!x.var.syms), .fns = list(mean = ~ sum(.x / P2POINTCNT, na.rm = TRUE),
                                                        var = ~ (sum(.x^2, na.rm = TRUE) - (P2POINTCNT * sum(.x / P2POINTCNT, na.rm = TRUE)^2)) / (P2POINTCNT * (P2POINTCNT - 1)),
                                                        cv = ~ (sum(.x * !!y.var.syms, na.rm = TRUE) - (P2POINTCNT * sum(.x / P2POINTCNT, na.rm = TRUE) * !!y.var.m.syms)) / (P2POINTCNT * (P2POINTCNT - 1)) )),
                       nPlots.x = length(unique(PLT_CN))) %>%
      dplyr::ungroup() %>%
      as.data.frame()

    ## EU means, variances for numerator
    xEU <- xStrat %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(ESTN_UNIT_CN, AREA_USED, P2PNTCNT_EU, !!!x.grp.syms) %>%
      dplyr::summarise(dplyr::across(c(!!!x.var.m.syms),
                              .fns = ~ AREA_USED * sum(.x * STRATUM_WGT, na.rm = TRUE)
                              ),
                       dplyr::across(c(!!!x.var.v.syms),
                              .fns = ~ (AREA_USED^2 / P2PNTCNT_EU) * (sum(.x * STRATUM_WGT * P2POINTCNT, na.rm = TRUE) + sum(.x * (1-STRATUM_WGT) * (P2POINTCNT / P2PNTCNT_EU), na.rm = TRUE))
                              ),
                       dplyr::across(c(!!!x.var.c.syms),
                              .fns = ~ (AREA_USED^2 / P2PNTCNT_EU) * (sum(.x * STRATUM_WGT * P2POINTCNT, na.rm = TRUE) + sum(.x * (1-STRATUM_WGT) * (P2POINTCNT / P2PNTCNT_EU), na.rm = TRUE))
                              ),
                       nPlots.x = sum(nPlots.x, na.rm = TRUE)) %>%
      dplyr::ungroup() %>%
      as.data.frame()


    ## Compute moving average weights if not TI ----------------------------------
    if (stringr::str_to_upper(method) %in% c("SMA", 'EMA', 'LMA')){

      ## Only needed INVYR for pop est
      x.grpBy <- x.grpBy[x.grpBy != 'INVYR']
      y.grpBy <- y.grpBy[y.grpBy != 'INVYR']
      x.grp.syms <- dplyr::syms(x.grpBy)
      y.grp.syms <- dplyr::syms(y.grpBy)

      ## Compute the weights
      wgts <- maWeights(pops, method, lambda)

      ## If moving average ribbons, add lambda to grpBy for easier summary
      if (stringr::str_to_upper(method) == 'EMA' & length(lambda) > 1){
        x.grpBy <- c('lambda', x.grpBy)
        y.grpBy <- c('lambda', y.grpBy)
      }


      ## Apply the weights
      if (stringr::str_to_upper(method) %in% c('LMA', 'EMA')){
        joinCols <- c('YEAR', 'STATECD', 'INVYR')
      } else {
        joinCols <- c('YEAR', 'STATECD')
      }

      xEU <- xEU %>%
        dplyr::left_join(dplyr::select(db$POP_ESTN_UNIT, CN, STATECD), by = c('ESTN_UNIT_CN' = 'CN')) %>%
        dplyr::left_join(wgts, by = joinCols) %>%
        dplyr::mutate(dplyr::across(c(!!!x.var.m.syms), ~(.x * wgt))) %>%
        dplyr::mutate(dplyr::across(c(!!!x.var.v.syms, !!!x.var.c.syms), ~(.x * (wgt^2)))) %>%
        dplyr::group_by(ESTN_UNIT_CN, P2PNTCNT_EU, !!!x.grp.syms) %>%
        dplyr::summarize(dplyr::across(c(!!!x.var.m.syms, !!!x.var.v.syms, !!!x.var.c.syms, nPlots.x), \(x) sum(x, na.rm = TRUE)))
      yEU <- yEU %>%
        dplyr::left_join(dplyr::select(db$POP_ESTN_UNIT, CN, STATECD), by = c('ESTN_UNIT_CN' = 'CN')) %>%
        dplyr::left_join(wgts, by = joinCols) %>%
        dplyr::mutate(dplyr::across(c(!!y.var.m.syms), ~(.x * wgt))) %>%
        dplyr::mutate(dplyr::across(c(!!y.var.v.syms), ~(.x * (wgt^2)))) %>%
        dplyr::group_by(ESTN_UNIT_CN, !!!y.grp.syms) %>%
        dplyr::summarize(dplyr::across(c(!!y.var.m.syms, !!y.var.v.syms, nPlots.y), \(x) sum(x, na.rm = TRUE)))



      ## If using an ANNUAL estimator --------------------------------------------
    } else if (stringr::str_to_upper(method) == 'ANNUAL') {

      ## Only needed INVYR for pop est
      x.grpBy <- x.grpBy[x.grpBy != 'INVYR']
      y.grpBy <- y.grpBy[y.grpBy != 'INVYR']


      # If INVYR is in YEAR, choose the estimates when INVYR == YEAR
      # Otherwise, choose the estimates produced with the most plots
      xEU <- filterAnnual(xEU, x.grpBy, nPlots.x, db$POP_ESTN_UNIT)
      yEU <- filterAnnual(yEU, y.grpBy, nPlots.y, db$POP_ESTN_UNIT)

    } else if (stringr::str_to_upper(method) == 'ANNUAL_COV') {

      xEU$YEAR <- xEU$INVYR
      yEU$YEAR <- yEU$INVYR
    }


  } else {
    ## Otherwise just strata mean and variance for x

    yEU <- NULL

    ## Strata means, variances for numerator
    xStrat <- x %>%
      dtplyr::lazy_dt() %>%
      dplyr::left_join(pops.sub, by = c('ESTN_UNIT_CN', 'STRATUM_CN', panel.join.vars)) %>%
      dplyr::group_by(ESTN_UNIT_CN, AREA_USED,
                      P2PNTCNT_EU, STRATUM_CN, STRATUM_WGT,
                      P2POINTCNT, !!!x.grp.syms) %>%
      dplyr::summarize(dplyr::across(c(!!!x.var.syms), .fns = list(mean = ~ sum(.x / P2POINTCNT, na.rm = TRUE),
                                                            var = ~ (sum(.x^2, na.rm = TRUE) - (P2POINTCNT * sum(.x / P2POINTCNT, na.rm = TRUE)^2)) / (P2POINTCNT * (P2POINTCNT - 1))
                                                            )),
                       nPlots.x = length(unique(PLT_CN))) %>%
      dplyr::ungroup() %>%
      as.data.frame()

    ## EU means, variances for numerator
    xEU <- xStrat %>%
      dtplyr::lazy_dt() %>%
      dplyr::group_by(ESTN_UNIT_CN, AREA_USED, P2PNTCNT_EU, !!!x.grp.syms) %>%
      dplyr::summarise(dplyr::across(c(!!!x.var.m.syms),
                              .fns = ~ AREA_USED * sum(.x * STRATUM_WGT, na.rm = TRUE)
                              ),
                       dplyr::across(c(!!!x.var.v.syms),
                              .fns = ~ (AREA_USED^2 / P2PNTCNT_EU) * (sum(.x * STRATUM_WGT * P2POINTCNT, na.rm = TRUE) + sum(.x * (1-STRATUM_WGT) * (P2POINTCNT / P2PNTCNT_EU), na.rm = TRUE))
                              ),
                       nPlots.x = sum(nPlots.x, na.rm = TRUE)) %>%
      dplyr::ungroup() %>%
      as.data.frame()


    ## Compute moving average weights if not TI ----------------------------------
    if (stringr::str_to_upper(method) %in% c("SMA", 'EMA', 'LMA')){

      ## Only needed INVYR for pop est
      x.grpBy <- x.grpBy[x.grpBy != 'INVYR']
      x.grp.syms <- dplyr::syms(x.grpBy)

      ## Compute the weights
      wgts <- maWeights(pops, method, lambda)

      ## If moving average ribbons, add lambda to grpBy for easier summary
      if (stringr::str_to_upper(method) == 'EMA' & length(lambda) > 1){
        x.grpBy <- c('lambda', x.grpBy)
      }


      ## Apply the weights
      if (stringr::str_to_upper(method) %in% c('LMA', 'EMA')){
        joinCols <- c('YEAR', 'STATECD', 'INVYR')
      } else {
        joinCols <- c('YEAR', 'STATECD')
      }

      xEU <- xEU %>%
        dplyr::left_join(dplyr::select(db$POP_ESTN_UNIT, CN, STATECD), by = c('ESTN_UNIT_CN' = 'CN')) %>%
        dplyr::left_join(wgts, by = joinCols) %>%
        dplyr::mutate(dplyr::across(c(!!!x.var.m.syms), ~(.x * wgt))) %>%
        dplyr::mutate(dplyr::across(c(!!!x.var.v.syms), ~(.x * (wgt^2)))) %>%
        dplyr::group_by(ESTN_UNIT_CN, P2PNTCNT_EU, !!!x.grp.syms) %>%
        dplyr::summarize(dplyr::across(c(!!!x.var.m.syms, !!!x.var.v.syms, nPlots.x), \(x) sum(x, na.rm = TRUE))) %>%
        dplyr::ungroup()



      ## If using an ANNUAL estimator --------------------------------------------
    } else if (stringr::str_to_upper(method) == 'ANNUAL') {

      ## Only needed INVYR for pop est
      x.grpBy <- x.grpBy[x.grpBy != 'INVYR']


      # If INVYR is in YEAR, choose the estimates when INVYR == YEAR
      # Otherwise, choose the estimates produced with the most plots
      xEU <- filterAnnual(xEU, x.grpBy, nPlots.x, db$POP_ESTN_UNIT)

    } else if (stringr::str_to_upper(method) == 'ANNUAL_COV') {

      xEU$YEAR <- xEU$INVYR
    }

  }

  return(list(x = xEU, y = yEU))
}


# Sometimes these read as dates, and that breaks readFIA when mutliple states are read
# Maybe also think about dropping uncommon columns from TREE. Could save a lot of memory
dropTheseCols <- function() {

  dropThese <- c("CREATED_BY", "CREATED_IN_INSTANCE", "CREATED_DATE", "MODIFIED_BY", "MODIFIED_IN_INSTANCE", "MODIFIED_DATE")
  dropThese <- c(dropThese, tolower(dropThese))
  return(dropThese)
}

# Read FIA data from a database connection
readFIA.db <- function(con,
                       schema,
                       states,
                       tables,
                       common,
                       inMemory) {

  # Load all data now, or wait until later (process state-by-state)?
  if (is.null(inMemory) | !is.logical(inMemory)) stop('`inMemory` must be TRUE/FALSE.')
  if (inMemory) {

    # Check that specified states are valid -------------------------------------
    if (is.null(states)) states <- pull_all_states(con, schema)
    states <- unique(stringr::str_to_upper(states))
    stateNames <- intData$stateNames
    badStates <- states[!c(states %in% stateNames$STATEAB)]
    if (length(badStates) > 0) {
      if (any(stringr::str_length(badStates) > 2)) {
        stop (paste0('The following states are not supported: ',
                     paste(badStates, collapse = ', '),
                     '. Note that state abbreviations are required in place of full names (e.g., "MI" instead of "Michigan").'))
      } else {
        stop (paste0('The following states are not supported: ',
                     paste(badStates, collapse = ', '), '.'))
      }
    }

    # Check that specified tables are valid -------------------------------------
    commonTables <- c('COND', 'COND_DWM_CALC', 'INVASIVE_SUBPLOT_SPP', 'PLOT',
                      'POP_ESTN_UNIT','POP_EVAL', 'POP_EVAL_GRP', 'POP_EVAL_TYP',
                      'POP_PLOT_STRATUM_ASSGN', 'POP_STRATUM', 'SUBPLOT', 'TREE',
                      'TREE_GRM_COMPONENT', 'TREE_GRM_MIDPT', 'TREE_GRM_BEGIN',
                      'SUBP_COND_CHNG_MTRX', 'SEEDLING', 'SURVEY', 'SUBP_COND',
                      'P2VEG_SUBP_STRUCTURE', 'PLOTGEOM')
    allTables <- sort(c(commonTables,
                        'COUNTY', 'SUBP_COND', 'BOUNDARY', 'TREE_WOODLAND_STEMS',
                        'TREE_REGIONAL_BIOMASS', 'TREE_GRM_ESTN', 'SITETREE',
                        'P2VEG_SUBPLOT_SPP', 'GRND_CVR', 'DWM_VISIT',
                        'DWM_COARSE_WOODY_DEBRIS', 'DWM_DUFF_LITTER_FUEL',
                        'DWM_FINE_WOODY_DEBRIS', 'DWM_MICROPLOT_FUEL',
                        'DWM_RESIDUAL_PILE', 'DWM_TRANSECT_SEGMENT',
                        'COND_DWM_CALC', 'PLOT_REGEN', 'SUBPLOT_REGEN',
                        'SEEDLING_REGEN', 'POP_EVAL_ATTRIBUTE', 'PLOTGEOM',
                        'PLOTSNAP'))

    # If `tables` isn't specified, default to common or all
    if (is.null(common) | !is.logical(common)) stop("'common' must be TRUE/FALSE.")
    if (is.null(tables) & common) tables <- commonTables
    if (is.null(tables) & !common) tables <- allTables

    # Checking all tables are cool
    tables <- unique(stringr::str_to_upper(tables))
    badTables <- tables[!c(tables %in% allTables)]
    if (length(badTables) > 0) {
      stop (paste0('The following tables are not supported: ',
                   paste(badTables, collapse = ', '), '.'))
    }

    # Load tables -------------------------------------------------------------
    out <- lapply(X = tables,
                  FUN = read_fia_table_from_db,
                  con = con,
                  schema = schema,
                  states = states)
    names(out) <- tables

    # POP_EVAL_TYP doesn't have a STATECD variable, but we can filter it by
    # EVAL_CN if available. Otherwise you get the whole thing.
    if ('POP_EVAL' %in% tables & 'POP_EVAL_TYP' %in% tables) {
      out$POP_EVAL_TYP <- out$POP_EVAL_TYP %>%
        dplyr::filter(EVAL_CN %in% out$POP_EVAL$CN)
    }

    # Add package class and done
    class(out) <- 'FIA.Database'



    # Set up "remote" FIA Database
  } else {

    # If states not specified, grab all available from the plot table
    if (is.null(states)) {
      message('`states` not provided. Defaulting to all states listed in PLOT table.')
      allStates <- dplyr::tbl(
        con,
        dplyr::sql(
          paste0("SELECT DISTINCT statecd FROM ", schema, '.plot')
        )
      ) %>%
        dplyr::collect() %>%
        dplyr::rename_with(stringr::str_to_upper) %>%
        dplyr::left_join(intData$stateNames, by = 'STATECD')

      states <- allStates$STATEAB
    }

    ## Saving the call to readFIA, for eval later
    out <- list(con = con,
                schema = schema,
                common = common,
                tables = tables,
                states = states)
    class(out) <- 'Remote.FIA.Database'

  }

  return(out)

}

# Load individual tables from a connection to a mirror of the FIADB
read_fia_table_from_db <- function(table, con, schema, states) {

  # Reformat states so we can use "IN"
  statecds <- intData$stateNames$STATECD[intData$stateNames$STATEAB %in% states]
  states <- paste0('(', paste(statecds, collapse = ', '), ')')

  # All tables except POP_EVAL_TYP have a STATECD variable, so we can filter
  # tables by state and then load. For POP_EVAL_TYP we have to load the whole
  # thing each time, and then apply an adhoc filter from EVAL_CN if POP_EVAL
  # is also available.
  query <- paste0("SELECT * FROM ", schema, '.', table)
  if (table != 'POP_EVAL_TYP') query <- paste0(query, ' WHERE statecd IN ', states)

  # Load the table
  fiaTable <- dplyr::tbl(
    con,
    dplyr::sql(query)
  ) %>%
    dplyr::collect() %>%
    dplyr::rename_with(stringr::str_to_upper)

  return(fiaTable)
}

# If states not specified, grab all available from the plot table
pull_all_states <- function(con, schema) {
  allStates <- dplyr::tbl(
    con,
    dplyr::sql(
      paste0("SELECT DISTINCT statecd FROM ", schema, '.plot')
    )
  ) %>%
    dplyr::collect() %>%
    dplyr::rename_with(stringr::str_to_upper) %>%
    dplyr::left_join(intData$stateNames, by = 'STATECD')
  return(allStates$STATEAB)
}

