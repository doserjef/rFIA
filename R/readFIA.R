# TODO: needs assessment
# Read FIA database (csv) from local directory
readFIA <- function(dir = NULL,
                    con = NULL,
                    schema = NULL,
                    common = TRUE,
                    tables = NULL,
                    states = NULL,
                    inMemory = TRUE,
                    nCores = 1,
                    ...){

  ## Must specify dir or con
  if (is.null(dir) & is.null(con)) {
    stop('Must specify `dir` or `con`. If reading from csv files, set `dir` to the directory where your FIA data lives.')
  }

  ## Need schema if reading from a database
  if (!is.null(con) & is.null(schema)) {
    stop('Must specify `schema` when reading data from a database connection (`con`).')
  }

  ## Attempt to read from database
  if (!is.null(con)) {
    db <- readFIA.db(con, schema, states, tables, common, inMemory)
    return(db)
  }


  # Add a slash to end of directory name if missing
  if (str_sub(dir,-1) != '/') dir <- paste(dir, '/', sep = "")
  # Grab all the file names in directory
  files <- list.files(dir)

  ## methods for reading the full database into memory
  if (inMemory){

    inTables <- list()

    # Some warnings
    if(!dir.exists(dir)) {
      stop(paste('Directory', dir, 'does not exist.'))
    }
    if(length(files[str_sub(files,-4, -1) == '.csv']) < 1){
      stop(paste('Directory', dir, 'contains no .csv files.'))
    }

    ## Some warnings up front
    ## Do not try to merge ENTIRE with other states
    if (length(states) > 1 & any(str_detect(str_to_upper(states), 'ENTIRE'))){
      stop('Cannot merge ENITRE with other state tables. ENTIRE includes all state tables combined. Do you only need data for a particular region?')
    }

    # Only csvs
    files <- files[str_sub(files,-4,-1) == '.csv']

    ## All states present in the directory, adding reference if present
    allStates <- unique(str_to_upper(str_sub(files, 1, 2)))
    allStates <- allStates[allStates %in% intData$stateNames$STATEAB]
    if (any(str_to_upper(str_sub(files, 1, 3)) == 'REF')) allStates <- c(allStates, 'REF')

    ## If states are given, then subset files now,
    ## otherwise, bring them all in
    if (!is.null(states)){

      ### I'm not very smart and like specify the name twice sometimes,
      ### --> making the function smarter than me
      states <- str_to_upper(unique(states))

      ## Make sure the states exist to begin with
      if (any(states %in% allStates == FALSE)){
        missStates <- states[states %in% allStates == FALSE]
        stop(paste('Data unavailable for: ', paste(as.character(missStates),collapse = ', '), '. States not found in specified directory.'))
      }

      state.files <- files[str_to_upper(str_sub(files, 1, 2)) %in% states]
      if ('REF' %in% states) {
        ref.files <- files[str_to_upper(str_sub(files, 1,3)) == 'REF']
        files <- c(state.files, ref.files)
      } else {
        files <- c(state.files)
      }

    } else {
      ## Checking if state files and merged state files are mixed in the directory.
      states <- unique(str_to_upper(str_sub(files, 1, 3)))
      ## Dropping lichen spp summary and project reference tables
      trueStates <- states[str_sub(states, 3,3) == '_']
      if ("REF" %in% states) {
        trueStates <- c(trueStates, 'REF')
      }

      ## If length is zero, then all merged states - great
      ## If length states is the same as true States, then all state files - great
      ## Otherwise, they're probably mixed. Throw a warning and read only the states
      if (length(trueStates) > 0 & length(trueStates) < length(states)) {
        warning('Found data from merged states and individual states in same directory. Reading only individual states files.')
        state.files <- files[str_to_upper(str_sub(files, 3, 3)) == '_']
        ref.files <- files[str_to_upper(str_sub(files, 1,3)) == 'REF']
        files <- c(state.files, ref.files)
      }
    }

    ## Common file names to consider
    ## All reference tables should be common
    cFiles <- c('COND', 'COND_DWM_CALC', 'INVASIVE_SUBPLOT_SPP', 'PLOT', 'POP_ESTN_UNIT',
                'POP_EVAL', 'POP_EVAL_GRP', 'POP_EVAL_TYP', 'POP_PLOT_STRATUM_ASSGN', 'POP_STRATUM',
                'SUBPLOT', 'TREE', 'TREE_GRM_COMPONENT', 'TREE_GRM_MIDPT', 'TREE_GRM_BEGIN', 'SUBP_COND_CHNG_MTRX',
                'SEEDLING', 'SURVEY', 'SUBP_COND', 'P2VEG_SUBP_STRUCTURE')

    if ('REF' %in% allStates) {
      cFiles <- c(cFiles,
                  "CITATION", "DIFFERENCE_TEST_PER_ACRE", "DIFFERENCE_TEST_TOTALS", "FIADB_VERSION",
                  "FOREST_TYPE_GROUP", "FOREST_TYPE", "GRM_TYPE", "HABTYP_DESCRIPTION", "HABTYP_PUBLICATION",
                  "INVASIVE_SPECIES", "LICHEN_SPECIES", "LICHEN_SPP_COMMENTS", "NVCS_HIERARCHY_STRCT",
                  "NVCS_LEVEL_1_CODES", "NVCS_LEVEL_2_CODES", "NVCS_LEVEL_3_CODES", "NVCS_LEVEL_4_CODES",
                  "NVCS_LEVEL_5_CODES", "NVCS_LEVEL_6_CODES", "NVCS_LEVEL_7_CODES", "NVCS_LEVEL_8_CODES",
                  "OWNGRPCD", "PLANT_DICTIONARY", "POP_ATTRIBUTE", "POP_EVAL_TYP_DESCR", "RESEARCH_STATION",
                  "SPECIES_GROUP", "SPECIES", "STATE_ELEV", "UNIT")
    }

    ## Drop state/ref abbreviations and .csv
    abbs <- c(paste0('start_here_', intData$stateNames$STATEAB, '_'), 'start_here_REF_')
    file.sub <- paste0('start_here_', str_remove(files, '.csv'))
    ## only allow subset at the start at the beginning of string
    for ( i in abbs ) file.sub <- str_remove(file.sub, i)


    # Only read in the specified tables
    if (!is.null(tables)){
      files <- files[file.sub %in% tables]

    # Otherwise if common=TRUE, grab all common
    } else if (common) {
      files <- files[file.sub %in% cFiles]

    # Otherwise only known FIA tables
    } else {
      files <- files[file.sub %in% intData$fiaTableNames]
    }



    ## Now read them in
    inTables <- list()
    for (n in 1:length(files)){
      ## If MODIFIED_DATE is not present, will warn
      suppressWarnings({
        # Read in and append each file to a list
        file <- data.table::fread(paste(dir, files[n], sep = ""), showProgress = FALSE,
                                  integer64 = 'double', logical01 = FALSE, nThread = nCores,
                                  drop = dropTheseCols(), ...)
      })

      # We don't want data.table formats
      #file <- as.data.frame(file)
      fileName <- str_sub(files[n], 1, -5)

      # Skip over files that are empty
      if(nrow(file) > 0){
        inTables[[fileName]] <- file
      }
    }

    # Give them some names
    #names(inTables) <- files
    #inTables <- lapply(inTables, as.data.frame)

    outTables <- list()
    if (any(str_sub(names(inTables), 3, 3) == '_')){ ## STATE NAMING CONVENTION
      # Remove the state prefix
      names(inTables) <- str_sub(names(inTables), 4, -1)
    }
    uniqueNames <- unique(names(inTables))
    ## Works regardless of whether or not there are duplicate names (multiple states)
    for (i in 1:length(uniqueNames)){
      outTables[[uniqueNames[i]]] <-  data.table::rbindlist(inTables[names(inTables) == uniqueNames[i]], fill = TRUE)
    }


    # NEW CLASS NAME FOR FIA DATABASE OBJECTS
    out <- lapply(outTables, as.data.frame)

    ## Add REF to the reference tables, will have _ prefix
    names(out)[str_sub(names(out), 1, 1) == '_'] <- paste0('REF', names(out)[str_sub(names(out), 1, 1) == '_'])


    ### Methods for keeping data remote until they are needed
    ### Chunking up data into states
    ## inMemory = FALSE
  } else {

    ## Can't use remote methods w/ full database
    if (any(str_detect(str_to_upper(states), 'ENTIRE'))) {
      stop('Remote methods undefind for ENTIRE. Please use `inMemory=TRUE`. Resulting FIA.Database will be quite large (~50GB).')
    }

    ## IF states isn't given, default to all
    ## states in the directory
    if (is.null(states)){
      states <- unique(str_to_upper(str_sub(files, 1, 3)))
      states <- states[str_sub(states, 3,3) == '_']
      states <- str_sub(states, 1, 2)
      ## Only states where abbreviations make sense
      states <- states[states %in% intData$stateNames$STATEAB]
      ## Don't fail if states have been merged
      if (length(states) < 1) states <- 1
    }


    ## Saving the call to readFIA, for eval later
    out <- list(dir = dir,
                common = common,
                tables = tables,
                states = states,
                ... = ...)

    class(out) <- 'Remote.FIA.Database'
  }


  if (inMemory) {

    ## Names of all tables forced to uppercase, since FIA changed this in Oct 2022
    out <- lapply(out, FUN = function(x){dplyr::rename_with(x, toupper)})

    ## If WY updates the END_INVYR for 2018 & 2019 (i.e., not set it as 2020)
    ## then we can drop this. Until then, we have to strong arm END_INVYR
    out <- handleWY(out)

    class(out) <- 'FIA.Database'
  }

  return(out)

}

