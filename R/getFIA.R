## Access FIA Database files from the FIA Datamart
getFIA <- function(states,
                   dir = NULL,
                   common = TRUE,
                   tables = NULL,
                   load = TRUE,
                   nCores = 1){

  if (!is.null(dir)){
    # Add a slash to end of directory name if missing
    if (str_sub(dir,-1) != '/'){
      dir <- paste(dir, '/', sep = "")
    }
    # Check to see directory exists, if not, make it
    if(!dir.exists(dir)) {
      dir.create(dir)
      message(paste('Creating directory:', dir))
    }

    message(paste0('Saving to ', dir, '. NOTE: modifying FIA tables in Excel may corrupt csv files.'))
  }

  if (is.null(dir) & load == FALSE){
    stop('Must specify a directory ("dir") to save data when "load = FALSE".')
  }

  if ('REF' %in% states & length(unique(states)) > 1) {
    stop('Please download reference tables and state subsets seperately.')
  }

  ## All or nothing w/ new FIADB for some reason
  if (states == 'REF') {
    tables <- NULL
  }

  ## If dir is not specified, hold in a temporary directory
  #if (is.null(dir)){tempDir <- tempdir()}
  tempDir <- tempdir()
  ## Some warnings up front
  ## Do not try to merge ENTIRE with other states
  if (length(states) > 1 & any(str_detect(str_to_upper(states), 'ENTIRE'))){
    stop('Cannot merge ENITRE with other state tables. ENTIRE includes all state tables combined. Do you only need data for a particular region?')
  }
  ## Check to make sure states exist
  allStates <- c('AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
                 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
                 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
                 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
                 'WI', 'WY', 'AS', 'FM', 'GU', 'MP', 'PW', 'PR', 'VI', 'ENTIRE', 'REF')
  if (any(str_to_upper(states) %in% allStates == FALSE)){
    missStates <- states[str_to_upper(states) %in% allStates == FALSE]
    stop(paste('Data unavailable for: ', paste(as.character(missStates),collapse = ', '), '. Did you use state/terrority abbreviations? e.g. use states = "AL" (correct) instead of states = "ALABAMA".'))
  }

  ## Check to make sure tables exist
  allTables <- c("BOUNDARY", "COND_DWM_CALC", "COND","COUNTY","DWM_COARSE_WOODY_DEBRIS",
                 "DWM_DUFF_LITTER_FUEL","DWM_FINE_WOODY_DEBRIS","DWM_MICROPLOT_FUEL",
                 "DWM_RESIDUAL_PILE", "DWM_TRANSECT_SEGMENT", "DWM_VISIT","GRND_CVR",
                 "INVASIVE_SUBPLOT_SPP","LICHEN_LAB","LICHEN_PLOT_SUMMARY","LICHEN_VISIT",
                 "OZONE_BIOSITE_SUMMARY","OZONE_PLOT_SUMMARY","OZONE_PLOT","OZONE_SPECIES_SUMMARY",
                 "OZONE_VALIDATION","OZONE_VISIT", "P2VEG_SUBP_STRUCTURE","P2VEG_SUBPLOT_SPP",
                 "PLOT_REGEN","PLOT", "PLOTGEOM", "PLOTSNAP","POP_ESTN_UNIT","POP_EVAL_ATTRIBUTE",
                 "POP_EVAL_GRP","POP_EVAL_TYP","POP_EVAL","POP_PLOT_STRATUM_ASSGN","POP_STRATUM",
                 "SEEDLING_REGEN","SEEDLING","SITETREE","SOILS_EROSION","SOILS_LAB","SOILS_SAMPLE_LOC" ,
                 "SOILS_VISIT", "SUBP_COND_CHNG_MTRX","SUBP_COND","SUBPLOT_REGEN","SUBPLOT",
                 "SURVEY","TREE_GRM_BEGIN","TREE_GRM_COMPONENT","TREE_GRM_ESTN", "TREE_GRM_MIDPT",
                 "TREE_REGIONAL_BIOMASS", "TREE_WOODLAND_STEMS","TREE","VEG_PLOT_SPECIES",
                 "VEG_QUADRAT","VEG_SUBPLOT_SPP","VEG_SUBPLOT", "VEG_VISIT",
                 'CITATION', 'DIFFERENCE_TEST_PER_ACRE', 'DIFFERENCE_TEST_TOTALS',
                 'FIADB_VERSION', 'FOREST_TYPE', 'FOREST_TYPE_GROUP',
                 'GRM_TYPE', 'HABTYP_DESCRIPTION', 'HABTYP_PUBLICATION',
                 'INVASIVE_SPECIES', 'LICHEN_SPECIES', 'LICHEN_SPP_COMMENTS',
                 'NVCS_HEIRARCHY_STRCT', 'NVCS_LEVEL_1_CODES', 'NVCS_LEVEL_2_CODES',
                 'NVCS_LEVEL_3_CODES', 'NVCS_LEVEL_4_CODES', 'NVCS_LEVEL_5_CODES',
                 'NVCS_LEVEL_6_CODES', 'NVCS_LEVEL_7_CODES', 'NVCS_LEVEL_8_CODES',
                 'OWNGRPCD', 'PLANT_DICTIONARY', 'POP_ATTRIBUTE', 'POP_EVAL_TYP_DESCR',
                 'RESEARCH_STATION', 'SPECIES', 'SPECIES_GROUP', 'STATE_ELEV', 'UNIT')
  if (any(str_to_upper(tables) %in% allTables == FALSE)){
    missTables <- tables[str_to_upper(tables) %in% allTables == FALSE]
    stop(paste('Tables: ', paste(as.character(missTables),collapse = ', '), ' unavailble. Check the FIA Datamart at https://apps.fs.usda.gov/fia/datamart/CSV/datamart_csv.html for a list of available tables for each state. Alternatively, specify common = TRUE to download the most commonly used tables.

Did you accidentally include the state abbreviation in front of the table name? e.g. tables = "AL_PLOT" (wrong) instead of tables = "PLOT" (correct).'))
  }

  ### I'm not very smart and like specify the name twice sometimes,
  states <- unique(states)

  ## If individual tables are specified, then just grab those .csvs, otherwise download the .zip file, extract and read with fread. Should be quite a bit quicker.
  if (!is.null(tables)){
    ## Make a list of tables names to read in
    ## Append table names with state abbs and then add url link
    tables <- str_to_upper(tables)

    # Make sure state Abbs are in right format
    states <- str_to_upper(states)
    if ('ENTIRE' %in% states == FALSE) {
      states <- paste0(states, '_')
    } else {
      states <- ''
    }


    tblNames <- sapply(states, FUN = function(x, y){paste0(x,y,'.zip')}, y = tables)
    tblNames <- c(tblNames)
    urls <- paste0('https://apps.fs.usda.gov/fia/datamart/CSV/', tblNames)


    inTables = list()
    for (n in 1:length(urls)){

      newName <- paste0(str_sub(tblNames[n], 1, -5), '.csv')

      ## Download the zip to a temporary file
      temp <- tempfile()
      download.file(urls[n], temp, timeout = 3600)

      # Write the data out the directory they've chosen
      if(is.null(dir)){
        unzip(temp, exdir = tempDir)
        file <- fread(paste0(tempDir, '/', newName), showProgress = FALSE, logical01 = FALSE, integer64 = 'double', nThread = nCores)
        ## Unlinking the directory is bad news, so just delete the file from tempdir
        file.remove(paste0(tempDir, '/', newName))
      } else {
        #download.file(urls[n], paste0(dir, tblNames[n]))
        unzip(temp, exdir = str_sub(dir, 1, -2))
        if (load) file <- fread(paste0(dir, newName), showProgress = FALSE, logical01 = FALSE, integer64 = 'double', nThread = nCores)
      }

      unlink(temp)

      if (load){
        # We don't want data.table formats
        file <- as.data.frame(file)
        inTables[[str_sub(urls[n], 43, -5)]] <- file
      }

    }

    if (load){
      # Check for corresponding tables (multiple states)
      # If they exist, loop through and merge corresponding tables
      ## Check if the directory has the entire US naming convention or state naming convention
      tableNames <- names(inTables)
      outTables <- list()
      if (any(str_sub(tableNames, 3, 3) == '_')){ ## STATE NAMING CONVENTION
        if (anyDuplicated(str_sub(tableNames, 4)) != 0){
          for (i in 1:length(unique(str_sub(tableNames, 4)))){
            subList <- inTables[str_sub(tableNames, 4) == unique(str_sub(tableNames,4))[i]]
            name <- unique(str_sub(tableNames, 4))[i]
            # Give a ton of warnings about factors and characters, don't do that
            outTables[[name]] <- suppressWarnings(do.call(rbind, subList))
          }
        } else {
          outTables <- inTables
          names(outTables) <- unique(str_sub(tableNames, 4))
        }
      } else{ ## ENTIRE NAMING CONVENTION
        if (anyDuplicated(tableNames) != 0){
          for (i in 1:length(unique(tableNames))){
            subList <- inTables[tableNames == unique(tableNames)[i]]
            name <- unique(str_sub(tableNames, 1))[i]
            # Give a ton of warnings about factors and characters, don't do that
            outTables[[name]] <- suppressWarnings(do.call(rbind, subList))
          }
        } else {
          outTables <- inTables
          names(outTables) <- unique(str_sub(tableNames, 1))
        }
      }
      # NEW CLASS NAME FOR FIA DATABASE OBJECTS
      #outTables <- lapply(outTables, as.data.frame)
      class(outTables) <- 'FIA.Database'
    }



    #### DOWNLOADING THE WHOLE ZIP FILE
  } else {
    ## Download to a temporary file location, then extract to the permanent if wanted. Read extracted tables into R w/ readFIA.

    # Make sure state Abbs are in right format
    states <- str_to_upper(states)

    ## Download each state and extract to directory
    for (i in 1:length(states)){
      # Temporary directory to download to
      temp <- paste0(tempDir, '/', states[i],'_CSV.zip') #tempfile()
      ## Make the URL
      url <- paste0('https://apps.fs.usda.gov/fia/datamart/CSV/', states[i],'_CSV.zip')
      if (states == 'REF') {url <- paste0('https://apps.fs.usda.gov/fia/datamart/CSV/FIADB_REFERENCE.zip')}
      #newName <- paste0(str_sub(url, 1, -4), 'csv')
      ## Download as temporary file
      download.file(url, temp, timeout = 3600)
      ## Extract
      if (is.null(dir)){
        unzip(temp, exdir = tempDir)
      } else {
        unzip(temp, exdir = str_sub(dir, 1, -2))
      }
      unlink(temp)
    }

    if (load){

      ## Read in the files w/ readFIA
      if (is.null(dir)){
        outTables <- readFIA(tempDir, nCores = nCores, common = common, states = states)
        #unlink(tempDir, recursive = TRUE)
      } else {
        outTables <- readFIA(dir, nCores = nCores, common = common, states = states)
      }
    }



  }


  if (is.null(dir)){
    tmp <- list.files(tempDir, full.names = TRUE, pattern = '.csv')
    invisible(file.remove(tmp))
  }

  if (load) return(outTables)
}

