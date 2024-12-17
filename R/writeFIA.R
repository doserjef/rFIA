# TODO: needs assessment
writeFIA <- function(db,
                     dir,
                     byState = FALSE,
                     nCores = 1,
                     ...){
  if (is(db, 'Remote.FIA.Database')) {
    stop('Cannot write remote database.')
  }

  # if (byState & !c('SURVEY' %in% names(db))){
  #   stop('Need survey table for state abbreviations.')
  # }

  #cat(sys.call()$dir)
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

  ## Method to chunk up the database into states before writing it out
  if (byState){

    db$PLOT <- db$PLOT %>%
      select(-c(any_of('STATEAB'))) %>%
      left_join(intData$stateNames, by = 'STATECD')

    ## Unique state abbreviations
    states <- unique(db$PLOT$STATEAB)

    ## Chunk up plot
    pltList <- split(db$PLOT, as.factor(db$PLOT$STATEAB))

    ## Loop over states, do the writing
    for (s in 1:length(states)){
      db_clip <- db
      ## Overwriting plot with shortened table
      db_clip$PLOT <- pltList[[s]]
      ## Subsetting the remaining database based
      ## on plot table
      db_clip <- clipFIA(db_clip, mostRecent = FALSE)

      ## Write it all out
      tableNames <- names(db_clip)[names(db_clip) != 'mostRecent']
      tableNames <- paste(unique(db_clip$PLOT$STATEAB), tableNames, sep = '_')
      for (i in 1:length(tableNames)){
        if (is.data.frame(db_clip[[i]])){
          fwrite(x = db_clip[[i]], file = paste0(dir, tableNames[i], '.csv'), showProgress = FALSE, nThread = nCores)
        }
      }
    }


    ## Merge states together on writing
  } else {
    tableNames <- names(db)

    for (i in 1:length(tableNames)){
      if (is.data.frame(db[[i]])){
        fwrite(x = db[[i]], file = paste0(dir, tableNames[i], '.csv'), showProgress = FALSE, nThread = nCores)
      }
    }
  }

}

