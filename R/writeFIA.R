writeFIA <- function(db,
                     dir,
                     byState = FALSE,
                     nCores = 1,
                     ...){
  if (is(db, 'Remote.FIA.Database')) {
    stop('Cannot write remote database.')
  }

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

  # Method to chunk up the database into states before writing it out
  if (byState){

    if (!is.null(db$PLOT)) {
      db$PLOT <- db$PLOT %>%
        dplyr::select(-c(any_of('STATEAB'))) %>%
        dplyr::left_join(intData$stateNames, by = 'STATECD')

      # Unique state abbreviations
      states <- unique(db$PLOT$STATEAB)

      # Chunk up plot
      pltList <- split(db$PLOT, as.factor(db$PLOT$STATEAB))

    } else if (!is.null(db$COND)) {
      db$COND <- db$COND %>%
        dplyr::select(-c(any_of('STATEAB'))) %>%
        dplyr::left_join(intData$stateNames, by = 'STATECD')

      # Unique state abbreviations
      states <- unique(db$COND$STATEAB)

      # Chunk up plot
      pltList <- split(db$COND, as.factor(db$COND$STATEAB))

    } else {
      stop("writing FIADB tables with byState = TRUE requires either the PLOT or COND tables to be loaded.")
    }

    # Loop over states, do the writing
    for (s in 1:length(states)){
      db_clip <- db
      if (!is.null(db$PLOT)) {
        # Overwriting plot with shortened table
        db_clip$PLOT <- pltList[[s]]
      } else {
        db_clip$COND <- pltList[[s]]
      }
      # Subsetting the remaining database based
      # on plot table
      # db_clip <- clipFIA(db_clip, mostRecent = FALSE)

      # Write it all out
      tableNames <- names(db_clip)[names(db_clip) != 'mostRecent']
      if (!is.null(db$PLOT)) {
        tableNames <- paste(unique(db_clip$PLOT$STATEAB), tableNames, sep = '_')
      } else {
        tableNames <- paste(unique(db_clip$COND$STATEAB), tableNames, sep = '_')
      }
      for (i in 1:length(tableNames)){
        if (is.data.frame(db_clip[[i]])){
          fwrite(x = exactCN(db_clip[[i]]), file = paste0(dir, tableNames[i], '.csv'), showProgress = FALSE, nThread = nCores)
        }
      }
    }


    # Merge states together on writing
  } else {
    tableNames <- names(db)

    for (i in 1:length(tableNames)){
      if (is.data.frame(db[[i]])){
        fwrite(x = exactCN(db[[i]]), file = paste0(dir, tableNames[i], '.csv'), showProgress = FALSE, nThread = nCores)
      }
    }
  }

}


# data.table::fwrite() writes doubles with at most 15 significant digits, but
# FIADB CNs (read in as doubles by readFIA()) can have 16 digits. Writing them
# as-is silently rounds them (e.g., 1097572188290487 -> 1097572188290490),
# breaking joins between tables. Convert numeric CN columns to exact integer
# strings before writing.
exactCN <- function(x) {
  cnCols <- names(x)[grepl('^(.*_)?CN$', names(x))]
  cnCols <- cnCols[vapply(cnCols, \(col) is.double(x[[col]]), logical(1))]
  if (length(cnCols) == 0) return(x)
  x %>%
    dplyr::mutate(dplyr::across(dplyr::all_of(cnCols),
                                \(cn) ifelse(is.na(cn), NA_character_, sprintf('%.0f', cn))))
}
