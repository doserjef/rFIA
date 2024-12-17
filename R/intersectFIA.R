# TODO: needs checking
intersectFIA <- function(db, polys, nCores = 1) {


  ## Some warnings if inputs are bogus -----------------------------------------

  ## Must have an FIA.Database or a remote one
  if (!c(class(db) %in% c('FIA.Database', 'Remote.FIA.Database'))) {
    stop('Must provide an `FIA.Database` or `Remote.FIA.Database`. See `readFIA` and/or `getFIA` to read and load your FIA data.')
  }

  ## Must use either sp or sf
  if (!is.null(polys) & first(class(polys)) %in% c('sf', 'SpatialPolygons', 'SpatialPolygonsDataFrame') == FALSE){
    stop('polys must be spatial polygons object of class sp or sf. ')
  }



  ## Handle remote database if it exists ---------------------------------------
  # If remote, read in to memory. Otherwise, DO NOT drop all unnecessary tables
  remote <- ifelse(class(db) == 'Remote.FIA.Database', 1, 0)
  if (remote) {
    db.backup <- db # So we can still return the original list
    req.tables <- c('PLOT')
    out.file <- db$dir
    db <- readRemoteHelper(db$states, db, remote, req.tables, nCores = 1)
  } else {
    if (!c('PLOT' %in% names(db))){
      stop('PLOT table required for spatial intersection, but is missing from `db`.')
    }
  }

  # Need a unique identifier for plots
  db$PLOT <- db$PLOT %>%
    dplyr::mutate(pltID = stringr::str_c(UNITCD, STATECD, COUNTYCD, PLOT, sep = '_'))



  ## Carry out the spatial intersection ----------------------------------------
  # Prep spatial data
  polys <- arealSumPrep1(polys)

  # Do the intersection
  db <- arealSumPrep2(db,
                      grpBy = NULL,
                      polys,
                      nCores,
                      remote = FALSE)


  ## If remote, save the PLOT table back to disk
  if (remote) {
    suppressMessages({
      writeFIA(db,
               dir = out.file,
               byState = TRUE,
               nCores = nCores)
    })
    db <- db.backup
  }

  return(db)



}





