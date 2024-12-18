diversity <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                      bySizeClass = FALSE, landType = 'forest', 
                      treeType = 'live', method = 'TI', lambda = 0.5, 
                      stateVar = TPA_UNADJ, grpVar = SPCD, treeDomain = NULL, 
                      areaDomain = NULL, byPlot = FALSE, condList = FALSE,
                      totals = FALSE, variance = FALSE, nCores = 1) {

  # Defuse user-supplied expressions in grpBy, areaDomain, treeDomain, 
  # stateVar, and grpVar
  grpBy_quo <- rlang::enquo(grpBy)
  areaDomain <- rlang::enquo(areaDomain)
  treeDomain <- rlang::enquo(treeDomain)
  stateVar <- rlang::enquo(stateVar)
  grpVar <- rlang::enquo(grpVar)

  # Handle iterator if db is remote
  remote <- ifelse(class(db) == 'Remote.FIA.Database', 1, 0)
  # Takes value 1 if not remote. iter for remote is a vector of different state IDs. 
  iter <- remoteIter(db, remote)

  # Check for a most recent subset
  mr <- checkMR(db, remote)

  # Prep for areal summary (converts polys to sf, converts factors to chrs, 
  # and adds the polyID column giving a unique ID to each areal unit). 
  polys <- arealSumPrep1(polys)

  # Run the main portion of the function
  out <- lapply(X = iter, FUN = diversityStarter, db, grpBy_quo = grpBy_quo, 
                polys, returnSpatial, bySizeClass, landType, treeType, method, 
                lambda, stateVar, grpVar, treeDomain, areaDomain, byPlot, 
                condList, totals, nCores, remote, mr)
}
