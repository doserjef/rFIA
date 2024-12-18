diversityStarter <- function(x, db, grpBy_quo = NULL, polys = NULL, 
                             returnSpatial = FALSE, bySizeClass = FALSE, 
                             landType = 'forest', treeType = 'live', method = 'TI', 
                             lambda = 0.5, stateVar = TPA_UNADJ, grpVar = SPCD, 
                             treeDomain = NULL, areaDomain = NULL, byPlot = FALSE,
                             condList = FALSE, totals = FALSE, nCores = 1, 
                             remote, mr) {

  # Read required data and prep the database ------------------------------
  reqTables <- c('PLOT', 'TREE', 'COND', 'POP_PLOT_STRATUM_ASSGN', 
                 'POP_ESTN_UNIT', 'POP_EVAL', 'POP_STRATUM', 'POP_EVAL_TYP', 
                 'POP_EVAL_GRP')

}
