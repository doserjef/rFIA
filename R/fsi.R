fsiStarter <- function(x,
                       db,
                       grpBy_quo = NULL,
                       scaleBy_quo = NULL,
                       polys = NULL,
                       returnSpatial = FALSE,
                       bySpecies = FALSE,
                       bySizeClass = FALSE,
                       landType = 'forest',
                       treeType = 'live',
                       method = 'TI',
                       lambda = .5,
                       treeDomain = NULL,
                       areaDomain = NULL,
                       totals = FALSE,
                       byPlot = FALSE,
                       useSeries = FALSE,
                       mostRecent = FALSE,
                       nCores = 1,
                       remote,
                       mr){




  ## Read required data, prep the database -------------------------------------
  reqTables <- c('PLOT', 'TREE', 'COND', 'POP_PLOT_STRATUM_ASSGN',
                 'POP_ESTN_UNIT', 'POP_EVAL',
                 'POP_STRATUM', 'POP_EVAL_TYP', 'POP_EVAL_GRP', 'PLOTGEOM')

  ## If remote, read in state by state. Otherwise, drop all unnecessary tables
  db <- readRemoteHelper(x, db, remote, reqTables, nCores)


  ## Handle TX issues - we only keep inventory years that are present in BOTH
  ## EAST AND WEST TX
  db <- handleTX(db)




  ## Some warnings if inputs are bogus -----------------------------------------
  if (!is.null(polys) &
      first(class(polys)) %in%
      c('sf', 'SpatialPolygons', 'SpatialPolygonsDataFrame') == FALSE){
    stop('polys must be spatial polygons object of class sp or sf. ')
  }
  if (landType %in% c('timber', 'forest') == FALSE){
    stop('landType must be one of: "forest" or "timber".')
  }
  if (treeType %in% c('live', 'dead', 'gs', 'all') == FALSE){
    stop('treeType must be one of: "live", "dead", "gs", or "all".')
  }
  if (any(reqTables %in% names(db) == FALSE)){
    missT <- reqTables[reqTables %in% names(db) == FALSE]
    stop(paste('Tables', paste (as.character(missT), collapse = ', '),
               'not found in object db.'))
  }
  if (str_to_upper(method) %in% c('TI', 'SMA', 'LMA', 'EMA', 'ANNUAL') == FALSE) {
    warning(paste('Method', method,
                  'unknown. Defaulting to Temporally Indifferent (TI).'))
  }


  ## Prep other variables ------------------------------------------------------
  # Join PLOT with PLOTGEOM to allow plot-level geographic attributes to be used
  # in grpBy statements
  db$PLOTGEOM <- db$PLOTGEOM %>%
    dplyr::select(-STATECD, -INVYR, -UNITCD, -COUNTYCD, -PLOT, -LAT, -LON, 
                  -dplyr::starts_with('CREATED'), -dplyr::starts_with('MODIFIED'))
  db$PLOT <- db$PLOT %>%
    dplyr::left_join(db$PLOTGEOM, by = 'CN')

  ## Need a plotCN, and a new ID
  db$PLOT <- db$PLOT %>%
    mutate(PLT_CN = CN,
           pltID = paste(UNITCD, STATECD, COUNTYCD, PLOT, sep = '_'))
  db$TREE <- db$TREE %>%
    mutate(TRE_CN = CN)

  ## Convert grpBy to character
  grpBy <- grpByToChar(db, grpBy_quo)

  ## Convert scaleBy to character
  scaleBy <- grpByToChar(db[names(db) %in% 'TREE' == FALSE], scaleBy_quo)

  # Save original grpBy for pretty return with spatial objects
  grpByOrig <- grpBy


  # I like a unique ID for a plot through time
  if (byPlot) {grpBy <- c('pltID', grpBy)}


  ## Intersect plots with polygons if polygons are given
  if (!is.null(polys)){

    ## Add shapefile names to grpBy
    grpBy = c(grpBy, 'polyID')
    ## Do the intersection
    db <- arealSumPrep2(db, grpBy, polys, nCores, remote)

    ## If there's nothing there, skip the state
    if (is.null(db)) return('no plots in polys')
  }

  ## If we want to return spatial plots
  if (byPlot & returnSpatial){
    grpBy <- c(grpBy, 'LON', 'LAT')
  }






  ## Build a domain indicator for each observation (1 or 0) --------------------
  ## Land type
  db$COND$landD <- landTypeDomain(landType,
                                  db$COND$COND_STATUS_CD,
                                  db$COND$SITECLCD,
                                  db$COND$RESERVCD)
  ## Tree type
  db$TREE$typeD <- treeTypeDomain(treeType,
                                  db$TREE$STATUSCD,
                                  db$TREE$DIA,
                                  db$TREE$TREECLCD)

  ## Spatial boundary
  if(!is.null(polys)){
    db$PLOT$sp <- ifelse(!is.na(db$PLOT$polyID), 1, 0)
  } else {
    db$PLOT$sp <- 1
  }

  # User defined domain indicator for area (ex. specific forest type)
  db <- udAreaDomain(db, areaDomain)

  # User defined domain indicator for tree (ex. trees > 20 ft tall)
  db <- udTreeDomain(db, treeDomain)




  ## Handle population tables --------------------------------------------------
  ## We only want inventory/ population info from t2 plots, but we need the plot tree cond data
  ## for t1 and t2
  remPlts <- db$PLOT %>%
    select(PLT_CN, PREV_PLT_CN, DESIGNCD, REMPER, PLOT_STATUS_CD) %>%
    ## Has to have a remeasurement, be in the current sample, and of the national design
    filter(!is.na(REMPER) & !is.na(PREV_PLT_CN) & PLOT_STATUS_CD != 3 & DESIGNCD %in% c(1, 501:505)) %>%
    left_join(select(db$PLOT, PLT_CN, DESIGNCD, PLOT_STATUS_CD), by = c('PREV_PLT_CN' = 'PLT_CN'), suffix = c('', '.prev')) %>%
    ## past remasurement must be in the previous sample and of national design
    filter(PLOT_STATUS_CD.prev != 3 & DESIGNCD.prev %in% c(1, 501:505))

  ## Filtering out all inventories that are not relevant to the current estimation
  ## type. If using estimator other than TI, handle the differences in P2POINTCNT
  ## and in assigning YEAR column (YEAR = END_INVYR if method = 'TI')
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
    grpByOrig <- c(grpByOrig, 'SPCD', 'COMMON_NAME', 'SCIENTIFIC_NAME')
  }

  ## Break into size classes
  if (bySizeClass){
    grpBy <- c(grpBy, 'sizeClass')
    grpByOrig <- c(grpByOrig, 'sizeClass')
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




#' @export
fsi <- function(db,
                   grpBy = NULL,
                   polys = NULL,
                   returnSpatial = FALSE,
                   bySpecies = FALSE,
                   bySizeClass = FALSE,
                   landType = 'forest',
                   treeType = 'live',
                   method = 'TI',
                   lambda = .5,
                   treeDomain = NULL,
                   areaDomain = NULL,
                   totals = TRUE,
                   variance = TRUE,
                   byPlot = FALSE,
                   useSeries = FALSE,
                   mostRecent = FALSE,
                   scaleBy = NULL,
                   betas = NULL,
                   returnBetas = FALSE,
                   nCores = 1) {

  ## Need rjags and coda installed if we're going to model size-density relationships
  if (!is.null(betas) & is.null(find.package('R2jags')) & is.null(find.package('coda'))) {
    stop('Packages "R2jags" and "coda" required to model maximum size-density curves. Please install with install.packages(c("R2jags", "coda")) and try again.

If not already installed, you can install JAGS from SourceForge:
         Windows: https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Windows/
         Mac: https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Mac%20OS%20X/
         Linux: http://mcmc-jags.sourceforge.net/')
  } else if (!is.null(betas) & is.null(find.package('R2jags'))) {
    stop('Package "R2jags" required to model maximum size-density curves. Please install with install.packages("R2jags") and try again.

If not already installed, you can install JAGS from SourceForge:
         Windows: https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Windows/
         Mac: https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Mac%20OS%20X/
         Linux: http://mcmc-jags.sourceforge.net/')
  } else if (!is.null(betas) &  is.null(find.package('coda'))) {
    stop('Package "coda" required to model maximum size-density curves. Please install with install.packages("coda") and try again.')
  }

  ##  don't have to change original code
  grpBy_quo <- rlang::enquo(grpBy)
  scaleBy_quo <- rlang::enquo(scaleBy)
  areaDomain <- rlang::enquo(areaDomain)
  treeDomain <- rlang::enquo(treeDomain)

  ## Handle iterator if db is remote
  remote <- ifelse(class(db) == 'Remote.FIA.Database', 1, 0)
  iter <- remoteIter(db, remote)

  ## Check for a most recent subset
  mr <- checkMR(db, remote)

  ## prep for areal summary
  polys <- arealSumPrep1(polys)


  ## Run the main portion
  out <- lapply(X = iter, FUN = fsiStarter, db,
                grpBy_quo = grpBy_quo, scaleBy_quo, polys, returnSpatial,
                bySpecies, bySizeClass,
                landType, treeType, method,
                lambda, treeDomain, areaDomain,
                totals, byPlot, useSeries, mostRecent,
                nCores, remote, mr)

  ## Bring the results back
  out <- unlist(out, recursive = FALSE)
  # TODO: 
  # if (remote) out <- dropStatesOutsidePolys(out)
  t <- bind_rows(out[names(out) == 't'])
  t1 <- bind_rows(out[names(out) == 't1'])
  a <- bind_rows(out[names(out) == 'a'])
  grpBy <- out[names(out) == 'grpBy'][[1]]
  scaleBy <- out[names(out) == 'scaleBy'][[1]]
  grpByOrig <- out[names(out) == 'grpByOrig'][[1]]
  mr <- out[names(out) == 'mr'][[1]]




  ## Prep data to model maximum size-density curves ----------------------------
  scaleSyms <- syms(scaleBy)
  grpSyms <- syms(grpBy)

  ## Get groups prepped to fit model
  grpRates <- select(t1, PLT_CN, !!!scaleSyms, BA2, TPA2) %>%
    ungroup() %>%
    filter(TPA2 > 0) %>%
    tidyr::drop_na(!!!scaleSyms) %>%
    ## Stand-level variables here
    mutate(t = log(TPA2),
           b = log(BA2)) %>%
    select(t, b, PLT_CN, !!!scaleSyms)


  if (!is.null(scaleBy)){
    ## group IDS
    grpRates <- mutate(grpRates, grps = as.factor(paste(!!!scaleSyms)))
    t <- mutate(t, grps = as.factor(paste(!!!scaleSyms)))

    ## Remove groups with less than 10 obs
    nGrps <- grpRates %>%
      group_by(grps) %>%
      summarise(n = n())
    grpRates <- grpRates %>%
      left_join(nGrps, by = 'grps')

  } else {

    grpRates$grps <- 1
    t$grps = 1

  }


  ## If parameters are not provided, use quantile regression to estimate them ---
  if (is.null(betas)){
    ## Prep data for model
    prep <- grpRates %>%
      ungroup() %>%
      arrange(grps) %>%
      ## Need numeric group-level assignments
      mutate(grp_index = as.numeric(as.factor(grps)))


    ## If more than one group use a mixed model
    if (length(unique(grpRates$grps)) > 1){

      modFile <- system.file("extdata", "qrLMM.jag", package = "rFIA")
      # Parameters to estimate
      params <- c('fe_alpha', 'fe_beta', 'alpha', 'beta')
      ## Set up in a list
      data <- list(I = nrow(prep), ## number of obs
                   J = length(unique(prep$grp_index)), # Number of groups
                   y = prep$t, ## log scale TPA
                   x = prep$b, ## log scale BA
                   p = .99, ## Percentile for qr
                   grp_index = prep$grp_index) # Numeric ID for groups

    } else {

      modFile <- system.file("extdata", "qrLM.jag", package = "rFIA")
      # Parameters to estimate
      params <- c('alpha', 'beta')
      ## Set up in a list
      data <- list(I = nrow(prep), ## number of obs
                   y = prep$t, ## log scale TPA
                   x = prep$b, ## log scale BA
                   p = .99) ## Percentile for qr

    }

    # MCMC settings
    ni <- 1000
    nc <- 3

    message('Modeling maximum size-density curve(s)...')

    # Start Gibbs sampling
    jags_mod_start <- R2jags::jags(data,
                                   parameters.to.save=params,
                                   model.file=modFile,
                                   n.chains=nc,
                                   n.iter=ni)
    jags_mod <- R2jags::autojags(jags_mod_start, n.iter = 1000)


    ## Convert to mcmc list
    jags_mcmc <- coda::as.mcmc(jags_mod)

    chains <- jags_mcmc
    ## Convert to data.frame
    for (i in 1:length(jags_mcmc)){
      chains[[i]] <- as.data.frame(jags_mcmc[[i]])
      names(chains)[i] <- i
    }
    class(chains) <- 'list'


    if (length(unique(grpRates$grps)) > 1){
      ## Make it tidy
      betas <- bind_rows(chains) %>%
        pivot_longer(cols = everything(), names_to = 'var', values_to = 'estimate') %>%
        filter(str_detect(var, 'fe_beta|fe_alpha|deviance', negate = TRUE)) %>%
        mutate(grp_index = unlist(regmatches(var, gregexpr("\\[.+?\\]", var))),
               grp_index = as.numeric(str_sub(grp_index, 2, -2)),
               term = case_when(str_detect(var, 'alpha') ~ 'int',
                                TRUE ~ 'rate')) %>%
        left_join(distinct(prep, grp_index, grps, n), by = 'grp_index') %>%
        select(grps, term, estimate, n) %>%
        mutate(estimate = case_when(term == 'int' ~ exp(estimate),
                                    TRUE ~ estimate)) %>%
        group_by(grps, term) %>%
        summarise(mean = mean(estimate),
                  upper = quantile(estimate, probs = .975),
                  lower = quantile(estimate, probs = .025),
                  n = first(n)) %>%
        pivot_wider(id_cols = c(grps, n), names_from = term, values_from = mean:lower) %>%
        rename(int = mean_int,
               rate = mean_rate)

      ## Predict the fixed effects for missing groups
      post_fe <- bind_rows(chains) %>%
        pivot_longer(cols = everything(), names_to = 'var', values_to = 'estimate') %>%
        filter(str_detect(var, 'fe_beta|fe_alpha')) %>%
        mutate(term = case_when(str_detect(var, 'alpha') ~ 'fe_int',
                                TRUE ~ 'fe_rate')) %>%
        select(term, estimate) %>%
        mutate(estimate = case_when(term == 'fe_int' ~ exp(estimate),
                                    TRUE ~ estimate)) %>%
        group_by(term) %>%
        summarise(mean = mean(estimate),
                  upper = quantile(estimate, probs = .975),
                  lower = quantile(estimate, probs = .025))%>%
        pivot_wider(names_from = term, values_from = mean:lower) %>%
        rename(fe_int = mean_fe_int,
               fe_rate = mean_fe_rate)

      ## Adding fixed effect info
      betas <- betas %>%
        mutate(fe_int = post_fe$fe_int,
               upper_fe_int = post_fe$upper_fe_int,
               lower_fe_int = post_fe$lower_fe_int,
               fe_rate = post_fe$fe_rate,
               upper_fe_rate = post_fe$upper_fe_rate,
               lower_fe_rate = post_fe$lower_fe_rate)

      ## Clean up names of betas
      betas <- betas %>%
        select(grps, alpha = int, rate, alpha_lower = lower_int, alpha_upper = upper_int,
               rate_lower = lower_rate, rate_upper = upper_rate,
               fixed_alpha = fe_int, fixed_rate = fe_rate,
               fixed_alpha_lower = lower_fe_int, fixed_alpha_upper = upper_fe_int,
               fixed_rate_lower = lower_fe_rate, fixed_rate_upper = upper_fe_rate,
               n)

    } else {
      ## Make it tidy
      betas <- bind_rows(chains) %>%
        pivot_longer(cols = everything(), names_to = 'var', values_to = 'estimate') %>%
        filter(str_detect(var, 'deviance', negate = TRUE)) %>%
        mutate(term = case_when(str_detect(var, 'alpha') ~ 'int',
                                TRUE ~ 'rate')) %>%
        select(term, estimate) %>%
        mutate(estimate = case_when(term == 'int' ~ exp(estimate),
                                    TRUE ~ estimate)) %>%
        group_by(term) %>%
        summarise(mean = mean(estimate),
                  upper = quantile(estimate, probs = .975),
                  lower = quantile(estimate, probs = .025)) %>%
        pivot_wider(names_from = term, values_from = mean:lower) %>%
        rename(int = mean_int,
               rate = mean_rate)

      betas$n <- nrow(grpRates)
      betas$grps = 1

      ## Clean up names of betas
      betas <- betas %>%
        select(grps, alpha = int, rate, alpha_lower = lower_int, alpha_upper = upper_int,
               rate_lower = lower_rate, rate_upper = upper_rate, n)

    }


  }


  ## If groups are missing, assume the fixed effects
  if ('fixed_alpha' %in% names(betas)) {
    t <- t %>%
      left_join(select(betas, c(grps, alpha, fixed_alpha, fixed_rate, rate)), by = 'grps') %>%
      mutate(alpha = case_when(!is.na(alpha) ~ fixed_alpha,
                             TRUE ~ alpha),
             rate = case_when(!is.na(rate) ~ fixed_rate,
                             TRUE ~ rate)) %>%
      mutate(ba = BAA / TPA_UNADJ,
             tmax = alpha * (ba^rate),
             rd = TPA_UNADJ / tmax)
  } else {
    ## Add the betas onto t
    t <- t %>%
      left_join(select(betas, c(grps, alpha, rate)), by = 'grps') %>%
      mutate(ba = BAA / TPA_UNADJ,
             tmax = alpha * (ba^rate),
             rd = TPA_UNADJ / tmax)
  }






  ## If byPlot, return plot-level estimates ------------------------------------
  ## Otherwise continue to population estimation -------------------------------
  if (byPlot){

    tOut <- t

    grpSyms <- syms(grpBy[!c(grpBy %in% 'YEAR')])
    grpScaleSyms <- syms(unique(c(grpBy[!c(grpBy %in% 'YEAR')], scaleBy)))

    tOut <- tOut %>%
      lazy_dt() %>%
      ## Summing within scaleBy
      group_by(!!!grpScaleSyms, YEAR, PLT_CN, PLOT_STATUS_CD, PREV_PLT_CN,
               REMPER) %>%
      summarize(PREV_RD = -sum(rd[ONEORTWO == 1] * tDI[ONEORTWO == 1], na.rm = TRUE),
                CURR_RD = sum(rd[ONEORTWO == 2] * tDI[ONEORTWO == 2], na.rm = TRUE),
                PREV_TPA = -sum(TPA_UNADJ[ONEORTWO == 1] * tDI[ONEORTWO == 1], na.rm = TRUE),
                PREV_BAA = -sum(BAA[ONEORTWO == 1] * tDI[ONEORTWO == 1], na.rm = TRUE),
                CURR_TPA = sum(TPA_UNADJ[ONEORTWO == 2] * tDI[ONEORTWO == 2], na.rm = TRUE),
                CURR_BAA = sum(BAA[ONEORTWO == 2] * tDI[ONEORTWO == 2], na.rm = TRUE)) %>%
      ## Summing across scaleBy
      group_by(!!!grpSyms, YEAR, PLT_CN, PLOT_STATUS_CD, PREV_PLT_CN,
               REMPER) %>%
      summarize(PREV_RD = mean(PREV_RD, na.rm = TRUE),
                CURR_RD = mean(CURR_RD, na.rm = TRUE),
                PREV_TPA = sum(PREV_TPA, na.rm = TRUE),
                PREV_BAA = sum(PREV_BAA, na.rm = TRUE),
                CURR_TPA = sum(CURR_TPA, na.rm = TRUE),
                CURR_BAA = sum(CURR_BAA, na.rm = TRUE)) %>%
      mutate(FSI = (CURR_RD - PREV_RD) / REMPER,
             PERC_FSI = FSI / PREV_RD * 100) %>%
      as.data.frame()

    ## If we want to use multiple remeasurements to estimate change,
    ## handle that here
    if (useSeries) {
      ## Get a unique ID for each remeasurement in the series
      nMeas <- tOut %>%
        distinct(pltID, PLT_CN, YEAR, REMPER) %>%
        group_by(pltID) %>%
        mutate(n = length(unique(PLT_CN)),
               series = min_rank(YEAR)) %>%
        ungroup() %>%
        select(pltID, PLT_CN, REMPER, n, series)

      ## Only if more than one remeasurement available
      if (any(nMeas$n > 1)){

        ## Now we loop over the unique values of n
        ## Basically have to chunk up the data each time
        ## in order to get intermediate estimates
        nRems <- unique(nMeas$n)
        remsList <- list()
        for (i in 1:length(nRems)){
          ## Temporal weights for each plot
          wgts <- nMeas %>%
            filter(series <= nRems[i] & n >= nRems[i]) %>%
            group_by(pltID) %>%
            ## Total remeasurement interval and weights for
            ## individual remeasurements
            mutate(fullRemp = sum(REMPER, na.rm = TRUE),
                   wgt = REMPER / fullRemp) %>%
            ungroup() %>%
            select(PLT_CN, n, series, wgt, fullRemp)

          grpSyms <- syms(grpBy[grpBy %in% c('YEAR', 'INVYR', 'MEASYEAR', 'PLOT_STATUS_CD') == FALSE])

          dat <- tOut %>%
            lazy_dt() %>%
            left_join(wgts, by = c('PLT_CN')) %>%
            filter(series <= nRems[i] & n >= nRems[i]) %>%
            group_by(!!!grpSyms) %>%
            mutate(PLOT_STATUS_CD = case_when(any(PLOT_STATUS_CD == 1) ~ as.double(1),
                                              TRUE ~ as.double(PLOT_STATUS_CD))) %>%
            summarize(FSI = sum(FSI*wgt, na.rm = TRUE),
                      PLT_CN = PLT_CN[which.max(series)],
                      CURR_RD = CURR_RD[which.max(series)],
                      PREV_RD = PREV_RD[which.min(series)],
                      PREV_TPA = PREV_TPA[which.min(series)],
                      PREV_BAA = PREV_BAA[which.min(series)],
                      CURR_TPA = CURR_TPA[which.max(series)],
                      CURR_BAA = CURR_BAA[which.max(series)],
                      REMPER = first(fullRemp),
                      PLOT_STATUS_CD = first(PLOT_STATUS_CD)) %>%
            ungroup() %>%
            as.data.frame()
          remsList[[i]] <- dat
        }
        ## Bring it all back together
        dat <- bind_rows(remsList)

        ## Update columns in tEst
        tOut <- tOut %>%
          ungroup() %>%
          select(-c(PREV_RD:CURR_BAA, REMPER, PLOT_STATUS_CD)) %>%
          left_join(dat, by = c('PLT_CN', grpBy[!c(grpBy %in% c('YEAR', 'INVYR', 'MEASYEAR', 'PLOT_STATUS_CD'))])) %>%
          mutate(PERC_FSI = FSI / PREV_RD * 100)
        }
    }

    ## Make it spatial
    if (returnSpatial){
      tOut <- tOut %>%
        filter(!is.na(LAT) & !is.na(LON)) %>%
        st_as_sf(coords = c('LON', 'LAT'),
                 crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
      grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]

    }



    tOut <- select(tOut, YEAR, PLT_CN, any_of('PREV_PLT_CN'), PLOT_STATUS_CD, grpBy[grpBy %in% c('YEAR', 'REMPER') == FALSE],
                   REMPER, FSI, PERC_FSI, PREV_RD, CURR_RD, PREV_TPA, CURR_TPA, PREV_BAA, CURR_BAA)



  ## Population estimation
  } else {
    ## back to dataframes
    pops <- bind_rows(out[names(out) == 'pops'])

    ## Adding YEAR to groups
    grpBy <- c('YEAR', grpBy)
    popState <- split(pops, as.factor(pops$STATECD))


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
        out <- parLapply(cl, X = names(popState), fun = fsiHelper2, popState, t, a, grpBy, scaleBy, method, useSeries)
        stopCluster(cl)
      } else { # Unix systems
        out <- mclapply(names(popState), FUN = fsiHelper2, popState, t, a, grpBy, scaleBy, method, useSeries, mc.cores = nCores)
      }
    })
    ## back to dataframes
    out <- unlist(out, recursive = FALSE)
    tEst <- bind_rows(out[names(out) == 'tEst'])
    tEst <- ungroup(tEst)



    ## Compute moving average weights if not TI ----------------------------------
    if (str_to_upper(method) %in% c("SMA", 'EMA', 'LMA')){

      ## Compute the weights
      wgts <- maWeights(pops, method, lambda)

      ## If moving average ribbons, add lambda to grpBy for easier summary
      if (str_to_upper(method) == 'EMA' & length(lambda) > 1){
        grpBy <- c('lambda', grpBy)
      }


      ## Apply the weights
      if (str_to_upper(method) %in% c('LMA', 'EMA')){
        joinCols <- c('YEAR', 'STATECD', 'INVYR')
      } else {
        joinCols <- c('YEAR', 'STATECD')
      }
      tEst <- tEst %>%
        left_join(select(db$POP_ESTN_UNIT, CN, STATECD), by = c('ESTN_UNIT_CN' = 'CN')) %>%
        left_join(wgts, by = joinCols) %>%
        mutate(across(ctEst:rempEst, ~(.*wgt))) %>%
        mutate(across(ctVar:cvEst_remp, ~(.*(wgt^2)))) %>%
        group_by(ESTN_UNIT_CN, .dots = grpBy) %>%
        summarize(across(ctEst:plotIn_t, sum, na.rm = TRUE))


      ## If using an ANNUAL estimator --------------------------------------------
    } else if (str_to_upper(method) == 'ANNUAL') {

      # If INVYR is in YEAR, choose the estimates when INVYR == YEAR
      # Otherwise, choose the estimates produced with the most plots
      tEst <- filterAnnual(tEst, grpBy, plotIn_t, db$POP_ESTN_UNIT)

    }





    ## Combine most-recent population estimates across states with potentially
    ## different reporting schedules, i.e., if 2016 is most recent in MI and 2017 is
    ## most recent in WI, combine them and label as 2017
    if (mr) {
      tEst <- combineMR(tEst)
    }



    ## Totals and ratios -------------------------------------------------------
    # Tree
    tTotal <- tEst %>%
      group_by(.dots = grpBy) %>%
      summarize_all(sum, na.rm = TRUE)

    suppressWarnings({
      tOut <- tTotal %>%
        group_by(.dots = grpBy) %>%
        summarize_all(sum,na.rm = TRUE) %>%
        mutate(TPA_RATE = ctEst / ptEst,
               BA_RATE = cbEst / pbEst,
               FSI = siEst / faEst,
               PERC_FSI = siEst / ra1Est,
               PREV_RD = ra1Est / faEst,
               CURR_RD = ra2Est / faEst,
               REMPER = rempEst / faEst,

               ## Ratio variance
               ctVar = (1/ptEst^2) * (ctVar + (TPA_RATE^2 * ptVar) - (2 * TPA_RATE * cvEst_ct)),
               cbVar = (1/pbEst^2) * (cbVar + (BA_RATE^2 * pbVar) - (2 * BA_RATE * cvEst_cb)),
               psiVar = (1/ra1Est^2) * (siVar + (PERC_FSI^2 * ra1Var) - (2 * PERC_FSI * cvEst_psi)),
               siVar = (1/faEst^2) * (siVar + (FSI^2 * faVar) - (2 * FSI * cvEst_si)),
               ra1Var = (1/faEst^2) * (ra1Var + (PREV_RD^2 * faVar) - (2 * PREV_RD * cvEst_ra1)),
               ra2Var = (1/faEst^2) * (ra2Var + (CURR_RD^2 * faVar) - (2 * CURR_RD * cvEst_ra2)),
               rempVar = (1/faEst^2) * (rempVar + (REMPER^2 * faVar) - (2 * REMPER * cvEst_remp)),

               ## Make it a percent
               PERC_FSI = PERC_FSI * 100,
               psiVar = psiVar * (100^2),

               ## RATIO variance
               FSI_VAR = siVar,
               PERC_FSI_SE = sqrt(psiVar) / abs(PERC_FSI) * 100,
               PERC_FSI_VAR = psiVar,
               TPA_RATE_VAR = ctVar,
               BA_RATE_VAR = cbVar,
               PREV_RD_VAR = ra1Var,
               CURR_RD_VAR = ra2Var,
               REMPER_VAR = rempVar,

               nPlots = plotIn_t,
               N = p2eu,
               FSI_INT = qt(.975, df=N-1) * (sqrt(siVar)/sqrt(N)),
               PERC_FSI_INT = qt(.975, df=N-1) * (sqrt(psiVar)/sqrt(N))) %>%
        mutate(FSI_STATUS = case_when(
          FSI < 0 & FSI + FSI_INT < 0 ~ 'Decline',
          FSI < 0 & FSI + FSI_INT > 0 ~ 'Stable',
          FSI > 0 & FSI - FSI_INT > 0  ~ 'Expand',
          TRUE ~ 'Stable'
        ))
    })


    if (totals) {
      tOut <- tOut %>%
        select(grpBy, FSI, PERC_FSI, FSI_STATUS,
               FSI_INT, PERC_FSI_INT,
               PREV_RD, CURR_RD, TPA_RATE, BA_RATE,
               FSI_VAR, PERC_FSI_VAR, PREV_RD_VAR, CURR_RD_VAR,
               TPA_RATE_VAR, BA_RATE_VAR,
               nPlots, N)

    } else {
      tOut <- tOut %>%
        select(grpBy, FSI, PERC_FSI, FSI_STATUS,
               FSI_INT, PERC_FSI_INT,
               PREV_RD, CURR_RD, TPA_RATE, BA_RATE,
               FSI_VAR, PERC_FSI_VAR, PREV_RD_VAR, CURR_RD_VAR,
               TPA_RATE_VAR, BA_RATE_VAR,
               nPlots, N)
    }

    # Snag the names
    tNames <- names(tOut)[names(tOut) %in% grpBy == FALSE]

  }

  ## Pretty output
  tOut <- tOut %>%
    ungroup() %>%
    mutate_if(is.factor, as.character) %>%
    drop_na(grpBy) %>%
    arrange(YEAR) %>%
    as_tibble()

  ## Make implicit NA explicit for spatial summaries
  ## Not sure if I like this or not, but I'm going with it for now
  tOut <- prettyNamesSF(tOut, polys, byPlot, grpBy, grpByOrig, tNames, returnSpatial)

  ## For spatial plots
  if (returnSpatial & byPlot) grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]

  ## Above converts to tibble
  if (returnSpatial) tOut <- st_sf(tOut)

  ## remove any duplicates in byPlot
  ## Also make PLOT_STATUS_CD more informative
  if (byPlot) {
    tOut <- unique(tOut)
    tOut <- tOut %>%
      mutate(PLOT_STATUS = case_when(is.na(PLOT_STATUS_CD) ~ NA_character_,
                                     PLOT_STATUS_CD == 1 ~ 'Forest',
                                     PLOT_STATUS_CD == 2 ~ 'Non-forest',
                                     PLOT_STATUS_CD == 3 ~ 'Non-sampled')) %>%
      relocate(PLOT_STATUS, .after = PLOT_STATUS_CD)
  }

  if (returnBetas) {
    tOut <- list(results = tOut, betas = betas)
  }

  return(tOut)

}


