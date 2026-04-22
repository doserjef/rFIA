fsi <- function(db, grpBy = NULL, polys = NULL, returnSpatial = FALSE, 
                bySpecies = FALSE, bySizeClass = FALSE, landType = 'forest', 
                treeType = 'live', method = 'TI', lambda = .5, 
                treeDomain = NULL, areaDomain = NULL, totals = TRUE, 
                variance = TRUE, byPlot = FALSE, useSeries = FALSE, 
                mostRecent = FALSE, scaleBy = NULL, betas = NULL, 
                returnBetas = FALSE, nCores = 1) {
  
  # Need rjags and coda installed if we're going to model size-density relationships
  if (!is.null(betas) & is.null(find.package('R2jags')) & is.null(find.package('coda'))) {
    stop('Packages "R2jags" and "coda" required to model maximum size-density curves. Please install with install.packages(c("R2jags", "coda")) and try again.

If not already installed, you can install JAGS from SourceForge:
         Windows: https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Windows/
         Mac: https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Mac%20OS%20X/
         Linux: https://mcmc-jags.sourceforge.io/')
  } else if (!is.null(betas) & is.null(find.package('R2jags'))) {
    stop('Package "R2jags" required to model maximum size-density curves. Please install with install.packages("R2jags") and try again.

If not already installed, you can install JAGS from SourceForge:
         Windows: https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Windows/
         Mac: https://sourceforge.net/projects/mcmc-jags/files/JAGS/4.x/Mac%20OS%20X/
         Linux: https://mcmc-jags.sourceforge.io/')
  } else if (!is.null(betas) &  is.null(find.package('coda'))) {
    stop('Package "coda" required to model maximum size-density curves. Please install with install.packages("coda") and try again.')
  }
  
  # Defuse user-supplied expressions in necessary arguments. 
  grpBy_quo <- rlang::enquo(grpBy)
  scaleBy_quo <- rlang::enquo(scaleBy)
  areaDomain <- rlang::enquo(areaDomain)
  treeDomain <- rlang::enquo(treeDomain)
  
  # Handle iterator if db is remote
  remote <- ifelse(class(db) == 'Remote.FIA.Database', 1, 0)
  # Takes value 1 if not remote. iter for remote is a vector of different state IDs. 
  iter <- remoteIter(db, remote)
  
  # Check for a most recent subset (simple logical value)
  mr <- checkMR(db, remote)
  
  # Prep for areal summary (converts polys to sf, converts factors to chrs, 
  # and adds the polyID column giving a unique ID to each areal unit). 
  polys <- arealSumPrep1(polys)
  
  # Run the starter function to extract the necessary 
  # information from FIADB to estimate max size-density curves.
  out <- lapply(X = iter, FUN = fsiStarter, db, grpBy_quo = grpBy_quo, 
                scaleBy_quo, polys, returnSpatial, bySpecies, bySizeClass,
                landType, treeType, method, lambda, treeDomain, areaDomain,
                totals, byPlot, useSeries, mostRecent, nCores, remote, mr)
  
  # Extract results from list.  
  out <- unlist(out, recursive = FALSE)
  # Might want to add this back in in the future.  
  # if (remote) out <- dropStatesOutsidePolys(out)
  # Tree-level information used when calculating FSI
  t <- bind_rows(out[names(out) == 't'])
  # Plot-level information
  t1 <- bind_rows(out[names(out) == 't1'])
  # Area information used when byPlot = FALSE
  a <- bind_rows(out[names(out) == 'a'])
  grpBy <- out[names(out) == 'grpBy'][[1]]
  scaleBy <- out[names(out) == 'scaleBy'][[1]]
  mr <- out[names(out) == 'mr'][[1]]
  
  # Prep data to model maximum size-density curves ------------------------
  scaleSyms <- rlang::syms(scaleBy)
  grpSyms <- rlang::syms(grpBy)
  
  # Get groups prepped to fit model (t1 is plot-level data)
  grpRates <- t1 %>% 
    dplyr::select(PLT_CN, !!!scaleSyms, BA2, TPA2) %>%
    dplyr::ungroup() %>%
    dplyr::filter(TPA2 > 0) %>%
    tidyr::drop_na(!!!scaleSyms) %>%
    # Stand-level variables here
    dplyr::mutate(t = log(TPA2),
                  b = log(BA2)) %>%
    dplyr::select(t, b, PLT_CN, !!!scaleSyms)
  
  if (!is.null(scaleBy)) {
    # group IDS
    grpRates <- grpRates %>% 
      dplyr::mutate(grps = as.factor(paste(!!!scaleSyms)))
    
    t <- t %>% 
      dplyr::mutate(grps = as.factor(paste(!!!scaleSyms)))
    
    # Determine groups with less than 10 observations. 
    nGrps <- grpRates %>%
      dplyr::group_by(grps) %>%
      dplyr::summarise(n = n())
    grpRates <- grpRates %>%
      dplyr::left_join(nGrps, by = 'grps')
    
  } else {
    grpRates$grps <- 1
    t$grps = 1
  }
  
  # Estimate parameters ---------------------------------------------------
  if (is.null(betas)) {
    # Prep data for model. Note that this is the plot-level data that gets used 
    # to fit the quantile regression model. 
    prep <- grpRates %>% 
      dplyr::ungroup() %>% 
      dplyr::arrange(grps) %>% 
      # Need numeric group-level assignments for JAGS
      dplyr::mutate(grp_index = as.numeric(as.factor(grps))) 
    
    # If more than one group, use a mixed model
    if (length(unique(grpRates$grps)) > 1) {
      modFile <- system.file("extdata", "qrLMM.jag", package = "rFIA")
      # Parameters to estimate
      params <- c('fe_alpha', 'fe_beta', 'alpha', 'beta')
      # Set up in a list
      data <- list(I = nrow(prep), # Number of obs
                   J = length(unique(prep$grp_index)), # Number of groups
                   y = prep$t, # log scale TPA
                   x = prep$b, # log scale BA
                   p = .99, # Percentile for qr
                   grp_index = prep$grp_index) # Numeric ID for groups
    } else {
      modFile <- system.file("extdata", "qrLM.jag", package = "rFIA")
      # Parameters to estimate
      params <- c('alpha', 'beta')
      # Set up in a list
      data <- list(I = nrow(prep), # number of obs
                   y = prep$t, # log scale TPA
                   x = prep$b, # log scale BA
                   p = .99) # Percentile for qr
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
    
    # Convert to MCMC list
    jags_mcmc <- coda::as.mcmc(jags_mod)
    
    chains <- jags_mcmc
    # Convert to data.frame
    for (i in 1:length(jags_mcmc)){
      chains[[i]] <- as.data.frame(jags_mcmc[[i]])
      names(chains)[i] <- i
    }
    class(chains) <- 'list'
    
    if (length(unique(grpRates$grps)) > 1){ # Tidy up LMMs
      # Make it tidy
      betas <- chains %>%
        dplyr::bind_rows() %>% 
        tidyr::pivot_longer(cols = everything(), names_to = 'var', values_to = 'estimate') %>%
        dplyr::filter(stringr::str_detect(var, 'fe_beta|fe_alpha|deviance', negate = TRUE)) %>%
        mutate(grp_index = unlist(regmatches(var, gregexpr("\\[.+?\\]", var))),
               grp_index = as.numeric(stringr::str_sub(grp_index, 2, -2)),
               term = dplyr::case_when(stringr::str_detect(var, 'alpha') ~ 'int',
                                       TRUE ~ 'rate')) %>%
        dplyr::left_join(dplyr::distinct(prep, grp_index, grps, n), by = 'grp_index') %>%
        dplyr::select(grps, term, estimate, n) %>%
        dplyr::mutate(estimate = case_when(term == 'int' ~ exp(estimate),
                                           TRUE ~ estimate)) %>%
        dplyr::group_by(grps, term) %>%
        dplyr::summarise(mean = mean(estimate),
                         upper = quantile(estimate, probs = .975),
                         lower = quantile(estimate, probs = .025),
                         n = first(n)) %>%
        tidyr::pivot_wider(id_cols = c(grps, n), names_from = term, values_from = mean:lower) %>%
        dplyr::rename(int = mean_int,
                      rate = mean_rate)
      
      # Predict the fixed effects for missing groups
      post_fe <- chains %>%
        dplyr::bind_rows() %>% 
        tidyr::pivot_longer(cols = everything(), names_to = 'var', values_to = 'estimate') %>%
        dplyr::filter(stringr::str_detect(var, 'fe_beta|fe_alpha')) %>%
        dplyr::mutate(term = dplyr::case_when(stringr::str_detect(var, 'alpha') ~ 'fe_int',
                                              TRUE ~ 'fe_rate')) %>%
        dplyr::select(term, estimate) %>%
        dplyr::mutate(estimate = dplyr::case_when(term == 'fe_int' ~ exp(estimate),
                                                  TRUE ~ estimate)) %>%
        dplyr::group_by(term) %>%
        dplyr::summarise(mean = mean(estimate),
                         upper = quantile(estimate, probs = .975),
                         lower = quantile(estimate, probs = .025))%>%
        tidyr::pivot_wider(names_from = term, values_from = mean:lower) %>%
        dplyr::rename(fe_int = mean_fe_int,
                      fe_rate = mean_fe_rate)
      
      # Adding fixed effect info
      betas <- betas %>%
        dplyr::mutate(fe_int = post_fe$fe_int,
                      upper_fe_int = post_fe$upper_fe_int,
                      lower_fe_int = post_fe$lower_fe_int,
                      fe_rate = post_fe$fe_rate,
                      upper_fe_rate = post_fe$upper_fe_rate,
                      lower_fe_rate = post_fe$lower_fe_rate)
      
      # Clean up names of betas
      betas <- betas %>%
        dplyr::select(grps, alpha = int, rate, alpha_lower = lower_int, 
                      alpha_upper = upper_int,
                      rate_lower = lower_rate, rate_upper = upper_rate,
                      fixed_alpha = fe_int, fixed_rate = fe_rate,
                      fixed_alpha_lower = lower_fe_int, fixed_alpha_upper = upper_fe_int,
                      fixed_rate_lower = lower_fe_rate, fixed_rate_upper = upper_fe_rate,
                      n)
      
    } else { # Tidy up LMs
      # Make it tidy
      betas <- chains %>%
        dplyr::bind_rows()
      tidyr::pivot_longer(cols = everything(), names_to = 'var', values_to = 'estimate') %>%
        dplyr::filter(stringr::str_detect(var, 'deviance', negate = TRUE)) %>%
        dplyr::mutate(term = dplyr::case_when(stringr::str_detect(var, 'alpha') ~ 'int',
                                              TRUE ~ 'rate')) %>%
        dplyr::select(term, estimate) %>%
        dplyr::mutate(estimate = dplyr::case_when(term == 'int' ~ exp(estimate),
                                                  TRUE ~ estimate)) %>%
        dplyr::group_by(term) %>%
        dplyr::summarise(mean = mean(estimate),
                         upper = quantile(estimate, probs = .975),
                         lower = quantile(estimate, probs = .025)) %>%
        tidyr::pivot_wider(names_from = term, values_from = mean:lower) %>%
        dplyr::rename(int = mean_int,
                      rate = mean_rate)
      
      betas$n <- nrow(grpRates)
      betas$grps = 1
      
      # Clean up names of betas
      betas <- betas %>%
        dplyr::select(grps, alpha = int, rate, alpha_lower = lower_int, 
                      alpha_upper = upper_int,
                      rate_lower = lower_rate, rate_upper = upper_rate, n)
    } # Fixed vs. mixed model
  } # Fitting or not fitting the model
  
  # If groups are missing, assume the fixed effects. This happens when, for example, 
  # there are trees in forest types where the plots in that forest type do not 
  # fit the criteria for being used for the creation of the max density curve. 
  # This is only relevant when scaleBy = TRUE, which is what the if statement 
  # below is effectively testing. 
  if ('fixed_alpha' %in% names(betas)) {
    t <- t %>%
      dplyr::left_join(dplyr::select(betas, c(grps, alpha, fixed_alpha, 
                                              fixed_rate, rate)), by = 'grps') %>%
      dplyr::mutate(alpha = dplyr::case_when(is.na(alpha) ~ unique(betas$fixed_alpha),
                                             TRUE ~ alpha),
                    rate = dplyr::case_when(is.na(rate) ~ unique(betas$fixed_rate),
                                            TRUE ~ rate)) %>%
      dplyr::mutate(ba = BAA / TPA_UNADJ,
                    tmax = alpha * (ba^rate),
                    rd = TPA_UNADJ / tmax)
  } else {
    # Add the betas onto t
    t <- t %>%
      dplyr::left_join(dplyr::select(betas, c(grps, alpha, rate)), by = 'grps') %>%
      dplyr::mutate(ba = BAA / TPA_UNADJ,
                    tmax = alpha * (ba^rate),
                    rd = TPA_UNADJ / tmax)
  }
  
  if (byPlot) {
    
    # Tree level data with relative density.  
    tOut <- t
    
    grpSyms <- syms(grpBy[!c(grpBy %in% 'YEAR')])
    grpScaleSyms <- syms(unique(c(grpBy[!c(grpBy %in% 'YEAR')], scaleBy)))
    
    tOut <- tOut %>%
      dtplyr::lazy_dt() %>%
      # Summing within scaleBy
      dplyr::group_by(!!!grpScaleSyms, YEAR, PLT_CN, PLOT_STATUS_CD, PREV_PLT_CN,
                      REMPER) %>%
      dplyr::summarize(PREV_RD = -sum(rd[ONEORTWO == 1] * tDI[ONEORTWO == 1], 
                                      na.rm = TRUE),
                       CURR_RD = sum(rd[ONEORTWO == 2] * tDI[ONEORTWO == 2], 
                                     na.rm = TRUE),
                       PREV_TPA = -sum(TPA_UNADJ[ONEORTWO == 1] * tDI[ONEORTWO == 1], 
                                       na.rm = TRUE),
                       PREV_BAA = -sum(BAA[ONEORTWO == 1] * tDI[ONEORTWO == 1], 
                                       na.rm = TRUE),
                       CURR_TPA = sum(TPA_UNADJ[ONEORTWO == 2] * tDI[ONEORTWO == 2], 
                                      na.rm = TRUE),
                       CURR_BAA = sum(BAA[ONEORTWO == 2] * tDI[ONEORTWO == 2], 
                                      na.rm = TRUE)) %>%
      # Summing across scaleBy
      dplyr::group_by(!!!grpSyms, YEAR, PLT_CN, PLOT_STATUS_CD, PREV_PLT_CN,
                      REMPER) %>%
      # Note that overall previous and current density are defined as the 
      # average PREV_RD and average CURR_RD across all standBy groups. This is
      # only the case however if there are multiple scaleBys within each PLT_CN, 
      # which will often not be the case. 
      dplyr::summarize(PREV_RD = mean(PREV_RD, na.rm = TRUE),
                       CURR_RD = mean(CURR_RD, na.rm = TRUE),
                       PREV_TPA = sum(PREV_TPA, na.rm = TRUE),
                       PREV_BAA = sum(PREV_BAA, na.rm = TRUE),
                       CURR_TPA = sum(CURR_TPA, na.rm = TRUE),
                       CURR_BAA = sum(CURR_BAA, na.rm = TRUE)) %>%
      dplyr::mutate(FSI = (CURR_RD - PREV_RD) / REMPER,
                    PERC_FSI = FSI / PREV_RD * 100) %>%
      as.data.frame()
    
    # If we want to use multiple remeasurements to estimate change,
    # handle that here
    if (useSeries) {
      # Get a unique ID for each remeasurement in the series
      nMeas <- tOut %>%
        dplyr::distinct(pltID, PLT_CN, YEAR, REMPER) %>%
        dplyr::group_by(pltID) %>%
        dplyr::mutate(n = length(unique(PLT_CN)),
                      series = dplyr::min_rank(YEAR)) %>%
        dplyr::ungroup() %>%
        dplyr::select(pltID, PLT_CN, REMPER, n, series)
      
      # Only if more than one remeasurement available
      if (any(nMeas$n > 1)) {
        
        # Now we loop over the unique values of n
        # Basically have to chunk up the data each time
        # in order to get intermediate estimates
        nRems <- unique(nMeas$n)
        remsList <- list()
        for (i in 1:length(nRems)) {
          # Temporal weights for each plot
          wgts <- nMeas %>%
            dplyr::filter(series <= nRems[i] & n >= nRems[i]) %>%
            dplyr::group_by(pltID) %>%
            # Total remeasurement interval and weights for
            # individual remeasurements
            dplyr::mutate(fullRemp = sum(REMPER, na.rm = TRUE),
                          wgt = REMPER / fullRemp) %>%
            dplyr::ungroup() %>%
            dplyr::select(PLT_CN, n, series, wgt, fullRemp)
          
          grpSyms <- rlang::syms(grpBy[grpBy %in% c('YEAR', 'INVYR', 
                                                    'MEASYEAR', 'PLOT_STATUS_CD') == FALSE])
          
          dat <- tOut %>%
            dplyr::left_join(wgts, by = c('PLT_CN')) %>%
            dplyr::filter(series <= nRems[i] & n >= nRems[i]) %>%
            dplyr::group_by(!!!grpSyms) %>%
            dplyr::mutate(PLOT_STATUS_CD = dplyr::case_when(any(PLOT_STATUS_CD == 1) ~ as.double(1),
                                                            TRUE ~ as.double(PLOT_STATUS_CD))) %>%
            dplyr::summarize(FSI = sum(FSI*wgt, na.rm = TRUE),
                             PLT_CN = PLT_CN[which.max(series)],
                             CURR_RD = CURR_RD[which.max(series)],
                             PREV_RD = PREV_RD[which.min(series)],
                             PREV_TPA = PREV_TPA[which.min(series)],
                             PREV_BAA = PREV_BAA[which.min(series)],
                             CURR_TPA = CURR_TPA[which.max(series)],
                             CURR_BAA = CURR_BAA[which.max(series)],
                             REMPER = first(fullRemp),
                             PLOT_STATUS_CD = first(PLOT_STATUS_CD)) %>%
            dplyr::ungroup() %>%
            as.data.frame()
          remsList[[i]] <- dat
        }
        # Bring it all back together
        dat <- dplyr::bind_rows(remsList)
        
        # Update columns in tEst
        tOut <- tOut %>%
          dplyr::ungroup() %>%
          dplyr::select(-c(PREV_RD:CURR_BAA, REMPER, PLOT_STATUS_CD, FSI)) %>%
          dplyr::left_join(dat, by = c('PLT_CN', grpBy[!c(grpBy %in% c('YEAR', 'INVYR', 'MEASYEAR', 'PLOT_STATUS_CD'))])) %>%
          dplyr::mutate(PERC_FSI = FSI / PREV_RD * 100)
      }
    }
    
    # Make it spatial
    if (returnSpatial){
      tOut <- tOut %>%
        dplyr::filter(!is.na(LAT) & !is.na(LON)) %>%
        sf::st_as_sf(coords = c('LON', 'LAT'),
                        crs = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
      grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]
    }
    
    
    
    tOut <- tOut %>%
      dplyr::select(YEAR, PLT_CN, dplyr::any_of('PREV_PLT_CN'), PLOT_STATUS_CD, 
                    grpBy[grpBy %in% c('YEAR', 'REMPER') == FALSE],
                    REMPER, FSI, PERC_FSI, PREV_RD, CURR_RD, PREV_TPA, 
                    CURR_TPA, PREV_BAA, CURR_BAA)
  } else { # Population Estimation
    # Convert pops to data frames
    pops <- bind_rows(out[names(out) == 'pops'])
    
    # Adding YEAR to groups
    grpBy <- c('YEAR', grpBy)
    
    # Split the population up by state and do the population estimation there. 
    popState <- split(pops, as.factor(pops$STATECD))
    
    suppressWarnings({
      # estimates in parallel -- Clusters in windows, forking otherwise
      if (Sys.info()['sysname'] == 'Windows'){
        cl <- makeCluster(nCores)
        clusterEvalQ(cl, {
          library(dplyr)
          library(stringr)
          library(rFIA)
          library(tidyr)
        })
        out <- parLapply(cl, X = names(popState), fun = fsiHelper2, popState, t, a, 
                         grpBy, scaleBy, method, useSeries)
        stopCluster(cl)
      } else { # Unix systems
        out <- mclapply(names(popState), FUN = fsiHelper2, popState, t, a, grpBy, 
                        scaleBy, method, useSeries, mc.cores = nCores)
      }
    })
    # Back to dataframes
    out <- unlist(out, recursive = FALSE)
    tEst <- dplyr::bind_rows(out[names(out) == 'tEst'])
    tEst <- dplyr::ungroup(tEst)
    
    # Compute moving average weights if not TI ----------------------------
    if (stringr::str_to_upper(method) %in% c("SMA", 'EMA', 'LMA')){
      
      # Compute the weights
      wgts <- maWeights(pops, method, lambda)
      
      # If moving average ribbons, add lambda to grpBy for easier summary
      if (stringr::str_to_upper(method) == 'EMA' & length(lambda) > 1){
        grpBy <- c('lambda', grpBy)
      }
      
      # Apply the weights
      if (stringr::str_to_upper(method) %in% c('LMA', 'EMA')){
        joinCols <- c('YEAR', 'STATECD', 'INVYR')
      } else {
        joinCols <- c('YEAR', 'STATECD')
      }
      tEst <- tEst %>%
        dplyr::left_join(select(db$POP_ESTN_UNIT, CN, STATECD), by = c('ESTN_UNIT_CN' = 'CN')) %>%
        dplyr::left_join(wgts, by = joinCols) %>%
        dplyr::mutate(dplyr::across(ctEst:rempEst, ~(.*wgt))) %>%
        dplyr::mutate(dplyr::across(ctVar:cvEst_remp, ~(.*(wgt^2)))) %>%
        dplyr::group_by(ESTN_UNIT_CN, .dots = grpBy) %>%
        dplyr::summarize(across(ctEst:plotIn_t, sum, na.rm = TRUE))
      
      # If using an ANNUAL estimator --------------------------------------------
    } else if (stringr::str_to_upper(method) == 'ANNUAL') {
      
      # If INVYR is in YEAR, choose the estimates when INVYR == YEAR
      # Otherwise, choose the estimates produced with the most plots
      tEst <- filterAnnual(tEst, grpBy, plotIn_t, db$POP_ESTN_UNIT)
      
    }
    
    # Combine most-recent population estimates across states with potentially
    # different reporting schedules, i.e., if 2016 is most recent in MI and 2017 is
    # most recent in WI, combine them and label as 2017
    if (mr) {
      tEst <- combineMR(tEst)
    }
    
    ## Totals and ratios -------------------------------------------------------
    # Tree
    tTotal <- tEst %>%
      dplyr::group_by(.dots = grpBy) %>%
      dplyr::summarize_all(sum, na.rm = TRUE)
    
    suppressWarnings({
      tOut <- tTotal %>%
        dplyr::group_by(.dots = grpBy) %>%
        dplyr::summarize_all(sum,na.rm = TRUE) %>%
        dplyr::mutate(TPA_RATE = ctEst / ptEst,
                      BA_RATE = cbEst / pbEst,
                      FSI = siEst / faEst,
                      PERC_FSI = siEst / ra1Est,
                      PREV_RD = ra1Est / faEst,
                      CURR_RD = ra2Est / faEst,
                      REMPER = rempEst / faEst,
                      
                      # Ratio variance
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
                      N = P2PNTCNT_EU,
                      FSI_INT = qt(.975, df=N-1) * (sqrt(siVar)/sqrt(N)),
                      PERC_FSI_INT = qt(.975, df=N-1) * (sqrt(psiVar)/sqrt(N))) %>%
        dplyr::mutate(FSI_STATUS = dplyr::case_when(
          FSI < 0 & FSI + FSI_INT < 0 ~ 'Decline',
          FSI < 0 & FSI + FSI_INT > 0 ~ 'Stable',
          FSI > 0 & FSI - FSI_INT > 0  ~ 'Expand',
          TRUE ~ 'Stable'
        ))
    })
    
    
    if (totals) {
      tOut <- tOut %>%
        dplyr::select(grpBy, FSI, PERC_FSI, FSI_STATUS,
                      FSI_INT, PERC_FSI_INT,
                      PREV_RD, CURR_RD, TPA_RATE, BA_RATE,
                      FSI_VAR, PERC_FSI_VAR, PREV_RD_VAR, CURR_RD_VAR,
                      TPA_RATE_VAR, BA_RATE_VAR,
                      nPlots, N)
      
    } else {
      tOut <- tOut %>%
        dplyr::select(grpBy, FSI, PERC_FSI, FSI_STATUS,
                      FSI_INT, PERC_FSI_INT,
                      PREV_RD, CURR_RD, TPA_RATE, BA_RATE,
                      FSI_VAR, PERC_FSI_VAR, PREV_RD_VAR, CURR_RD_VAR,
                      TPA_RATE_VAR, BA_RATE_VAR,
                      nPlots, N)
    }
    
    # Snag the names
    tNames <- names(tOut)[names(tOut) %in% grpBy == FALSE]
    
  }
  
  # Pretty output
  tOut <- tOut %>%
    dplyr::ungroup() %>%
    dplyr::mutate_if(is.factor, as.character) %>%
    tidyr::drop_na(grpBy) %>%
    dplyr::arrange(YEAR) %>%
    dplyr::as_tibble()
  
  # For spatial plots
  if (returnSpatial & byPlot) grpBy <- grpBy[grpBy %in% c('LAT', 'LON') == FALSE]
  
  # Above converts to tibble
  if (returnSpatial) tOut <- sf::st_sf(tOut)
  
  # Remove any duplicates in byPlot
  # Also make PLOT_STATUS_CD more informative
  if (byPlot) {
    tOut <- unique(tOut)
    tOut <- tOut %>%
      dplyr::mutate(PLOT_STATUS = dplyr::case_when(is.na(PLOT_STATUS_CD) ~ NA_character_,
                                                   PLOT_STATUS_CD == 1 ~ 'Forest',
                                                   PLOT_STATUS_CD == 2 ~ 'Non-forest',
                                                   PLOT_STATUS_CD == 3 ~ 'Non-sampled')) %>%
      dplyr::relocate(PLOT_STATUS, .after = PLOT_STATUS_CD)
  }
  
  if (returnBetas) {
    tOut <- list(results = tOut, betas = betas)
  }
  
  return(tOut)
  
}
